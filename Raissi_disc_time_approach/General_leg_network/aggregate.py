#!/usr/bin/env python
"""Collect the trained runs into two tables.

    results/summary.csv   one row per run: the whole-split numbers that
                          train.py already recorded, plus the run's settings
                          and whether it converged.
    results/by_leg.csv    one row per (run, split, leg type, momentum band):
                          the same score dict recomputed on that subset of the
                          split, together with the exact-scheme ceiling for
                          that leg at q = 8.
    results/stage_errors.csv  one row per (run, leg type, stage node): the
                          median position error at that Gauss node inside the
                          leg, so the error can be read along the step and not
                          only at its end.

The per-subset numbers have to be recomputed here: the json a run writes holds
only whole-split scores, so the network is reloaded from its checkpoint and
re-scored on the subsets with the shared scorer (`score_against_reference`),
which is the identical arithmetic.

The ceiling columns come from ../Simple_first_pass:
  ceiling_leg_um   the published whole-leg value at q = 8
                   (results/scheme_error_vs_q.csv, median_err_mm)
  ceiling_band_um  the same scan restricted to this momentum band
                   (results/scheme_scan.csv), measured the way our score is
                   measured: the larger of |dx| and |dy|.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"

import csv
import glob
import json

import numpy as np
import torch

import use_shared                                        # noqa: F401
from _shared.evaluate import predict, score_against_reference
from _shared.model import OneStepNetwork

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.environ.get("A3_RESULTS") or os.path.join(HERE, "results")
SCHEME = os.path.join(os.path.dirname(HERE), "Simple_first_pass", "results")
BANDS = (("1-5GeV", 1.0, 5.0), ("5-20GeV", 5.0, 20.0), ("20-200GeV", 20.0, 1e9))
SPLITS = ("val", "test")
SCORE_KEYS = ("endpoint_med_um", "endpoint_p95_um", "stage_med_um",
              "slope_med_mrad", "rho_mean", "rho_median", "straight_med_um", "n")


def leg_ceilings_um():
    """Whole-leg ceiling at q=8, and per-band ceilings, both in micrometres."""
    whole = {}
    with open(os.path.join(SCHEME, "scheme_error_vs_q.csv")) as f:
        for r in csv.DictReader(f):
            if int(r["q"]) == 8:
                whole[r["leg"]] = float(r["median_err_mm"]) * 1e3
    band = {}
    with open(os.path.join(SCHEME, "scheme_scan.csv")) as f:
        rows = [r for r in csv.DictReader(f) if int(r["q"]) == 8]
    for leg in "ABCD":
        for name, lo, hi in BANDS:
            e = [max(abs(float(r["err_x_mm"])), abs(float(r["err_y_mm"]))) * 1e3
                 for r in rows
                 if r["leg"] == leg and lo <= float(r["p_GeV"]) < hi]
            band[(leg, name)] = (float(np.median(e)) if e else float("nan"), len(e))
    return whole, band




def build_model(pt_path, data, width, depth):
    model = OneStepNetwork(int(data["q"]), data["in_scale"], data["out_scale"],
                           width=width, depth=depth, n_extra=2)
    model.load_state_dict(torch.load(pt_path, weights_only=True))
    model.eval()
    return model


def main():
    data = np.load(os.path.join(RESULTS, "general_legs.npz"))
    data = {k: data[k] for k in data.files}
    whole_ceiling, band_ceiling = leg_ceilings_um()

    runs = []
    for jf in sorted(glob.glob(os.path.join(RESULTS, "w*_s*.json"))):
        with open(jf) as f:
            runs.append((os.path.splitext(os.path.basename(jf))[0], json.load(f)))
    if not runs:
        raise SystemExit("no run json files in %s" % RESULTS)

    # ---- summary.csv -------------------------------------------------------
    srows = []
    for tag, j in runs:
        row = {"tag": tag, "mode": j["mode"], "seed": j["seed"],
               "width": j["width"], "depth": j["depth"],
               "restarts": j["restarts"], "final_loss": j["final_loss"],
               "converged": j["converged"], "wall_s": j["wall_s"]}
        for split in ("train", "val", "test"):
            sc = j.get(split) or j.get("scores", {}).get(split, {})
            for k in SCORE_KEYS:
                row["%s_%s" % (split, k)] = sc.get(k)
        srows.append(row)
    with open(os.path.join(RESULTS, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(srows[0].keys()))
        w.writeheader()
        w.writerows(srows)
    print("wrote summary.csv (%d runs)" % len(srows))

    # ---- by_leg.csv and stage_errors.csv -----------------------------------
    brows, strows = [], []
    for tag, j in runs:
        pt = os.path.join(RESULTS, tag + ".pt")
        if not os.path.exists(pt):
            print("  no checkpoint for %s, skipped" % tag)
            continue
        model = build_model(pt, data, j["width"], j["depth"])
        for split in SPLITS:
            S = np.asarray(data["%s_S" % split])
            ref = np.asarray(data["%s_ref" % split])
            dz = np.asarray(data["%s_dz" % split])
            extra = np.asarray(data["%s_extra" % split])
            L = np.asarray(data["%s_LEG" % split])
            P = np.asarray(data["%s_P" % split])
            out = predict(model, S, extra)
            if split == "test":
                node_err = np.abs(out[:, :, :2] - ref[:, :, :2]).max(axis=2) * 1e3
                cnodes = np.asarray(data["c"])
                for li, leg in enumerate("ABCD"):
                    m = L == li
                    if m.sum() < 5:
                        continue
                    for k in range(node_err.shape[1]):
                        last = k == node_err.shape[1] - 1
                        strows.append({
                            "tag": tag, "mode": j["mode"], "seed": j["seed"],
                            "width": j["width"], "depth": j["depth"],
                            "leg": leg, "node": k,
                            "node_c": 1.0 if last else float(cnodes[k]),
                            "is_endpoint": last,
                            "med_um": float(np.median(node_err[m, k])),
                            "p95_um": float(np.quantile(node_err[m, k], 0.95)),
                            "n": int(m.sum()),
                        })
            for li, leg in enumerate("ABCD"):
                for name, lo, hi in BANDS:
                    m = (L == li) & (P >= lo) & (P < hi)
                    if m.sum() < 5:
                        continue
                    sc = score_against_reference(out[m], S[m], ref[m], dz[m])
                    cb, cn = band_ceiling[(leg, name)]
                    brows.append({
                        "tag": tag, "mode": j["mode"], "seed": j["seed"],
                        "width": j["width"], "depth": j["depth"],
                        "converged": j["converged"], "split": split,
                        "leg": leg, "band": name,
                        **{k: sc[k] for k in SCORE_KEYS},
                        "ceiling_leg_um": whole_ceiling.get(leg),
                        "ceiling_band_um": cb, "ceiling_band_n": cn,
                    })
                # the leg as a whole, all momenta together
                m = L == li
                if m.sum() >= 5:
                    sc = score_against_reference(out[m], S[m], ref[m], dz[m])
                    brows.append({
                        "tag": tag, "mode": j["mode"], "seed": j["seed"],
                        "width": j["width"], "depth": j["depth"],
                        "converged": j["converged"], "split": split,
                        "leg": leg, "band": "all",
                        **{k: sc[k] for k in SCORE_KEYS},
                        "ceiling_leg_um": whole_ceiling.get(leg),
                        "ceiling_band_um": whole_ceiling.get(leg),
                        "ceiling_band_n": 32,
                    })
        print("  scored %s" % tag)
    with open(os.path.join(RESULTS, "by_leg.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(brows[0].keys()))
        w.writeheader()
        w.writerows(brows)
    print("wrote by_leg.csv (%d rows)" % len(brows))

    with open(os.path.join(RESULTS, "stage_errors.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(strows[0].keys()))
        w.writeheader()
        w.writerows(strows)
    print("wrote stage_errors.csv (%d rows)" % len(strows))


if __name__ == "__main__":
    main()
