#!/usr/bin/env python
"""Score the residual arm, and put wave 1 beside it in the same tables.

This is `aggregate.py` for the straight-line-residual runs. It cannot simply be
that script with a different glob, because the two arms are different models
(`residual_model.ResidualOneStepNetwork` against `_shared.model.OneStepNetwork`)
reading different datasets (`general_legs_residual.npz`, which is
`general_legs.npz` plus the scale arrays, against `general_legs.npz` itself).
It is a new file rather than an edit to `aggregate.py` so that nothing the
wave-1 analysis depends on moves under it.

The population, the splits, the momentum bands, the score dict and the ceiling
columns are aggregate.py's, unchanged - the two arms are scored by the same
arithmetic on the same states, which is the whole point.

Every row carries an `arm` column:

    residual   tags `residual_w<width>_<mode>_s<seed>`
    wave1      tags `w<width>_<mode>_s<seed>` - whatever wave-1 runs are on
               disk at the time this is run, so a 4x200 wave that is still
               draining simply contributes fewer seeds

Outputs
    results/residual_summary.csv        one row per run, both arms
    results/by_leg_residual.csv         one row per (run, split, leg, band)
    results/stage_errors_residual.csv   one row per (run, leg, stage node)
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
from residual_model import build_residual_model

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


def own_ceilings():
    path = os.path.join(RESULTS, "scheme_ceiling_same_population.csv")
    if not os.path.exists(path):
        print("  (no scheme_ceiling_same_population.csv; run measure_ceiling.py)")
        return {}
    with open(path) as f:
        return {(r["leg"], r["band"]): (float(r["ceiling_med_um"]),
                                        float(r["ceiling_p95_um"]))
                for r in csv.DictReader(f)}


def load_runs():
    """(arm, tag, json, checkpoint) for every finished run of either arm."""
    runs = []
    for pattern, arm in (("residual_w*_s*.json", "residual"),
                         ("w*_s*.json", "wave1")):
        for jf in sorted(glob.glob(os.path.join(RESULTS, pattern))):
            tag = os.path.splitext(os.path.basename(jf))[0]
            pt = os.path.join(RESULTS, tag + ".pt")
            if not os.path.exists(pt):
                continue
            with open(jf) as f:
                runs.append((arm, tag, json.load(f), pt))
    return runs


def main():
    dres = np.load(os.path.join(RESULTS, "general_legs_residual.npz"))
    dres = {k: dres[k] for k in dres.files}
    dw1 = np.load(os.path.join(RESULTS, "general_legs.npz"))
    dw1 = {k: dw1[k] for k in dw1.files}
    profile = str(dres["residual_node_profile"])
    whole_ceiling, band_ceiling = leg_ceilings_um()
    own = own_ceilings()

    runs = load_runs()
    if not any(a == "residual" for a, *_ in runs):
        raise SystemExit("no residual run json files in %s" % RESULTS)
    print("%d runs (%d residual, %d wave 1)"
          % (len(runs), sum(a == "residual" for a, *_ in runs),
             sum(a == "wave1" for a, *_ in runs)))

    srows, brows, strows = [], [], []
    for arm, tag, j, pt in runs:
        data = dres if arm == "residual" else dw1
        if arm == "residual":
            model = build_residual_model(data, j["width"], j["depth"],
                                         node_profile=profile)
        else:
            model = OneStepNetwork(int(data["q"]), data["in_scale"],
                                   data["out_scale"], width=j["width"],
                                   depth=j["depth"], n_extra=2)
        model.load_state_dict(torch.load(pt, weights_only=True))
        model.eval()

        row = {"arm": arm, "tag": tag, "mode": j["mode"], "seed": j["seed"],
               "width": j["width"], "depth": j["depth"],
               "restarts": j["restarts"], "final_loss": j["final_loss"],
               "converged": j["converged"], "wall_s": j["wall_s"]}
        for split in ("train", "val", "test"):
            sc = j.get(split) or {}
            for k in SCORE_KEYS:
                row["%s_%s" % (split, k)] = sc.get(k)
        srows.append(row)

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
                            "arm": arm, "tag": tag, "mode": j["mode"],
                            "seed": j["seed"], "width": j["width"],
                            "depth": j["depth"], "leg": leg, "node": k,
                            "node_c": 1.0 if last else float(cnodes[k]),
                            "is_endpoint": last,
                            "med_um": float(np.median(node_err[m, k])),
                            "p95_um": float(np.quantile(node_err[m, k], 0.95)),
                            "n": int(m.sum())})
            for li, leg in enumerate("ABCD"):
                cells = [(name, lo, hi) for name, lo, hi in BANDS]
                for name, lo, hi in cells:
                    m = (L == li) & (P >= lo) & (P < hi)
                    if m.sum() < 5:
                        continue
                    sc = score_against_reference(out[m], S[m], ref[m], dz[m])
                    cb, cn = band_ceiling[(leg, name)]
                    brows.append({
                        "arm": arm, "tag": tag, "mode": j["mode"],
                        "seed": j["seed"], "width": j["width"],
                        "depth": j["depth"], "converged": j["converged"],
                        "split": split, "leg": leg, "band": name,
                        **{k: sc[k] for k in SCORE_KEYS},
                        "ceiling_leg_um": whole_ceiling.get(leg),
                        "ceiling_band_um": cb, "ceiling_band_n": cn,
                        "ceiling_own_um": own.get((leg, name), (None,))[0],
                        "ceiling_own_p95_um": own.get((leg, name), (None, None))[1]
                        if (leg, name) in own else None})
                m = L == li
                if m.sum() >= 5:
                    sc = score_against_reference(out[m], S[m], ref[m], dz[m])
                    brows.append({
                        "arm": arm, "tag": tag, "mode": j["mode"],
                        "seed": j["seed"], "width": j["width"],
                        "depth": j["depth"], "converged": j["converged"],
                        "split": split, "leg": leg, "band": "all",
                        **{k: sc[k] for k in SCORE_KEYS},
                        "ceiling_leg_um": whole_ceiling.get(leg),
                        "ceiling_band_um": whole_ceiling.get(leg),
                        "ceiling_band_n": 32,
                        "ceiling_own_um": own.get((leg, "all"), (None,))[0],
                        "ceiling_own_p95_um": own.get((leg, "all"), (None, None))[1]
                        if (leg, "all") in own else None})
        print("  scored %-28s (%s)" % (tag, arm), flush=True)

    def write(name, rows):
        with open(os.path.join(RESULTS, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("wrote %s (%d rows)" % (name, len(rows)))

    write("residual_summary.csv", srows)
    write("by_leg_residual.csv", brows)
    write("stage_errors_residual.csv", strows)


if __name__ == "__main__":
    main()
