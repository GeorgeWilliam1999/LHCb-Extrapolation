#!/usr/bin/env python
"""Walk the straight-line-residual networks along the same real particle paths.

`chain.py` for the residual arm. The chains, the leg-D set, the RK4 references,
the 500-particle subsample and its seed, the seed-selection rule and every
metric are that script's, unchanged - the only difference is which networks are
loaded and from which dataset:

    ../General_leg_network/results/residual_w*_s*.json  +  .pt
    ../General_leg_network/results/general_legs_residual.npz

It is a new file rather than a flag on `chain.py` because the two arms need
different model classes, and nothing the wave-1 chaining analysis depends on
should move while it is being written up.

`_shared.evaluate.chain` needs no change at all: it calls `model(S, extra)` with
the per-leg normalised (z0, dz), and the residual network recovers the physical
leg from exactly those two numbers and computes its own scale from the field.
That is what makes the leg-D reproduction meaningful here - the D geometry is in
no dataset, so the scale has to be derived from the inputs, and it is.

Outputs
    results/chain_summary_residual.csv
    results/chains_residual.csv
    results/selection_residual.csv
    results/leg_d_residual.csv
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"

import argparse
import csv
import glob
import json
import sys

import numpy as np
import torch

import use_shared                                        # noqa: F401
from _shared.evaluate import chain, chain_reference
from _shared.reference import rk4_rows

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.environ.get("A3_CHAIN_OUT") or os.path.join(HERE, "results")
CHAINS = os.path.join(HERE, "results")
TRAINED = os.environ.get("A3_TRAINED") or os.path.join(
    os.path.dirname(HERE), "General_leg_network", "results")
GENERAL = os.path.join(os.path.dirname(HERE), "General_leg_network")
if GENERAL not in sys.path:
    sys.path.insert(0, GENERAL)
from residual_model import build_residual_model                  # noqa: E402

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

N_PARTICLE_ROWS = 500
SUBSAMPLE_SEED = 20260905


def load_runs():
    runs = []
    for jf in sorted(glob.glob(os.path.join(TRAINED, "residual_w*_s*.json"))):
        tag = os.path.splitext(os.path.basename(jf))[0]
        pt = os.path.join(TRAINED, tag + ".pt")
        if not os.path.exists(pt):
            continue
        with open(jf) as f:
            runs.append((tag, json.load(f), pt))
    return runs


def load_model(pt, j, data, profile):
    m = build_residual_model(data, j["width"], j["depth"], node_profile=profile)
    m.load_state_dict(torch.load(pt, weights_only=True))
    m.eval()
    return m


def legs_of(planes, n):
    return [(planes[:, i], planes[:, i + 1]) for i in range(n)]


def group_chains(npz):
    S0, planes, n_legs, P = npz["S0"], npz["planes"], npz["n_legs"], npz["P"]
    groups = []
    for n in sorted(set(n_legs.tolist())):
        m = n_legs == n
        legs = legs_of(planes[m], n)
        ref = chain_reference(S0[m], legs)
        groups.append({"n": int(n), "mask": m, "S0": S0[m], "legs": legs,
                       "ref": ref, "P": P[m],
                       "cum_dz": np.cumsum(np.diff(planes[m, :n + 1], axis=1),
                                           axis=1),
                       "idx": np.where(m)[0]})
    return groups


def errors(pred, ref):
    end = np.abs(pred[:, :, :2] - ref[:, :, :2]).max(axis=2) * 1e3      # um
    slope = np.abs(pred[:, :, 2:4] - ref[:, :, 2:4]).max(axis=2) * 1e3  # mrad
    return end, slope


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def selection(summary):
    one_step = {}
    path = os.path.join(TRAINED, "residual_summary.csv")
    if os.path.exists(path):
        with open(path) as f:
            one_step = {r["tag"]: r for r in csv.DictReader(f)}
    rows = []
    for width in sorted({r["width"] for r in summary}):
        for mode in ("physics", "data"):
            sel = [r for r in summary if r["width"] == width
                   and r["mode"] == mode and r["step"] == 3]
            tags = sorted({r["network"] for r in sel})
            if not tags:
                continue
            vals, tests, ones = [], [], []
            for t in tags:
                v = [r for r in sel if r["network"] == t and r["split"] == "val"][0]
                te = [r for r in sel if r["network"] == t and r["split"] == "test"][0]
                vals.append(v["end_med_um"]); tests.append(te["end_med_um"])
                ones.append(float(one_step[t]["val_endpoint_med_um"])
                            if t in one_step and one_step[t]["val_endpoint_med_um"]
                            else float("nan"))
            order = np.argsort(vals)
            for rank, i in enumerate(order):
                rows.append({
                    "arm": "residual", "width": width, "mode": mode,
                    "network": tags[i], "rank_by_val_chain": rank,
                    "chain_val_leg4_med_um": vals[i],
                    "chain_test_leg4_med_um": tests[i],
                    "one_step_val_med_um": ones[i],
                    "selected": rank == 0})
            rows.append({
                "arm": "residual", "width": width, "mode": mode,
                "network": "SPEARMAN", "rank_by_val_chain": None,
                "chain_val_leg4_med_um": spearman(vals, tests),
                "chain_test_leg4_med_um": spearman(ones, tests),
                "one_step_val_med_um": spearman(ones, vals),
                "selected": ("columns: val-chain vs test-chain | one-step-val "
                             "vs test-chain | one-step-val vs val-chain")})
    with open(os.path.join(RESULTS, "selection_residual.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote selection_residual.csv (%d rows)" % len(rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, default=None,
                    help="use only the first N particles of each set; for "
                         "exercising the pipeline, never for a result")
    a = ap.parse_args()

    data = np.load(os.path.join(TRAINED, "general_legs_residual.npz"))
    data = {k: data[k] for k in data.files}
    em, es = data["extra_mean"], data["extra_scale"]
    profile = str(data["residual_node_profile"])

    chains = {s: dict(np.load(os.path.join(CHAINS, "chains_%s.npz" % s),
                              allow_pickle=True))
              for s in ("val", "test")}
    if a.limit:
        chains = {s: {k: v[:a.limit] for k, v in d.items()}
                  for s, d in chains.items()}
    print("building the RK4 reference paths ...", flush=True)
    groups = {s: group_chains(chains[s]) for s in ("val", "test")}
    for s in groups:
        print("  %s: %d groups, %d particles"
              % (s, len(groups[s]), sum(len(g["S0"]) for g in groups[s])),
              flush=True)

    legd = dict(np.load(os.path.join(CHAINS, "leg_d_test.npz"),
                        allow_pickle=True))
    if a.limit:
        legd = {k: v[:a.limit] for k, v in legd.items()}
    d_S0, d_zt = legd["S0"], legd["z_t"]
    d_zmid, d_zpv = legd["z_mid"], legd["z_pv"]
    d_label = legd["label"]
    d_ref_end = d_label[:, :4]
    d_straight = (d_S0[:, :2] + d_S0[:, 2:4] * (d_zpv - d_zt)[:, None])
    d_straight_err = np.abs(d_straight - d_ref_end[:, :2]).max(axis=1) * 1e3
    d_rk4 = rk4_rows(d_S0, d_zt, d_zpv)
    d_label_vs_rk4 = float(np.median(
        np.abs(d_rk4[:, :2] - d_ref_end[:, :2]).max(axis=1) * 1e3))
    print("stored D label vs a fresh RK4 pass: median %.3f um" % d_label_vs_rk4)

    rng = np.random.default_rng(SUBSAMPLE_SEED)
    n_test = len(chains["test"]["S0"])
    keep_rows = np.zeros(n_test, dtype=bool)
    keep_rows[rng.permutation(n_test)[:N_PARTICLE_ROWS]] = True

    runs = load_runs()
    print("%d residual runs found" % len(runs), flush=True)
    if not runs:
        raise SystemExit("no residual checkpoints in %s" % TRAINED)
    summary, particle_rows, drows = [], [], []

    for tag, j, pt in runs:
        model = load_model(pt, j, data, profile)
        base = {"arm": "residual", "network": tag, "mode": j["mode"],
                "seed": j["seed"], "width": j["width"], "depth": j["depth"],
                "converged": j["converged"]}
        for split in ("val", "test"):
            end_all, slope_all, cum_all = {}, {}, {}
            for g in groups[split]:
                pred = chain(model, g["S0"], g["legs"], em, es)
                end, slope = errors(pred, g["ref"])
                for k in range(g["n"]):
                    end_all.setdefault(k, []).append(end[:, k])
                    slope_all.setdefault(k, []).append(slope[:, k])
                    cum_all.setdefault(k, []).append(g["cum_dz"][:, k])
                if split == "test":
                    sel = keep_rows[g["idx"]]
                    if sel.any():
                        for gi in np.where(sel)[0]:
                            for k in range(g["n"]):
                                particle_rows.append({
                                    **base, "particle": int(g["idx"][gi]),
                                    "n_legs": g["n"], "step": k,
                                    "cum_dz_mm": float(g["cum_dz"][gi, k]),
                                    "p_GeV": float(g["P"][gi]),
                                    "end_err_um": float(end[gi, k]),
                                    "slope_err_mrad": float(slope[gi, k])})
            for k in sorted(end_all):
                e = np.concatenate(end_all[k])
                sl = np.concatenate(slope_all[k])
                cz = np.concatenate(cum_all[k])
                summary.append({
                    **base, "split": split, "step": k, "n": int(len(e)),
                    "end_med_um": float(np.median(e)),
                    "end_p95_um": float(np.quantile(e, 0.95)),
                    "slope_med_mrad": float(np.median(sl)),
                    "cum_dz_med_mm": float(np.median(cz))})
        # ---- leg D --------------------------------------------------------
        legs_d = [(d_zt, d_zmid), (d_zmid, d_zpv)]
        pred_d = chain(model, d_S0, legs_d, em, es)
        one = chain(model, d_S0, [(d_zt, d_zpv)], em, es)
        comp_err = np.abs(pred_d[:, -1, :2] - d_ref_end[:, :2]).max(axis=1) * 1e3
        one_err = np.abs(one[:, -1, :2] - d_ref_end[:, :2]).max(axis=1) * 1e3
        comp_sl = np.abs(pred_d[:, -1, 2:4] - d_ref_end[:, 2:4]).max(axis=1) * 1e3
        mid_ref = rk4_rows(d_S0, d_zt, d_zmid)
        mid_err = np.abs(pred_d[:, 0, :2] - mid_ref[:, :2]).max(axis=1) * 1e3
        for name, arr in (("composite_two_steps", comp_err),
                          ("single_giant_step", one_err)):
            drows.append({
                **base, "variant": name, "n": int(len(arr)),
                "med_um": float(np.median(arr)),
                "p95_um": float(np.quantile(arr, 0.95)),
                "slope_med_mrad": float(np.median(comp_sl)) if name.startswith("comp") else None,
                "first_step_med_um": float(np.median(mid_err)) if name.startswith("comp") else None,
                "straight_med_um": float(np.median(d_straight_err)),
                "label_vs_rk4_med_um": d_label_vs_rk4})
        print("  %-28s chain(test, leg 4) %9.1f um   leg-D one step %9.1f um"
              % (tag,
                 [r for r in summary if r["network"] == tag
                  and r["split"] == "test" and r["step"] == 3][0]["end_med_um"],
                 float(np.median(one_err))), flush=True)

    os.makedirs(RESULTS, exist_ok=True)

    def write(name, rows):
        with open(os.path.join(RESULTS, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("wrote %s (%d rows)" % (name, len(rows)))

    write("chain_summary_residual.csv", summary)
    write("chains_residual.csv", particle_rows)
    write("leg_d_residual.csv", drows)
    selection(summary)


if __name__ == "__main__":
    main()
