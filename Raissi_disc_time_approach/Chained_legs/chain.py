#!/usr/bin/env python
"""Walk every trained network along the real particle paths, leg after leg.

For each network trained in ../General_leg_network the same procedure is run on
both the val and the test chains: start from the particle's own first state,
apply the network to the first leg, feed its predicted endpoint in as the start
state of the second leg, and so on to the end of the particle's path. The truth
is the fp64 RK4 path from the same start state through the same planes, so the
only thing being measured is the propagation.

The same script also runs the leg-D reproduction: the one giant backward step
from the first T station state to the primary vertex, walked instead as two
steps through the particle's own UT plane, scored against the stored D-leg
label.

Outputs
    results/chain_summary.csv  per (network, split, step index): how many
        particles are still in the chain at that step, the median and 95th
        percentile endpoint error, the median slope error and the median
        distance travelled so far.
    results/chains.csv         per-particle rows, as asked, for a fixed random
        subsample of %d test particles (the same particles for every network, so
        the networks can be compared particle by particle without carrying a
        million rows).
    results/selection.csv      the seed ranking: median chain error on val, the
        test number for the same seed, the one-step val error, and the rank
        correlations between them.
    results/leg_d_reproduction.csv  the composite-vs-one-giant-step comparison.
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
from _shared.evaluate import chain, chain_reference
from _shared.model import OneStepNetwork
from _shared.reference import rk4_rows

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
TRAINED = os.environ.get("A3_TRAINED") or os.path.join(
    os.path.dirname(HERE), "General_leg_network", "results")
N_PARTICLE_ROWS = 500
SUBSAMPLE_SEED = 20260905
__doc__ = __doc__ % N_PARTICLE_ROWS


def load_runs():
    runs = []
    for jf in sorted(glob.glob(os.path.join(TRAINED, "w*_s*.json"))):
        tag = os.path.splitext(os.path.basename(jf))[0]
        pt = os.path.join(TRAINED, tag + ".pt")
        if not os.path.exists(pt):
            continue
        with open(jf) as f:
            j = json.load(f)
        runs.append((tag, j, pt))
    return runs


def load_model(pt, j, data):
    m = OneStepNetwork(int(data["q"]), data["in_scale"], data["out_scale"],
                       width=j["width"], depth=j["depth"], n_extra=2)
    m.load_state_dict(torch.load(pt, weights_only=True))
    m.eval()
    return m


def legs_of(planes, n):
    """The n (z0, z1) pairs of a group of chains that all have n legs."""
    return [(planes[:, i], planes[:, i + 1]) for i in range(n)]


def group_chains(npz):
    """Split the chains into groups of equal length; the reference per group."""
    S0 = npz["S0"]
    planes = npz["planes"]
    n_legs = npz["n_legs"]
    P = npz["P"]
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


def main():
    data = np.load(os.path.join(TRAINED, "general_legs.npz"))
    data = {k: data[k] for k in data.files}
    em, es = data["extra_mean"], data["extra_scale"]

    chains = {s: dict(np.load(os.path.join(RESULTS, "chains_%s.npz" % s),
                              allow_pickle=True))
              for s in ("val", "test")}
    print("building the RK4 reference paths ...", flush=True)
    groups = {s: group_chains(chains[s]) for s in ("val", "test")}
    for s in groups:
        print("  %s: %d groups, %d particles"
              % (s, len(groups[s]), sum(len(g["S0"]) for g in groups[s])),
              flush=True)

    legd = dict(np.load(os.path.join(RESULTS, "leg_d_test.npz"),
                        allow_pickle=True))
    d_S0, d_zt = legd["S0"], legd["z_t"]
    d_zmid, d_zpv = legd["z_mid"], legd["z_pv"]
    d_label, d_P = legd["label"], legd["P"]
    d_ref_end = d_label[:, :4]                     # the stored D-leg label
    d_straight = (d_S0[:, :2] + d_S0[:, 2:4] * (d_zpv - d_zt)[:, None])
    d_straight_err = np.abs(d_straight - d_ref_end[:, :2]).max(axis=1) * 1e3
    # the label recomputed with the reference engine, as a cross-check
    d_rk4 = rk4_rows(d_S0, d_zt, d_zpv)
    d_label_vs_rk4 = float(np.median(
        np.abs(d_rk4[:, :2] - d_ref_end[:, :2]).max(axis=1) * 1e3))
    print("stored D label vs a fresh RK4 pass: median %.3f um" % d_label_vs_rk4)

    rng = np.random.default_rng(SUBSAMPLE_SEED)
    n_test = len(chains["test"]["S0"])
    keep_rows = np.zeros(n_test, dtype=bool)
    keep_rows[rng.permutation(n_test)[:N_PARTICLE_ROWS]] = True

    # A handful of example paths, kept in full so the growth can be drawn:
    # six particles from the longest chain group, spread across momentum.
    gbig = max(groups["test"], key=lambda g: g["n"])
    qs = np.quantile(gbig["P"], [0.05, 0.25, 0.45, 0.65, 0.85, 0.97])
    ex = sorted({int(np.argmin(np.abs(gbig["P"] - q))) for q in qs})
    ex_S0 = gbig["S0"][ex]
    ex_legs = [(z0[ex], z1[ex]) for z0, z1 in gbig["legs"]]
    ex_planes = np.concatenate(
        [ex_legs[0][0][:, None]] + [l[1][:, None] for l in ex_legs], axis=1)
    examples = {"ex_particle": np.asarray(gbig["idx"])[ex],
                "ex_n_legs": np.full(len(ex), gbig["n"]),
                "ex_P": gbig["P"][ex], "ex_S0": ex_S0,
                "ex_planes": ex_planes, "ex_ref": gbig["ref"][ex]}

    runs = load_runs()
    print("%d trained runs found" % len(runs), flush=True)
    summary, particle_rows, drows = [], [], []

    for tag, j, pt in runs:
        model = load_model(pt, j, data)
        base = {"network": tag, "mode": j["mode"], "seed": j["seed"],
                "width": j["width"], "depth": j["depth"],
                "converged": j["converged"]}
        for split in ("val", "test"):
            end_all, slope_all, cum_all, p_all, gid_all = {}, {}, {}, {}, {}
            for g in groups[split]:
                pred = chain(model, g["S0"], g["legs"], em, es)
                end, slope = errors(pred, g["ref"])
                for k in range(g["n"]):
                    end_all.setdefault(k, []).append(end[:, k])
                    slope_all.setdefault(k, []).append(slope[:, k])
                    cum_all.setdefault(k, []).append(g["cum_dz"][:, k])
                    p_all.setdefault(k, []).append(g["P"])
                    gid_all.setdefault(k, []).append(g["idx"])
                if split == "test":
                    sel = keep_rows[g["idx"]]
                    if sel.any():
                        for row_i, gi in enumerate(np.where(sel)[0]):
                            for k in range(g["n"]):
                                particle_rows.append({
                                    **base, "particle": int(g["idx"][gi]),
                                    "n_legs": g["n"], "step": k,
                                    "cum_dz_mm": float(g["cum_dz"][gi, k]),
                                    "p_GeV": float(g["P"][gi]),
                                    "end_err_um": float(end[gi, k]),
                                    "slope_err_mrad": float(slope[gi, k]),
                                })
            for k in sorted(end_all):
                e = np.concatenate(end_all[k])
                sl = np.concatenate(slope_all[k])
                cz = np.concatenate(cum_all[k])
                summary.append({
                    **base, "split": split, "step": k, "n": int(len(e)),
                    "end_med_um": float(np.median(e)),
                    "end_p95_um": float(np.quantile(e, 0.95)),
                    "slope_med_mrad": float(np.median(sl)),
                    "cum_dz_med_mm": float(np.median(cz)),
                })
        examples["pred_%s" % tag] = chain(model, ex_S0, ex_legs, em, es)
        # ---- leg D: two composite steps against the one giant step ----------
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
                "label_vs_rk4_med_um": d_label_vs_rk4,
            })
        print("  %-18s chain(test, leg 4) %8.1f um   leg-D composite %9.1f um"
              % (tag,
                 [r for r in summary if r["network"] == tag
                  and r["split"] == "test" and r["step"] == 3][0]["end_med_um"],
                 float(np.median(comp_err))), flush=True)

    def write(name, rows):
        with open(os.path.join(RESULTS, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("wrote %s (%d rows)" % (name, len(rows)))

    np.savez_compressed(os.path.join(RESULTS, "example_paths.npz"), **examples)
    print("wrote example_paths.npz (%d particles)" % len(examples["ex_P"]))
    write("chain_summary.csv", summary)
    write("chains.csv", particle_rows)
    write("leg_d_reproduction.csv", drows)
    selection(summary)


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
    """Rank the physics seeds on val chains; report what test says."""
    with open(os.path.join(TRAINED, "summary.csv")) as f:
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
                            if t in one_step else float("nan"))
            order = np.argsort(vals)
            for rank, i in enumerate(order):
                rows.append({
                    "width": width, "mode": mode, "network": tags[i],
                    "rank_by_val_chain": rank,
                    "chain_val_leg4_med_um": vals[i],
                    "chain_test_leg4_med_um": tests[i],
                    "one_step_val_med_um": ones[i],
                    "selected": rank == 0,
                })
            rows.append({
                "width": width, "mode": mode, "network": "SPEARMAN",
                "rank_by_val_chain": None,
                "chain_val_leg4_med_um": spearman(vals, tests),
                "chain_test_leg4_med_um": spearman(ones, tests),
                "one_step_val_med_um": spearman(ones, vals),
                "selected": ("columns: val-chain vs test-chain | one-step-val "
                             "vs test-chain | one-step-val vs val-chain"),
            })
    with open(os.path.join(RESULTS, "selection.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote selection.csv (%d rows)" % len(rows))


if __name__ == "__main__":
    main()
