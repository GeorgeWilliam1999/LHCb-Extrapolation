#!/usr/bin/env python
"""The exact scheme's own error, measured on THIS experiment's test states.

`../Simple_first_pass/results/scheme_error_vs_q.csv` already records the exact
scheme's endpoint error per (leg, q), and those are the numbers quoted as the
ceiling. They were measured on a different population, though: 32 legs per type
drawn **stratified in momentum**, which deliberately over-weights the soft
tracks that bend hardest. The networks here are scored on the natural mix of
this experiment's test split. Two medians over two populations are not a
like-for-like comparison, and the verdict turns on exactly that comparison, so
the scheme is solved again here on the network's own states, leg type by leg
type and momentum band by momentum band.

The solver is not reimplemented: `solve_leg` is imported from
`../Simple_first_pass/exact_scheme.py`, so this ceiling and that one come from
the same code, character for character. The error measure is the one the
networks are scored with: max(|dx|, |dy|) against the fp64 RK4 reference.

    PYTHONNOUSERSITE=1 .../python measure_ceiling.py [--q 8] [--n 120]

Writes results/scheme_ceiling_same_population.csv.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"

import argparse
import csv
import sys
import time

import numpy as np

import use_shared                                        # noqa: F401
from _shared.reference import gauss_legendre

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.environ.get("A3_RESULTS") or os.path.join(HERE, "results")
FIRST_PASS = os.path.join(os.path.dirname(HERE), "Simple_first_pass")
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "Baseline_data_exploration"))
sys.path.insert(0, FIRST_PASS)
from exact_scheme import solve_leg                        # noqa: E402

BANDS = (("1-5GeV", 1.0, 5.0), ("5-20GeV", 5.0, 20.0), ("20-200GeV", 20.0, 1e9),
         ("all", 1.0, 1e9))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--q", type=int, default=8)
    ap.add_argument("--n", type=int, default=120,
                    help="states solved per (leg, band) cell")
    ap.add_argument("--split", default="test")
    ap.add_argument("--seed", type=int, default=20260905)
    a = ap.parse_args()

    d = np.load(os.path.join(RESULTS, "general_legs.npz"))
    S = d["%s_S" % a.split]
    ref = d["%s_ref" % a.split]
    z0 = d["%s_z0" % a.split]
    dz = d["%s_dz" % a.split]
    L = d["%s_LEG" % a.split]
    P = d["%s_P" % a.split]
    tab = gauss_legendre(a.q)
    rng = np.random.default_rng(a.seed)

    rows = []
    for li, leg in enumerate("ABC"):
        for name, lo, hi in BANDS:
            cell = np.where((L == li) & (P >= lo) & (P < hi))[0]
            if len(cell) == 0:
                continue
            idx = cell if len(cell) <= a.n else np.sort(
                rng.permutation(cell)[:a.n])
            errs, stage, ok, t0 = [], [], 0, time.time()
            for i in idx:
                stages, S1, conv, _, _ = solve_leg(
                    S[i].copy(), float(z0[i]), float(z0[i] + dz[i]), tab)
                ok += int(conv)
                errs.append(np.abs(S1[:2] - ref[i, -1, :2]).max() * 1e3)
                stage.append(np.abs(stages[:, :2]
                                    - ref[i, :a.q, :2]).max(axis=1).max() * 1e3)
            errs = np.asarray(errs)
            rows.append({
                "leg": leg, "band": name, "q": a.q, "split": a.split,
                "n": int(len(idx)), "n_available": int(len(cell)),
                "converged_frac": ok / len(idx),
                "ceiling_med_um": float(np.median(errs)),
                "ceiling_p95_um": float(np.quantile(errs, 0.95)),
                "ceiling_stage_med_um": float(np.median(stage)),
                "wall_s": round(time.time() - t0, 1),
            })
            print("  %s %-10s n=%-4d converged=%.2f  median %.4g um  p95 %.4g um"
                  % (leg, name, len(idx), rows[-1]["converged_frac"],
                     rows[-1]["ceiling_med_um"], rows[-1]["ceiling_p95_um"]),
                  flush=True)
    path = os.path.join(RESULTS, "scheme_ceiling_same_population.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", os.path.relpath(path, HERE))


if __name__ == "__main__":
    main()
