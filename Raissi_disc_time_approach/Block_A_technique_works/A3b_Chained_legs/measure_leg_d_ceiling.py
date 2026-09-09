#!/usr/bin/env python
"""The exact scheme's own error on the leg-D population used here.

The published leg-D ceilings (0.74 mm at q = 8, 92 um at q = 16) come from
../../Block_0_first_pass/S1_Simple_first_pass, measured on 32 D legs drawn stratified in momentum. The
composite test in this experiment is scored on the 2,726 test particles that
have a D leg, whose momentum mix is the natural one. This script solves the same
exact scheme, with the same solver (`solve_leg` imported from
../../Block_0_first_pass/S1_Simple_first_pass/exact_scheme.py), on those very states, so that "the
network beats the one-giant-step ceiling" is a like-for-like statement.

Writes results/leg_d_ceiling_same_population.csv (one row per q and momentum
band): the median and 95th percentile endpoint error against the stored D-leg
label, in microns, measured as max(|dx|, |dy|) - the measure the networks are
scored with.
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
RESULTS = os.path.join(HERE, "results")
FIRST_PASS = os.path.join(os.path.dirname(os.path.dirname(HERE)),
                    "Block_0_first_pass", "S1_Simple_first_pass")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)),
                    "Block_0_first_pass", "S0_Baseline_data_exploration"))
sys.path.insert(0, FIRST_PASS)
from exact_scheme import solve_leg                        # noqa: E402

BANDS = (("1-5GeV", 1.0, 5.0), ("5-20GeV", 5.0, 20.0), ("20-200GeV", 20.0, 1e9),
         ("all", 1.0, 1e9))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--q", type=int, nargs="+", default=[8, 16])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260905)
    a = ap.parse_args()

    d = np.load(os.path.join(RESULTS, "leg_d_test.npz"), allow_pickle=True)
    S0, z_t, z_pv = d["S0"], d["z_t"], d["z_pv"]
    label, P = d["label"], d["P"]
    rng = np.random.default_rng(a.seed)

    rows = []
    for q in a.q:
        tab = gauss_legendre(q)
        for name, lo, hi in BANDS:
            cell = np.where((P >= lo) & (P < hi))[0]
            if len(cell) == 0:
                continue
            idx = cell if len(cell) <= a.n else np.sort(rng.permutation(cell)[:a.n])
            errs, ok, t0 = [], 0, time.time()
            for i in idx:
                _, S1, conv, _, _ = solve_leg(S0[i].copy(), float(z_t[i]),
                                              float(z_pv[i]), tab)
                ok += int(conv)
                errs.append(np.abs(S1[:2] - label[i, :2]).max() * 1e3)
            errs = np.asarray(errs)
            rows.append({"leg": "D", "q": q, "band": name, "n": int(len(idx)),
                         "n_available": int(len(cell)),
                         "converged_frac": ok / len(idx),
                         "ceiling_med_um": float(np.median(errs)),
                         "ceiling_p95_um": float(np.quantile(errs, 0.95)),
                         "wall_s": round(time.time() - t0, 1)})
            print("  q=%-3d %-10s n=%-4d converged=%.2f  median %.4g um  "
                  "p95 %.4g um" % (q, name, len(idx), rows[-1]["converged_frac"],
                                   rows[-1]["ceiling_med_um"],
                                   rows[-1]["ceiling_p95_um"]), flush=True)
    path = os.path.join(RESULTS, "leg_d_ceiling_same_population.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", os.path.relpath(path, HERE))


if __name__ == "__main__":
    main()
