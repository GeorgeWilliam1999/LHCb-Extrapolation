#!/usr/bin/env python
"""The exact scheme's own error, measured on THIS experiment's test states.

Why this script exists. `../Simple_first_pass/results/scheme_error_vs_q.csv`
already records the exact scheme's endpoint error per q, and that is the number
quoted as the ceiling. But it was measured on a different sample: 32 leg-B legs
drawn **stratified in momentum**, which deliberately over-weights the soft
tracks that bend hardest, on their own start planes rather than rebased to a
common one. The networks here are scored on the 2018 test states of the frozen
leg. Two medians over two populations are not a like-for-like comparison, and
the whole verdict turns on comparing them, so the scheme is solved again here
on the network's own states.

The solver is not reimplemented: `solve_leg` is imported from
`../Simple_first_pass/exact_scheme.py`, so the ceiling measured here and the
ceiling in that experiment come from the same code, character for character.

    PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python \\
        measure_scheme_ceiling.py [--q 8] [--n 256] [--split test]

Writes `results/scheme_ceiling_same_population_q<qq>.json`: the median and p95
endpoint error and the median stage-state error, all in microns and all under
the same measure `_shared/evaluate.py` scores the networks with - max(|dx|,
|dy|) against the same fp64 RK4 reference - plus the fraction of legs the
root-finder actually solved and the number of states used.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

import use_shared  # noqa: F401  (puts _shared on sys.path)
from _shared.reference import gauss_legendre

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIRST_PASS = os.path.join(HERE, "..", "Simple_first_pass")

# exact_scheme.py resolves its own imports off sys.path; give it what it needs
# and import the solver rather than copying it.
sys.path.insert(0, os.path.join(FIRST_PASS, "..", "Baseline_data_exploration"))
sys.path.insert(0, FIRST_PASS)
from exact_scheme import solve_leg  # noqa: E402


def measure(q, n, split, seed=20260718):
    npz = os.path.join(RESULTS, "frozen_leg_q%02d.npz" % q)
    d = np.load(npz, allow_pickle=False)
    S = d["%s_S" % split]
    ref = d["%s_ref" % split]           # (N, q+1, 5); the endpoint is the last
    z0, z1 = float(d["z0"]), float(d["z1"])
    tab = gauss_legendre(q)

    rng = np.random.default_rng(seed)
    idx = np.arange(len(S)) if n >= len(S) else np.sort(
        rng.permutation(len(S))[:n])

    # The error measure is the one `_shared/evaluate.py` scores the networks
    # with, and the one `../Simple_first_pass` aggregated its own scan with:
    # max(|dx|, |dy|) against the fp64 RK4 reference, in microns. Not a
    # Euclidean distance - mixing the two would silently shift the comparison.
    errs, stage_errs, ok, t0 = [], [], 0, time.time()
    for i in idx:
        stages, S1, conv, _, _ = solve_leg(S[i].copy(), z0, z1, tab)
        ok += int(conv)
        errs.append(np.abs(S1[:2] - ref[i, -1, :2]).max() * 1e3)
        stage_errs.append(np.abs(stages[:, :2] - ref[i, :q, :2]).max(axis=1) * 1e3)
    errs = np.array(errs)
    stage_errs = np.concatenate(stage_errs)

    out = {
        "q": q, "split": split, "n": int(len(idx)),
        "n_available": int(len(S)),
        "sample_seed": seed,
        "converged_frac": ok / len(idx),
        "endpoint_med_um": float(np.median(errs)),
        "endpoint_p95_um": float(np.percentile(errs, 95)),
        "endpoint_mean_um": float(errs.mean()),
        "stage_med_um": float(np.median(stage_errs)),
        "wall_s": round(time.time() - t0, 1),
        "solver": "solve_leg imported from ../Simple_first_pass/exact_scheme.py",
        "dataset": os.path.basename(npz),
    }
    path = os.path.join(RESULTS, "scheme_ceiling_same_population_q%02d.json" % q)
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("q=%-3d n=%-5d converged=%.3f  median=%.1f um  p95=%.1f um  (%.0f s)"
          % (q, out["n"], out["converged_frac"], out["endpoint_med_um"],
             out["endpoint_p95_um"], out["wall_s"]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--q", type=int, default=None)
    ap.add_argument("--n", type=int, default=256,
                    help="states to solve (a subsample; the root-finder is slow "
                         "at large q)")
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    qs = (a.q,) if a.q else (2, 4, 8, 16)
    for q in qs:
        measure(q, a.n, a.split)


if __name__ == "__main__":
    main()
