#!/usr/bin/env python
"""Step 1 of the port: the exact Gauss-Legendre scheme on the LHCb ODE,
with no network in it.

For a leg (state at z0 -> z1) the q-stage implicit scheme defines stage states
Y_j at z0 + c_j*dz through

    Y_j = S0 + dz * sum_k a_jk f(Y_k, z0 + c_k*dz),        j = 1..q
    S1  = S0 + dz * sum_j b_j  f(Y_j, z0 + c_j*dz),

and we solve those equations directly with a root-finder. This maps, over
(q, leg type, momentum), two things the network step needs to know first:

  FEASIBILITY - does the implicit solve converge on our C0 (trilinear) field,
                including one giant step across the whole magnet?
  CEILING     - the exact scheme's endpoint error vs the fp64 RK4 reference:
                the best any network trained on these equations could do.

Details that differ from the van der Pol solver:
  - non-autonomous f(S, z); qop is held fixed (4 unknowns per stage);
  - residuals scaled to [mm, mm, mrad, mrad] so the mixed-unit system is
    well-conditioned for the solver;
  - initial guess = straight-line states at the nodes (uses only the input
    state, as the network would);
  - start states are REAL legs sampled from the event-derived training set
    (train split), stratified in momentum per leg type.

Run:  /data/bfys/gscriven/conda/envs/TE/bin/python exact_scheme.py   (~10 min)
Outputs: results/scheme_scan.csv (row per solve), results/scheme_error_vs_q.csv
"""
import csv
import os
import sys
import time

import numpy as np
from scipy.optimize import root

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "Baseline_data_exploration"))
from reference_card import deriv, load_training, rk4_rows  # noqa: E402
from irk import tableau, verify_tableau  # noqa: E402

RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

QS = (2, 4, 8, 16, 32)
N_PER_LEG = 32
SEED = 20260718
SCALE = np.array([1.0, 1.0, 1e-3, 1e-3])  # mm, mm, rad->mrad-equivalent
LEG_NAME = {0: "A", 1: "B", 2: "C", 3: "D"}


def solve_leg(S0, z0, z1, tab, maxfev=20000):
    """Solve the q-stage implicit equations for one leg. Returns
    (stages(q,5), S1(5), converged, n_evals, residual_inf)."""
    c, A, b = tab
    q = len(c)
    dz = z1 - z0
    znodes = z0 + c * dz
    qop = S0[4]

    n_eval = [0]

    def rates(Y4):
        n_eval[0] += 1
        S = np.column_stack([Y4, np.full(q, qop)])
        return deriv(S, znodes)[:, :4]

    def residual(v):
        Y4 = v.reshape(q, 4)
        R = Y4 - S0[None, :4] - dz * (A @ rates(Y4))
        return (R / SCALE[None, :]).ravel()

    # initial guess: straight line from the input state
    guess = np.column_stack([
        S0[0] + S0[2] * (znodes - z0),
        S0[1] + S0[3] * (znodes - z0),
        np.full(q, S0[2]),
        np.full(q, S0[3]),
    ])
    sol = root(residual, guess.ravel(), method="hybr",
               options=dict(maxfev=maxfev), tol=1e-12)
    Y4 = sol.x.reshape(q, 4)
    res = np.abs(residual(sol.x)).max()
    S1 = S0.copy()
    S1[:4] = S0[:4] + dz * (b @ rates(Y4))
    stages = np.column_stack([Y4, np.full(q, qop)])
    return stages, S1, res < 1e-8, n_eval[0], res


def stratified_rows(d, leg, n, rng):
    idx = np.where(d["LEG"] == leg)[0]
    order = idx[np.argsort(d["P"][idx])]
    take = order[np.linspace(0, len(order) - 1, n).astype(int)]
    return rng.permutation(take)


def main():
    for q in QS:
        assert verify_tableau(q)["passes"], f"tableau q={q} failed verification"

    d = load_training(split="train")
    rng = np.random.default_rng(SEED)
    tabs = {q: tableau(q) for q in QS}

    rows = []
    t0 = time.time()
    for leg in range(4):
        sel = stratified_rows(d, leg, N_PER_LEG, rng)
        X = d["X"][sel].astype(np.float64)
        ref = rk4_rows(X[:, :5], X[:, 5], X[:, 6])  # fp64 reference, recomputed
        for q in QS:
            for i, r in enumerate(sel):
                S0, z0, z1 = X[i, :5].copy(), X[i, 5], X[i, 6]
                t1 = time.time()
                _, S1, ok, nev, res = solve_leg(S0, z0, z1, tabs[q])
                rows.append({
                    "leg": LEG_NAME[leg], "q": q, "p_GeV": float(d["P"][r]),
                    "dz_mm": float(z1 - z0), "converged": int(ok),
                    "n_evals": nev, "residual": res,
                    "err_x_mm": abs(S1[0] - ref[i, 0]),
                    "err_y_mm": abs(S1[1] - ref[i, 1]),
                    "err_tx": abs(S1[2] - ref[i, 2]),
                    "err_ty": abs(S1[3] - ref[i, 3]),
                    "wall_s": time.time() - t1,
                })
            done = sum(1 for x in rows)
            print("leg %s q %2d done (%d solves, %.0f s elapsed)"
                  % (LEG_NAME[leg], q, done, time.time() - t0), flush=True)

    with open(os.path.join(RES, "scheme_scan.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    # aggregate: per (leg, q) over converged solves
    agg = []
    for leg in "ABCD":
        for q in QS:
            sub = [r for r in rows if r["leg"] == leg and r["q"] == q]
            conv = [r for r in sub if r["converged"]]
            errs = sorted(max(r["err_x_mm"], r["err_y_mm"]) for r in conv) or [float("nan")]
            agg.append({
                "leg": leg, "q": q, "n": len(sub), "converged_frac": len(conv) / len(sub),
                "median_err_mm": errs[len(errs) // 2],
                "worst_err_mm": errs[-1],
                "median_nev": int(np.median([r["n_evals"] for r in sub])),
            })
    with open(os.path.join(RES, "scheme_error_vs_q.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=agg[0].keys())
        w.writeheader()
        w.writerows(agg)
    for a in agg:
        print(a)


if __name__ == "__main__":
    main()
