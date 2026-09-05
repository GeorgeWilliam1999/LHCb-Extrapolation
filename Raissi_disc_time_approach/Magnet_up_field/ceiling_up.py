#!/usr/bin/env python
"""A4.4 - the exact-scheme ceiling on the magnet-up field.

The q-stage Gauss-Legendre scheme has its own discretisation error, independent
of any network: solve its stage equations exactly with a root-finder and
compare the endpoint with the fp64 RK4 reference. That error is the floor under
anything a network trained on those same equations can achieve, so the up-field
answer is what the up-field trainings have to be judged against.

The solver is `../Simple_first_pass/exact_scheme.py` (Task-1 result: leg B,
q = 8, 29 um median on MagDown), copied here in its minimal form because that
file reaches into `Baseline_data_exploration/reference_card.py`, whose `deriv`
is hard-wired to the MagDown map and takes no field argument. The arithmetic is
unchanged - same residual scaling, same straight-line initial guess, same
`scipy.optimize.root(method='hybr', tol=1e-12)`, same convergence test
(residual < 1e-8) - and the only edit is that `deriv` is called with an
explicit field so that both polarities can be measured in one process.

Both polarities are run on the SAME 32 momentum-stratified leg-B start states,
so the up/down comparison has nothing in it but the field.

Two populations, because they answer different questions and give different
numbers:

  stratified  32 leg-B legs drawn stratified in momentum, on their own start
              planes - the population `../Simple_first_pass` used, which
              deliberately over-weights the soft tracks that bend hardest;
  test-states the frozen leg's own test split, the exact states the networks
              here are scored on - the like-for-like comparator for the
              network floor. `../Stage_count_sweep/measure_scheme_ceiling.py`
              measured the MagDown value this way (22.5 um at q = 8) and this
              script reproduces it as a cross-check alongside the MagUp one.

    results/ceiling_up.csv        one row per stratified solve, both polarities
    results/ceiling_test_states.csv   one row per test-split solve, both
    results/ceiling_summary.json  both populations, both polarities
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import csv    # noqa: E402
import json   # noqa: E402
import time   # noqa: E402

import numpy as np                # noqa: E402
from scipy.optimize import root   # noqa: E402

import use_shared                 # noqa: F401,E402
from _shared.irk import tableau, verify_tableau                # noqa: E402
from _shared.reference import (deriv, load_training,           # noqa: E402
                               make_field, rk4_rows)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

Q = 8
N_STATES = 32
SEED = 20260718
SCALE = np.array([1.0, 1.0, 1e-3, 1e-3])   # mm, mm, mrad-equivalent


def solve_leg(S0, z0, z1, tab, field, maxfev=20000):
    """One leg's implicit stage equations. Verbatim from Simple_first_pass,
    except that `deriv` is given the field explicitly."""
    c, A, b = tab
    q = len(c)
    dz = z1 - z0
    znodes = z0 + c * dz
    qop = S0[4]
    n_eval = [0]

    def rates(Y4):
        n_eval[0] += 1
        S = np.column_stack([Y4, np.full(q, qop)])
        return deriv(S, znodes, field)[:, :4]

    def residual(v):
        Y4 = v.reshape(q, 4)
        R = Y4 - S0[None, :4] - dz * (A @ rates(Y4))
        return (R / SCALE[None, :]).ravel()

    guess = np.column_stack([
        S0[0] + S0[2] * (znodes - z0),
        S0[1] + S0[3] * (znodes - z0),
        np.full(q, S0[2]),
        np.full(q, S0[3]),
    ])
    sol = root(residual, guess.ravel(), method="hybr",
               options=dict(maxfev=maxfev), tol=1e-12)
    Y4 = sol.x.reshape(q, 4)
    res = float(np.abs(residual(sol.x)).max())
    S1 = S0.copy()
    S1[:4] = S0[:4] + dz * (b @ rates(Y4))
    stages = np.column_stack([Y4, np.full(q, qop)])
    return stages, S1, res < 1e-8, n_eval[0], res


def stratified_rows(d, leg, n, rng):
    """Momentum-stratified pick, verbatim from Simple_first_pass."""
    idx = np.where(d["LEG"] == leg)[0]
    order = idx[np.argsort(d["P"][idx])]
    take = order[np.linspace(0, len(order) - 1, n).astype(int)]
    return rng.permutation(take)


def test_state_ceiling(which, split="test"):
    """The scheme's error on the frozen leg's own states, in the measure
    `_shared/evaluate.py` scores the networks with: max(|dx|, |dy|)."""
    d = np.load(os.path.join(RES, "frozen_leg_%s.npz" % which), allow_pickle=False)
    S, ref = d["%s_S" % split], d["%s_ref" % split]
    z0, z1 = float(d["z0"]), float(d["z1"])
    tab = tableau(Q)
    fld = make_field(which)
    rows, t0 = [], time.time()
    for i in range(len(S)):
        stages, S1, ok, nev, res = solve_leg(S[i].copy(), z0, z1, tab, fld)
        rows.append({
            "field": which, "split": split, "i": i, "q": Q,
            "p_GeV": float(d["%s_P" % split][i]), "converged": int(ok),
            "n_evals": nev, "residual": res,
            "err_xy_um": float(np.abs(S1[:2] - ref[i, -1, :2]).max() * 1e3),
            "stage_err_um": float(np.abs(
                stages[:, :2] - ref[i, :Q, :2]).max() * 1e3),
        })
    print("%s test-state ceiling: %d states (%.0f s)"
          % (which, len(rows), time.time() - t0), flush=True)
    return rows


def main():
    os.makedirs(RES, exist_ok=True)
    assert verify_tableau(Q)["passes"]
    tab = tableau(Q)

    d = load_training(split="train")
    rng = np.random.default_rng(SEED)
    sel = stratified_rows(d, 1, N_STATES, rng)          # leg B = index 1
    X = d["X"][sel].astype(np.float64)
    P = d["P"][sel].astype(np.float64)

    rows = []
    t0 = time.time()
    for which in ("up", "down"):
        fld = make_field(which)
        ref = rk4_rows(X[:, :5], X[:, 5], X[:, 6], field=fld)
        for i in range(len(sel)):
            S0, z0, z1 = X[i, :5].copy(), X[i, 5], X[i, 6]
            t1 = time.time()
            _, S1, ok, nev, res = solve_leg(S0, z0, z1, tab, fld)
            rows.append({
                "field": which, "leg": "B", "q": Q, "p_GeV": float(P[i]),
                "dz_mm": float(z1 - z0), "converged": int(ok),
                "n_evals": nev, "residual": res,
                "err_x_mm": abs(S1[0] - ref[i, 0]),
                "err_y_mm": abs(S1[1] - ref[i, 1]),
                "err_xy_mm": max(abs(S1[0] - ref[i, 0]), abs(S1[1] - ref[i, 1])),
                "err_endpoint_um": 1e3 * float(np.hypot(S1[0] - ref[i, 0],
                                                        S1[1] - ref[i, 1])),
                "err_tx": abs(S1[2] - ref[i, 2]),
                "err_ty": abs(S1[3] - ref[i, 3]),
                "wall_s": time.time() - t1,
            })
        print("%s done (%.0f s)" % (which, time.time() - t0), flush=True)

    with open(os.path.join(RES, "ceiling_up.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- the second population: the networks' own test states -----------
    trows = []
    for which in ("up", "down"):
        trows += test_state_ceiling(which)
    with open(os.path.join(RES, "ceiling_test_states.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(trows[0].keys()))
        w.writeheader()
        w.writerows(trows)

    summary = {}
    for which in ("up", "down"):
        sub = [r for r in rows if r["field"] == which]
        conv = [r for r in sub if r["converged"]]
        e = np.array(sorted(r["err_xy_mm"] for r in conv))
        summary[which] = {
            "n": len(sub), "n_converged": len(conv),
            "converged_frac": len(conv) / len(sub),
            "median_err_um": 1e3 * float(np.median(e)),
            "p95_err_um": 1e3 * float(np.percentile(e, 95)),
            "worst_err_um": 1e3 * float(e.max()),
            "median_n_evals": int(np.median([r["n_evals"] for r in sub])),
        }
    on_test = {}
    for which in ("up", "down"):
        sub = [r for r in trows if r["field"] == which]
        e = np.array([r["err_xy_um"] for r in sub])
        on_test[which] = {
            "n": len(sub),
            "converged_frac": float(np.mean([r["converged"] for r in sub])),
            "endpoint_med_um": float(np.median(e)),
            "endpoint_p95_um": float(np.percentile(e, 95)),
            "endpoint_mean_um": float(e.mean()),
            "stage_med_um": float(np.median([r["stage_err_um"] for r in sub])),
        }
    with open(os.path.join(RES, "ceiling_summary.json"), "w") as f:
        json.dump({"q": Q, "leg": "B", "n_states": N_STATES, "seed": SEED,
                   "per_field": summary,
                   "on_the_test_states": on_test,
                   "note": "per_field = 32 momentum-stratified legs "
                           "(the Simple_first_pass population); "
                           "on_the_test_states = the frozen leg's own test "
                           "split, the like-for-like network comparator"},
                  f, indent=1)
    print(json.dumps(on_test, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
