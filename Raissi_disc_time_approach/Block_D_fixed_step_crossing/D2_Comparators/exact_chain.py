#!/usr/bin/env python
"""D2 - the exact collocation scheme chained across the magnet: the ceiling.

For each (N, q) of the Block D grid the q-stage Gauss-Legendre scheme is
solved EXACTLY, with a root-finder and no network (`C2_Exact_scheme_table/
exact_solver.py`), on every test particle, leg after leg from z0 to z1, the
solved end state of one leg being the start state of the next - the same
chaining the networks are put through. Its error against the RK6 truth is the
scheme's own discretisation error: the best a network trained on these
equations could do, at that N and q.

Usage (one job per (N, q), or `--all` for the whole grid in one process):
    python exact_chain.py --N 4 --q 8
    python exact_chain.py --all

Output:
    results/exact_N<NNN>_q<qq>.json    per-plane and endpoint errors, cost
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"

import argparse
import json
import sys
import time

import numpy as np

import use_shared                                        # noqa: F401
from _shared.reference import gauss_legendre, make_field

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(use_shared.SHARED_ROOT, "Block_C_step_size_and_stages",
                                "C2_Exact_scheme_table"))
from exact_solver import solve_state                     # noqa: E402
sys.path.insert(0, os.path.join(HERE, "..", "D1_Chain_grid"))
from metrics import component_stats, per_plane_curves     # noqa: E402

DEFAULT_DATA = os.path.join(HERE, "..", "D0_Crossing_dataset", "results",
                            "crossing_particles.npz")
N_VALUES = (1, 4, 16, 64, 128)
QS = tuple(range(1, 21))


def run_one(N, q, D, out_dir, split="test", n_cap=None):
    Z0, L = float(D["z0"]), float(D["L"])
    n_max = int(D["n_max"])
    stride = n_max // N
    dz = L / N
    fld = make_field(str(D["field"]))
    tab = gauss_legendre(q)
    S = np.asarray(D["%s_S0" % split])[:n_cap]
    truth = np.asarray(D["%s_truth" % split])[:n_cap]
    n = len(S)
    t0 = time.time()
    cur = S.copy()
    all_states = np.empty((n, N + 1, 5))
    all_states[:, 0] = S
    n_fail, n_eval = 0, 0
    for k in range(N):
        zk = Z0 + k * dz
        nxt = np.empty_like(cur)
        for i in range(n):
            _, S1, ok, used, _ = solve_state(cur[i], zk, dz, tab, fld)
            nxt[i] = S1
            n_fail += (not ok)
            n_eval += used
        cur = nxt
        all_states[:, k + 1] = cur
    per_plane = per_plane_curves(all_states, truth, stride, N)
    end_err = np.abs(cur[:, :2] - truth[:, n_max, :2]).max(axis=1) * 1e3
    slope_err = np.abs(cur[:, 2:4] - truth[:, n_max, 2:4]).max(axis=1) * 1e3
    rec = {"N": N, "q": q, "dz_mm": dz, "split": split, "n": int(n),
           "endpoint_pos_med_um": float(np.median(end_err)),
           "endpoint_pos_p95_um": float(np.quantile(end_err, 0.95)),
           "endpoint_slope_med_mrad": float(np.median(slope_err)),
           "components": component_stats(cur, truth[:, n_max]),
           "per_plane": per_plane,
           "per_plane_pos_med_um": per_plane["pos_med_um"],
           "per_plane_pos_p95_um": per_plane["pos_p95_um"],
           "solves_not_converged": int(n_fail),
           "residual_evaluations_per_particle": n_eval / n,
           "wall_s": round(time.time() - t0, 1)}
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "exact_N%03d_q%02d_states.npz" % (N, q)),
                        states=all_states)
    with open(os.path.join(out_dir, "exact_N%03d_q%02d.json" % (N, q)), "w") as f:
        json.dump(rec, f, indent=1)
    print("EXACT N=%d q=%d: endpoint median %.4g um (p95 %.4g), %d unconverged solves, "
          "%.0f s" % (N, q, rec["endpoint_pos_med_um"], rec["endpoint_pos_p95_um"],
                      n_fail, rec["wall_s"]), flush=True)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--N", type=int)
    ap.add_argument("--q", type=int)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--n-cap", type=int, default=None, help="smoke tests only")
    a = ap.parse_args(argv)
    D = {k: v for k, v in np.load(os.path.abspath(a.data), allow_pickle=False).items()}
    if a.all:
        for N in N_VALUES:
            for q in QS:
                p = os.path.join(a.out, "exact_N%03d_q%02d.json" % (N, q))
                have = os.path.exists(p) and "components" in json.load(open(p))
                if not have:
                    run_one(N, q, D, a.out, n_cap=a.n_cap)
    else:
        run_one(a.N, a.q, D, a.out, n_cap=a.n_cap)


if __name__ == "__main__":
    main()
