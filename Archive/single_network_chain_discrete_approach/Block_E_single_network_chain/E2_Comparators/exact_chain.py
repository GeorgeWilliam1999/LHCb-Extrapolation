#!/usr/bin/env python
"""E2 - the exact collocation scheme chained across the magnet: the ceiling.

Block D's `D2_Comparators/exact_chain.py` on Block E's tracks. For each (N, q)
the q-stage Gauss-Legendre scheme is solved exactly, with a root-finder and no
network (Block C's `exact_solver.solve_state`), on every test track, step after
step from z0 to z1, each solved end state starting the next step - the same
chaining the network goes through. Its error against the RK6 track is the
scheme's own error: the best a network trained on these equations can do.

Block E needs new runs at N = 2 and 256 only; Block D's runs at N = 64 and 128
are on the same test particles (E0's gate) and are reused.

Usage:
    python exact_chain.py --N 256 --q 16
Output:
    results/exact_N<NNN>_q<qq>.json          errors at z1 and on every plane, cost
    results/exact_N<NNN>_q<qq>_states.npz    the solved states on every plane
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"

import argparse   # noqa: E402
import json       # noqa: E402
import sys        # noqa: E402
import time       # noqa: E402

import numpy as np  # noqa: E402

import use_shared   # noqa: E402,F401
from _shared.reference import gauss_legendre, make_field   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(use_shared.SHARED_ROOT, "..", "multi_network_chain_discrete_approach",
                                "Block_C_step_size_and_stages", "C2_Exact_scheme_table"))
from exact_solver import solve_state                      # noqa: E402
sys.path.insert(0, os.path.join(HERE, "..", "E1_Network_grid"))
from metrics import component_stats, per_plane_curves      # noqa: E402

DEFAULT_TRACKS = os.path.join(HERE, "..", "E0_Track_dataset", "results", "tracks.npz")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--q", type=int, required=True)
    ap.add_argument("--tracks", default=DEFAULT_TRACKS)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-cap", type=int, default=None, help="smoke runs only")
    a = ap.parse_args(argv)
    D = np.load(os.path.abspath(a.tracks))
    Z0, L, n_max = float(D["z0"]), float(D["L"]), int(D["n_max"])
    N, q = a.N, a.q
    stride, dz = n_max // N, L / N
    fld = make_field(str(D["field"]))
    tab = gauss_legendre(q)
    S = np.asarray(D["%s_S0" % a.split])[:a.n_cap]
    truth = np.asarray(D["%s_truth" % a.split])[:a.n_cap]
    n = len(S)
    t0 = time.time()
    states = np.empty((n, N + 1, 5))
    states[:, 0] = S
    cur, n_fail, n_eval = S.copy(), 0, 0
    for k in range(N):
        nxt = np.empty_like(cur)
        for i in range(n):
            _, S1, ok, used, _ = solve_state(cur[i], Z0 + k * dz, dz, tab, fld)
            nxt[i] = S1
            n_fail += (not ok)
            n_eval += used
        cur = nxt
        states[:, k + 1] = cur
    end = truth[:, n_max]
    pos = np.abs(cur[:, :2] - end[:, :2]).max(axis=1) * 1e3
    slope = np.abs(cur[:, 2:4] - end[:, 2:4]).max(axis=1) * 1e3
    rec = {"N": N, "q": q, "dz_mm": dz, "split": a.split, "n": int(n),
           "endpoint_pos_med_um": float(np.median(pos)),
           "endpoint_pos_p95_um": float(np.quantile(pos, 0.95)),
           "endpoint_slope_med_mrad": float(np.median(slope)),
           "components": component_stats(cur, end),
           "per_plane": per_plane_curves(states, truth, stride, N),
           "solves_not_converged": int(n_fail),
           "residual_evaluations_per_track": n_eval / n,
           "wall_s": round(time.time() - t0, 1)}
    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(os.path.join(a.out, "exact_N%03d_q%02d_states.npz" % (N, q)), states=states)
    with open(os.path.join(a.out, "exact_N%03d_q%02d.json" % (N, q)), "w") as f:
        json.dump(rec, f, indent=1)
    print("EXACT N=%d q=%d: endpoint median %.4g um (p95 %.4g), %d unconverged solves, %.0f s"
          % (N, q, rec["endpoint_pos_med_um"], rec["endpoint_pos_p95_um"], n_fail, rec["wall_s"]), flush=True)
    return rec


if __name__ == "__main__":
    main()
