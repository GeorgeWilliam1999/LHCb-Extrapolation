"""Exact Gauss-Legendre scheme and RK4 on the LINEAR map over the 13,447 test tracks of
afc61a558211, for the linear-map rows (no stored numbers exist at these layouts on this split).
Same method as what_are_the_classical_references_on_the_cubic_map/analysis/classical_references.py:
Newton sweeps from the forward-Euler path until the largest stage residual < 1e-10 (at most 8)."""
import os, sys, time
import numpy as np, torch
from multiprocessing import Pool
from rkpinn.equation_of_motion.field_map import load_field_map
from rkpinn.equation_of_motion.lhcb import DifferentiableLhcbEquationOfMotion, LhcbEquationOfMotion
from rkpinn.evaluation.newton_sweep import forward_euler_path, one_sweep
from rkpinn.integrators.gauss_legendre_tableau import gauss_legendre_tableau
from rkpinn.predicted_track.track_layout import TrackLayout
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_work", "classical_linear"); os.makedirs(OUT, exist_ok=True)
T = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_work", "test_afc61a558211.npz"))
start = T["start_state"]; ref = T["reference_states_on_planes"][:, -1]; Z0, Z1 = 2648.2, 7826.0
CELLS = [(8, 2), (16, 2), (32, 2), (64, 2), (128, 2), (256, 2), (64, 4), (64, 8), (64, 16), (128, 8), (256, 16)]
_d = None
def share(a):
    global _d
    N, q, idx = a; torch.set_num_threads(1)
    if _d is None: _d = DifferentiableLhcbEquationOfMotion(load_field_map("v8r1_up"))
    lay = TrackLayout(Z0, Z1, N, gauss_legendre_tableau(q))
    tr = forward_euler_path(_d, start[idx], lay); res, k = np.inf, 0
    while res > 1e-10 and k < 8:
        s = one_sweep(tr, _d); tr, res, k = s.track, s.largest_residual_after, k + 1
    return np.asarray(tr.final_state()), res, k
if __name__ == "__main__":
    eq = LhcbEquationOfMotion(load_field_map("v8r1_up"))
    for N in (16, 32, 64, 128, 256):
        f = os.path.join(OUT, "rk4_%03d.npz" % N)
        if os.path.exists(f): continue
        h = (Z1 - Z0) / N; S = start.copy()
        for z in Z0 + h * np.arange(N):
            k1 = eq.rates(S, z); k2 = eq.rates(S + .5 * h * k1, z + .5 * h); k3 = eq.rates(S + .5 * h * k2, z + .5 * h); k4 = eq.rates(S + h * k3, z + h)
            S = S + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        np.savez(f, final=S); print("rk4", N, np.median(np.hypot(*((S[:, :2] - ref[:, :2]) * 1e3).T)), flush=True)
    for N, q in CELLS:
        f = os.path.join(OUT, "exact_%03d_%02d.npz" % (N, q))
        if os.path.exists(f): continue
        t = time.time()
        parts = np.array_split(np.arange(len(start)), int(np.ceil(len(start) * max(1, N * q / 1024) / 500)))
        with Pool(4) as p: r = p.map(share, [(N, q, s) for s in parts], chunksize=1)
        fin = np.concatenate([x[0] for x in r])
        np.savez(f, final=fin, residual=max(x[1] for x in r), sweeps=max(x[2] for x in r))
        print("exact", N, q, np.median(np.hypot(*((fin[:, :2] - ref[:, :2]) * 1e3).T)), "res %.1e" % max(x[1] for x in r), "%.0f s" % (time.time() - t), flush=True)
