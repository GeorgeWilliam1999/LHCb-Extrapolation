"""The chain from the stage residual to the endpoint error, step by step.

For one run, on the test tracks, along the network's own chain:

    r      stage residual        Y^ - S_n - dz A F(Y^)
    eps    stage error           Y^ - Y*            (Y* the exact stages from S_n)
    eps1   first-order eps       (I - dz A (x) J)^-1 r
    delta  local defect          network end - exact end, both from S_n
    delta1 first-order delta     dz sum_k b_k J_k eps_k
    carried defects              exact on the first TRACKS_CARRIED tracks, linear on all

Norms: x, y in mm and tx, ty in mrad (SCALE). Arrays are saved per run in
CACHE; a file there is never recomputed (delete it to recompute).

    PYTHONNOUSERSITE=1 PYTHONPATH=../../../RK_Pinn_module/src THREADS=8 \
        /data/bfys/gscriven/conda/envs/TE/bin/python decompose.py <run key> [<run key> ...]

Reads the store and the field map. Writes only to CACHE.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

from rkpinn.equation_of_motion.field_map import load_field_map
from rkpinn.equation_of_motion.lhcb import DifferentiableLhcbEquationOfMotion
from rkpinn.run_record.configuration import read_configuration
from rkpinn.run_record.store import Store
from rkpinn.track_data.load_tracks import load_tracks
from rkpinn.training.checkpoints import RunFolder
from rkpinn.training.round_trainer import build_run

STORE = "/data/bfys/gscriven/rkpinn_store"
CACHE = "/data/bfys/gscriven/rkpinn_analysis_cache/how_does_the_stage_residual_become_the_endpoint_error"
TRACKS_KEY = "12a8d35c3165"
TRACKS_CARRIED = 200
LIMIT = int(os.environ.get("LIMIT", "0")) or None   # smoke runs only; then nothing is cached
SCALE = torch.tensor([1.0, 1.0, 1e-3, 1e-3], dtype=torch.float64)   # mm, mrad
TOLERANCE = 1e-12
MOST_NEWTON_STEPS = 30

store = Store(STORE)
tracks = load_tracks(store, TRACKS_KEY)
test = tracks.split("test")
EQUATION = None


def network_of(key):
    folder = RunFolder(store, key)
    configuration = read_configuration(folder.path("configuration.yaml"))
    run = build_run(configuration, store, folder.constants())
    rounds = int(folder.state()["rounds_done"])
    run.network.load_state_dict(folder.weights_of_round(rounds))
    run.network.eval()
    return run.network, configuration, rounds


class Scheme:
    """The q-stage Gauss-Legendre stage equations of one step length, in torch."""

    def __init__(self, network):
        global EQUATION
        if EQUATION is None:
            EQUATION = DifferentiableLhcbEquationOfMotion(load_field_map("v8r1_up"))
        t = network.tableau
        self.q = int(t.number_of_stages)
        self.A = torch.as_tensor(t.stage_matrix, dtype=torch.float64)
        self.b = torch.as_tensor(t.weights, dtype=torch.float64)
        self.c = torch.as_tensor(t.nodes, dtype=torch.float64)
        self.dz = float(network.step_length_mm)

    def planes(self, z0):
        return z0[:, None] + self.c[None, :] * self.dz

    def rates(self, Y, qop, zs):
        return EQUATION.rates(Y, qop, zs)

    def rates_and_jacobian(self, Y, qop, zs):
        Y = Y.detach().clone().requires_grad_(True)
        f = EQUATION.rates(Y, qop, zs)
        J = torch.empty(Y.shape + (4,), dtype=torch.float64)
        for a in range(4):
            (g,) = torch.autograd.grad(f[..., a].sum(), Y, retain_graph=a < 3)
            J[..., a, :] = g
        return f.detach(), J

    def residual(self, S0, Y, f):
        return Y - S0[:, None, :4] - self.dz * torch.einsum("jk,nkd->njd", self.A, f)

    def newton_matrix(self, J):
        n, q = J.shape[0], self.q
        M = torch.eye(4 * q, dtype=torch.float64).repeat(n, 1, 1)
        for j in range(q):
            for k in range(q):
                M[:, 4 * j:4 * j + 4, 4 * k:4 * k + 4] -= self.dz * self.A[j, k] * J[:, k]
        return M

    def end(self, S0, f):
        return S0[:, :4] + self.dz * torch.einsum("k,nkd->nd", self.b, f)

    def exact(self, S0, z0):
        """Exact stages (n,q,4), end states (n,4), largest scaled residual."""
        qop, zs = S0[:, 4:5], self.planes(z0)
        d = (self.c * self.dz)[None, :]
        Y = torch.stack([S0[:, None, 0] + S0[:, None, 2] * d, S0[:, None, 1] + S0[:, None, 3] * d,
                         S0[:, None, 2].expand(-1, self.q), S0[:, None, 3].expand(-1, self.q)], dim=-1)
        n = len(S0)
        for _ in range(MOST_NEWTON_STEPS):
            f, J = self.rates_and_jacobian(Y, qop, zs)
            G = self.residual(S0, Y, f)
            if (G.abs() / SCALE).amax().item() < TOLERANCE:
                break
            with torch.no_grad():
                dY = torch.linalg.solve(self.newton_matrix(J), -G.reshape(n, -1, 1))[..., 0]
                Y = Y + dY.reshape(n, self.q, 4)
        with torch.no_grad():
            f = self.rates(Y, qop, zs)
            worst = (self.residual(S0, Y, f).abs() / SCALE).amax().item()
            return Y, self.end(S0, f), worst

    def carry(self, S, first_step, start_planes):
        """S (n,5) on the start plane of step `first_step`, carried by the exact
        scheme to the last plane. Returns (n,4)."""
        S = S.clone()
        for k in range(first_step, len(start_planes)):
            z0 = torch.full((len(S),), float(start_planes[k]), dtype=torch.float64)
            _, end, _ = self.exact(S, z0)
            S = torch.cat([end, S[:, 4:5]], dim=1)
        return S[:, :4]


def decompose(key):
    network, configuration, rounds = network_of(key)
    scheme = Scheme(network)
    steps = int(round(tracks.length_mm / scheme.dz))
    start_planes = tracks.start_planes_mm(steps)
    last_plane = float(tracks.first_plane_mm + tracks.length_mm)
    S = torch.as_tensor(np.asarray(test.start_state, dtype=np.float64)[:LIMIT])
    n = len(S)
    out = {name: np.empty((n, steps, scheme.q, 4)) for name in ("r", "eps", "eps1")}
    out.update({name: np.empty((n, steps, 4)) for name in ("delta", "delta1")})
    states = np.empty((n, steps + 1, 5))
    states[:, 0] = S.numpy()
    exact_ends = np.empty((n, steps, 4))
    worst = 0.0
    for k in range(steps):
        z0 = torch.full((n,), float(start_planes[k]), dtype=torch.float64)
        with torch.no_grad():
            step = network.predict(S, z0)
            Yhat = step.stage_states[:, 0].to(torch.float64)
            Ehat = step.end_states[:, 0].to(torch.float64)
        qop, zs = S[:, 4:5], scheme.planes(z0)
        fhat, Jhat = scheme.rates_and_jacobian(Yhat, qop, zs)
        Ystar, Estar, w = scheme.exact(S, z0)
        worst = max(worst, w)
        with torch.no_grad():
            r = scheme.residual(S, Yhat, fhat)
            eps = Yhat - Ystar
            eps1 = torch.linalg.solve(scheme.newton_matrix(Jhat), r.reshape(n, -1, 1))[..., 0]
            delta = Ehat - Estar
            delta1 = scheme.dz * torch.einsum("k,nkab,nkb->na", scheme.b, Jhat, eps)
        out["r"][:, k], out["eps"][:, k] = r.numpy(), eps.numpy()
        out["eps1"][:, k] = eps1.reshape(n, scheme.q, 4).numpy()
        out["delta"][:, k], out["delta1"][:, k] = delta.numpy(), delta1.numpy()
        exact_ends[:, k] = Estar.numpy()
        S = torch.cat([Ehat, S[:, 4:5]], dim=1)
        states[:, k + 1] = S.numpy()
        print("%s step %d of %d" % (key, k + 1, steps), flush=True)
    # carried defects: linear on all tracks, exact on the first TRACKS_CARRIED
    lever = last_plane - (start_planes + scheme.dz)                       # (steps,)
    d = out["delta"]
    linear = np.stack([d[..., 0] + d[..., 2] * lever, d[..., 1] + d[..., 3] * lever], axis=-1)
    m = min(TRACKS_CARRIED, n)
    exact_carried = np.empty((m, steps, 4))
    for k in range(steps):
        net = torch.as_tensor(states[:m, k + 1])
        ex = torch.as_tensor(np.concatenate([exact_ends[:m, k], states[:m, k, 4:5]], axis=1))
        exact_carried[:, k] = (scheme.carry(net, k + 1, start_planes)
                               - scheme.carry(ex, k + 1, start_planes)).numpy()
        print("%s carried %d of %d" % (key, k + 1, steps), flush=True)
    exact_chain = scheme.carry(torch.as_tensor(states[:, 0]), 0, start_planes).numpy()
    return dict(out, states=states, linear_carried=linear, exact_carried=exact_carried,
                exact_chain_end=exact_chain, reference_end=np.asarray(test.reference_end_state)[:LIMIT, :4],
                momentum_gev=np.asarray(test.momentum_gev)[:LIMIT], step_length_mm=scheme.dz,
                number_of_stages=scheme.q, round=rounds,
                end_state=str(network.end_state), worst_exact_residual=worst)


def main(keys):
    torch.set_num_threads(int(os.environ.get("THREADS", "4")))
    os.makedirs(CACHE, exist_ok=True)
    for key in keys:
        file = os.path.join(CACHE, "%s.npz" % key) if LIMIT is None else os.path.join(
            CACHE, "smoke_%s_%d.npz" % (key, LIMIT))
        if os.path.exists(file):
            print("%s: in the cache" % key)
            continue
        t = time.time()
        result = decompose(key)
        np.savez(file + ".partial.npz", **result)
        os.replace(file + ".partial.npz", file)
        print("%s: written in %.0f s" % (key, time.time() - t), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
