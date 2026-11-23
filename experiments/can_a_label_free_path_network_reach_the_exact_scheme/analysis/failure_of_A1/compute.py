"""Where and how A1 fails: the arrays.

A1 (3ccd44d61cd8) at its latest saved round, A2 (0b41ecdd2505) at its best
round, on the cubic test tracks of 5558be31abeb. Each part writes one .npz to
CACHE; a file there is never recomputed (delete it to recompute).

    1  carried    every test track carried by A1 and A2, with the reference and
                  the exact scheme (64 steps, 2 stages) on the 65 planes
    2  chain      on SAMPLE random test tracks, step by step along each
                  network's own chain: the stage residual r, the stage error
                  eps against the exact stages from the same plane state, the
                  local defect delta, and delta carried to the last plane
    3  training   on the 4,096 first-round training states: L_n, the gradient
                  of the loss with respect to the outputs of step n, and the
                  parameter gradient of each L_n alone; P1 on its own tracks
    4  sweep      A1 and A2 after one Newton sweep, on the SAMPLE tracks

    PYTHONNOUSERSITE=1 PYTHONPATH=../../../../RK_Pinn_module/src THREADS=4 \
        nice /data/bfys/gscriven/conda/envs/TE/bin/python compute.py [part ...]

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
from rkpinn.evaluation.newton_sweep import scores_of_sweeps
from rkpinn.losses.stage_residual_of_a_path import loss_per_step, residual_of_a_path
from rkpinn.run_record.configuration import read_configuration
from rkpinn.run_record.store import Store
from rkpinn.track_data.exact_states import load_exact_states
from rkpinn.track_data.load_tracks import load_tracks
from rkpinn.training.checkpoints import RunFolder
from rkpinn.training.round_trainer import build_run

STORE = "/data/bfys/gscriven/rkpinn_store"
CACHE = os.environ.get("CACHE", "/data/bfys/gscriven/rkpinn_analysis_cache/failure_of_A1")
TRACKS_KEY = "5558be31abeb"
OLD_TRACKS_KEY = "12a8d35c3165"
# A1 is pinned: it was still training (round 37 was its latest saved round on
# 23 Nov 23:00); A2 at its best round by validation (36)
RUNS = {"A1": ("3ccd44d61cd8", "37"), "A2": ("0b41ecdd2505", "best")}
P1 = "25934ed22604"
SAMPLE = int(os.environ.get("SAMPLE", "2000"))
SEED = 0
TOLERANCE = 1e-12
MOST_NEWTON_STEPS = 30
SCALE = torch.tensor([1.0, 1.0, 1e-3, 1e-3], dtype=torch.float64)   # mm, mrad

store = Store(STORE)


def run_of(key, which):
    folder = RunFolder(store, key)
    configuration = read_configuration(folder.path("configuration.yaml"))
    run = build_run(configuration, store, folder.constants())
    if which == "last":
        number = int(folder.state()["rounds_done"])
    elif which == "best":
        rows = folder.rounds()
        quantity = configuration["training"]["stopping_rule"]["quantity"]
        number = int(rows[int(np.argmin([float(r[quantity]) for r in rows]))]["round"])
    else:
        number = int(which)
    # the newest snapshot on disk: a round may be counted before its snapshot is written
    while not os.path.exists(folder.path("snapshots", "round_%04d" % number)):
        number -= 1
    run.network.load_state_dict(folder.weights_of_round(number))
    run.network.eval()
    return run, number


def sample_rows(n):
    return np.sort(np.random.default_rng(SEED).choice(n, size=min(SAMPLE, n), replace=False))


# -- part 1 ---------------------------------------------------------------------------
def carried():
    tracks = load_tracks(store, TRACKS_KEY)
    test = tracks.split("test")
    on = np.append(tracks.planes_a_step_starts_on(64), tracks.number_of_steps_between_planes)
    out = dict(reference=test.reference_states_on_planes[:, on],
               momentum_gev=test.momentum_gev, start=test.start_state,
               event=test.event, particle_in_event=test.particle_in_event)
    exact, planes, _ = load_exact_states(store, TRACKS_KEY, "test", 64, 2)
    out["exact"], out["planes_mm"] = exact, planes
    for name, (key, which) in RUNS.items():
        run, number = run_of(key, which)
        whole = run.network.whole_track(test.start_state, run.layout)
        out[name] = whole.states_on_planes()
        out[name + "_round"] = number
        print("%s round %d carried" % (name, number), flush=True)
    # the old 1,452 test tracks, found by (event, particle in event)
    old = load_tracks(store, OLD_TRACKS_KEY, check_the_key=False).split("test")
    where = {(int(e), int(p)): i for i, (e, p) in enumerate(zip(test.event,
                                                                test.particle_in_event))}
    out["old_subset"] = np.array([where[(int(e), int(p))] for e, p in
                                  zip(old.event, old.particle_in_event)])
    # |B| along the reference path of every track
    field = load_field_map("v8r1_up_cubic")
    ref = out["reference"]
    zz = np.broadcast_to(planes[None, :], ref.shape[:2])
    Bx, By, Bz = field(ref[..., 0], ref[..., 1], zz)
    out["B_on_reference"] = np.stack([Bx, By, Bz], axis=-1)
    return out


# -- part 2 ---------------------------------------------------------------------------
class Scheme:
    """The 2-stage Gauss-Legendre stage equations of one step, in torch
    (as in how_does_the_stage_residual_become_the_endpoint_error/analysis/decompose.py,
    on the cubic map)."""

    def __init__(self, network, equation):
        t = network.tableau
        self.equation = equation
        self.q = int(t.number_of_stages)
        self.A = torch.as_tensor(t.stage_matrix, dtype=torch.float64)
        self.b = torch.as_tensor(t.weights, dtype=torch.float64)
        self.c = torch.as_tensor(t.nodes, dtype=torch.float64)
        self.dz = float(network.step_length_mm)

    def planes(self, z0):
        return z0[:, None] + self.c[None, :] * self.dz

    def rates_and_jacobian(self, Y, qop, zs):
        Y = Y.detach().clone().requires_grad_(True)
        f = self.equation.rates(Y, qop, zs)
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

    def exact(self, S0, z0):
        qop, zs = S0[:, 4:5], self.planes(z0)
        d = (self.c * self.dz)[None, :]
        Y = torch.stack([S0[:, None, 0] + S0[:, None, 2] * d, S0[:, None, 1] + S0[:, None, 3] * d,
                         S0[:, None, 2].expand(-1, self.q), S0[:, None, 3].expand(-1, self.q)],
                        dim=-1)
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
            f = self.equation.rates(Y, qop, zs)
            worst = (self.residual(S0, Y, f).abs() / SCALE).amax().item()
            return Y, S0[:, :4] + self.dz * torch.einsum("k,nkd->nd", self.b, f), worst


def chain():
    tracks = load_tracks(store, TRACKS_KEY)
    test = tracks.split("test")
    rows = sample_rows(test.number_of_tracks)
    equation = DifferentiableLhcbEquationOfMotion(load_field_map("v8r1_up_cubic"))
    out = dict(rows=rows)
    for name, (key, which) in RUNS.items():
        run, number = run_of(key, which)
        net = run.network
        scheme = Scheme(net, equation)
        whole = net.whole_track(test.start_state[rows], run.layout)
        S_in = torch.as_tensor(whole.input_states)                  # (n, N, 5)
        Yhat = torch.as_tensor(whole.stage_states)                  # (n, N, q, 4)
        Ehat = torch.as_tensor(whole.end_states)                    # (n, N, 4)
        n, N = S_in.shape[:2]
        starts = net.layout.start_planes_mm
        r = np.empty((n, N, 2, 4)); eps = np.empty_like(r); delta = np.empty((n, N, 4))
        worst = 0.0
        for k in range(N):
            S0 = S_in[:, k]
            z0 = torch.full((n,), float(starts[k]), dtype=torch.float64)
            f, _ = scheme.rates_and_jacobian(Yhat[:, k], S0[:, 4:5], scheme.planes(z0))
            Ystar, Estar, w = scheme.exact(S0, z0)
            worst = max(worst, w)
            r[:, k] = scheme.residual(S0, Yhat[:, k], f).numpy()
            eps[:, k] = (Yhat[:, k] - Ystar).numpy()
            delta[:, k] = (Ehat[:, k] - Estar).numpy()
        lever = float(net.layout.last_plane_mm) - (starts + scheme.dz)
        carried_x = delta[..., 0] + delta[..., 2] * lever
        carried_y = delta[..., 1] + delta[..., 3] * lever
        out.update({name + "_r": r, name + "_eps": eps, name + "_delta": delta,
                    name + "_carried": np.stack([carried_x, carried_y], -1),
                    name + "_worst_exact_residual": worst, name + "_round": number})
        print("%s chain done, worst exact residual %.2g" % (name, worst), flush=True)
    return out


# -- part 3 ---------------------------------------------------------------------------
def training():
    out = {}
    for name, (key, which) in RUNS.items():
        run, number = run_of(key, which)
        net, constants = run.network, run.constants["loss"]
        states = torch.as_tensor(run.first_round.states)
        sigma = torch.as_tensor(constants["divisor_per_component"], dtype=torch.float64)
        params = [p for p in net.parameters() if p.requires_grad]
        # the loss, with the outputs of every step kept for their gradient
        track = net.predict(states)
        stages = track.stage_states
        stages.retain_grad()
        L = loss_per_step(track, run.loss.equation_of_motion, constants["divisor_per_component"],
                          constants["gradient_through_plane_states"])
        loss = L.mean()
        grads = torch.autograd.grad(loss, [stages] + params, retain_graph=True)
        g_out = (grads[0] * sigma).pow(2).sum(dim=(0, 2, 3)).sqrt().numpy()
        g_total = torch.sqrt(sum((g * g).sum() for g in grads[1:])).item()
        # the parameter gradient of each step's loss alone, as it enters the mean
        g_param = np.empty(len(L))
        flat_total = torch.cat([g.reshape(-1) for g in grads[1:]])
        cos = np.empty(len(L))
        for k in range(len(L)):
            gk = torch.autograd.grad(L[k] / len(L), params, retain_graph=True)
            flat = torch.cat([g.reshape(-1) for g in gk])
            g_param[k] = flat.norm().item()
            cos[k] = (flat @ flat_total).item() / (flat.norm().item() * flat_total.norm().item())
        out.update({name + "_L": L.detach().numpy(), name + "_grad_outputs": g_out,
                    name + "_grad_params_per_step": g_param, name + "_cos_with_total": cos,
                    name + "_grad_params_total": g_total, name + "_round": number})
        print("%s training done" % name, flush=True)
        del track, stages, L, loss, grads
    # P1: the supervised path network, on its own (linear map) test tracks
    run, number = run_of(P1, "last")
    test = run.tracks.split("test")
    with torch.no_grad():
        track = run.network.predict(torch.as_tensor(test.start_state))
        sigma = run.constants["loss"]["divisor_per_component"] if \
            "divisor_per_component" in run.constants["loss"] else None
        R = residual_of_a_path(track, run.differentiable_equation, True)
    out["P1_R"] = R.numpy()
    out["P1_round"] = number
    print("P1 done", flush=True)
    return out


# -- part 4 ---------------------------------------------------------------------------
def sweep():
    tracks = load_tracks(store, TRACKS_KEY)
    test = tracks.split("test")
    rows = sample_rows(test.number_of_tracks)
    exact, _, _ = load_exact_states(store, TRACKS_KEY, "test", 64, 2)
    out = dict(rows=rows)
    for name, (key, which) in RUNS.items():
        run, number = run_of(key, which)
        whole = run.network.whole_track(test.start_state[rows], run.layout)
        scored = scores_of_sweeps(whole, run.differentiable_equation,
                                  test.reference_end_state[rows], exact[rows, -1],
                                  number_of_sweeps=1)
        for s in scored:
            for against in ("against_the_reference", "against_the_exact_scheme"):
                for stat in ("median_micrometres", "percentile_95_micrometres"):
                    out["%s_sweeps%d_%s_%s" % (name, s["sweeps"], against, stat)] = \
                        s[against][stat]
        out[name + "_round"] = number
        print(name, [(s["sweeps"], s["against_the_reference"]["median_micrometres"])
                     for s in scored], flush=True)
    return out


PARTS = dict(carried=carried, chain=chain, training=training, sweep=sweep)


def main(parts):
    torch.set_num_threads(int(os.environ.get("THREADS", "4")))
    os.makedirs(CACHE, exist_ok=True)
    for part in parts or list(PARTS):
        file = os.path.join(CACHE, "%s.npz" % part)
        if os.path.exists(file):
            print("%s: in the cache" % part)
            continue
        started = time.time()
        np.savez(file, **PARTS[part]())
        print("%s: %.0f s" % (part, time.time() - started), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
