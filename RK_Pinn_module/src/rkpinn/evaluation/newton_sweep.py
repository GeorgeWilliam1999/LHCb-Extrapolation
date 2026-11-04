"""One Newton sweep: one Newton step on the stage equations of a whole track,
with every stage of every step evaluated at once.

The unknowns of a track on a layout of N steps and q stages are
U = (S_01 .. S_0q, s_1, S_11 .. S_1q, s_2, ..., s_N). They obey

    S_ni  = s_n + h sum_j a_ij f(z_nj, S_nj)          i = 1..q
    s_n+1 = s_n + h sum_i b_i  f(z_ni, S_ni)

which is G(U) = 0. One Newton step is U_1 = U_0 - J^{-1} G(U_0). The rates f
and the Jacobians df/ds at all N x q stage points are evaluated in one call,
which is the part that reads the field map; the solve is then a forward
substitution over the steps of (4q x 4q) systems with no field lookups.

From the forward-Euler path of the Kalman experiment one sweep reaches the
exact-scheme floor, 0.146 um at 64 steps and 2 stages. Here the start is any
PredictedTrack with a layout: the chain, the path network or the Euler path.

Counts returned beside the track: field lookups (N q per track) and
Jacobians (the same), so a report can print the cost.

What the Kalman experiment measured is not this. There the gain corrected one
step at a time along the chain, each step from Euler stages built on the state
the previous corrected step gave. Here the whole path is corrected at once,
from states that may all be off; how far one global sweep gets from an Euler
path is measured by the gates, not assumed.

The Jacobian comes from torch autograd through the differentiable equation of
motion, four backward passes over all stage points at once; inside a cell of
the trilinear field map it is exact. Ported onto the stbc checkout on
2026-11-04. Gates: tests/test_newton_sweep.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from rkpinn.predicted_track.predicted_track import END_STATE_PREDICTED, PredictedTrack


@dataclass
class Sweep:
    track: PredictedTrack
    field_lookups_per_track: int
    jacobians_per_track: int
    largest_residual_before: float
    largest_residual_after: float


def rates_and_jacobians(equation, stage_states, charge_over_momentum, stage_planes_mm):
    """f and df/ds at every stage point. stage_states (n, N, q, 4);
    charge_over_momentum (n,); stage_planes_mm (N, q). One batched evaluation
    of the field; the rows of the Jacobian by four backward passes, which are
    exact because every point's rates depend on that point alone."""
    n, N, q, _ = stage_states.shape
    with torch.enable_grad():
        flat = stage_states.reshape(-1, 1, 4).detach().clone().requires_grad_(True)
        qop = charge_over_momentum[:, None, None].expand(n, N, q).reshape(-1, 1)
        z = stage_planes_mm[None].expand(n, N, q).reshape(-1, 1)
        f = equation.rates(flat, qop, z)[:, 0]                              # (M, 4)
        rows = [torch.autograd.grad(f[:, d].sum(), flat, retain_graph=d < 3)[0][:, 0]
                for d in range(4)]
    jac = torch.stack(rows, dim=1)                                          # (M, 4, 4)
    return f.detach().reshape(n, N, q, 4), jac.reshape(n, N, q, 4, 4)


def one_sweep(track: PredictedTrack, equation) -> Sweep:
    """equation: the differentiable equation of motion. The track must have a
    layout. Returns the corrected track and the counts."""
    layout = track.layout
    if layout is None:
        raise ValueError("one sweep needs a whole track with a layout")
    tableau = layout.tableau
    A = torch.as_tensor(np.asarray(tableau.stage_matrix, dtype=np.float64))
    b = torch.as_tensor(np.asarray(tableau.weights, dtype=np.float64))
    h = float(layout.step_length_mm)
    N, q = layout.number_of_steps, layout.number_of_stages
    stage_planes = torch.as_tensor(np.stack([layout.stage_planes_mm(k) for k in range(N)]),
                                   dtype=torch.float64)
    with torch.no_grad():
        S = torch.as_tensor(np.asarray(track.stage_states, dtype=np.float64))    # (n, N, q, 4)
        inputs = torch.as_tensor(np.asarray(track.input_states, dtype=np.float64))
        ends = torch.as_tensor(np.asarray(track.end_states, dtype=np.float64))   # (n, N, 4)
        qop = inputs[:, 0, 4]
        planes = torch.cat([inputs[:, :1, :4], ends], dim=1)                    # (n, N + 1, 4)
        n = S.shape[0]
        F, J = rates_and_jacobians(equation, S, qop, stage_planes)
        G1 = S - planes[:, :N, None, :] - h * torch.einsum("ij,bnjd->bnid", A, F)
        G2 = planes[:, 1:] - planes[:, :N] - h * torch.einsum("i,bnid->bnd", b, F)
        before = max(float(G1.abs().max()), float(G2.abs().max()))
        I = torch.eye(4 * q, dtype=torch.float64)
        ds = torch.zeros(n, 4, dtype=torch.float64)
        new_S, new_planes = [], [planes[:, 0]]
        for m in range(N):
            Jm = J[:, m]                                                          # (n, q, 4, 4)
            M = I - h * torch.einsum("ij,bjkl->bikjl", A, Jm).reshape(n, 4 * q, 4 * q)
            r = (-G1[:, m] + ds[:, None, :]).reshape(n, 4 * q)
            dS = torch.linalg.solve(M, r).reshape(n, q, 4)
            ds = ds + h * torch.einsum("i,bikl,bil->bk", b, Jm, dS) - G2[:, m]
            new_S.append(S[:, m] + dS); new_planes.append(planes[:, m + 1] + ds)
        S1 = torch.stack(new_S, 1); P1 = torch.stack(new_planes, 1)
        F1, _ = rates_and_jacobians(equation, S1, qop, stage_planes)
        after = max(float((S1 - P1[:, :N, None, :] - h * torch.einsum("ij,bnjd->bnid", A, F1)).abs().max()),
                    float((P1[:, 1:] - P1[:, :N] - h * torch.einsum("i,bnid->bnd", b, F1)).abs().max()))
        new_inputs = torch.cat([P1[:, :N], qop[:, None, None].expand(n, N, 1)], dim=2)
        corrected = PredictedTrack(
            step_length_mm=h, tableau=tableau,
            start_planes_mm=np.asarray(track.start_planes_mm, dtype=np.float64).copy(),
            input_states=new_inputs.numpy(), stage_states=S1.numpy(),
            end_states=P1[:, 1:].numpy(),
            end_state_was=END_STATE_PREDICTED, layout=layout, stage_rates=F1.numpy(),
            produced_by=track.produced_by + " + one sweep")
    return Sweep(track=corrected, field_lookups_per_track=N * q, jacobians_per_track=N * q,
                 largest_residual_before=before, largest_residual_after=after)


def forward_euler_path(equation, start_states, layout) -> PredictedTrack:
    """The forward-Euler path of a layout, the network-free start of a sweep:
    one field lookup per step, at the state on its start plane,

        S_ni   = s_n + c_i h f(z_n, s_n)
        s_n+1  = s_n + h f(z_n, s_n)

    the stages built as the Euler stages of the Kalman experiment. start_states
    (n, 5) on the first plane. Returns a whole track, as numpy."""
    if layout.tableau is None:
        raise ValueError("the forward-Euler path is built on a layout with stages")
    nodes = torch.as_tensor(np.asarray(layout.tableau.nodes, dtype=np.float64))
    h, N = float(layout.step_length_mm), layout.number_of_steps
    with torch.no_grad():
        s = torch.as_tensor(np.asarray(start_states, dtype=np.float64))
        qop = s[:, 4:5]
        current = s[:, :4]
        inputs, stages, ends = [], [], []
        for m, z in enumerate(layout.start_planes_mm):
            f = equation.rates(current[:, None, :], qop,
                               torch.tensor([float(z)], dtype=torch.float64))[:, 0]
            inputs.append(torch.cat([current, qop], dim=1))
            stages.append(current[:, None, :] + (nodes * h)[None, :, None] * f[:, None, :])
            current = current + h * f
            ends.append(current)
    n = len(s)
    return PredictedTrack(
        step_length_mm=h, tableau=layout.tableau,
        start_planes_mm=np.broadcast_to(layout.start_planes_mm, (n, N)).copy(),
        input_states=torch.stack(inputs, 1).numpy(), stage_states=torch.stack(stages, 1).numpy(),
        end_states=torch.stack(ends, 1).numpy(), end_state_was=END_STATE_PREDICTED,
        layout=layout, produced_by="forward Euler")


def scores_of_sweeps(track: PredictedTrack, equation, reference_end_states,
                     exact_end_states=None, number_of_sweeps=2) -> list:
    """The endpoint error of a whole track alone and after each of
    `number_of_sweeps` sweeps, one row each: the radial error against the
    reference (and against the exact scheme when its end states are given) as
    the statistics of a magnitude, in micrometres; the largest residual of the
    stage equations; the field lookups and Jacobians per track spent by the
    sweeps so far; and the seconds of the sweeps so far."""
    from rkpinn.evaluation import conventions

    def row(of, sweeps, residual, seconds):
        final = np.asarray(of.final_state())
        scored = {"sweeps": sweeps, "largest_residual": residual,
                  "field_lookups_per_track": sweeps * layout_lookups,
                  "jacobians_per_track": sweeps * layout_lookups,
                  "seconds_of_the_sweeps": seconds,
                  "against_the_reference": conventions.statistics_of_magnitude(
                      conventions.errors(final, reference_end_states)["radial"])}
        if exact_end_states is not None:
            scored["against_the_exact_scheme"] = conventions.statistics_of_magnitude(
                conventions.errors(final, exact_end_states)["radial"])
        return scored

    import time
    layout_lookups = track.layout.number_of_steps * track.layout.number_of_stages
    rows, current, seconds, residual = [], track, 0.0, None
    for sweeps in range(number_of_sweeps + 1):
        if sweeps:
            started = time.time()
            sweep = one_sweep(current, equation)
            seconds += time.time() - started
            current = sweep.track
            residual = sweep.largest_residual_after
            if sweeps == 1:
                rows[0]["largest_residual"] = sweep.largest_residual_before
        rows.append(row(current, sweeps, residual, seconds))
    return rows
