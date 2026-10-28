"""Part B: a linearised, Kalman-like correction of the summed end state, built
from the stage residual that the summed form already has. No training.

For one step from S0 on z0 of length dz, with the stages Y (q = 2, Gauss-Legendre)
and the rates F_k = f(Y_k, z0 + c_k dz) that the summed end state already needs:

    residual   r_k = Y_k - S0 - dz sum_j A_kj F_j          (rkpinn/losses/stage_residual.py)
    Jacobian   J_k = df/dY at Y_k, 4 x 4, q/p fixed       (torch autograd through the
                                                           differentiable field map)
    gain       K   = dz (b^T (x) J) (I - dz A (x) J)^-1     (an 8 x 8 solve per step)
    end state  S_end = S0 + dz sum_k b_k F_k - K r

K r is the change of the summed end state that one Newton step of the stage
equations would make, with f at the new stages replaced by its linearisation.
So the correction needs no new field lookup: J comes from the same lookups as F.

Variants:
  plain        the summed end state, chained
  B1           the corrected end state of each step, chained into the next step
  B2           the plain chain; the corrections of its steps are carried to the
               last plane with the reference Jacobians (as in A.5 of the bias study)
               and added to the plain endpoint. A diagnosis, not a method
  B3           not run: a second update needs r at the corrected stages, so f at
               the corrected stages, 2 more field lookups a step. The linear model
               itself predicts r = 0 there
  controls     the same K r update from stages with no network:
                 straight line  Y_k = S0 + c_k dz (tx, ty, 0, 0)
                 explicit Euler Y_k = S0 + c_k dz f(S0)          (1 more lookup)

Counts per step: field lookups (f at one state), Jacobian evaluations (J at one
state; it reuses the lookup of f at that state through autograd), network calls.

    part_b.py run       writes results/b_endpoint.csv, b_single_step.csv,
                        b_low_momentum_track.npz
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

from common import (NUMBER_OF_STEPS, RUNS, SHORT, START_PLANES_MM, STEP_MM, TO_REPORTED,
                    BIAS_CACHE, RESULTS, chain, endpoint_scores, lag_one_scores, network_of,
                    on_planes, setup_threads, split, stages_and_end, write_rows)
from stage_equations import A, B, C, _t, rates, rates_and_jacobian, residual, stage_planes

ROWS = 32768


# ------------------------------------------------------------------ the gain
def gain_update(states, z0, stages):
    """Summed end (n,4), corrected end (n,4), correction -K r (n,4) and residual
    r (n,2,4) for stages (n,2,4) of states (n,5) on planes z0 (n,). numpy."""
    n = len(states)
    summed, corrected, residuals = np.empty((n, 4)), np.empty((n, 4)), np.empty((n, 2, 4))
    for i in range(0, n, ROWS):
        S0, z = _t(states[i:i + ROWS]), _t(np.asarray(z0[i:i + ROWS]).copy())
        Y = _t(stages[i:i + ROWS])
        f, J = rates_and_jacobian(Y, S0[:, 4:5], stage_planes(z))
        with torch.no_grad():
            r = residual(S0, Y, f)                                         # (m,2,4)
            m = len(Y)
            M = torch.eye(8, dtype=torch.float64).repeat(m, 1, 1)
            for j in range(2):
                for k in range(2):
                    M[:, 4 * j:4 * j + 4, 4 * k:4 * k + 4] -= STEP_MM * A[j, k] * J[:, k]
            w = torch.linalg.solve(M, r.reshape(m, 8, 1))[..., 0].reshape(m, 2, 4)
            Kr = STEP_MM * torch.einsum("k,mkab,mkb->ma", B, J, w)
            end = S0[:, :4] + STEP_MM * torch.einsum("k,mkd->md", B, f)
        summed[i:i + m] = end.numpy()
        corrected[i:i + m] = (end - Kr).numpy()
        residuals[i:i + m] = r.numpy()
    return summed, corrected, residuals


def straight_line_stages(states, z0):
    d = np.asarray(C)[None, :] * STEP_MM
    return np.stack([states[:, None, 0] + states[:, None, 2] * d,
                     states[:, None, 1] + states[:, None, 3] * d,
                     np.repeat(states[:, None, 2], 2, 1), np.repeat(states[:, None, 3], 2, 1)],
                    axis=-1)


def euler_stages(states, z0):
    S0 = _t(states)
    with torch.no_grad():
        f0 = rates(S0[:, None, :4], S0[:, 4:5], _t(np.asarray(z0).copy())[:, None])   # (n,1,4)
        Y = S0[:, None, :4] + (C * STEP_MM)[None, :, None] * f0
    return Y.numpy()


def network_stages(network):
    return lambda states, z0: stages_and_end(network, states, z0)[0]


# Each source: stages(states, z0) and its counts per step without the gain.
def sources():
    out = []
    for series, key in RUNS:
        out.append(dict(source=SHORT[key], key=key, stages=network_stages(network_of(key)),
                        field=2, net=1))
    out.append(dict(source="straight line (no network)", key="", stages=straight_line_stages,
                    field=2, net=0))
    out.append(dict(source="explicit Euler (no network)", key="", stages=euler_stages,
                    field=3, net=0))
    return out


def plain_step(stages_of):
    return lambda s, z: gain_update(s, z, stages_of(s, z))[0]


def corrected_step(stages_of):
    return lambda s, z: gain_update(s, z, stages_of(s, z))[1]


# ------------------------------------------------------------------ helpers
def carried_to_the_end():
    J = np.load(os.path.join(BIAS_CACHE, "jacobians_of_the_steps_test.npy"))
    T = np.empty_like(J)
    T[:, -1] = np.eye(5)
    for k in range(NUMBER_OF_STEPS - 2, -1, -1):
        T[:, k] = T[:, k + 1] @ J[:, k + 1]
    return T


def single_step(stages_of):
    """Signed single-step errors (n,64,4) before and after the gain, from the
    reference states of the test tracks (um / urad), and the residuals."""
    ref = on_planes("test")
    n = len(ref)
    s = ref[:, :-1].reshape(-1, 5)
    z = np.broadcast_to(START_PLANES_MM, (n, NUMBER_OF_STEPS)).reshape(-1).copy()
    summed, corrected, r = gain_update(s, z, stages_of(s, z))
    target = ref[:, 1:, :4]
    before = (summed.reshape(n, NUMBER_OF_STEPS, 4) - target) * TO_REPORTED
    after = (corrected.reshape(n, NUMBER_OF_STEPS, 4) - target) * TO_REPORTED
    return before, after, r.reshape(n, NUMBER_OF_STEPS, 2, 4)


def single_step_row(e):
    radial = np.hypot(e[..., 0], e[..., 1])
    row = dict(single_step_radial_median_um=float(np.median(radial)),
               single_step_radial_rms_um=float(np.sqrt((radial ** 2).mean())),
               single_step_rms_tx_urad=float(np.sqrt((e[..., 2] ** 2).mean())),
               single_step_rms_ty_urad=float(np.sqrt((e[..., 3] ** 2).mean())))
    row.update(lag_one_scores(e))
    return row


# ------------------------------------------------------------------ run
def run():
    setup_threads()
    test = split("test")
    start = test.start_state
    T = carried_to_the_end()
    endpoint_rows, step_rows = [], []
    low = np.flatnonzero(test.momentum_gev < 3.0)
    track = int(low[np.argsort(test.momentum_gev[low])[len(low) // 2]])   # median track below 3 GeV
    along_track = dict(track=track, momentum_gev=float(test.momentum_gev[track]),
                       z_start_mm=START_PLANES_MM)
    exact = np.load(os.path.join(BIAS_CACHE, "e_exact_gl2_test.npy"))
    along_track["exact"] = exact[track, :, 0]
    step_rows.append(dict(source="exact 2-stage Gauss-Legendre", run_key="", variant="exact",
                          **single_step_row(exact)))
    for src in sources():
        name, stages_of = src["source"], src["stages"]
        f, net = src["field"], src["net"]
        counts_plain = dict(field_per_step=f, jacobian_per_step=0, network_per_step=net)
        counts_gain = dict(field_per_step=f, jacobian_per_step=2, network_per_step=net)

        # plain chain, and B2 along it
        states = chain(plain_step(stages_of), start)
        row, d_plain = endpoint_scores(states[:, -1])
        endpoint_rows.append(dict(source=name, run_key=src["key"], variant="plain",
                                  **counts_plain, **row))
        s = states[:, :-1].reshape(-1, 5)
        z = np.broadcast_to(START_PLANES_MM, (len(start), NUMBER_OF_STEPS)).reshape(-1).copy()
        summed, corrected, _ = gain_update(s, z, stages_of(s, z))
        delta = (corrected - summed).reshape(len(start), NUMBER_OF_STEPS, 4)
        delta5 = np.concatenate([delta, np.zeros(delta.shape[:2] + (1,))], axis=2)
        carried = np.einsum("nkab,nkb->na", T, delta5)
        row, _ = endpoint_scores(states[:, -1] + carried)
        endpoint_rows.append(dict(source=name, run_key=src["key"],
                                  variant="B2 corrections carried to the end",
                                  **counts_gain, **row))
        raw = states[:, -1].copy()
        raw[:, :4] += delta.sum(axis=1)
        row, _ = endpoint_scores(raw)
        endpoint_rows.append(dict(source=name, run_key=src["key"],
                                  variant="B2 corrections summed, not carried",
                                  **counts_gain, **row))

        # B1
        states = chain(corrected_step(stages_of), start)
        row, _ = endpoint_scores(states[:, -1])
        endpoint_rows.append(dict(source=name, run_key=src["key"], variant="B1 corrected, chained",
                                  **counts_gain, **row))
        for r in endpoint_rows[-4:]:
            print({k: r[k] for k in ("source", "variant", "endpoint_radial_median_um")}, flush=True)

        # single steps from the reference states
        before, after, res = single_step(stages_of)
        step_rows.append(dict(source=name, run_key=src["key"], variant="plain",
                              **single_step_row(before)))
        step_rows.append(dict(source=name, run_key=src["key"], variant="gain",
                              **single_step_row(after)))
        print(step_rows[-2], step_rows[-1], flush=True)
        along_track[name + " | plain"] = before[track, :, 0]
        along_track[name + " | gain"] = after[track, :, 0]
    write_rows("b_endpoint.csv", endpoint_rows)
    write_rows("b_single_step.csv", step_rows)
    np.savez(os.path.join(RESULTS, "b_low_momentum_track.npz"),
             **{k: np.asarray(v) for k, v in along_track.items()})


def check():
    """The summed end state of gain_update equals the network's own end state."""
    setup_threads()
    ref = on_planes("test")[:200]
    s = ref[:, :-1].reshape(-1, 5)
    z = np.broadcast_to(START_PLANES_MM, (200, NUMBER_OF_STEPS)).reshape(-1).copy()
    for series, key in RUNS:
        net = network_of(key)
        stages, ends = stages_and_end(net, s, z)
        summed, corrected, r = gain_update(s, z, stages)
        print(key, "max |summed - network end| (mm, slope):",
              np.abs(summed - ends).max(axis=0),
              "median |K r| x (um):", np.median(np.abs(corrected - summed)[:, 0]) * 1e3, flush=True)


if __name__ == "__main__":
    {"run": run, "check": check}[sys.argv[1]]()
