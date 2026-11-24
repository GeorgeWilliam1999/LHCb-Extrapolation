"""Python copies of the integrators that LHCb runs, with their default settings.

They are written from the LHCb source and they are the reference that the
compiled copies are checked against. They run one track at a time in double
precision. They give final states and counts (steps, failed steps, field
reads); they are not timed.

    rec_cash_karp           Rec v41r3, Tr/TrackExtrapolators,
                            TrackRungeKuttaExtrapolator::extrapolate with the
                            scheme CashKarp: Tolerance 0.001 mm, RelToleranceTx
                            5e-5, MinStep 10 mm, MaxStep and InitialStep 1 m,
                            Sigma 5.5, MinStepScale 0.125, MaxStepScale 4,
                            StepScaleSafetyFactor 1. The first stage of a failed
                            step is used again. The growth of the step uses
                            FastRoots::invfifthroot of LHCbMath, copied here.
                            This is the long-distance integrator of
                            TrackMasterExtrapolator, which the HLT2 fit calls.
    allen_adaptive_nystrom  Allen v9r3, RungeKuttaExtrapolator.cuh,
                            AdaptiveRKNystromExtrapolator::extrapolate:
                            tolerance 0.005 mm, steps 10 mm to 1 m, three field
                            points per step, the last one is the first of the
                            next step. In Allen v9r3 only the test algorithm
                            extrapolate_states calls it.
    allen_fast_nystrom      Allen v9r3, RungeKuttaNystromExtrapolator::propagate:
                            fixed steps of 500 mm, ONE field read per step, at
                            the middle of the straight line of the step
                            (make_fast_step). The HLT1 T-track code calls it.

The states are those of the present package: (x, y, tx, ty, q/p) with q/p =
0.299792458 * charge / momentum in GeV, which is the q/p of LHCb in 1/MeV times
c_light. The field callable gives tesla, so the factor kappa = 1e-3 stands for
the Gaudi unit of the field.

Correction (2026-11-24): the timing study of 2026-11-24 had a copy named "Allen
fixed Nystrom, 500 mm" that read the field at three points per step. That is
the step of the adaptive integrator, not of the fixed one. `allen_fast_nystrom`
is the code that Allen runs.
"""
from __future__ import annotations

import math

import numpy as np

KAPPA = 1.0e-3

# Cash-Karp, as in TrackRungeKuttaExtrapolator.cpp
CASH_KARP_A = [[], [1 / 5], [3 / 40, 9 / 40], [3 / 10, -9 / 10, 6 / 5],
               [-11 / 54, 5 / 2, -70 / 27, 35 / 27],
               [1631 / 55296, 175 / 512, 575 / 13824, 44275 / 110592, 253 / 4096]]
CASH_KARP_B = [37 / 378, 0.0, 250 / 621, 125 / 594, 0.0, 512 / 1771]
CASH_KARP_B_STAR = [2825 / 27648, 0.0, 18575 / 48384, 13525 / 55296, 277 / 14336, 1 / 4]

PROPAGATION_TOLERANCE_MM = 1e-6     # TrackParameters::propagationTolerance


class Counted:
    """The field of a map of the present package at one point, with a count of
    the reads."""

    def __init__(self, field_map):
        self.field_map, self.reads = field_map, 0

    def __call__(self, x, y, z):
        self.reads += 1
        bx, by, bz = self.field_map(np.array([x]), np.array([y]), np.array([z]))
        return float(bx[0]), float(by[0]), float(bz[0])


def inverse_fifth_root(x: float) -> float:
    """FastRoots::invfifthroot(double) of LHCbMath: a bit trick and four Newton steps."""
    xa = abs(x)
    i = int(np.array([xa], dtype=np.float64).view(np.int64)[0])
    i = 0x4cb8bc2ffc470ddb - i // 5
    r = float(np.array([i], dtype=np.int64).view(np.float64)[0])
    for _ in range(4):
        r += (r - xa * ((r * r) * (r * r) * (r * r))) / 5
    return math.copysign(r, x)


def _rec_derivative(s, k, B):
    """evaluateDerivatives of Rec: (tx, ty, qop Ax, qop Ay)."""
    tx, ty = s[2], s[3]
    bx, by, bz = B
    tx2, ty2 = tx * tx, ty * ty
    norm = math.sqrt(1 + tx2 + ty2)
    ax = norm * (ty * (tx * bx + bz) - (1 + tx2) * by)
    ay = norm * (-tx * (ty * by + bz) + (1 + ty2) * bx)
    return np.array([tx, ty, k * ax, k * ay])


def rec_cash_karp(state, z, z_out, field, tolerance=0.001, relative_tolerance_tx=5e-5,
                  min_step=10.0, max_step=1000.0, initial_step=1000.0, sigma=5.5,
                  min_step_scale=0.125, max_step_scale=4.0, safety=1.0):
    """Returns (final (x, y, tx, ty), accepted steps, failed steps)."""
    s = np.array(state[:4], dtype=np.float64)
    k = KAPPA * float(state[4])
    total = z_out - z
    tolerance_tx = tolerance / abs(total)
    absstep = min(abs(total), initial_step)
    direction = 1.0 if total > 0 else -1.0
    steps = failed = 0
    kept = None                       # the derivative of the first stage of a failed step
    while abs(z - z_out) > PROPAGATION_TOLERANCE_MM:
        dz = absstep * direction
        first = kept if kept is not None else _rec_derivative(s, k, field(s[0], s[1], z))
        ks = [dz * first]
        for m in range(1, 6):
            stage, stage_z = s.copy(), z
            for a, kk in zip(CASH_KARP_A[m], ks):
                stage = stage + a * kk
                stage_z = stage_z + a * dz
            ks.append(dz * _rec_derivative(stage, k, field(stage[0], stage[1], stage_z)))
        err = np.zeros(4)
        new = s.copy()
        for m in range(6):
            err = err + (CASH_KARP_B[m] - CASH_KARP_B_STAR[m]) * ks[m]
            new = new + CASH_KARP_B[m] * ks[m]
        success = absstep <= min_step
        if not success:
            normdx = abs(err[0]) / tolerance
            normdy = abs(err[1]) / tolerance
            normdtx = abs(err[2]) / (tolerance_tx + abs(new[2] - s[2]) * relative_tolerance_tx)
            eot = max(normdx, max(normdy, normdtx))
            success = eot <= sigma
            if eot > 1:
                factor = max(min_step_scale, safety / math.sqrt(math.sqrt(eot)))
            elif eot > 0:
                factor = min(max_step_scale, safety * inverse_fifth_root(eot))
            else:
                factor = max_step_scale
            absstep = max(min_step, min(absstep * factor, max_step))
        if success:
            s, z, steps, kept = new, z + dz, steps + 1, None
        else:
            failed, kept = failed + 1, first
        if absstep > direction * (z_out - z):
            absstep = abs(z_out - z)
    return s, steps, failed


def _slope_derivative(tx, ty, k, B):
    """slope_derivative of Allen (ExtrapolatorCommon.cuh)."""
    bx, by, bz = B
    tx2, ty2 = tx * tx, ty * ty
    norm = math.sqrt(1.0 + tx2 + ty2)
    ax = norm * (ty * (tx * bx + bz) - (1.0 + tx2) * by)
    ay = norm * (-tx * (ty * by + bz) + (1.0 + ty2) * bx)
    return np.array([k * ax, k * ay])


def allen_adaptive_nystrom(state, z, z_out, field, tolerance=0.005, min_step=10.0,
                           max_step=1000.0, sigma=5.5, min_step_scale=0.125,
                           max_step_scale=4.0, max_steps=1000):
    """Returns (final (x, y, tx, ty), accepted steps, failed steps)."""
    x, y, tx, ty = (float(v) for v in state[:4])
    k = KAPPA * float(state[4])
    direction = 1.0 if z_out > z else -1.0
    absstep = min(max_step, abs(z_out - z))
    k0 = None
    B0 = None
    steps = failed = counter = 0
    while counter < max_steps and abs(z - z_out) > 1e-3:
        dz = direction * absstep
        if counter == 0:
            B0 = field(x, y, z)
        if k0 is None:
            k0 = _slope_derivative(tx, ty, k, B0)
        dz2, half = dz * dz, 0.5 * dz
        B1 = field(x + half * tx + 0.125 * dz2 * k0[0], y + half * ty + 0.125 * dz2 * k0[1],
                   z + half)
        k1 = _slope_derivative(tx + half * k0[0], ty + half * k0[1], k, B1)
        k2 = _slope_derivative(tx + half * k1[0], ty + half * k1[1], k, B1)
        B2 = field(x + dz * tx + 0.5 * dz2 * k2[0], y + dz * ty + 0.5 * dz2 * k2[1], z + dz)
        k3 = _slope_derivative(tx + dz * k2[0], ty + dz * k2[1], k, B2)
        err = (k0 - k1 - k2 + k3) * dz2
        success = absstep <= min_step
        previous = absstep
        if not success:
            eot = max(abs(err[0]), abs(err[1])) / tolerance
            success = eot <= sigma
            factor = max_step_scale
            if eot > 1:
                factor = 1.0 / math.sqrt(math.sqrt(eot))
            elif eot > 0:
                factor = eot ** -0.2
            factor = max(min_step_scale, min(factor, max_step_scale))
            absstep = max(min_step, min(absstep * factor, max_step))
        if success:
            step = direction * previous
            position = (k0 + k1 + k2) * (step * step * (1.0 / 6.0))
            slope = (k0 + k1 * 2.0 + k2 * 2.0 + k3) * (step * (1.0 / 6.0))
            x, y = x + step * tx + position[0], y + step * ty + position[1]
            z = z + step
            tx, ty = tx + slope[0], ty + slope[1]
            B0, k0, steps = B2, None, steps + 1
        else:
            failed += 1
        z_diff = z_out - z
        if absstep - direction * z_diff > 0:
            absstep = abs(z_diff)
            direction = 1.0 if z_out > z else -1.0
        counter += 1
    return np.array([x, y, tx, ty]), steps, failed


def _gamma(tx, ty, B):
    """gamma of RungeKuttaNystromExtrapolator (Allen)."""
    bx, by, bz = B
    norm = math.sqrt(1.0 + tx * tx + ty * ty)
    return np.array([norm * (tx * ty * bx - (1.0 + tx * tx) * by + ty * bz),
                     norm * ((1.0 + ty * ty) * bx - tx * ty * by - tx * bz)])


def allen_fast_nystrom(state, z, z_out, field, step=500.0, max_steps=100):
    """Returns (final (x, y, tx, ty), steps, 0)."""
    x, y, tx, ty = (float(v) for v in state[:4])
    k = KAPPA * float(state[4])
    direction = 1.0 if z_out > z else -1.0
    steps = 0
    for _ in range(max_steps):
        dz = direction * min(step, abs(z_out - z))
        B = field(x + 0.5 * tx * dz, y + 0.5 * ty * dz, z + 0.5 * dz)
        c = (0.0, 0.5, 0.5, 1.0)
        ks = [_gamma(tx, ty, B) * k]
        for stage in range(1, 4):
            t = np.array([tx, ty]) + ks[stage - 1] * (dz * c[stage])
            ks.append(_gamma(t[0], t[1], B) * k)
        dR = np.array([tx, ty]) * dz + (ks[0] + ks[1] + ks[2]) * (dz * dz * (1.0 / 6.0))
        dT = (ks[0] + ks[1] * 2.0 + ks[2] * 2.0 + ks[3]) * (dz * (1.0 / 6.0))
        x, y, z = x + dR[0], y + dR[1], z + dz
        tx, ty = tx + dT[0], ty + dT[1]
        steps += 1
        if abs(z - z_out) < 1.0:
            break
    return np.array([x, y, tx, ty]), steps, 0


INTEGRATORS = {"rec_cash_karp": rec_cash_karp,
               "allen_adaptive_nystrom": allen_adaptive_nystrom,
               "allen_fast_nystrom": allen_fast_nystrom}


def carry_tracks(name, start_states, first_plane_mm, last_plane_mm, field_map, **settings):
    """Every start state through one integrator. Returns the final states
    (n, 5) and the counts (n, 3): field reads, accepted steps, failed steps."""
    integrate = INTEGRATORS[name]
    finals = np.empty((len(start_states), 5))
    counts = np.empty((len(start_states), 3), dtype=np.int64)
    for i, state in enumerate(start_states):
        field = Counted(field_map)
        final, steps, failed = integrate(state, first_plane_mm, last_plane_mm, field, **settings)
        finals[i, :4], finals[i, 4] = final, state[4]
        counts[i] = field.reads, steps, failed
    return finals, counts
