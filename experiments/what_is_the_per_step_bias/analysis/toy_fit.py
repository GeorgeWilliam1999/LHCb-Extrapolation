"""A.7: does the network's endpoint bias move the momentum of a track fit?

A toy least-squares fit of the state on the first plane (2648.2 mm),
theta = (x, y, tx, ty, q/p), to hits in the UT and the SciFi:

  UT      4 layers at z = 2327.5, 2372.5, 2597.5, 2642.5 mm, stereo 0, +5, -5, 0 deg,
          resolution 50 um per hit
  SciFi   12 layers, stations at 7826 / 8508 / 9193 mm, layers 70 mm apart,
          stereo 0, +5, -5, 0 deg, resolution 75 um per hit

A hit measures u = x cos(a) - y sin(a). The layout and resolutions are those
of the brief (UT about 50 um, SciFi 70 to 80 um), which follow the LHCb Tracker
Upgrade TDR (CERN-LHCC-2014-001); the values were not re-checked against the
document here. There is no multiple scattering, no energy loss and no VELO, so
the momentum resolution is that of the hits alone. This makes the resolution
small, and so the test "bias >= 10% of the resolution" strict.

Two fit models differ only from 2648.2 to 7826 mm:
  reference   the sixth-order method at 0.1 mm
  network     the network chained over the 64 steps
Both carry the state back to the UT and on from 7826 to the SciFi with the
sixth-order method at 0.1 mm (the field map covers z up to 14 m).

The fit is linear around the true state. With H the derivative of the hits by
theta (finite differences on the reference), W = 1/sigma^2 and
K = (H^T W H)^-1 H^T W, the network fit differs from the reference fit by
-K D Delta, where Delta is the network's endpoint error on 7826 mm and D the
derivative of the SciFi hits by the state on 7826 mm. A full Gauss-Newton fit
with both models evaluated exactly (3 iterations, one noise draw per track)
checks the linear answer.

    toy_fit.py fit      writes results/a7_toy_fit_by_band.csv and a7_nonlinear_check.csv
"""
from __future__ import annotations

import os
import sys

import numpy as np

from common import (CACHE, EQUATION, NUMBER_OF_STEPS, RUNS, SHORT, band_masks, chain,
                    network_of, plain_step, setup_threads, split, write_rows)

from rkpinn.integrators.runge_kutta_sixth_order import (                       # noqa: E402
    integrate_with_sixth_order, integrate_with_sixth_order_over_planes)

UT_Z = np.array([2327.5, 2372.5, 2597.5, 2642.5])
FIBRE_Z = np.concatenate([s + np.arange(4) * 70.0 for s in (7826.0, 8508.0, 9193.0)])
STEREO_DEG = np.array([0.0, 5.0, -5.0, 0.0])
UT_ANGLE = np.radians(STEREO_DEG)
FIBRE_ANGLE = np.radians(np.tile(STEREO_DEG, 3))
SIGMA_UT_MM, SIGMA_FIBRE_MM = 0.050, 0.075
FIRST, LAST = 2648.2, 7826.0
H_STEP = np.array([1e-3, 1e-3, 1e-6, 1e-6])       # x, y, tx, ty; q/p is relative 1e-4
NOISE_DRAWS = 200


def projection(states, angles):
    """u = x cos a - y sin a for states (..., L, 5) and angles (L,)."""
    return states[..., 0] * np.cos(angles) - states[..., 1] * np.sin(angles)


def perturbed(s):
    """11 copies of s (n,5): s itself, then +h and -h in each of the five."""
    out = [s.copy()]
    for j in range(5):
        h = H_STEP[j] if j < 4 else 1e-4 * np.abs(s[:, 4])
        for sign in (1.0, -1.0):
            t = s.copy()
            t[:, j] += sign * h
            out.append(t)
    return np.concatenate(out), [H_STEP[j] if j < 4 else 1e-4 * np.abs(s[:, 4]) for j in range(5)]


def derivative(values, steps, n):
    """values (11 n, ...) from perturbed(): nominal (n, ...) and d/dtheta (n, ..., 5)."""
    v = values.reshape((11, n) + values.shape[1:])
    d = np.empty(v.shape[1:] + (5,))
    for j in range(5):
        h = steps[j]
        h = h.reshape((n,) + (1,) * (v.ndim - 2)) if np.ndim(h) else h
        d[..., j] = (v[1 + 2 * j] - v[2 + 2 * j]) / (2 * h)
    return v[0], d


def to_planes(states, planes):
    """The sixth-order method from planes[0] through the others; (n, L, 5) without planes[0]."""
    return integrate_with_sixth_order_over_planes(EQUATION, states, planes,
                                                  step_length_mm=0.1)[:, 1:]


def ingredients():
    """H (n, 16, 5), D (n, 12, 5) and the hits of the truth, cached."""
    file = os.path.join(CACHE, "a7_ingredients.npz")
    if not os.path.exists(file):
        test = split("test")
        theta, end = test.start_state, test.reference_end_state
        n = len(theta)
        rows, steps = perturbed(theta)
        ut = to_planes(rows, np.concatenate([[FIRST], UT_Z[::-1]]))[:, ::-1]    # (11n, 4, 5)
        ut_nominal, dut = derivative(ut, steps, n)
        rows_e, steps_e = perturbed(end)
        fibre = to_planes(rows_e, FIBRE_Z)                                       # starts on 7826
        fibre = np.concatenate([rows_e[:, None], fibre], axis=1)
        fibre_nominal, dfibre = derivative(fibre, steps_e, n)                    # d/d(state on 7826)
        J = np.load(os.path.join(CACHE, "jacobians_of_the_steps_test.npy"))     # (n, 64, 5, 5)
        Jall = np.broadcast_to(np.eye(5), (n, 5, 5)).copy()
        for k in range(NUMBER_OF_STEPS):
            Jall = J[:, k] @ Jall
        H_ut = projection(np.moveaxis(dut, -1, -2), UT_ANGLE[None, :, None])    # (n, 4, 5)
        D = projection(np.moveaxis(dfibre, -1, -2), FIBRE_ANGLE[None, :, None])  # (n, 12, 5)
        H = np.concatenate([H_ut, D @ Jall], axis=1)
        np.savez(file, H=H, D=D, ut=projection(ut_nominal, UT_ANGLE),
                 fibre=projection(fibre_nominal, FIBRE_ANGLE), Jall=Jall)
    with np.load(file) as f:
        return {k: f[k] for k in f.files}


def fit():
    setup_threads()
    test = split("test")
    theta = test.start_state
    n = len(theta)
    g = ingredients()
    H, D = g["H"], g["D"]
    sigma = np.concatenate([np.full(4, SIGMA_UT_MM), np.full(12, SIGMA_FIBRE_MM)])
    W = 1.0 / sigma ** 2
    HtW = np.moveaxis(H, 1, 2) * W
    cov = np.linalg.inv(HtW @ H)
    K = cov @ HtW                                                    # (n, 5, 16)
    sigma_qop = np.sqrt(cov[:, 4, 4])
    qop = theta[:, 4]
    resolution = sigma_qop / np.abs(qop)                             # sigma(p)/p
    rng = np.random.default_rng(1)
    noise = rng.normal(size=(NOISE_DRAWS, n, 16)) * sigma
    pull_ref = np.einsum("nij,dnj->dni", K, noise)[..., 4] / sigma_qop
    rows, checks = [], []
    for series, key in RUNS:
        net = network_of(key)
        final = chain(plain_step(net), theta)[:, -1]
        delta = final - test.reference_end_state                     # (n, 5), q/p column 0
        shift = np.concatenate([np.zeros((n, 4)), np.einsum("nlj,nj->nl", D, delta)], axis=1)
        bias = -np.einsum("nij,nj->ni", K, shift)                    # theta_net - theta_ref
        bias_dp_over_p = -bias[:, 4] / qop
        pull_net = pull_ref + (bias[:, 4] / sigma_qop)[None]
        for label, m in [("all", np.ones(n, bool))] + band_masks(test.momentum_gev):
            rows.append(dict(
                series=series, run_key=key, band=label, tracks=int(m.sum()),
                resolution_dp_over_p_median=float(np.median(resolution[m])),
                bias_dp_over_p_mean=float(bias_dp_over_p[m].mean()),
                bias_dp_over_p_se=float(bias_dp_over_p[m].std(ddof=1) / np.sqrt(m.sum())),
                bias_dp_over_p_rms=float(np.sqrt((bias_dp_over_p[m] ** 2).mean())),
                mean_bias_over_resolution=float(bias_dp_over_p[m].mean()
                                                / np.median(resolution[m])),
                rms_of_per_track_bias_over_own_resolution=float(np.sqrt(
                    ((bias_dp_over_p[m] / resolution[m]) ** 2).mean())),
                pull_mean_reference=float(pull_ref[:, m].mean()),
                pull_width_reference=float(pull_ref[:, m].std()),
                pull_mean_network=float(pull_net[:, m].mean()),
                pull_width_network=float(pull_net[:, m].std()),
                bias_matters_if_abs_ratio_ge_0_1=bool(abs(bias_dp_over_p[m].mean())
                                                      >= 0.1 * np.median(resolution[m]))))
            print(rows[-1], flush=True)
        checks.append(nonlinear_check(key, net, K, H, sigma, bias, g))
        print(checks[-1], flush=True)
    write_rows("a7_toy_fit_by_band.csv", rows)
    write_rows("a7_nonlinear_check.csv", checks)


def nonlinear_check(key, net, K, H, sigma, linear_bias, g):
    """Gauss-Newton with both models evaluated exactly between 2648.2 and
    7826 mm (the UT and SciFi legs linearised around the truth, the same for
    both), one noise draw per track, 3 iterations."""
    test = split("test")
    theta0 = test.start_state
    n = len(theta0)
    rng = np.random.default_rng(2)
    hits = np.concatenate([g["ut"], g["fibre"]], axis=1) + rng.normal(size=(n, 16)) * sigma
    Jall, D = g["Jall"], g["D"]
    ut_rows = H[:, :4]

    def predicted_hits(theta, model):
        end = (chain(plain_step(net), theta)[:, -1] if model == "network" else
               integrate_with_sixth_order(EQUATION, theta, FIRST, LAST, step_length_mm=0.1))
        d0 = theta - theta0
        ut = g["ut"] + np.einsum("nlj,nj->nl", ut_rows, d0)
        fibre = g["fibre"] + np.einsum("nlj,nj->nl", D, end - test.reference_end_state)
        return np.concatenate([ut, fibre], axis=1)

    fitted = {}
    for model in ("reference", "network"):
        theta = theta0.copy()
        for _ in range(3):
            theta = theta + np.einsum("nij,nj->ni", K, hits - predicted_hits(theta, model))
        fitted[model] = theta
    nonlinear = fitted["network"][:, 4] - fitted["reference"][:, 4]
    return dict(run_key=key, series=SHORT[key],
                rms_linear_bias_dp_over_p=float(np.sqrt(((linear_bias[:, 4] / theta0[:, 4]) ** 2)
                                                        .mean())),
                rms_nonlinear_bias_dp_over_p=float(np.sqrt(((nonlinear / theta0[:, 4]) ** 2)
                                                           .mean())),
                rms_difference_dp_over_p=float(np.sqrt((((nonlinear - linear_bias[:, 4])
                                                         / theta0[:, 4]) ** 2).mean())),
                correlation=float(np.corrcoef(nonlinear, linear_bias[:, 4])[0, 1]))


if __name__ == "__main__":
    {"fit": fit}[sys.argv[1]]()
