"""Part A: the toy UT + SciFi fit of the per-step bias study (A.7), with a
model-error covariance Q for the network step.

Copied from ../../what_is_the_per_step_bias/analysis/toy_fit.py on 2026-10-28;
the original is not edited. The layout, the hit resolutions, the linearisation
and the noise draws are those of A.7:

  UT      4 layers at z = 2327.5, 2372.5, 2597.5, 2642.5 mm, stereo 0, +5, -5, 0 deg,
          resolution 50 um per hit
  SciFi   12 layers, stations at 7826 / 8508 / 9193 mm, layers 70 mm apart,
          stereo 0, +5, -5, 0 deg, resolution 75 um per hit

No multiple scattering, no energy loss, no VELO. A hit measures
u = x cos(a) - y sin(a). The fit parameter is theta = (x, y, tx, ty, q/p) on
2648.2 mm. The models differ only from 2648.2 to 7826 mm (64 steps). With H the
derivative of the 16 hits by theta (reference), D the derivative of the 12
SciFi hits by the state on 7826 mm and Delta the model's endpoint error on
7826 mm, the fitted theta differs from the truth by

    theta_fit - theta_true = K (noise - [0, D Delta]),   K = C H^T V^-1,
    C = (H^T V^-1 H)^-1.

The covariance V of the hits:

  no Q         V = diag(sigma^2), as in A.7
  Q            V = diag(sigma^2) + D Q_end D^T on the SciFi block, with
               Q_end = sum_k T_k Q T_k^T, the per-step Q carried to 7826 mm by the
               reference Jacobians T_k (end of step k to 7826 mm). Q is the 4 x 4
               covariance of the signed single-step errors (x, y, tx, ty) of the
               network on the validation tracks of the track's momentum band.
               This is the process noise of a Kalman filter that treats the step
               errors as independent
  Q measured at the end (extra)
               Q_end = covariance of the chained endpoint error on 7826 mm of the
               validation tracks of the band. This keeps the correlation of the
               step errors along the track, which the per-step Q drops

Pulls: (theta_fit - theta_true)_i / sqrt(C_ii) for x, y, tx, ty, q/p on
2648.2 mm, over 200 noise draws of every test track. Momentum bias: the mean of
dp/p from the model alone (no noise), over the median resolution sqrt(C_qq)/|q/p|
of the same fit.

    toy_fit_with_q.py fit      writes results/a_toy_fit_with_q.csv, a_q_per_band.csv
"""
from __future__ import annotations

import os
import sys

import numpy as np

from common import (BIAS_CACHE, NUMBER_OF_STEPS, RUNS, SHORT, START_PLANES_MM, TO_REPORTED,
                    band_masks, cached_single_step_errors, chain, network_of, plain_step,
                    setup_threads, split, write_rows)
from stage_equations import exact_step

SIGMA_UT_MM, SIGMA_FIBRE_MM = 0.050, 0.075
NOISE_DRAWS = 200
COMPONENTS = ("x", "y", "tx", "ty", "qop")


def ingredients():
    """H (n, 16, 5), D (n, 12, 5): cached by A.7 of the per-step bias study."""
    with np.load(os.path.join(BIAS_CACHE, "a7_ingredients.npz")) as f:
        return {k: f[k] for k in f.files}


def carried_to_the_end():
    J = np.load(os.path.join(BIAS_CACHE, "jacobians_of_the_steps_test.npy"))
    T = np.empty_like(J)
    T[:, -1] = np.eye(5)
    for k in range(NUMBER_OF_STEPS - 2, -1, -1):
        T[:, k] = T[:, k + 1] @ J[:, k + 1]
    return T


def band_index(momentum_gev):
    idx = np.full(len(momentum_gev), -1)
    for b, (_, m) in enumerate(band_masks(momentum_gev)):
        idx[m] = b
    return idx


def per_step_q(key):
    """Q (5 bands, 4, 4) in mm / rad: covariance of the validation single-step errors."""
    e = cached_single_step_errors(key, "validation") / TO_REPORTED
    p = split("validation").momentum_gev
    return np.stack([np.cov(e[m].reshape(-1, 4), rowvar=False) for _, m in band_masks(p)])


def endpoint_q(key):
    """Q_end (5 bands, 4, 4): covariance of the chained endpoint error, validation."""
    v = split("validation")
    final = chain(plain_step(network_of(key)), v.start_state)[:, -1]
    d = final[:, :4] - v.reference_end_state[:, :4]
    return np.stack([np.cov(d[m], rowvar=False) for _, m in band_masks(v.momentum_gev)])


def embed(Q4):
    Q5 = np.zeros(Q4.shape[:-2] + (5, 5))
    Q5[..., :4, :4] = Q4
    return Q5


def fit_one(H, D, sigma, Q_end, delta, noise):
    """Returns the fit errors over the draws (draws, n, 5), the model-only shift
    (n, 5) and C (n, 5, 5)."""
    n = len(H)
    V = np.broadcast_to(np.diag(sigma ** 2), (n, 16, 16)).copy()
    if Q_end is not None:
        V[:, 4:, 4:] += D @ Q_end @ np.moveaxis(D, 1, 2)
    Vinv = np.linalg.inv(V)
    HtVinv = np.moveaxis(H, 1, 2) @ Vinv
    C = np.linalg.inv(HtVinv @ H)
    K = C @ HtVinv
    shift = np.concatenate([np.zeros((n, 4)), np.einsum("nlj,nj->nl", D, delta)], axis=1)
    model = -np.einsum("nij,nj->ni", K, shift)
    errors = np.einsum("nij,dnj->dni", K, noise) + model[None]
    return errors, model, C


def fit():
    setup_threads()
    test = split("test")
    theta = test.start_state
    n, qop = len(theta), theta[:, 4]
    g = ingredients()
    H, D = g["H"], g["D"]
    sigma = np.concatenate([np.full(4, SIGMA_UT_MM), np.full(12, SIGMA_FIBRE_MM)])
    noise = np.random.default_rng(1).normal(size=(NOISE_DRAWS, n, 16)) * sigma
    T = carried_to_the_end()
    bands = band_index(test.momentum_gev)

    cases = [("reference (sixth order, 0.1 mm)", "", "no Q", np.zeros((n, 5)), None)]
    exact_end = chain(lambda s, z: exact_step(s, z)[0], theta)[:, -1]
    cases.append(("exact 2-stage Gauss-Legendre, 64 steps", "", "no Q",
                  exact_end - test.reference_end_state, None))
    q_rows = []
    for series, key in RUNS:
        delta = chain(plain_step(network_of(key)), theta)[:, -1] - test.reference_end_state
        Q = per_step_q(key)
        Qe = endpoint_q(key)
        Q_carried = np.einsum("nkab,nbc,nkdc->nad", T, embed(Q)[bands], T)
        for b, (label, _, _) in enumerate(__import__("common").BANDS):
            row = dict(series=SHORT[key], run_key=key, band=label)
            for kind, M in (("per_step", Q[b]), ("endpoint", Qe[b])):
                for i, a in enumerate(COMPONENTS[:4]):
                    row["%s_sigma_%s" % (kind, a)] = float(np.sqrt(M[i, i]))
                row["%s_corr_x_tx" % kind] = float(M[0, 2] / np.sqrt(M[0, 0] * M[2, 2]))
                row["%s_corr_y_ty" % kind] = float(M[1, 3] / np.sqrt(M[1, 1] * M[3, 3]))
            m = bands == b
            row["carried_per_step_sigma_x_at_7826"] = float(np.sqrt(Q_carried[m, 0, 0].mean()))
            row["carried_per_step_sigma_y_at_7826"] = float(np.sqrt(Q_carried[m, 1, 1].mean()))
            q_rows.append(row)
        cases.append((SHORT[key], key, "no Q", delta, None))
        cases.append((SHORT[key], key, "Q per step", delta, Q_carried))
        cases.append((SHORT[key], key, "Q measured at the end", delta, embed(Qe)[bands]))

    rows = []
    for model, key, q, delta, Q_end in cases:
        errors, shift, C = fit_one(H, D, sigma, Q_end, delta, noise)
        sd = np.sqrt(np.einsum("nii->ni", C))
        pulls = errors / sd[None]
        resolution = sd[:, 4] / np.abs(qop)
        bias = -shift[:, 4] / qop
        for label, m in [("all", np.ones(n, bool))] + band_masks(test.momentum_gev):
            row = dict(model=model, run_key=key, q=q, band=label, tracks=int(m.sum()),
                       resolution_dp_over_p_median=float(np.median(resolution[m])),
                       bias_dp_over_p_mean=float(bias[m].mean()),
                       bias_dp_over_p_se=float(bias[m].std(ddof=1) / np.sqrt(m.sum())),
                       bias_over_resolution=float(bias[m].mean() / np.median(resolution[m])))
            for i, a in enumerate(COMPONENTS):
                row["pull_mean_" + a] = float(pulls[:, m, i].mean())
                row["pull_width_" + a] = float(pulls[:, m, i].std())
            rows.append(row)
        print({k: rows[-6][k] for k in ("model", "q", "bias_over_resolution",
                                        "pull_width_x", "pull_width_qop")}, flush=True)
    write_rows("a_toy_fit_with_q.csv", rows)
    write_rows("a_q_per_band.csv", q_rows)


if __name__ == "__main__":
    {"fit": fit}[sys.argv[1]]()
