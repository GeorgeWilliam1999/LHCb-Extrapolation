"""A start plus a correction, with a polynomial as the correction.

    end state = start(start state) + correction(start state)

The start is the straight line or the Nystrom integrator with N equal steps.
The correction is a polynomial of the start state:

    u_j = input_j / scale_j                       j = x, y, tx, ty, q/p
    correction_c = u_qop * sum over terms of
                   a_c,term  L_a(u_x) L_b(u_y) L_c(u_tx) L_d(u_ty) L_e(u_qop)

with L the Legendre polynomials and a + b + c + d + e <= degree. The factor
u_qop makes the correction zero for q/p = 0, where every start is exact. The
terms are in the order of the nested loops a, b, c, d, e, which is the order
of the compiled program (cpp/corrections/corrected_start.h).

The fit minimises the Huber loss of each component over the training tracks
(quadratic below 2 um for a position and 1e-6 for a slope, linear above), by
least squares with weights that are renewed. A plain least-squares fit is
ruled by the 1 % of the tracks whose correction is 100 times the typical one.

    fit(...)        the coefficients, from start states and reference end states
    export(...)     the folder <exports>/corrections/<name> for the compiled program
    correction(...) the same arithmetic in numpy, for the gate
"""
from __future__ import annotations

import os

import numpy as np

from fastex import FOLDER_OF_EXPORTS

HUBER = np.array([2e-3, 2e-3, 1e-6, 1e-6])      # mm, mm, slope, slope
RENEWALS_OF_THE_WEIGHTS = 12
FOLDER_OF_CORRECTIONS = os.path.join(FOLDER_OF_EXPORTS, "corrections")


def terms_of(degree):
    """The exponents (a, b, c, d, e) of every term, shape (terms, 5)."""
    return np.array([(a, b, c, d, e)
                     for a in range(degree + 1)
                     for b in range(degree - a + 1)
                     for c in range(degree - a - b + 1)
                     for d in range(degree - a - b - c + 1)
                     for e in range(degree - a - b - c - d + 1)], dtype=np.int64)


def legendre(u, degree):
    """L_k(u) for k = 0 .. degree, shape (degree + 1,) + u.shape."""
    table = np.ones((degree + 1,) + u.shape)
    if degree >= 1:
        table[1] = u
    for k in range(1, degree):
        table[k + 1] = ((2 * k + 1) / (k + 1)) * u * table[k] - (k / (k + 1)) * table[k - 1]
    return table


def design(states, scale, degree):
    """The value of every term times u_qop, shape (n, terms)."""
    u = np.asarray(states, dtype=np.float64) / scale
    table = legendre(u, degree)                     # (degree + 1, n, 5)
    powers = terms_of(degree)
    out = u[:, 4:5] * np.ones((1, len(powers)))
    for j in range(5):
        out *= table[powers[:, j], :, j].T
    return out


def scale_of_inputs(states):
    """One fixed scale a variable: the value that 99.5 % of the tracks are below."""
    return np.quantile(np.abs(np.asarray(states, dtype=np.float64)), 0.995, axis=0)


def fit(states, target, degree, scale=None):
    """The coefficients (terms, 4) of the polynomial that gives `target` (n, 4)
    from the start states (n, 5), and the scale of the inputs."""
    scale = scale_of_inputs(states) if scale is None else np.asarray(scale, dtype=np.float64)
    q, r = np.linalg.qr(design(states, scale, degree))
    in_q = np.zeros((q.shape[1], 4))
    for c in range(4):
        weight = np.ones(len(q))
        for _ in range(RENEWALS_OF_THE_WEIGHTS):
            qw = q * weight[:, None]
            gram = q.T @ qw + 1e-12 * np.eye(q.shape[1])
            in_q[:, c] = np.linalg.solve(gram, qw.T @ target[:, c])
            residual = np.abs(target[:, c] - q @ in_q[:, c])
            weight = np.minimum(1.0, HUBER[c] / np.maximum(residual, 1e-300))
    return np.linalg.solve(r, in_q), scale


def correction(states, coefficients, scale, degree):
    """The correction of every track, shape (n, 4)."""
    return design(states, scale, degree) @ coefficients


def export(name, coefficients, scale, degree, *, start, steps, reference, reader, first_plane_mm,
           last_plane_mm, more=()):
    """Writes <exports>/corrections/<name>: meta, scale_of_inputs.f64 and
    coefficients.f64 (component by component, each `terms` long)."""
    folder = os.path.join(FOLDER_OF_CORRECTIONS, name)
    os.makedirs(folder, exist_ok=True)
    np.ascontiguousarray(scale, dtype=np.float64).tofile(os.path.join(folder, "scale_of_inputs.f64"))
    np.ascontiguousarray(np.asarray(coefficients).T, dtype=np.float64).tofile(
        os.path.join(folder, "coefficients.f64"))
    pairs = [("name", name), ("start", start), ("steps", int(steps)), ("correction", "polynomial"),
             ("degree", int(degree)), ("terms", len(coefficients)), ("reference", reference),
             ("reader", reader), ("first_plane_mm", repr(float(first_plane_mm))),
             ("last_plane_mm", repr(float(last_plane_mm)))] + list(more)
    with open(os.path.join(folder, "meta"), "w") as handle:
        for key, value in pairs:
            handle.write("%s %s\n" % (key, value))
    return folder


def read(name):
    """The meta, the coefficients (terms, 4) and the scale of a correction."""
    folder = os.path.join(FOLDER_OF_CORRECTIONS, name)
    meta = dict(line.rstrip("\n").split(" ", 1) for line in open(os.path.join(folder, "meta")))
    scale = np.fromfile(os.path.join(folder, "scale_of_inputs.f64"))
    coefficients = np.fromfile(os.path.join(folder, "coefficients.f64")).reshape(4, -1).T
    return meta, coefficients, scale
