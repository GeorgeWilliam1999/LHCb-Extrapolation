"""Python versions of the classical integrators of the compiled program that
the present package does not have. They are the reference for the gate of
cpp/classical/more_integrators.h; they are not timed.

    nystrom3      fourth-order Runge-Kutta-Nystrom on given planes, three field
                  points a step (start, middle, end); the end of a step is the
                  start of the next, so N steps read the field 2 N + 1 times
    stage_sweeps  the Gauss-Legendre scheme of q stages on N equal steps, solved
                  with K sweeps of its stage map from the straight line; many
                  sweeps give the exact scheme

Both work on arrays of states (n, 5) with the numpy field map of the present
package, in double precision.
"""
from __future__ import annotations

import numpy as np

KAPPA = 1.0e-3


def _slopes(tx, ty, k, B):
    """slope_derivative of Allen: the rates of the two slopes."""
    bx, by, bz = B
    tx2, ty2 = tx * tx, ty * ty
    norm = np.sqrt(1.0 + tx2 + ty2)
    return (k * (norm * (ty * (tx * bx + bz) - (1.0 + tx2) * by)),
            k * (norm * (-tx * (ty * by + bz) + (1.0 + ty2) * bx)))


def nystrom3(states, planes, field_map):
    """Returns the final states (n, 5)."""
    s = np.array(states, dtype=np.float64)
    x, y, tx, ty = s[:, 0].copy(), s[:, 1].copy(), s[:, 2].copy(), s[:, 3].copy()
    k = KAPPA * s[:, 4]
    planes = np.asarray(planes, dtype=np.float64)
    B0 = field_map(x, y, np.full_like(x, planes[0]))
    for i in range(len(planes) - 1):
        z, dz = planes[i], planes[i + 1] - planes[i]
        dz2, half = dz * dz, 0.5 * dz
        k0x, k0y = _slopes(tx, ty, k, B0)
        B1 = field_map(x + half * tx + 0.125 * dz2 * k0x, y + half * ty + 0.125 * dz2 * k0y,
                       np.full_like(x, z + half))
        k1x, k1y = _slopes(tx + half * k0x, ty + half * k0y, k, B1)
        k2x, k2y = _slopes(tx + half * k1x, ty + half * k1y, k, B1)
        B2 = field_map(x + dz * tx + 0.5 * dz2 * k2x, y + dz * ty + 0.5 * dz2 * k2y,
                       np.full_like(x, z + dz))
        k3x, k3y = _slopes(tx + dz * k2x, ty + dz * k2y, k, B2)
        fpos, fslo = dz2 * (1.0 / 6.0), dz * (1.0 / 6.0)
        x = x + dz * tx + (k0x + k1x + k2x) * fpos
        y = y + dz * ty + (k0y + k1y + k2y) * fpos
        tx = tx + (k0x + k1x * 2.0 + k2x * 2.0 + k3x) * fslo
        ty = ty + (k0y + k1y * 2.0 + k2y * 2.0 + k3y) * fslo
        B0 = B2
    return np.stack([x, y, tx, ty, s[:, 4]], axis=1)


def stage_sweeps(states, z0, z1, steps, tableau, sweeps, field_map):
    """Returns the final states (n, 5). `tableau` is a Gauss-Legendre tableau of
    the present package."""
    s = np.array(states, dtype=np.float64)
    S = s[:, :4].copy()
    k = KAPPA * s[:, 4]
    nodes, weights, A = tableau.nodes, tableau.weights, tableau.stage_matrix
    q, dz = len(nodes), (z1 - z0) / steps

    def rates(Y, z):
        bx, by, bz = field_map(Y[:, 0], Y[:, 1], np.full(len(Y), z))
        tx, ty = Y[:, 2], Y[:, 3]
        N = np.sqrt(1 + tx * tx + ty * ty)
        return np.stack([tx, ty, k * N * (tx * ty * bx - (1 + tx * tx) * by + ty * bz),
                         k * N * ((1 + ty * ty) * bx - tx * ty * by - tx * bz)], axis=1)

    for n in range(steps):
        start = z0 + n * dz
        Y = []
        for j in range(q):
            offset = nodes[j] * dz
            Y.append(np.stack([S[:, 0] + S[:, 2] * offset, S[:, 1] + S[:, 3] * offset,
                               S[:, 2], S[:, 3]], axis=1))
        sweep = 0
        while True:
            F = [rates(Y[j], start + nodes[j] * dz) for j in range(q)]
            if sweep == sweeps:
                break
            new = []
            for j in range(q):
                total = np.zeros_like(S)
                for m in range(q):
                    total = total + A[j, m] * F[m]
                new.append(S + dz * total)
            Y = new
            sweep += 1
        total = np.zeros_like(S)
        for j in range(q):
            total = total + weights[j] * F[j]
        S = S + dz * total
    return np.concatenate([S, s[:, 4:5]], axis=1)
