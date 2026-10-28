"""The 2-stage Gauss-Legendre stage equations, solved or polished in torch.

For a step from S0 on z0 of length dz, the stage states Y_1, Y_2 satisfy

    G_j(Y) = Y_j - S0 - dz sum_k A_jk f(Y_k, z0 + c_k dz) = 0,      j = 1, 2
    S_end  = S0 + dz sum_k b_k f(Y_k, z0 + c_k dz).

Three operations, all vectorised over rows:

  fixed-point sweep   Y <- S0 + dz A f(Y)                 2 field calls a sweep
  Newton step         (I - dz A (x) J_f) dY = -G(Y)        2 field calls with the
                                                          gradient of the map a step
  solve               Newton from the straight line until the scaled residual is
                      below 1e-12; the exact scheme, used here as a cross-check of
                      the package's root finder and for the exact single steps.

The end state then needs f at the final stages: 2 more field calls. The field
gradient is that of the trilinear map in closed form inside a cell, taken by
torch autograd through the package's differentiable map. On a cell face torch
returns the gradient of one side.

Field calls are counted per step as calls of f at one stage state (one field
lookup), the unit of the brief.
"""
from __future__ import annotations

import numpy as np
import torch

from common import DIFFERENTIABLE, STEP_MM, TABLEAU

A = torch.as_tensor(TABLEAU.stage_matrix)
B = torch.as_tensor(TABLEAU.weights)
C = torch.as_tensor(TABLEAU.nodes)
RESIDUAL_SCALE = torch.tensor([1.0, 1.0, 1e-3, 1e-3], dtype=torch.float64)


def _t(a):
    return torch.as_tensor(np.asarray(a, dtype=np.float64))


def stage_planes(z0):
    return z0[:, None] + C[None, :] * STEP_MM


def rates(Y, qop, zs):
    return DIFFERENTIABLE.rates(Y, qop, zs)


def rates_and_jacobian(Y, qop, zs):
    """f (n,q,4) and J (n,q,4,4), J[..., a, b] = d f_a / d Y_b."""
    Y = Y.detach().clone().requires_grad_(True)
    f = DIFFERENTIABLE.rates(Y, qop, zs)
    J = torch.empty(Y.shape + (4,), dtype=torch.float64)
    for a in range(4):
        (g,) = torch.autograd.grad(f[..., a].sum(), Y, retain_graph=a < 3)
        J[..., a, :] = g
    return f.detach(), J


def end_from_rates(S0, f):
    return S0[:, :4] + STEP_MM * torch.einsum("k,nkd->nd", B, f)


def residual(S0, Y, f):
    return Y - S0[:, None, :4] - STEP_MM * torch.einsum("jk,nkd->njd", A, f)


def newton_update(S0, Y, f, J):
    n = len(Y)
    G = residual(S0, Y, f).reshape(n, 8)
    M = torch.eye(8, dtype=torch.float64).repeat(n, 1, 1)
    for j in range(2):
        for k in range(2):
            M[:, 4 * j:4 * j + 4, 4 * k:4 * k + 4] -= STEP_MM * A[j, k] * J[:, k]
    dY = torch.linalg.solve(M, -G[..., None])[..., 0]
    return Y + dY.reshape(n, 2, 4)


def polish(states, z0, stages, *, sweeps=0, newton_steps=0):
    """End states (n,4) after `sweeps` fixed-point sweeps or `newton_steps`
    Newton steps from the given stages. numpy in, numpy out."""
    S0, z0, Y = _t(states), _t(z0), _t(stages)
    qop, zs = S0[:, 4:5], stage_planes(z0)
    with torch.no_grad():
        for _ in range(sweeps):
            f = rates(Y, qop, zs)
            Y = S0[:, None, :4] + STEP_MM * torch.einsum("jk,nkd->njd", A, f)
    for _ in range(newton_steps):
        f, J = rates_and_jacobian(Y, qop, zs)
        with torch.no_grad():
            Y = newton_update(S0, Y, f, J)
    with torch.no_grad():
        f = rates(Y, qop, zs)
        return end_from_rates(S0, f).numpy(), Y.numpy()


def exact_step(states, z0, tolerance=1e-12, most=30, rows_at_a_time=200000):
    """The exact scheme by Newton from the straight line. Returns the end
    states (n,4), the stages (n,2,4) and the largest scaled residual."""
    states = np.asarray(states, dtype=np.float64)
    z0 = np.broadcast_to(np.asarray(z0, dtype=np.float64), (len(states),))
    ends = np.empty((len(states), 4))
    stages = np.empty((len(states), 2, 4))
    worst = 0.0
    for i in range(0, len(states), rows_at_a_time):
        S0, z = _t(states[i:i + rows_at_a_time]), _t(z0[i:i + rows_at_a_time].copy())
        qop, zs = S0[:, 4:5], stage_planes(z)
        d = (C * STEP_MM)[None, :, None]
        Y = torch.stack([S0[:, None, 0] + S0[:, None, 2] * d[..., 0],
                         S0[:, None, 1] + S0[:, None, 3] * d[..., 0],
                         S0[:, None, 2].expand(-1, 2), S0[:, None, 3].expand(-1, 2)], dim=-1)
        for _ in range(most):
            f, J = rates_and_jacobian(Y, qop, zs)
            r = (residual(S0, Y, f).abs() / RESIDUAL_SCALE).amax().item()
            if r < tolerance:
                break
            with torch.no_grad():
                Y = newton_update(S0, Y, f, J)
        with torch.no_grad():
            f = rates(Y, qop, zs)
            r = (residual(S0, Y, f).abs() / RESIDUAL_SCALE).amax().item()
            ends[i:i + rows_at_a_time] = end_from_rates(S0, f).numpy()
            stages[i:i + rows_at_a_time] = Y.numpy()
        worst = max(worst, r)
    return ends, stages, worst
