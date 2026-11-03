"""The linearised gain: one Newton step of the stage equations, mapped to the
end of the step, with no new field lookup.

For one step from S0 on z0 of length dz with q stages Y_k on the planes
z0 + c_k dz and the rates F_k = f(Y_k, z0 + c_k dz) that the summed end state
already uses:

    residual    r_j = Y_j - S0 - dz sum over k of A_jk F_k          (4q numbers)
    Jacobian    J_k = df/dY at Y_k, 4 x 4, q/p fixed
    stage map   M   = I - dz (A (x) J),  the block (j, k) is  delta_jk I - dz A_jk J_k
    gain        K   = dz (b^T (x) J) M^-1,  shape 4 x 4q,  the block k of
                      b^T (x) J is b_k J_k
    end state   S_end = S0 + dz sum over k of b_k F_k  -  K r

Why. A Newton step of the stage equations G(Y) = 0 moves the stages by
dY = -M^-1 r. With f at the new stages replaced by F_k + J_k dY_k, the summed
end state moves by dz sum_k b_k J_k dY_k = -K r. So K r needs the Jacobians at
the stages the step already has, and no lookup of f at a new state.

The Kalman analogy. r is the innovation: what the stage equations, used as an
observation of the stages, say is wrong. K is the gain that maps the
innovation to a correction of the state that is carried on. Here the gain is
exact to first order and deterministic: there is no noise covariance.

The arithmetic and its order are those of `gain_update` in the Part B script of
the experiment "does a Kalman-like gain remove the bias", so that the two give
the same numbers to the last bit. The residual is formed in that order,
Y - S0 - dz A F, which differs from the order of `losses/stage_residual.py`
(Y - dz A F - S0) by rounding only.

Counts per step: q field lookups (f at the stages, which the summed end state
needs anyway), q Jacobian evaluations (they reuse the lookups through
autograd: the gradient of the trilinear map comes from the same 8 cell
corners), one 4q x 4q solve.
Gates: tests/test_learned_gain_and_error_conditioned_network.py.
"""
from __future__ import annotations

import torch


def rates_and_jacobian(equation_of_motion, stage_states, charge_over_momentum,
                       stage_planes_mm):
    """f, shape (n, q, 4), and J, shape (n, q, 4, 4), J[..., a, b] = df_a / dY_b,
    at the stage states (n, q, 4). The differentiable equation of motion. No
    graph is kept."""
    Y = stage_states.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        f = equation_of_motion.rates(Y, charge_over_momentum, stage_planes_mm)
        J = torch.empty(Y.shape + (4,), dtype=torch.float64)
        for a in range(4):
            (g,) = torch.autograd.grad(f[..., a].sum(), Y, retain_graph=a < 3)
            J[..., a, :] = g
    return f.detach(), J


def residual_of_the_stages(input_states, stage_states, stage_rates, step_length_mm,
                           stage_matrix):
    """r, shape (n, q, 4): Y - S0 - dz A F."""
    return (stage_states - input_states[:, None, :4]
            - step_length_mm * torch.einsum("jk,nkd->njd", stage_matrix, stage_rates))


def stage_map(jacobians, step_length_mm, stage_matrix):
    """M = I - dz (A (x) J), shape (n, 4q, 4q), from J of shape (n, q, 4, 4)."""
    n, q = jacobians.shape[:2]
    M = torch.eye(4 * q, dtype=torch.float64).repeat(n, 1, 1)
    for j in range(q):
        for k in range(q):
            M[:, 4 * j:4 * j + 4, 4 * k:4 * k + 4] -= \
                step_length_mm * stage_matrix[j, k] * jacobians[:, k]
    return M


def gain_times_residual(jacobians, residual, step_length_mm, stage_matrix, weights):
    """K r, shape (n, 4), from J (n, q, 4, 4) and r (n, q, 4)."""
    n, q = residual.shape[:2]
    M = stage_map(jacobians, step_length_mm, stage_matrix)
    w = torch.linalg.solve(M, residual.reshape(n, 4 * q, 1))[..., 0].reshape(n, q, 4)
    return step_length_mm * torch.einsum("k,mkab,mkb->ma", weights, jacobians, w)


def gain_matrix(jacobians, step_length_mm, stage_matrix, weights):
    """K, shape (n, 4, 4q). K r of `gain_times_residual` is K times r flattened."""
    n, q = jacobians.shape[:2]
    M = stage_map(jacobians, step_length_mm, stage_matrix)
    bJ = step_length_mm * torch.cat([weights[k] * jacobians[:, k] for k in range(q)], dim=2)
    # K M = bJ, so K = bJ M^-1, solved as M^T K^T = bJ^T
    return torch.linalg.solve(M.transpose(1, 2), bJ.transpose(1, 2)).transpose(1, 2)


def analytic_correction(equation_of_motion, input_states, stage_states, start_planes_mm,
                        step_length_mm, tableau):
    """The summed end state (n, 4), the corrected end state (n, 4), K r (n, 4)
    and r (n, q, 4) of one step. What Part B calls B1 for one step."""
    A = torch.as_tensor(tableau.stage_matrix)
    b = torch.as_tensor(tableau.weights)
    c = torch.as_tensor(tableau.nodes)
    planes = start_planes_mm[:, None] + c[None, :] * step_length_mm
    f, J = rates_and_jacobian(equation_of_motion, stage_states, input_states[:, 4:5], planes)
    with torch.no_grad():
        r = residual_of_the_stages(input_states, stage_states.detach(), f, step_length_mm, A)
        Kr = gain_times_residual(J, r, step_length_mm, A, b)
        end = input_states[:, :4] + step_length_mm * torch.einsum("k,mkd->md", b, f)
    return end, end - Kr, Kr, r
