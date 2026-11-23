"""The stage residual of a path with a drift term: the label-free loss of a
whole crossing, plus a term that estimates, without labels, how the errors of
the steps add up along the track.

    loss = L + lambda * D

L is the stage residual of a path (stage_residual_of_a_path.py), with its
weights and its `gradient_through_plane_states`. D is one of three drift terms,
chosen by `drift_term`. Each is zero exactly when the stages solve the scheme.

Why (experiments/can_a_label_free_path_network_reach_the_exact_scheme/analysis/
failure_of_A1): the local defects of consecutive steps of a trained network
have one sign over long parts of a track, so they add up; L does not see the
sign. The sum of the carried defects ranks the tracks by their endpoint error
with a Spearman correlation of 0.9995 on A1 at round 37.

`endpoint_sum`, `block_sums`

For step n of a track, from its stages Y, rates F = f(Y) and residual r:

    swept stages     Y' = Y - r                          one fixed-point sweep
    local defect     delta_n = h sum_i b_i (F_ni - f(Y'_ni))      (x, y, tx, ty)

Its x part is h sum_i b_i r_ni,tx exactly. For a block B of consecutive steps
that ends on the plane z_B, the defect of step n carried to z_B is

    c_n,x  = delta_n,x + delta_n,tx (z_B - z_n - h)      the same in y
    c_n,tx = delta_n,tx                                  the same in ty

with z_n the start plane of step n. Then

    D = mean over the tracks and the components d of
        sum over the scales k of sum over the blocks B of k steps of
            ( sum_{n in B} c_n,d / sigma_d )^2

with sigma_d the divisor of L. `endpoint_sum` is the whole track as one block,
in x and y only: the carried-defect estimate of the endpoint error,

    D = mean over the tracks and d in (x, y) of ( sum_n c_n,d / sigma_d )^2

as in pooled_with_a_carried_defect_term.py. For `block_sums` the scales are `block_scales_in_steps`; each
divides the number of steps of the track. At the scale of the whole track the
x and y parts are the carried-defect estimate of the endpoint error, as in
pooled_with_a_carried_defect_term.py. A track of one step (the separate steps
of the check before training) is one block of one step at every scale.

`newton_endpoint`

The correction one Newton sweep makes to the state on the last plane,

    D = mean over the tracks and the components d of ( (J^-1 G(U))_end,d / sigma_d )^2

with G(U) and J the stage equations and their Jacobian as in
evaluation/newton_sweep.py, solved by the same forward substitution over the
steps. J is detached: the gradient flows through G only. G is taken with the
gradient through the plane states, whatever L does.

lambda is fixed once, when the run is made, so that the drift term is the share
`share_of_the_drift_term_at_round_one` of the whole loss of the first round's
states under the network as it is drawn from the seed, as in
pooled_with_a_bias_term.py:

    lambda = share L1 / ((1 - share) D1)

With the share 0, lambda is 0 and the loss is the stage residual of a path to
the last bit. `squared_terms` gives the terms of L alone.

Added 2026-11-23 for the experiment can_a_label_free_path_network_reach_the_exact_scheme
(runs C1, C2 and C3, approved by George the same day). stage_residual_of_a_path.py
is unchanged, so no run made before changes its key.
Gates: tests/test_drift_term.py.
"""
from __future__ import annotations

import torch

from rkpinn.losses.stage_residual_of_a_path import (
    CAUSAL, StageResidualOfAPath, residual_of_a_path)
from rkpinn.registry import register

ENDPOINT_SUM = "endpoint_sum"
BLOCK_SUMS = "block_sums"
NEWTON_ENDPOINT = "newton_endpoint"
DRIFT_TERMS = (ENDPOINT_SUM, BLOCK_SUMS, NEWTON_ENDPOINT)


def _parts(track, equation_of_motion):
    """(Y, F, qop per row, stage planes (n, S, q), input states) with the gradient."""
    Y = torch.as_tensor(track.stage_states)
    inputs = torch.as_tensor(track.input_states)
    n, S, q = Y.shape[:3]
    z = torch.as_tensor(track.stage_planes_mm(), dtype=torch.float64)
    qop = inputs[:, :, 4:5].reshape(n * S, 1)
    if track.stage_rates is not None:
        F = torch.as_tensor(track.stage_rates)
    else:
        F = equation_of_motion.rates(Y.reshape(n * S, q, 4), qop, z.reshape(n * S, q)) \
            .reshape(n, S, q, 4)
    return Y, F, qop, z, inputs


def local_defects(track, equation_of_motion):
    """delta, shape (n, S, 4): the end state summed from the stages minus the
    end state summed from the swept stages Y - r."""
    Y, F, qop, z, _ = _parts(track, equation_of_motion)
    n, S, q = Y.shape[:3]
    r = residual_of_a_path(track, equation_of_motion, True)
    swept = equation_of_motion.rates((Y - r).reshape(n * S, q, 4), qop,
                                     z.reshape(n * S, q)).reshape(n, S, q, 4)
    b = torch.as_tensor(track.tableau.weights, dtype=torch.float64)
    return float(track.step_length_mm) * torch.einsum("i,bnid->bnd", b, F - swept)


def _starts(track, n, S):
    starts = torch.as_tensor(track.start_planes_mm, dtype=torch.float64)
    return starts.expand(n, S) if starts.dim() == 2 else starts[None].expand(n, S)


def endpoint_sum(track, equation_of_motion, divisor):
    """Per track, (sum_n c_n / sigma)^2 meaned over x and y, with c carried to
    the end of the track's last step; shape (n,)."""
    delta = local_defects(track, equation_of_motion)                       # (n, S, 4)
    n, S = delta.shape[:2]
    h = float(track.step_length_mm)
    starts = _starts(track, n, S)
    lever = (starts[:, -1:] + h - starts - h)[..., None]                    # (n, S, 1)
    carried = delta[..., :2] + delta[..., 2:] * lever
    sigma = torch.as_tensor(divisor, dtype=torch.float64)[:2]
    return ((carried.sum(1) / sigma) ** 2).mean(-1)


def block_sums(track, equation_of_motion, divisor, scales):
    """Per track, sum over the scales and the blocks of (sum c / sigma)^2,
    meaned over the components; shape (n,)."""
    delta = local_defects(track, equation_of_motion)                       # (n, S, 4)
    n, S = delta.shape[:2]
    h = float(track.step_length_mm)
    starts = _starts(track, n, S)
    sigma = torch.as_tensor(divisor, dtype=torch.float64)
    total = torch.zeros(n, dtype=torch.float64)
    for k in scales:
        k = min(int(k), S)
        if S % k:
            raise ValueError("a block of %d steps does not divide a track of %d steps" % (k, S))
        blocks = S // k
        ends = starts.reshape(n, blocks, k)[:, :, -1:] + h                  # (n, blocks, 1)
        lever = (ends - starts.reshape(n, blocks, k) - h)[..., None]        # (n, blocks, k, 1)
        d = delta.reshape(n, blocks, k, 4)
        carried = torch.cat([d[..., :2] + d[..., 2:] * lever, d[..., 2:]], dim=-1)
        total = total + ((carried.sum(2) / sigma) ** 2).sum(1).mean(-1)
    return total


def newton_endpoint_correction(track, equation_of_motion):
    """(J^-1 G(U)) on the last plane, shape (n, 4), with J detached and the
    gradient through G. The forward substitution of evaluation/newton_sweep.py;
    the correction it adds is minus this."""
    Y, F, qop, z, inputs = _parts(track, equation_of_motion)
    n, S, q = Y.shape[:3]
    h = float(track.step_length_mm)
    A = torch.as_tensor(track.tableau.stage_matrix, dtype=torch.float64)
    b = torch.as_tensor(track.tableau.weights, dtype=torch.float64)
    ends = torch.as_tensor(track.end_states)
    G1 = residual_of_a_path(track, equation_of_motion, True)                # (n, S, q, 4)
    if S > 1:
        increments = h * torch.einsum("i,bnid->bnd", b, F[:, :-1])
        first = inputs[:, :1, :4]
        planes = torch.cat([first, first + torch.cumsum(increments, dim=1)], dim=1)
    else:
        planes = inputs[:, :, :4]
    G2 = ends - planes - h * torch.einsum("i,bnid->bnd", b, F)              # (n, S, 4)
    with torch.enable_grad():
        flat = Y.detach().reshape(-1, 1, 4).clone().requires_grad_(True)
        f = equation_of_motion.rates(flat, qop.repeat_interleave(q, 0), z.reshape(-1, 1))[:, 0]
        rows = [torch.autograd.grad(f[:, d].sum(), flat, retain_graph=d < 3)[0][:, 0]
                for d in range(4)]
    J = torch.stack(rows, dim=1).reshape(n, S, q, 4, 4)
    I = torch.eye(4 * q, dtype=torch.float64)
    ds = torch.zeros(n, 4, dtype=torch.float64)
    for m in range(S):
        Jm = J[:, m]
        M = I - h * torch.einsum("ij,bjkl->bikjl", A, Jm).reshape(n, 4 * q, 4 * q)
        dS = torch.linalg.solve(M, (-G1[:, m] + ds[:, None, :]).reshape(n, 4 * q, 1))
        dS = dS.reshape(n, q, 4)
        ds = ds + h * torch.einsum("i,bikl,bil->bk", b, Jm, dS) - G2[:, m]
    return -ds


@register("loss", "stage_residual_of_a_path_with_a_drift_term")
class StageResidualOfAPathWithADriftTerm(StageResidualOfAPath):
    name = "stage_residual_of_a_path_with_a_drift_term"
    needs_labels = False
    settings_in_a_configuration = StageResidualOfAPath.settings_in_a_configuration + (
        "drift_term", "share_of_the_drift_term_at_round_one", "block_scales_in_steps")

    @classmethod
    def from_configuration(cls, settings, context):
        return cls(equation_of_motion=context.equation_of_motion, **settings)

    def __init__(self, *, drift_term, share_of_the_drift_term_at_round_one,
                 block_scales_in_steps=None, **settings):
        super().__init__(**settings)
        if drift_term not in DRIFT_TERMS:
            raise ValueError("the drift term is %s, not %r" % (" or ".join(DRIFT_TERMS),
                                                               drift_term))
        share = float(share_of_the_drift_term_at_round_one)
        if not 0.0 <= share < 1.0:
            raise ValueError("the share of the drift term is in [0, 1), not %r" % share)
        if drift_term == BLOCK_SUMS:
            if not block_scales_in_steps or any(int(k) < 1 for k in block_scales_in_steps):
                raise ValueError("the block sums need block_scales_in_steps, each at least 1")
            block_scales_in_steps = [int(k) for k in block_scales_in_steps]
        elif block_scales_in_steps is not None:
            raise ValueError("the drift term %r does not use block_scales_in_steps; "
                             "give it as null" % drift_term)
        self.drift_term = drift_term
        self.share = share
        self.block_scales_in_steps = block_scales_in_steps

    # -- the drift term -------------------------------------------------------------
    def drift(self, predicted_track, constants):
        divisor = constants["divisor_per_component"]
        if self.drift_term == ENDPOINT_SUM:
            return endpoint_sum(predicted_track, self.equation_of_motion, divisor).mean()
        if self.drift_term == BLOCK_SUMS:
            return block_sums(predicted_track, self.equation_of_motion, divisor,
                              self.block_scales_in_steps).mean()
        correction = newton_endpoint_correction(predicted_track, self.equation_of_motion)
        sigma = torch.as_tensor(divisor, dtype=torch.float64)
        return ((correction / sigma) ** 2).mean()

    def residual_part(self, predicted_track, constants):
        L = self.per_step(predicted_track, constants)
        w = self.weights_of_the_steps(L, constants)
        return (w * L).sum() / w.sum()

    # -- the constants --------------------------------------------------------------
    def constants_with_the_network(self, first_round, network) -> dict:
        """The constants of the stage residual of a path, and lambda from the
        first round's states under the network as it is drawn from the seed."""
        if self.weights == CAUSAL:
            constants = StageResidualOfAPath.constants_with_the_network(self, first_round,
                                                                        network)
        else:
            constants = StageResidualOfAPath.constants(self, first_round.states)
        constants["name"] = self.name
        with torch.no_grad():
            track = network.predict(torch.as_tensor(first_round.states),
                                    torch.as_tensor(first_round.start_planes_mm))
            residual = float(self.residual_part(track, constants))
            drift = float(self.drift(track, constants))
        self.stop_holding()
        weight = 0.0 if self.share == 0.0 else self.share * residual / ((1.0 - self.share)
                                                                         * drift)
        constants.update(drift_term=self.drift_term,
                         share_of_the_drift_term_at_round_one=self.share,
                         block_scales_in_steps=self.block_scales_in_steps,
                         residual_at_round_one=residual, drift_at_round_one=drift,
                         weight_of_the_drift_term=weight)
        return constants

    def constants(self, first_round_states):
        raise ValueError("the constants of %r need the network: the run measures them "
                         "with constants_with_the_network" % self.name)

    # -- the loss -------------------------------------------------------------------
    def value(self, predicted_track, target, constants):
        residual = self.residual_part(predicted_track, constants)
        weight = float(constants["weight_of_the_drift_term"])
        if weight == 0.0:
            return residual
        return residual + weight * self.drift(predicted_track, constants)
