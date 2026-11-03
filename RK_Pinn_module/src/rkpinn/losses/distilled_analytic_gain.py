"""The distilled analytic gain: the loss that teaches a learned gain the
linearised gain, with no reference label.

For a step with input S0, stages Y, rates F at the stages and end state S_end,

    summed end   E   = S0 + dz sum over k of b_k F_k
    correction   D   = E - S_end                  what the network took off E
    label        K r                              integrators/linearised_gain.py:
                                                  the Jacobians at Y, one solve
    loss         mean over n, d of  ( (D[n, d] - (K r)[n, d]) / s[d] ) ** 2

with s[d] the spread of component d over the first round's states, as in the
pooled loss. K r is built from the equation of motion and the stages alone. It
is the change one Newton step of the stage equations makes to the end state,
with f linearised at the stages. So the loss needs no reference integrator and
no label of a target: it is label-free in the sense of this package. It needs
q Jacobians a step in training; the network it trains needs none.

With `with_the_stage_loss` the pooled loss over the stages is added as q more
terms of the mean: r / s, squared. This is for a run that trains the stage
network with the gain. With a frozen stage network it is a constant.

K r is cached on the memory of the stage states when they carry no gradient:
a frozen stage network gives the same tensor at every call of a restart
(stage_network_with_a_learned_gain.py).

At the stages of the exact scheme r = 0, so K r = 0, and the end state is the
summed one, so D = 0: the loss is zero to rounding.
Gates: tests/test_learned_gain_and_error_conditioned_network.py.
"""
from __future__ import annotations

import torch

from rkpinn.integrators.linearised_gain import (gain_times_residual, rates_and_jacobian,
                                                residual_of_the_stages)
from rkpinn.losses.pooled import spread_of_components
from rkpinn.registry import register


@register("loss", "distilled_analytic_gain")
class DistilledAnalyticGain:
    name = "distilled_analytic_gain"
    needs_labels = False
    settings_in_a_configuration = ("with_the_stage_loss",)

    @classmethod
    def from_configuration(cls, settings, context):
        return cls(equation_of_motion=context.equation_of_motion,
                   with_the_stage_loss=settings["with_the_stage_loss"])

    def __init__(self, *, equation_of_motion, with_the_stage_loss):
        if not isinstance(with_the_stage_loss, bool):
            raise ValueError("with_the_stage_loss is true or false, not %r"
                             % (with_the_stage_loss,))
        self.equation_of_motion = equation_of_motion
        self.with_the_stage_loss = with_the_stage_loss
        self._cache = None

    def constants(self, first_round_states) -> dict:
        return {"name": self.name, "with_the_stage_loss": self.with_the_stage_loss,
                "divisor_per_component":
                    spread_of_components(first_round_states)[:4].tolist()}

    def label(self, S0, stages, start_planes_mm, step_length_mm, tableau):
        """K r (m, 4) and the rates (m, q, 4) at the stages."""
        # the track holds a new view of the network's cached tensor at every
        # call, so the key is where the numbers are. The cache keeps that
        # tensor alive, so no other tensor can be at the same place
        where = (stages.data_ptr(), tuple(stages.shape), tuple(stages.stride()))
        cached = self._cache
        if cached is not None and cached[0] == where:
            return cached[2]
        A = torch.as_tensor(tableau.stage_matrix)
        b = torch.as_tensor(tableau.weights)
        c = torch.as_tensor(tableau.nodes)
        planes = start_planes_mm[:, None] + c[None, :] * step_length_mm
        f, J = rates_and_jacobian(self.equation_of_motion, stages, S0[:, 4:5], planes)
        with torch.no_grad():
            r = residual_of_the_stages(S0, stages.detach(), f, step_length_mm, A)
            made = gain_times_residual(J, r, step_length_mm, A, b)
        if not stages.requires_grad:
            self._cache = (where, stages, made)
        return made

    def squared_terms(self, predicted_track, target, constants):
        """Shape (m, 1, 4), or (m, 1 + q, 4) with the stage loss: the term of
        the gain, then those of the stages."""
        track = predicted_track
        q = track.number_of_stages
        if q == 0:
            raise ValueError("a track with no stages has no gain to learn")
        dz = track.step_length_mm
        S0 = torch.as_tensor(track.input_states).reshape(-1, 5)
        stages = torch.as_tensor(track.stage_states).reshape(-1, q, 4)
        planes = torch.as_tensor(track.start_planes_mm).reshape(-1)
        end = torch.as_tensor(track.end_states).reshape(-1, 4)
        A = torch.as_tensor(track.tableau.stage_matrix)
        b = torch.as_tensor(track.tableau.weights)
        if track.stage_rates is not None:
            F = torch.as_tensor(track.stage_rates).reshape(-1, q, 4)
        else:
            c = torch.as_tensor(track.tableau.nodes)
            F = self.equation_of_motion.rates(stages, S0[:, 4:5],
                                              planes[:, None] + c[None, :] * dz)
        summed = S0[:, :4] + dz * torch.einsum("k,nkd->nd", b, F)
        divisor = torch.as_tensor(constants["divisor_per_component"], dtype=torch.float64)
        label = self.label(S0, stages, planes, dz, track.tableau)
        terms = (((summed - end) - label) / divisor)[:, None, :] ** 2
        if constants["with_the_stage_loss"]:
            r = residual_of_the_stages(S0, stages, F, dz, A)
            terms = torch.cat([terms, (r / divisor) ** 2], dim=1)
        return terms

    def value(self, predicted_track, target, constants):
        return self.squared_terms(predicted_track, target, constants).mean()
