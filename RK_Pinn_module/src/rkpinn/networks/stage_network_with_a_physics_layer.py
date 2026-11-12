"""The stage network with a physics layer: the stage map is the last layer.

One step from S0 on z0, with q stages and `sweeps_inside` = K:

    body        Y^(0) = N(S0, z0)                         the stage network
    layer k     F^(k-1) = f(Y^(k-1), z0 + c dz)           q field lookups
                Y^(k)   = S0 + dz A F^(k-1)                k = 1 .. K
    rates       F^(K)   = f(Y^(K), z0 + c dz)             q field lookups
    end state   S_end   = S0 + dz sum over k of b_k F^(K)

The layer has no weights. It is the stage equations of the scheme, applied to
what the body emits, and the gradient of the loss flows through it to the
body. The stages of the predicted track are Y^(K), so every loss and every
evaluation reads the stages after the layer. The stages the body emitted are
kept in the track's `first_call`.

The stage residual of the track is r^(K) = Y^(K) - S0 - dz A F^(K)
= dz A (F^(K-1) - F^(K)): the body is trained so that its stages, passed
through the stage map, are a fixed point of that map.

With `sweeps_inside` 0 the network is the stage network, to the last bit.

Why (George, 2026-11-12: a physics layer inside the network counts as "the
network is the process"; plan "RK-PINN: plan for the next phase", option B):
one fixed-point sweep of the stage equations takes every trained stage network
to the exact scheme (per-step bias study, item C.2). Here the sweep is part of
the network and the body is trained through it.

Cost of one step at inference: (K + 1) q field lookups, one call of the body,
no Jacobian, no solve. The end state is summed from the stages; a predicted
end state is refused.

`stage_network.py` is unchanged; this is a separate component, so no run made
before changes its key.
Gates: tests/test_physics_layer_and_carried_defect.py.
"""
from __future__ import annotations

import torch

from rkpinn.networks.collect_and_sum import collect_and_sum, rates_at_the_stages
from rkpinn.networks.stage_network import StageNetwork
from rkpinn.predicted_track.predicted_track import END_STATE_SUMMED, PredictedTrack
from rkpinn.registry import register


@register("network", "stage_network_with_a_physics_layer")
class StageNetworkWithAPhysicsLayer(StageNetwork):
    kind = "stage_network_with_a_physics_layer"
    settings_in_a_configuration = StageNetwork.settings_in_a_configuration + ("sweeps_inside",)

    @classmethod
    def from_configuration(cls, settings, context):
        from rkpinn.registry import component
        output_form = component("output_form", settings["output_form"]) \
            .from_configuration(settings["scale_of_outputs"], context)
        layout = context.layout
        torch.manual_seed(int(context.seed))
        return cls(
            equation_of_motion=context.equation_of_motion, tableau=layout.tableau,
            step_length_mm=layout.step_length_mm,
            first_start_plane_mm=layout.start_planes_mm[0],
            last_start_plane_mm=layout.start_planes_mm[-1],
            scale_of_inputs=context.scale(settings["scale_of_inputs"], 5),
            output_form=output_form, width=settings["width"], depth=settings["depth"],
            activation=settings["activation"], end_state=settings["end_state"],
            sweeps_inside=settings["sweeps_inside"])

    def __init__(self, *, sweeps_inside, **settings):
        super().__init__(**settings)
        if self.end_state != END_STATE_SUMMED:
            raise ValueError("the physics layer sums the end state from the stages; "
                             "a predicted end state is refused")
        if (not isinstance(sweeps_inside, int) or isinstance(sweeps_inside, bool)
                or sweeps_inside < 0):
            raise ValueError("the number of sweeps inside is a whole number of at "
                             "least 0, not %r" % (sweeps_inside,))
        self.sweeps_inside = sweeps_inside
        self.register_buffer("stage_matrix", torch.as_tensor(self.tableau.stage_matrix))

    def track_from_emitted(self, states, start_planes_mm, emitted) -> PredictedTrack:
        if self.sweeps_inside == 0:
            return super().track_from_emitted(states, start_planes_mm, emitted)
        planes = start_planes_mm[:, None] + self.nodes[None, :] * self.step_length_mm
        equation, dz, b = self.equation_of_motion, self.step_length_mm, self.weights_of_the_tableau

        def track(stage_states, stage_rates, produced_by):
            return PredictedTrack(
                step_length_mm=dz, tableau=self.tableau,
                start_planes_mm=start_planes_mm[:, None], input_states=states[:, None, :],
                stage_states=stage_states[:, None],
                end_states=collect_and_sum(states, stage_rates, dz, b)[:, None],
                end_state_was=END_STATE_SUMMED, layout=None,
                stage_rates=stage_rates[:, None], produced_by=produced_by)

        first_stages = emitted[:, :self.number_of_stages]
        first_rates = rates_at_the_stages(equation, states, first_stages, planes)
        stages, rates = first_stages, first_rates
        for _ in range(self.sweeps_inside):
            stages = states[:, None, :4] + dz * torch.einsum(
                "jk,nkd->njd", self.stage_matrix, rates)
            rates = rates_at_the_stages(equation, states, stages, planes)
        made = track(stages, rates, self.kind)
        made.first_call = track(first_stages, first_rates, self.kind + ", before the layer")
        return made

    def cost_per_step(self) -> dict:
        cost = super().cost_per_step()
        cost["field_lookups_per_step"] = (self.sweeps_inside + 1) * self.number_of_stages
        return cost

    def settings(self) -> dict:
        return dict(super().settings(), sweeps_inside=self.sweeps_inside)
