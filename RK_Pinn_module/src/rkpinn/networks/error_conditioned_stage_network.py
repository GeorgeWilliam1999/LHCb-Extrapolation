"""The error-conditioned stage network: a trained stage network is called once,
and a second network reads the stage residual of that call and corrects the
stages.

One step from S0 on z0, with q stages:

    first call    Y = N(S0, z0)                        the stage network
    rates         F_k = f(Y_k, z0 + c_k dz)           q field lookups
    residual      r = Y - S0 - dz A F
    corrector     W = C(S0 / s_in, z0 mapped, r / s_r),  shape 4q x 4q
    correction    dY_k,d = s_out,d  sum over m of W_(k,d),m r_m / s_r,m
    second stages Y' = Y + dY                          Y and r detached
    rates again   F'_k = f(Y'_k, z0 + c_k dz)         q more field lookups
    end state     S_end = S0 + dz sum over k of b_k F'_k

C is a body of `depth` tanh layers of `width` units and a linear layer of
4q x 4q outputs, read as the matrix W. With q = 2 it has 6 + 8 inputs.

This follows the error-conditioned neural solvers of Jiang et al.
(arXiv:2606.27354): a predictor gives a first estimate, a corrector reads the
residual of that estimate as an input and gives a correction, and the input of
the correction step is detached. What differs, stated for a ruling:

  * The paper trains on labels (the squared error to the true solution) and
    adds no residual loss. Here the loss is label-free: the pooled stage
    residual of both calls (losses/pooled_on_both_calls.py).
  * The paper's corrector gives a free field, scaled by a fixed step size,
    and is applied K = 5 times. Here it gives a matrix W applied to r, once,
    so a residual of zero gives a correction of zero exactly, and W can be
    the Newton step -M^-1 of integrators/linearised_gain.py.
  * The paper's corrector also reads the current solution. Here it reads the
    state and the plane, of which the first stages are a function.
  * The last layer of C starts at zero: the run starts as the stage network.

Cost of one step at inference: 2q field lookups (q more than the stage
network), 1 call of the stage network, 1 call of C, no Jacobian, no solve.

Settings: those of stage_network_with_a_learned_gain.py, without
`inputs_of_the_gain`. The rates of the first call are not an input; their
scale is measured and recorded for symmetry only.
Gates: tests/test_learned_gain_and_error_conditioned_network.py.
"""
from __future__ import annotations

import torch

from rkpinn.networks.collect_and_sum import collect_and_sum, rates_at_the_stages
from rkpinn.networks.stage_network import body_of_layers, check_activation
from rkpinn.networks.stage_network_of_a_run import (
    field_lookups_of_the_stage_network_scale, stage_network_of_a_run)
from rkpinn.networks.stage_network_with_a_learned_gain import BuiltOnAStageNetwork
from rkpinn.predicted_track.predicted_track import END_STATE_SUMMED, PredictedTrack
from rkpinn.registry import register


@register("network", "error_conditioned_stage_network")
class ErrorConditionedStageNetwork(BuiltOnAStageNetwork):
    kind = "error_conditioned_stage_network"
    end_state = END_STATE_SUMMED
    settings_in_a_configuration = (
        "stage_network_of_the_run", "train_the_stage_network", "width", "depth",
        "activation", "scale_of_inputs", "scale_of_outputs",
        "scale_of_the_residual_and_rates")

    @classmethod
    def from_configuration(cls, settings, context):
        stage_network = stage_network_of_a_run(settings["stage_network_of_the_run"], context)
        torch.manual_seed(int(context.seed))
        return cls(stage_network, context=context,
                   **{name: settings[name] for name in cls.settings_in_a_configuration
                      if name != "stage_network_of_the_run"},
                   stage_network_of_the_run=settings["stage_network_of_the_run"])

    def __init__(self, stage_network, *, context, stage_network_of_the_run,
                 train_the_stage_network, width, depth, activation, scale_of_inputs,
                 scale_of_outputs, scale_of_the_residual_and_rates):
        super().__init__()
        self._build_on(stage_network, train_the_stage_network=train_the_stage_network,
                       scale_of_inputs=scale_of_inputs, scale_of_outputs=scale_of_outputs,
                       scale_of_the_residual_and_rates=scale_of_the_residual_and_rates,
                       stage_network_of_the_run=stage_network_of_the_run, context=context)
        self.width, self.depth = int(width), int(depth)
        self.activation = check_activation(activation)
        q = self.number_of_stages
        self.corrector = body_of_layers(6 + 4 * q, (4 * q) ** 2, width, depth, activation)
        with torch.no_grad():
            self.corrector[-1].weight.zero_()
            self.corrector[-1].bias.zero_()

    def correction(self, states, start_planes_mm, residual):
        """dY, shape (n, q, 4), from the detached residual of the first call."""
        n, q = len(states), self.number_of_stages
        normalised = residual.detach().reshape(n, 4 * q) / self.scale_of_the_residual
        W = self.corrector(torch.cat([self.inputs_of_the_state(states, start_planes_mm),
                                      normalised], dim=1)).reshape(n, 4 * q, 4 * q)
        scale = self.scale_of_the_correction(states, start_planes_mm).reshape(-1, 1, 4)
        return scale * torch.einsum("nkm,nm->nk", W, normalised).reshape(n, q, 4)

    def predict(self, states, start_planes_mm) -> PredictedTrack:
        """One step from every state on its own start plane: separate steps.
        The track holds the second stages; its `first_call` holds the first."""
        stages, rates, residual = self.first_call(states, start_planes_mm)
        b = self.stage_network.weights_of_the_tableau
        second = stages.detach() + self.correction(states, start_planes_mm, residual)
        planes = start_planes_mm[:, None] + self.stage_network.nodes[None, :] * self.step_length_mm
        second_rates = rates_at_the_stages(self.equation_of_motion, states, second, planes)

        def track(stage_states, stage_rates, produced_by):
            return PredictedTrack(
                step_length_mm=self.step_length_mm, tableau=self.tableau,
                start_planes_mm=start_planes_mm[:, None], input_states=states[:, None, :],
                stage_states=stage_states[:, None],
                end_states=collect_and_sum(states, stage_rates, self.step_length_mm, b)[:, None],
                end_state_was=END_STATE_SUMMED, layout=None,
                stage_rates=stage_rates[:, None], produced_by=produced_by)

        made = track(second, second_rates, self.kind)
        made.first_call = track(stages, rates, self.kind + ", first call")
        return made

    def cost_per_step(self) -> dict:
        """What one step costs at inference."""
        q = self.number_of_stages
        return {
            "field_lookups_per_step": 2 * q,
            "field_lookups_of_the_output_scales_per_step":
                field_lookups_of_the_stage_network_scale(self.stage_network)
                + self.scale_of_the_correction.field_lookups_per_step(),
            "jacobian_evaluations_per_step": 0,
            "linear_solves_per_step": 0,
            "calls_of_the_stage_network_per_step": 1,
            "calls_of_the_small_network_per_step": 1,
            "jacobian_evaluations_per_step_in_training": 0,
        }

    def settings(self) -> dict:
        """What is needed, with the weights, to build this network again."""
        return dict(self._settings_of_the_base(), kind=self.kind)
