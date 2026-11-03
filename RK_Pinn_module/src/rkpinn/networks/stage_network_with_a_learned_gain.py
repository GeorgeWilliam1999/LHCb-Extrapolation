"""The stage network with a learned gain: a trained stage network, and a small
network that corrects the summed end state from the stage residual.

One step from S0 on z0, with q stages:

    stages      Y = N(S0, z0)                         the stage network, one call
    rates       F_k = f(Y_k, z0 + c_k dz)            q field lookups
    summed end  E = S0 + dz sum over k of b_k F_k
    residual    r = Y - S0 - dz A F                   free: F is already there
    gain        W = G(S0 / s_in, z0 mapped, r / s_r [, F / s_F]),  shape 4 x 4q
    correction  D_d = s_out,d  sum over m of W_dm r_m / s_r,m
    end state   S_end = E - D

G is a body of `depth` tanh layers of `width` units and a linear layer of
4 x 4q outputs, read as the matrix W. s_in is the scale of the inputs (five
numbers), s_r and s_F the root mean square of r and F of the stage network
over the first round's states (4q numbers each), s_out the scale of the
correction (stage_network_of_a_run.py).

Three structural choices, stated for a ruling:

  * The correction is a matrix W applied to r, not a free vector. A residual
    of zero then gives a correction of zero exactly. W can be the analytic
    gain K of integrators/linearised_gain.py, which depends on the stages
    through J, so G can represent K: W = diag(1/s_out) K diag(s_r). This is
    the form of the KalmanNet update (Revach et al., arXiv:2107.10043): a
    network gives the gain, and the gain multiplies the innovation.
  * r enters G twice: as the vector W multiplies, and as an input of G, so
    that W can depend on r (a correction that is not linear in r).
  * The last layer of G starts at zero. The run then starts as the stage
    network alone.

r, and F when it is an input, are detached before they enter G: no gradient
flows from G into the stage network through them.

Settings:

  stage_network_of_the_run   {key, round}: the trained stage network
  train_the_stage_network    false: the stage network is frozen (the default
                             of the experiment); true: it is trained with G
  inputs_of_the_gain         state_and_residual, or state_residual_and_rates
  width, depth, activation   of G
  scale_of_inputs            five numbers, or spread_of_first_round_states
  scale_of_outputs           s_out: same_as_the_stage_network, per_track,
                             spread_of_first_round_states, or four numbers
  scale_of_the_residual_and_rates
                             rms_over_first_round_states, or {residual, rates}

Cost of one step at inference: q field lookups, 1 call of the stage network,
1 call of G, no Jacobian, no solve. In training the loss
`distilled_analytic_gain` evaluates q Jacobians a step.

The end state is not the sum of the stages, so the track says it is
predicted. The stage states are those of the stage network.
Gates: tests/test_learned_gain_and_error_conditioned_network.py.
"""
from __future__ import annotations

import numpy as np
import torch

from rkpinn.networks.collect_and_sum import collect_and_sum
from rkpinn.networks.stage_network import body_of_layers, check_activation, positive_scale
from rkpinn.networks.stage_network_of_a_run import (
    ScaleOfACorrection, field_lookups_of_the_stage_network_scale, first_call,
    divisors_of_the_residual_and_rates, stage_network_of_a_run)
from rkpinn.predicted_track.predicted_track import END_STATE_PREDICTED, PredictedTrack
from rkpinn.registry import register

STATE_AND_RESIDUAL = "state_and_residual"
STATE_RESIDUAL_AND_RATES = "state_residual_and_rates"
INPUTS_OF_THE_GAIN = (STATE_AND_RESIDUAL, STATE_RESIDUAL_AND_RATES)


class BuiltOnAStageNetwork(torch.nn.Module):
    """What the two networks built on a trained stage network share."""

    has_stages = True

    def _build_on(self, stage_network, *, train_the_stage_network, scale_of_inputs,
                  scale_of_outputs, scale_of_the_residual_and_rates, stage_network_of_the_run,
                  context):
        self.stage_network = stage_network
        self.train_the_stage_network = bool(train_the_stage_network)
        for parameter in stage_network.parameters():
            parameter.requires_grad_(self.train_the_stage_network)
        self.stage_network_of_the_run = {"key": str(stage_network_of_the_run["key"]),
                                         "round": int(stage_network_of_the_run["round"])}
        self.tableau = stage_network.tableau
        self.number_of_stages = stage_network.number_of_stages
        self.step_length_mm = stage_network.step_length_mm
        self.register_buffer("scale_of_inputs",
                             positive_scale(scale_of_inputs, 5, "the scale of the inputs"))
        self.scale_of_the_correction = ScaleOfACorrection(scale_of_outputs, stage_network,
                                                          context)
        scales = divisors_of_the_residual_and_rates(scale_of_the_residual_and_rates,
                                                 stage_network, context)
        self.register_buffer("scale_of_the_residual",
                             torch.as_tensor(scales["residual"], dtype=torch.float64))
        self.register_buffer("scale_of_the_rates",
                             torch.as_tensor(scales["rates"], dtype=torch.float64))
        self._cache = None

    @property
    def equation_of_motion(self):
        return self.stage_network.equation_of_motion

    def first_call(self, states, start_planes_mm):
        """Stages, rates and residual of the stage network. When it is frozen
        they are cached on the identity of the inputs, because the loss calls
        the network hundreds of times a restart on the same tensors."""
        if self.train_the_stage_network:
            return first_call(self.stage_network, states, start_planes_mm)
        cached = self._cache
        if cached is not None and cached[0] is states and cached[1] is start_planes_mm:
            return cached[2]
        with torch.no_grad():
            made = first_call(self.stage_network, states, start_planes_mm)
        self._cache = (states, start_planes_mm, made)
        return made

    def inputs_of_the_state(self, states, start_planes_mm):
        return torch.cat([states / self.scale_of_inputs,
                          self.stage_network.start_plane_as_input(start_planes_mm)[:, None]],
                         dim=1)

    def whole_track(self, states, layout):
        """The network applied to itself over the layout; see self_chain.py."""
        from rkpinn.networks.self_chain import whole_track
        return whole_track(self, states, layout)

    def _settings_of_the_base(self):
        return {
            "number_of_stages": self.number_of_stages, "step_length_mm": self.step_length_mm,
            "stage_network_of_the_run": dict(self.stage_network_of_the_run),
            "train_the_stage_network": self.train_the_stage_network,
            "scale_of_inputs": self.scale_of_inputs.tolist(),
            "scale_of_outputs": self.scale_of_the_correction.setting_to_record(),
            "scale_of_the_residual_and_rates": {
                "residual": self.scale_of_the_residual.tolist(),
                "rates": self.scale_of_the_rates.tolist()},
            "width": self.width, "depth": self.depth, "activation": self.activation,
        }


@register("network", "stage_network_with_a_learned_gain")
class StageNetworkWithALearnedGain(BuiltOnAStageNetwork):
    kind = "stage_network_with_a_learned_gain"
    end_state = END_STATE_PREDICTED
    settings_in_a_configuration = (
        "stage_network_of_the_run", "train_the_stage_network", "inputs_of_the_gain",
        "width", "depth", "activation", "scale_of_inputs", "scale_of_outputs",
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
                 train_the_stage_network, inputs_of_the_gain, width, depth, activation,
                 scale_of_inputs, scale_of_outputs, scale_of_the_residual_and_rates):
        super().__init__()
        if inputs_of_the_gain not in INPUTS_OF_THE_GAIN:
            raise ValueError("the inputs of the gain are %s, not %r"
                             % (" or ".join(INPUTS_OF_THE_GAIN), inputs_of_the_gain))
        self._build_on(stage_network, train_the_stage_network=train_the_stage_network,
                       scale_of_inputs=scale_of_inputs, scale_of_outputs=scale_of_outputs,
                       scale_of_the_residual_and_rates=scale_of_the_residual_and_rates,
                       stage_network_of_the_run=stage_network_of_the_run, context=context)
        self.inputs_of_the_gain = inputs_of_the_gain
        self.width, self.depth = int(width), int(depth)
        self.activation = check_activation(activation)
        q = self.number_of_stages
        number_of_inputs = 6 + 4 * q * (2 if inputs_of_the_gain == STATE_RESIDUAL_AND_RATES
                                        else 1)
        self.gain = body_of_layers(number_of_inputs, 4 * 4 * q, width, depth, activation)
        with torch.no_grad():
            self.gain[-1].weight.zero_()
            self.gain[-1].bias.zero_()

    def gain_matrix(self, states, start_planes_mm, residual, rates):
        """W, shape (n, 4, 4q), from the detached residual and rates."""
        n, q = len(states), self.number_of_stages
        parts = [self.inputs_of_the_state(states, start_planes_mm),
                 residual.detach().reshape(n, 4 * q) / self.scale_of_the_residual]
        if self.inputs_of_the_gain == STATE_RESIDUAL_AND_RATES:
            parts.append(rates.detach().reshape(n, 4 * q) / self.scale_of_the_rates)
        return self.gain(torch.cat(parts, dim=1)).reshape(n, 4, 4 * q)

    def correction(self, states, start_planes_mm, residual, rates):
        """D, shape (n, 4): the correction subtracted from the summed end state."""
        n, q = len(states), self.number_of_stages
        normalised = residual.detach().reshape(n, 4 * q) / self.scale_of_the_residual
        W = self.gain_matrix(states, start_planes_mm, residual, rates)
        scale = self.scale_of_the_correction(states, start_planes_mm).reshape(-1, 4)
        return scale * torch.einsum("nam,nm->na", W, normalised)

    def predict(self, states, start_planes_mm) -> PredictedTrack:
        """One step from every state on its own start plane: separate steps."""
        stages, rates, residual = self.first_call(states, start_planes_mm)
        summed = collect_and_sum(states, rates, self.step_length_mm,
                                 self.stage_network.weights_of_the_tableau)
        end_states = summed - self.correction(states, start_planes_mm, residual, rates)
        return PredictedTrack(
            step_length_mm=self.step_length_mm, tableau=self.tableau,
            start_planes_mm=start_planes_mm[:, None], input_states=states[:, None, :],
            stage_states=stages[:, None], end_states=end_states[:, None],
            end_state_was=END_STATE_PREDICTED, layout=None,
            stage_rates=rates[:, None], produced_by=self.kind)

    def cost_per_step(self) -> dict:
        """What one step costs at inference, and the Jacobians of training."""
        q = self.number_of_stages
        return {
            "field_lookups_per_step": q,
            "field_lookups_of_the_output_scales_per_step":
                field_lookups_of_the_stage_network_scale(self.stage_network)
                + self.scale_of_the_correction.field_lookups_per_step(),
            "jacobian_evaluations_per_step": 0,
            "linear_solves_per_step": 0,
            "calls_of_the_stage_network_per_step": 1,
            "calls_of_the_small_network_per_step": 1,
            "jacobian_evaluations_per_step_in_training": q,
        }

    def settings(self) -> dict:
        """What is needed, with the weights, to build this network again."""
        return dict(self._settings_of_the_base(), kind=self.kind,
                    inputs_of_the_gain=self.inputs_of_the_gain)
