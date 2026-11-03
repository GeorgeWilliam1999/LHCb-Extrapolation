"""A trained stage network, read from a run in the store, and what the two
networks that are built on it share.

    stage_network_of_a_run(settings, context)
        settings  {key: <run key>, round: <a round of that run>}
        The stage network is built again from the settings the run recorded
        (`constants.json`, the block "network") and the weights of the
        snapshot of that round. It must have the step length, the stages and
        the tracks of the run that uses it, and its end state must be summed
        from the stages. The weights are read; nothing in the store is written.

    rms_of_the_residual_and_rates(network, states, start_planes_mm)
        The root mean square of the stage residual r and of the rates F of the
        network on the given states, per stage and component, each a list of
        4q numbers. A network built on a stage network divides r and F by them,
        so that its inputs are of order one. r is near 1e-9 of the state, so
        a divisor by the spread of the states would leave it at 1e-9.
        Rounded to six significant digits.

    ScaleOfACorrection(setting, stage_network, context), the scale of a correction
        same_as_the_stage_network   the scale the output form of the stage
                                    network gives: per track, or four numbers
        per_track                   the per-track scale of output_forms.py
                                    (16 field samples a step along the straight line)
        spread_of_first_round_states, or four numbers

The two networks built on it: stage_network_with_a_learned_gain.py and
error_conditioned_stage_network.py.
Gates: tests/test_learned_gain_and_error_conditioned_network.py.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
import yaml

from rkpinn.integrators.linearised_gain import residual_of_the_stages
from rkpinn.networks.collect_and_sum import rates_at_the_stages
from rkpinn.networks.output_forms import (PER_TRACK, StraightLinePlusScaledCorrection,
                                          four_positive_numbers, per_track_scale)
from rkpinn.predicted_track.predicted_track import END_STATE_SUMMED

SETTINGS_OF_THE_RUN = ("key", "round")
SAME_AS_THE_STAGE_NETWORK = "same_as_the_stage_network"
RMS_OVER_FIRST_ROUND_STATES = "rms_over_first_round_states"
SAMPLES_OF_THE_PER_TRACK_SCALE = 16
ROWS_AT_A_TIME = 8192


def stage_network_of_a_run(settings, context):
    """The stage network of the run `settings["key"]` at the round
    `settings["round"]`, with its weights."""
    from rkpinn.networks.stage_network import StageNetwork
    from rkpinn.registry import component
    if sorted(settings) != sorted(SETTINGS_OF_THE_RUN):
        raise ValueError("the stage network of a run is given by %s, not %r"
                         % (", ".join(SETTINGS_OF_THE_RUN), settings))
    if context.store is None:
        raise ValueError("a network built on a trained run needs the store")
    key, round_number = str(settings["key"]), int(settings["round"])
    folder = os.path.join(context.store.folder_of_runs, key)
    with open(os.path.join(folder, "constants.json")) as handle:
        recorded = json.load(handle)["network"]
    with open(os.path.join(folder, "configuration.yaml")) as handle:
        configuration = yaml.safe_load(handle)
    layout = context.layout
    if recorded["kind"] != StageNetwork.kind:
        raise ValueError("the run %s holds a %s, not a stage network" % (key, recorded["kind"]))
    if recorded["end_state"] != END_STATE_SUMMED:
        raise ValueError("the stage network of the run %s predicts its end state; a gain "
                         "corrects a summed end state" % key)
    if (recorded["number_of_stages"] != layout.number_of_stages
            or abs(recorded["step_length_mm"] - layout.step_length_mm) > 1e-9):
        raise ValueError("the run %s has %d stages and steps of %r mm, this run %d and %r mm"
                         % (key, recorded["number_of_stages"], recorded["step_length_mm"],
                            layout.number_of_stages, layout.step_length_mm))
    tracks_of_the_run = configuration["tracks"]["key"]
    network = StageNetwork(
        equation_of_motion=context.equation_of_motion, tableau=layout.tableau,
        step_length_mm=recorded["step_length_mm"],
        first_start_plane_mm=recorded["first_start_plane_mm"],
        last_start_plane_mm=recorded["last_start_plane_mm"],
        scale_of_inputs=recorded["scale_of_inputs"],
        output_form=component("output_form", recorded["output_form"])
        .from_configuration(recorded["scale_of_outputs"], context),
        width=recorded["width"], depth=recorded["depth"],
        activation=recorded["activation"], end_state=recorded["end_state"])
    weights = torch.load(os.path.join(folder, "snapshots", "round_%04d" % round_number,
                                      "network.pt"), weights_only=True)
    network.load_state_dict(weights)
    network.of_the_run = {"key": key, "round": round_number, "tracks_key": tracks_of_the_run}
    return network


def first_call(network, states, start_planes_mm):
    """The stages (n, q, 4), their rates (n, q, 4) and their residual (n, q, 4)
    of one call of a stage network."""
    stages = network(states, start_planes_mm)[:, :network.number_of_stages]
    planes = start_planes_mm[:, None] + network.nodes[None, :] * network.step_length_mm
    rates = rates_at_the_stages(network.equation_of_motion, states, stages, planes)
    residual = residual_of_the_stages(states, stages, rates, network.step_length_mm,
                                      torch.as_tensor(network.tableau.stage_matrix))
    return stages, rates, residual


def rms_of_the_residual_and_rates(network, states, start_planes_mm):
    """{"residual": 4q numbers, "rates": 4q numbers}: the root mean square of
    r and F of the network over the states, per stage and component."""
    states = torch.as_tensor(np.asarray(states, dtype=np.float64))
    planes = torch.as_tensor(np.asarray(start_planes_mm, dtype=np.float64))
    q = network.number_of_stages
    sum_r, sum_f = torch.zeros(q, 4, dtype=torch.float64), torch.zeros(q, 4, dtype=torch.float64)
    with torch.no_grad():
        for i in range(0, len(states), ROWS_AT_A_TIME):
            _, rates, residual = first_call(network, states[i:i + ROWS_AT_A_TIME],
                                            planes[i:i + ROWS_AT_A_TIME])
            sum_r += (residual ** 2).sum(dim=0)
            sum_f += (rates ** 2).sum(dim=0)
    def rms(total):
        # six significant digits: a divisor, measured again when a run is
        # resumed, so rounding in the last bit must not change the settings
        return [float("%.6g" % value)
                for value in torch.sqrt(total / len(states)).reshape(-1).tolist()]
    return {"residual": rms(sum_r), "rates": rms(sum_f)}


def divisors_of_the_residual_and_rates(setting, network, context):
    """The divisors of r and F, measured or given; see the docstring."""
    if setting == RMS_OVER_FIRST_ROUND_STATES:
        measured = rms_of_the_residual_and_rates(
            network, context.first_round_states, context.first_round_start_planes_mm)
    elif isinstance(setting, dict) and sorted(setting) == ["rates", "residual"]:
        measured = {name: [float(v) for v in setting[name]] for name in ("residual", "rates")}
    else:
        raise ValueError("the scale of the residual and the rates is %s or "
                         "{residual: [...], rates: [...]}, not %r"
                         % (RMS_OVER_FIRST_ROUND_STATES, setting))
    for name in ("residual", "rates"):
        values = np.asarray(measured[name], dtype=np.float64)
        if values.shape != (4 * network.number_of_stages,) or not (values > 0).all():
            raise ValueError("the scale of the %s has %d positive numbers"
                             % (name, 4 * network.number_of_stages))
    return measured


class ScaleOfACorrection(torch.nn.Module):
    """The scale a correction is multiplied by, shape (n, 1, 4) or (4,)."""

    def __init__(self, setting, stage_network, context):
        super().__init__()
        self.stage_network_holder = (stage_network,)
        self.per_track_here = False
        self.register_buffer("fixed", None)
        if setting == SAME_AS_THE_STAGE_NETWORK:
            self.setting = SAME_AS_THE_STAGE_NETWORK
        elif setting == PER_TRACK:
            self.setting, self.per_track_here = PER_TRACK, True
            self._cache = None
        else:
            self.setting = "numbers"
            self.fixed = four_positive_numbers(context.scale(setting, 4))

    def forward(self, states, start_planes_mm):
        stage_network = self.stage_network_holder[0]
        if self.setting == SAME_AS_THE_STAGE_NETWORK:
            form = stage_network.output_form
            if isinstance(form, StraightLinePlusScaledCorrection):
                return form.scale(states, start_planes_mm)
            return form.scale_of_outputs
        if self.per_track_here:
            cached = self._cache
            if cached is not None and cached[0] is states and cached[1] is start_planes_mm:
                return cached[2]
            scale = per_track_scale(states, start_planes_mm, stage_network.step_length_mm,
                                    stage_network.equation_of_motion.field)[:, None, :]
            self._cache = (states, start_planes_mm, scale)
            return scale
        return self.fixed

    def setting_to_record(self):
        return self.setting if self.setting != "numbers" else self.fixed.tolist()

    def field_lookups_per_step(self):
        """Field samples this scale needs a step, beyond those of the stage network."""
        return SAMPLES_OF_THE_PER_TRACK_SCALE if self.per_track_here else 0


def field_lookups_of_the_stage_network_scale(stage_network):
    form = stage_network.output_form
    return (SAMPLES_OF_THE_PER_TRACK_SCALE
            if isinstance(form, StraightLinePlusScaledCorrection) and form.scale_is_per_track
            else 0)
