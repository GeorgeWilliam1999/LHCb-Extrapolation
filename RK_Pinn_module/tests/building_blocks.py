"""What the gates of the networks and the losses share. Not a gate.

The settings here are the settings of the gates, chosen to match the frozen
study where a comparison with it is made. They are not defaults of the package.
"""
from __future__ import annotations

import numpy as np
import torch

from frozen_code import frozen_tracks
from rkpinn.equation_of_motion.field_map import load_field_map
from rkpinn.equation_of_motion.lhcb import (
    DifferentiableLhcbEquationOfMotion, LhcbEquationOfMotion)
from rkpinn.integrators.gauss_legendre_tableau import gauss_legendre_tableau
from rkpinn.losses.pooled import spread_of_components
from rkpinn.networks.output_forms import DirectStates
from rkpinn.networks.stage_network import build_stage_network
from rkpinn.track_data.draw_training_states import (
    draw_training_states, generator_of_a_round)

FIELD_MAP = "v8r1_up"
WIDTH, DEPTH = 128, 2
STATES_PER_ROUND = 32000

# (number of steps, number of stages)
CELLS = ((2, 2), (64, 2), (256, 16))

SETTINGS_OF_THE_COST_WEIGHTED_LOSS = dict(
    momentum_window_gev=(10.0, 50.0), roll_off=float(np.log(2.0)), floor=0.05,
    clamp=5.0, samples_of_the_field_integral=1024, lever_arm_is_on=True,
    track_bend_is_on=True, momentum_window_is_on=True,
    reference_bend_mm="median_of_first_round_states")

_equations = {}


def equations():
    """(numpy equation, differentiable equation) on the field map of the tracks."""
    if not _equations:
        field_map = load_field_map(FIELD_MAP)
        _equations["numpy"] = LhcbEquationOfMotion(field_map)
        _equations["torch"] = DifferentiableLhcbEquationOfMotion(field_map)
    return _equations["numpy"], _equations["torch"]


class Crossing:
    """The first round's states of one cell, drawn as the frozen trainer draws them."""

    def __init__(self, number_of_steps, number_of_stages, seed=0):
        tracks = frozen_tracks()
        self.number_of_steps, self.number_of_stages = number_of_steps, number_of_stages
        self.first_plane_mm = float(tracks["z0"])
        self.last_plane_mm = float(tracks["z1"])
        self.length_mm = float(tracks["L"])
        self.step_length_mm = self.length_mm / number_of_steps
        self.tableau = gauss_legendre_tableau(number_of_stages)
        finest = int(tracks["n_max"])
        self.start_planes_mm = (self.first_plane_mm
                                + np.arange(number_of_steps) * self.step_length_mm)
        on_start_planes = tracks["train_truth"][:, 0:finest:finest // number_of_steps]
        drawn = draw_training_states(on_start_planes, self.start_planes_mm,
                                     STATES_PER_ROUND, generator_of_a_round(seed, 1))
        self.first_round_states = drawn.states
        self.first_round_start_planes_mm = drawn.start_planes_mm
        self.spread = spread_of_components(drawn.states)

    def states(self, number, as_tensors=True):
        """`number` of the first round's states, spread over all the start planes."""
        pick = np.linspace(0, len(self.first_round_states) - 1, number).astype(int)
        states = self.first_round_states[pick].copy()
        planes = self.first_round_start_planes_mm[pick].copy()
        if as_tensors:
            return torch.as_tensor(states), torch.as_tensor(planes)
        return states, planes

    def network(self, end_state, seed=0, width=WIDTH, depth=DEPTH):
        return build_stage_network(
            seed=seed, equation_of_motion=equations()[1], tableau=self.tableau,
            step_length_mm=self.step_length_mm,
            first_start_plane_mm=self.start_planes_mm[0],
            last_start_plane_mm=self.start_planes_mm[-1],
            scale_of_inputs=self.spread, output_form=DirectStates(self.spread[:4]),
            width=width, depth=depth, end_state=end_state)


_crossings = {}


def crossing(number_of_steps, number_of_stages) -> Crossing:
    cell = (number_of_steps, number_of_stages)
    if cell not in _crossings:
        _crossings[cell] = Crossing(*cell)
    return _crossings[cell]
