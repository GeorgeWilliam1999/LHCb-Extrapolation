"""Gate: the draw of training states is the draw of the frozen trainer.

Proves
  * from the same seed and round, the package draws the same states, on the
    same start planes, as the frozen `draw_states`, to the last bit, at every
    number of steps in use;
  * the same number of states is drawn on every start plane, and no track is
    drawn twice on one plane;
  * when the budget covers everything, everything is taken.
"""
from __future__ import annotations

import numpy as np
import pytest

from frozen_code import frozen_round_trainer, frozen_tracks, identical
from rkpinn.track_data.draw_training_states import (
    draw_training_states, generator_of_a_round)

STATES_PER_ROUND = 32000


def states_and_planes(number_of_steps):
    tracks = frozen_tracks()
    finest = int(tracks["n_max"])
    on_start_planes = tracks["train_truth"][:, 0:finest:finest // number_of_steps]
    start_planes = (float(tracks["z0"])
                    + np.arange(number_of_steps) * (float(tracks["L"]) / number_of_steps))
    return on_start_planes, start_planes


@pytest.mark.parametrize("number_of_steps", (2, 64, 128, 256))
@pytest.mark.parametrize("seed, round_number", ((0, 1), (0, 2), (3, 7)))
def test_draw_is_identical_to_the_frozen_trainer(number_of_steps, seed, round_number):
    frozen = frozen_round_trainer()
    assert frozen.STATES_PER_ROUND == STATES_PER_ROUND
    on_start_planes, start_planes = states_and_planes(number_of_steps)
    ours = draw_training_states(on_start_planes, start_planes, STATES_PER_ROUND,
                                generator_of_a_round(seed, round_number))
    states, planes, step = frozen.draw_states(
        on_start_planes, start_planes, STATES_PER_ROUND,
        np.random.default_rng([seed, round_number]))
    assert identical(ours.states, states)
    assert identical(ours.start_planes_mm, planes)
    assert identical(ours.step_of_the_track, step)


@pytest.mark.parametrize("number_of_steps", (2, 64, 256))
def test_the_same_number_is_drawn_on_every_start_plane(number_of_steps):
    on_start_planes, start_planes = states_and_planes(number_of_steps)
    drawn = draw_training_states(on_start_planes, start_planes, STATES_PER_ROUND,
                                 generator_of_a_round(0, 1))
    numbers = np.bincount(drawn.step_of_the_track, minlength=number_of_steps)
    assert len(set(numbers.tolist())) == 1
    assert len(drawn.states) <= STATES_PER_ROUND
    for step in range(number_of_steps):
        tracks_on_plane = drawn.track[drawn.step_of_the_track == step]
        assert len(np.unique(tracks_on_plane)) == len(tracks_on_plane)
    assert identical(drawn.states, on_start_planes[drawn.track, drawn.step_of_the_track])


def test_everything_is_taken_when_the_budget_covers_it():
    on_start_planes, start_planes = states_and_planes(2)
    few = on_start_planes[:100]
    drawn = draw_training_states(few, start_planes, STATES_PER_ROUND,
                                 generator_of_a_round(0, 1))
    assert len(drawn.states) == 200
    assert np.bincount(drawn.track).tolist() == [2] * 100
