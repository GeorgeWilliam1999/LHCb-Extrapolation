"""Training protocols: which states a round trains on.

A protocol gives, for a round, the states, the plane each starts on, and the
labels when the loss needs them. It names no loss, no network and no target:
the labels are those the target of the run gives.

Two are built.

  rounds_on_own_predictions      for a network with stages, chained to itself
      round 1       the reference states of the training tracks, on the planes
                    the steps start on
      later rounds  the network's own predictions: every training track is
                    carried from its start state by the network as it is, and
                    the states it predicts on the start planes are used
      Each round draws `states_per_round` states, the same number on every
      start plane. The states are fixed within the round, and no gradient
      flows back through earlier steps.

  start_states_of_the_tracks     for the whole-crossing network
      every round  the start states of the training tracks, with the labels
                   the target of the run gives for them. If there are more tracks
                   than `states_per_round`, that many are drawn.

The random generator of a round is made from the seed of the run and the
number of the round, so a round can be drawn again exactly.

`rounds_on_own_predictions` is ported from `E1_Network_grid/train_network.py`
(`new_round_states`). `start_states_of_the_tracks` is new.
Gates: tests/test_training.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from rkpinn.registry import register
from rkpinn.track_data.draw_training_states import (
    draw_training_states, generator_of_a_round)

FROM_THE_REFERENCE = "reference states"
FROM_OWN_PREDICTIONS = "own predictions"
FROM_START_STATES = "start states of the tracks"


@dataclass
class StatesOfARound:
    states: np.ndarray                    # shape (m, 5)
    start_planes_mm: np.ndarray           # shape (m,)
    labels: Optional[np.ndarray]          # shape (m, 5), or None without labels
    source: str


@register("training_protocol", "rounds_on_own_predictions")
class RoundsOnOwnPredictions:
    name = "rounds_on_own_predictions"
    for_networks_with_stages = True
    settings_in_a_configuration = ("states_per_round",)

    @classmethod
    def from_configuration(cls, settings, context):
        return cls(**settings)

    def __init__(self, *, states_per_round):
        self.states_per_round = int(states_per_round)

    def states_of_round(self, round_number, seed, tracks, layout, network,
                        target) -> StatesOfARound:
        """network  the network as it is now; not used in round 1, where it may be None
        target   the target of the run; it gives no labels to this protocol"""
        generator = generator_of_a_round(seed, round_number)
        training = tracks.training
        start_planes = tracks.start_planes_mm(layout.number_of_steps)
        if round_number == 1:
            on = tracks.planes_a_step_starts_on(layout.number_of_steps)
            on_start_planes = training.reference_states_on_planes[:, on]
            source = FROM_THE_REFERENCE
        else:
            whole = network.whole_track(training.start_state, layout)
            on_start_planes = np.asarray(whole.input_states)
            source = FROM_OWN_PREDICTIONS
        drawn = draw_training_states(on_start_planes, start_planes, self.states_per_round,
                                     generator)
        return StatesOfARound(states=drawn.states, start_planes_mm=drawn.start_planes_mm,
                              labels=None, source=source)


@register("training_protocol", "start_states_of_the_tracks")
class StartStatesOfTheTracks:
    name = "start_states_of_the_tracks"
    for_networks_with_stages = False
    settings_in_a_configuration = ("states_per_round",)

    @classmethod
    def from_configuration(cls, settings, context):
        return cls(**settings)

    def __init__(self, *, states_per_round):
        self.states_per_round = int(states_per_round)

    def states_of_round(self, round_number, seed, tracks, layout, network,
                        target) -> StatesOfARound:
        training = tracks.training
        labels = target.for_states(tracks, "training")
        n = training.number_of_tracks
        if n <= self.states_per_round:
            rows = np.arange(n)
        else:
            rows = np.sort(generator_of_a_round(seed, round_number).choice(
                n, self.states_per_round, replace=False))
        return StatesOfARound(
            states=training.start_state[rows],
            start_planes_mm=np.full(len(rows), layout.first_plane_mm),
            labels=None if labels is None else labels[rows],
            source=FROM_START_STATES)
