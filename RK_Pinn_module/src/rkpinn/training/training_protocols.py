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

Two more are added beside them, each as a separate component so that no run
made before changes its key: `rounds_on_one_fixed_draw` (2026-10-23) and
`rounds_on_own_predictions_with_residual_draws` (2026-10-28, item B.3 of the
per-step bias study). Each is described where it is defined.

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


ONE_FIXED_DRAW_OWN_PREDICTIONS = "own_predictions_on_the_same_tracks_and_planes"
ONE_FIXED_DRAW_REFERENCE = "the_reference_states_of_round_one"
LATER_ROUNDS_OF_ONE_FIXED_DRAW = (ONE_FIXED_DRAW_OWN_PREDICTIONS, ONE_FIXED_DRAW_REFERENCE)
FROM_OWN_PREDICTIONS_FIXED = "own predictions, fixed draw"
FROM_THE_REFERENCE_FIXED = "reference states, fixed draw"


@register("training_protocol", "rounds_on_one_fixed_draw")
class RoundsOnOneFixedDraw:
    """One draw of (track, start plane) pairs, made in round 1 and kept.

    Added 2026-10-23 to test a fixed collocation set, a change the literature
    suggests: every round trains on the same points, so that a round continues
    the minimisation of the last one instead of starting on new states.

    Round 1 is that of `rounds_on_own_predictions`, to the last bit: the same
    generator, the same draw, the reference states. The later rounds keep the
    tracks and planes of that draw, and take, as the setting
    `states_after_round_one` says, either

      own_predictions_on_the_same_tracks_and_planes
            the network's own predictions on those tracks and planes, as
            `rounds_on_own_predictions` does on a fresh draw; or
      the_reference_states_of_round_one
            the reference states of round 1 again: one fixed set, as in the
            literature, with no states from the network's own predictions.

    `rounds_on_own_predictions` is unchanged; this is a separate component, so
    no run made before it changes its key.
    """

    name = "rounds_on_one_fixed_draw"
    for_networks_with_stages = True
    settings_in_a_configuration = ("states_per_round", "states_after_round_one")

    @classmethod
    def from_configuration(cls, settings, context):
        return cls(**settings)

    def __init__(self, *, states_per_round, states_after_round_one):
        if states_after_round_one not in LATER_ROUNDS_OF_ONE_FIXED_DRAW:
            raise ValueError("the states after round one are %s, not %r"
                             % (" or ".join(LATER_ROUNDS_OF_ONE_FIXED_DRAW),
                                states_after_round_one))
        self.states_per_round = int(states_per_round)
        self.states_after_round_one = states_after_round_one

    def _draw_of_round_one(self, seed, tracks, layout):
        on = tracks.planes_a_step_starts_on(layout.number_of_steps)
        on_start_planes = tracks.training.reference_states_on_planes[:, on]
        return draw_training_states(on_start_planes,
                                    tracks.start_planes_mm(layout.number_of_steps),
                                    self.states_per_round, generator_of_a_round(seed, 1))

    def states_of_round(self, round_number, seed, tracks, layout, network,
                        target) -> StatesOfARound:
        drawn = self._draw_of_round_one(seed, tracks, layout)
        if round_number == 1:
            source, states = FROM_THE_REFERENCE, drawn.states
        elif self.states_after_round_one == ONE_FIXED_DRAW_REFERENCE:
            source, states = FROM_THE_REFERENCE_FIXED, drawn.states
        else:
            used = np.unique(drawn.track)
            whole = network.whole_track(tracks.training.start_state[used], layout)
            row = np.searchsorted(used, drawn.track)
            states = np.asarray(whole.input_states)[row, drawn.step_of_the_track]
            source = FROM_OWN_PREDICTIONS_FIXED
        return StatesOfARound(states=np.ascontiguousarray(states),
                              start_planes_mm=drawn.start_planes_mm, labels=None,
                              source=source)


FROM_OWN_PREDICTIONS_BY_RESIDUAL = "own predictions, half drawn by stage residual"


def stage_residual_of_states(network, states, start_planes_mm, divisor, rows_at_a_time=65536):
    """The pooled squared stage residual of each state under the network,
    shape (m,): the mean over stages and components of (r / divisor)^2."""
    import torch
    from rkpinn.losses.stage_residual import TERMS_STAGES, stage_residual
    out = np.empty(len(states))
    divisor = torch.as_tensor(np.asarray(divisor, dtype=np.float64))
    with torch.no_grad():
        for i in range(0, len(states), rows_at_a_time):
            track = network.predict(torch.as_tensor(states[i:i + rows_at_a_time]),
                                    torch.as_tensor(start_planes_mm[i:i + rows_at_a_time]))
            r = stage_residual(track, network.equation_of_motion, TERMS_STAGES).residual
            out[i:i + rows_at_a_time] = ((r / divisor) ** 2).mean(dim=(1, 2)).numpy()
    return out


def steps_across_a_z_face(start_planes_mm, step_length_mm, field):
    """True for a step whose span in z crosses a z face of the field map's grid."""
    lowest = float(field.lowest_corner[2])
    inverse = float(field.inverse_cell_size[2])
    first = np.floor((np.asarray(start_planes_mm) - lowest) * inverse)
    last = np.floor((np.asarray(start_planes_mm) + step_length_mm - lowest) * inverse)
    return first != last


@register("training_protocol", "rounds_on_own_predictions_with_residual_draws")
class RoundsOnOwnPredictionsWithResidualDraws:
    """Residual-based collocation: item B.3 of the per-step bias study.

    Round 1 is that of `rounds_on_own_predictions`, to the last bit. In a later
    round the training tracks are carried by the network as it is, as in
    `rounds_on_own_predictions`, and the budget is split:

      uniform share       (1 - share_drawn_by_residual) of the budget, drawn as
                          `rounds_on_own_predictions` draws, the same number on
                          every start plane
      residual share      the rest. A pool of `candidates_per_residual_state`
                          times as many states is drawn uniformly in the same
                          way; each candidate gets the weight of its pooled
                          squared stage residual under the network as it is,
                          multiplied by `weight_of_a_step_across_a_z_face` when
                          its step crosses a z face of the field map; the share
                          is drawn from the pool without replacement with
                          probability proportional to the weight

    The residual is divided per component by the spread of the states of the
    round-1 draw, the divisor of the pooled loss. With `share_drawn_by_residual`
    0 the states are those of `rounds_on_own_predictions` to the last bit.

    Added 2026-10-28 (approved by George the same day). `rounds_on_own_predictions`
    is unchanged, so no run made before changes its key.
    Gates: tests/test_bias_study_changes.py.
    """

    name = "rounds_on_own_predictions_with_residual_draws"
    for_networks_with_stages = True
    settings_in_a_configuration = ("states_per_round", "share_drawn_by_residual",
                                   "candidates_per_residual_state",
                                   "weight_of_a_step_across_a_z_face")

    @classmethod
    def from_configuration(cls, settings, context):
        return cls(**settings)

    def __init__(self, *, states_per_round, share_drawn_by_residual,
                 candidates_per_residual_state, weight_of_a_step_across_a_z_face):
        share = float(share_drawn_by_residual)
        if not 0.0 <= share <= 1.0:
            raise ValueError("the share drawn by residual is in [0, 1], not %r" % share)
        if int(candidates_per_residual_state) < 1:
            raise ValueError("there is at least one candidate per residual state")
        if not float(weight_of_a_step_across_a_z_face) > 0.0:
            raise ValueError("the weight of a step across a z face is positive")
        self.states_per_round = int(states_per_round)
        self.share_drawn_by_residual = share
        self.candidates_per_residual_state = int(candidates_per_residual_state)
        self.weight_of_a_step_across_a_z_face = float(weight_of_a_step_across_a_z_face)
        self._standard = RoundsOnOwnPredictions(states_per_round=states_per_round)

    def states_of_round(self, round_number, seed, tracks, layout, network,
                        target) -> StatesOfARound:
        by_residual = int(round(self.share_drawn_by_residual * self.states_per_round))
        if round_number == 1 or by_residual == 0:
            return self._standard.states_of_round(round_number, seed, tracks, layout,
                                                  network, target)
        generator = generator_of_a_round(seed, round_number)
        start_planes = tracks.start_planes_mm(layout.number_of_steps)
        whole = network.whole_track(tracks.training.start_state, layout)
        on_start_planes = np.asarray(whole.input_states)
        uniform = draw_training_states(on_start_planes, start_planes,
                                       self.states_per_round - by_residual, generator)
        pool = draw_training_states(on_start_planes, start_planes,
                                    by_residual * self.candidates_per_residual_state,
                                    generator)
        on = tracks.planes_a_step_starts_on(layout.number_of_steps)
        divisor = draw_training_states(
            tracks.training.reference_states_on_planes[:, on], start_planes,
            self.states_per_round, generator_of_a_round(seed, 1)).states.std(axis=0)[:4]
        weight = stage_residual_of_states(network, pool.states, pool.start_planes_mm, divisor)
        weight = weight * np.where(
            steps_across_a_z_face(pool.start_planes_mm, layout.step_length_mm,
                                  network.equation_of_motion.field),
            self.weight_of_a_step_across_a_z_face, 1.0)
        take = min(by_residual, int((weight > 0).sum()))
        chosen = generator.choice(len(weight), take, replace=False, p=weight / weight.sum())
        return StatesOfARound(
            states=np.concatenate([uniform.states, pool.states[chosen]]),
            start_planes_mm=np.concatenate([uniform.start_planes_mm,
                                            pool.start_planes_mm[chosen]]),
            labels=None, source=FROM_OWN_PREDICTIONS_BY_RESIDUAL)
