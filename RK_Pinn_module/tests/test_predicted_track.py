"""Gate: the predicted track holds what was put in it, and the comparators fill it.

Proves
  * the planes of a layout are where the frozen code puts them;
  * a predicted track refuses arrays of the wrong shape, and an end state
    summed from stages that do not exist;
  * charge over momentum is carried unchanged to every end state;
  * the exact scheme fills the structure, and its states on the planes are
    the exact states in the store, to the last bit on the machine that wrote them;
  * the reference integrator fills the structure, and its states are the
    reference states of the track set, to the last bit.
"""
from __future__ import annotations

import numpy as np
import pytest

from building_blocks import equations
from frozen_code import identical
from rkpinn.integrators.gauss_legendre_tableau import gauss_legendre_tableau
from rkpinn.integrators.runge_kutta_sixth_order import REFERENCE_STEP_LENGTH_MM
from rkpinn.predicted_track.predicted_track import (
    END_STATE_PREDICTED, END_STATE_SUMMED, PredictedTrack, steps_of_the_exact_scheme,
    track_of_the_exact_scheme, track_of_the_reference)
from rkpinn.predicted_track.track_layout import TrackLayout
from rkpinn.run_record.provenance import processor_of_this_machine
from rkpinn.track_data.exact_states import load_exact_states
from rkpinn.track_data.load_tracks import load_tracks
from the_store import KEY_OF_THE_TRACKS, the_store


@pytest.fixture(scope="module")
def tracks():
    return load_tracks(the_store(), KEY_OF_THE_TRACKS, check_the_key=False)


def some_track(rows=3, steps=2, stages=2, end_state_was=END_STATE_SUMMED):
    generator = np.random.default_rng(0)
    return dict(
        step_length_mm=10.0, tableau=gauss_legendre_tableau(stages) if stages else None,
        start_planes_mm=np.tile(np.arange(steps) * 10.0, (rows, 1)),
        input_states=generator.random((rows, steps, 5)),
        stage_states=generator.random((rows, steps, stages, 4)) if stages else None,
        end_states=generator.random((rows, steps, 4)), end_state_was=end_state_was,
        layout=TrackLayout(0.0, steps * 10.0, steps,
                           gauss_legendre_tableau(stages) if stages else None))


@pytest.mark.parametrize("number_of_steps", (2, 64, 128, 256))
def test_planes_of_a_layout(tracks, number_of_steps):
    tableau = gauss_legendre_tableau(4)
    layout = TrackLayout(tracks.first_plane_mm, tracks.last_plane_mm, number_of_steps, tableau)
    assert layout.step_length_mm == tracks.length_mm / number_of_steps
    assert identical(layout.start_planes_mm, tracks.start_planes_mm(number_of_steps))
    assert len(layout.planes_mm) == number_of_steps + 1
    assert layout.number_of_stages == 4
    step = number_of_steps - 1
    assert identical(layout.stage_planes_mm(step),
                     layout.start_planes_mm[step] + tableau.nodes * layout.step_length_mm)
    with pytest.raises(IndexError):
        layout.stage_planes_mm(number_of_steps)
    assert TrackLayout(0.0, 1.0, 1, None).number_of_stages == 0
    with pytest.raises(ValueError):
        TrackLayout(0.0, 1.0, 0, None)


def test_a_track_holds_what_was_put_in_it():
    parts = some_track()
    track = PredictedTrack(**parts)
    assert (track.number_of_rows, track.number_of_steps, track.number_of_stages) == (3, 2, 2)
    assert track.is_a_whole_track
    assert identical(track.input_state(1), parts["input_states"][:, 1])
    assert identical(track.stage_states_of(0), parts["stage_states"][:, 0])
    assert track.stage_planes_mm().shape == (3, 2, 2)
    on_planes = track.states_on_planes()
    assert on_planes.shape == (3, 3, 5)
    assert identical(on_planes[:, 0], parts["input_states"][:, 0])
    assert identical(on_planes[:, 1:, :4], parts["end_states"])


def test_charge_over_momentum_is_carried_unchanged():
    parts = some_track()
    track = PredictedTrack(**parts)
    for step in range(2):
        end = track.end_state(step)
        assert end.shape == (3, 5)
        assert identical(end[:, 4], parts["input_states"][:, step, 4])
        assert identical(end[:, :4], parts["end_states"][:, step])
    assert identical(track.final_state(), track.end_state(1))


def test_wrong_shapes_are_refused():
    for name, wrong in (("input_states", np.zeros((3, 2, 4))),
                        ("end_states", np.zeros((3, 2, 5))),
                        ("stage_states", np.zeros((3, 2, 3, 4))),
                        ("start_planes_mm", np.zeros((3,)))):
        with pytest.raises(ValueError):
            PredictedTrack(**dict(some_track(), **{name: wrong}))
    with pytest.raises(ValueError):
        PredictedTrack(**dict(some_track(), end_state_was="guessed"))
    with pytest.raises(ValueError):
        PredictedTrack(**some_track(stages=0, end_state_was=END_STATE_SUMMED))
    without_stages = PredictedTrack(**some_track(stages=0, end_state_was=END_STATE_PREDICTED))
    assert without_stages.stage_states_of(0) is None
    assert without_stages.stage_planes_mm() is None
    separate = PredictedTrack(**dict(some_track(steps=1), layout=None))
    with pytest.raises(ValueError):
        separate.states_on_planes()


@pytest.mark.parametrize("number_of_steps, number_of_stages, number_of_tracks",
                         ((2, 2, 30), (2, 16, 30), (64, 4, 3)))
def test_exact_scheme_fills_the_track(tracks, number_of_steps, number_of_stages,
                                      number_of_tracks):
    layout = TrackLayout(tracks.first_plane_mm, tracks.last_plane_mm, number_of_steps,
                         gauss_legendre_tableau(number_of_stages))
    track = track_of_the_exact_scheme(
        equations()[0], tracks.test.start_state[:number_of_tracks], layout)
    assert track.end_state_was == END_STATE_SUMMED and track.is_a_whole_track
    assert track.stage_states.shape == (number_of_tracks, number_of_steps,
                                        number_of_stages, 4)
    stored, _, description = load_exact_states(
        the_store(), KEY_OF_THE_TRACKS, "test", number_of_steps, number_of_stages,
        check_the_key=False)
    ours = track.states_on_planes()
    if processor_of_this_machine() == description["provenance"]["processor"]:
        assert identical(ours, stored[:number_of_tracks])
    else:
        assert np.abs(ours - stored[:number_of_tracks])[..., :2].max() < 1e-11


def test_separate_steps_of_the_exact_scheme(tracks):
    tableau = gauss_legendre_tableau(4)
    on = tracks.planes_a_step_starts_on(64)
    states = tracks.test.reference_states_on_planes[:10, on[20]]
    planes = np.full(10, tracks.start_planes_mm(64)[20])
    track = steps_of_the_exact_scheme(
        equations()[0], states, planes, tracks.length_mm / 64, tableau)
    assert not track.is_a_whole_track and track.number_of_steps == 1
    assert track.stage_states.shape == (10, 1, 4, 4)
    assert identical(track.input_state(0), states)


def test_reference_fills_the_track(tracks):
    # the first 8 of the 256 steps between the planes of the track set
    layout = TrackLayout(tracks.first_plane_mm, float(tracks.planes_mm[8]), 8,
                         gauss_legendre_tableau(2))
    track = track_of_the_reference(
        equations()[0], tracks.test.start_state[:20], layout,
        step_length_of_the_integrator_mm=REFERENCE_STEP_LENGTH_MM)
    assert track.number_of_stages == 0 and track.stage_states is None
    assert identical(track.states_on_planes(),
                     tracks.test.reference_states_on_planes[:20, :9])
