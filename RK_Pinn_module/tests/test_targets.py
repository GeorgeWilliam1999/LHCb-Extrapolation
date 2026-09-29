"""Gate: every target gives the states it names, aligned with the tracks.

Proves
  * no target gives nothing and needs no label;
  * the reference end state is the reference state on the last plane;
  * the reference states on the planes of a layout are taken from the planes
    of the track set the steps fall on;
  * the true state comes with the plane it lies on;
  * every target has one row per track of the split, in the order of the
    tracks, and an unknown split is refused.
"""
from __future__ import annotations

import pytest

from frozen_code import frozen_tracks, identical
from rkpinn import registry
from rkpinn.track_data.load_tracks import load_tracks
from the_store import KEY_OF_THE_TRACKS, the_store

SPLITS = (("training", "train"), ("validation", "val"), ("test", "test"))


@pytest.fixture(scope="module")
def tracks():
    return load_tracks(the_store(), KEY_OF_THE_TRACKS, check_the_key=False)


def test_no_target(tracks):
    target = registry.component("target", "no_target")()
    assert target.needs_labels is False
    assert target.for_states(tracks, "training") is None
    with pytest.raises(KeyError):
        target.for_states(tracks, "train")


@pytest.mark.parametrize("split, frozen_split", SPLITS)
def test_reference_end_state(tracks, split, frozen_split):
    target = registry.component("target", "reference_end_state")()
    assert target.needs_labels is True
    states = target.for_states(tracks, split)
    assert identical(states, frozen_tracks()[frozen_split + "_truth"][:, 256])
    assert len(states) == tracks.split(split).number_of_tracks


@pytest.mark.parametrize("number_of_steps", (2, 64, 128, 256))
def test_reference_states_on_planes(tracks, number_of_steps):
    target = registry.component("target", "reference_states_on_planes")(
        number_of_steps=number_of_steps)
    states = target.for_states(tracks, "test")
    stride = 256 // number_of_steps
    assert states.shape == (tracks.test.number_of_tracks, number_of_steps + 1, 5)
    assert identical(states, frozen_tracks()["test_truth"][:, ::stride])
    assert identical(states[:, 0], tracks.test.start_state)
    assert identical(states[:, -1], tracks.test.reference_end_state)


def test_steps_off_the_planes_are_refused(tracks):
    target = registry.component("target", "reference_states_on_planes")(number_of_steps=3)
    with pytest.raises(ValueError):
        target.for_states(tracks, "test")


@pytest.mark.parametrize("split, frozen_split", SPLITS)
def test_true_state(tracks, split, frozen_split):
    states, planes = registry.component("target", "true_state")().for_states(tracks, split)
    frozen = frozen_tracks()
    assert identical(states, frozen[frozen_split + "_S_post"])
    assert identical(planes, frozen[frozen_split + "_z_post"])
    assert abs(planes - tracks.last_plane_mm).max() < 60.0
