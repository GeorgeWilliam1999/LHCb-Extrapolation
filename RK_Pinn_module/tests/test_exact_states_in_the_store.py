"""Gate: the exact scheme's states in the store are those the package solves.

Proves
  * the store holds the exact states of the test split at every number of
    steps and stages in use, for every test track, and every solve converged;
  * each file's content gives the key its record and the manifest state;
  * the first tracks, solved again now, are identical to the store to the last
    bit when this machine has the processor the store was written on, and
    equal to 1e-11 mm in position and 1e-15 in slope on any other;
  * the store agrees with the stored files of the frozen code to 1e-11 mm and
    1e-15, for the eight files the frozen self-chained study wrote;
  * the state at step 0 is the start state, and charge over momentum is
    carried unchanged.
"""
from __future__ import annotations

import numpy as np
import pytest

from frozen_code import identical, stored_exact_states
from rkpinn.run_record import manifest
from rkpinn.run_record.provenance import processor_of_this_machine
from rkpinn.run_record.store import key_of_content
from rkpinn.track_data.exact_states import load_exact_states, solve_exact_states
from rkpinn.track_data.load_tracks import load_tracks
from the_store import KEY_OF_THE_TRACKS, NUMBERS_OF_STAGES, NUMBERS_OF_STEPS, the_store

POSITION_TOLERANCE_MM = 1e-11
SLOPE_TOLERANCE = 1e-15

CELLS = [(steps, stages) for steps in NUMBERS_OF_STEPS for stages in NUMBERS_OF_STAGES]
# how many tracks are solved again, by number of steps
TRACKS_SOLVED_AGAIN = {2: 40, 64: 4, 128: 2, 256: 1}


@pytest.fixture(scope="module")
def tracks():
    return load_tracks(the_store(), KEY_OF_THE_TRACKS, check_the_key=False)


@pytest.mark.parametrize("number_of_steps, number_of_stages", CELLS)
def test_store_holds_the_exact_states(tracks, number_of_steps, number_of_stages):
    states, planes, description = load_exact_states(
        the_store(), KEY_OF_THE_TRACKS, "test", number_of_steps, number_of_stages)
    assert states.shape == (tracks.test.number_of_tracks, number_of_steps + 1, 5)
    assert description["solves_not_converged"] == 0
    assert description["provenance"]["traceable_to_the_commit"] is True
    assert identical(states[:, 0], tracks.test.start_state)
    assert (states[:, :, 4] == tracks.test.start_state[:, 4:5]).all()
    assert planes[0] == tracks.first_plane_mm
    assert abs(planes[-1] - tracks.last_plane_mm) < 1e-9
    assert np.isfinite(states).all()
    listed = [row for row in manifest.list_exact_states(the_store())
              if row["tracks_key"] == KEY_OF_THE_TRACKS and row["split"] == "test"
              and int(row["number_of_steps"]) == number_of_steps
              and int(row["number_of_stages"]) == number_of_stages]
    assert len(listed) == 1
    assert listed[0]["key_of_content"] == description["key_of_content"] \
        == key_of_content({"states": states, "planes_mm": planes})


@pytest.mark.parametrize("number_of_steps, number_of_stages", CELLS)
def test_tracks_solved_again_equal_the_store(tracks, number_of_steps, number_of_stages):
    stored, _, description = load_exact_states(
        the_store(), KEY_OF_THE_TRACKS, "test", number_of_steps, number_of_stages,
        check_the_key=False)
    number = TRACKS_SOLVED_AGAIN[number_of_steps]
    again, not_converged, _ = solve_exact_states(
        tracks, "test", number_of_steps, number_of_stages, workers=1, most_tracks=number)
    assert not_converged == 0
    if processor_of_this_machine() == description["provenance"]["processor"]:
        assert identical(again, stored[:number])
    else:
        difference = np.abs(again - stored[:number])
        assert difference[..., :2].max() < POSITION_TOLERANCE_MM
        assert difference[..., 2:4].max() < SLOPE_TOLERANCE


@pytest.mark.parametrize("number_of_steps", (2, 256))
@pytest.mark.parametrize("number_of_stages", NUMBERS_OF_STAGES)
def test_store_agrees_with_the_files_of_the_frozen_code(number_of_steps, number_of_stages):
    ours, _, _ = load_exact_states(
        the_store(), KEY_OF_THE_TRACKS, "test", number_of_steps, number_of_stages,
        check_the_key=False)
    theirs = stored_exact_states(number_of_steps, number_of_stages)
    assert ours.shape == theirs.shape
    difference = np.abs(ours - theirs)
    assert difference[..., :2].max() < POSITION_TOLERANCE_MM
    assert difference[..., 2:4].max() < SLOPE_TOLERANCE
    assert identical(ours[..., 4], theirs[..., 4])
