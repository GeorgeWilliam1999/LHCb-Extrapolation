"""Gate: the tracks the package builds are the tracks of the frozen file.

Proves
  * the planes of the track set are identical to the frozen planes;
  * every cut removes the number of rows the frozen record says;
  * a part of the track set, rebuilt by the package now, holds the same
    particles, in the same order, with the same states, to the last bit;
  * the whole track set in the store is identical to the frozen file, to the
    last bit, in every array of every split;
  * the content of the track set in the store gives its key.

The rebuild covers the first 200 particles, to keep the gate short. The whole
track set was built once into the store, and that is compared in full.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from frozen_code import (
    FROZEN_NAME_OF_ARRAY, FROZEN_NAME_OF_SPLIT, SAMPLE_FILE, TRACKS_FILE, frozen_tracks,
    identical, needed_file)
from rkpinn.run_record.store import hash_of_file, key_of_content
from rkpinn.track_data.build_tracks import (
    ARRAYS_OF_A_SPLIT, SPLITS, SettingsOfTheTracks, build_arrays)
from rkpinn.track_data.load_tracks import load_tracks
from the_store import KEY_OF_THE_TRACKS, SETTINGS_OF_THE_TRACKS, the_store

NUMBER_OF_PARTICLES_REBUILT = 200
HASH_OF_THE_SAMPLE_FILE = "9b3d84ce7080cde9db810e0e3c47fa7e"
COUNTS_OF_THE_FROZEN_FILE = {"training": 11567, "validation": 1463, "test": 1452}


@pytest.fixture(scope="module")
def rebuilt():
    settings = SettingsOfTheTracks(**SETTINGS_OF_THE_TRACKS)
    return build_arrays(needed_file(SAMPLE_FILE), settings, workers=1,
                        report=lambda text: None,
                        most_particles=NUMBER_OF_PARTICLES_REBUILT)


def test_the_sample_is_the_one_the_frozen_file_was_built_from():
    assert hash_of_file(needed_file(SAMPLE_FILE)) == HASH_OF_THE_SAMPLE_FILE


def test_planes_are_identical_to_the_frozen_planes():
    settings = SettingsOfTheTracks(**SETTINGS_OF_THE_TRACKS)
    frozen = frozen_tracks()
    assert identical(settings.planes_mm, frozen["planes"])
    assert settings.length_mm == float(frozen["L"])
    assert settings.step_length_mm == float(frozen["rk6_step_mm"])
    assert settings.window_mm == float(frozen["window_mm"])
    assert settings.field_map == "v8r1_" + str(frozen["field"])


def test_every_cut_removes_what_the_frozen_record_says(rebuilt):
    _, cascade, _ = rebuilt
    with open(needed_file(os.path.splitext(TRACKS_FILE)[0] + "_meta.json")) as handle:
        frozen = json.load(handle)["cut_cascade"]
    # the last cut, inside the field map, is applied to the part rebuilt here
    assert len(cascade) == len(frozen)
    for ours, theirs in zip(cascade[:-1], frozen[:-1]):
        for count in ("rows_in", "rows_removed", "rows_out", "particles_out"):
            assert ours[count] == theirs[count], (ours["cut"], theirs["cut"], count)


def test_rebuilt_part_is_identical_to_the_frozen_file(rebuilt):
    arrays, _, counts = rebuilt
    frozen = frozen_tracks()
    assert sum(counts.values()) == NUMBER_OF_PARTICLES_REBUILT
    assert all(counts[split] > 0 for split in SPLITS)
    for split in SPLITS:
        for name in ARRAYS_OF_A_SPLIT:
            theirs = frozen["%s_%s" % (FROZEN_NAME_OF_SPLIT[split],
                                       FROZEN_NAME_OF_ARRAY[name])]
            assert identical(arrays["%s.%s" % (split, name)], theirs[:counts[split]]), \
                (split, name)


def test_track_set_in_the_store_is_identical_to_the_frozen_file():
    tracks = load_tracks(the_store(), KEY_OF_THE_TRACKS)
    frozen = frozen_tracks()
    assert identical(tracks.planes_mm, frozen["planes"])
    for split in SPLITS:
        ours = tracks.split(split)
        assert ours.number_of_tracks == COUNTS_OF_THE_FROZEN_FILE[split]
        for name in ARRAYS_OF_A_SPLIT:
            theirs = frozen["%s_%s" % (FROZEN_NAME_OF_SPLIT[split],
                                       FROZEN_NAME_OF_ARRAY[name])]
            assert identical(getattr(ours, name), theirs), (split, name)


def test_track_set_in_the_store_is_described():
    store = the_store()
    tracks = load_tracks(store, KEY_OF_THE_TRACKS)
    description = tracks.description
    assert description["key"] == KEY_OF_THE_TRACKS
    assert description["whole_track_set"] is True
    assert description["counts"] == COUNTS_OF_THE_FROZEN_FILE
    assert description["sample"]["hash_of_file"] == HASH_OF_THE_SAMPLE_FILE
    assert not os.path.isabs(description["sample"]["file"])
    assert description["provenance"]["traceable_to_the_commit"] is True
    assert len(description["provenance"]["commit"]) == 40
    assert tracks.first_plane_mm == 2648.2 and tracks.last_plane_mm == 7826.0


def test_splits_share_no_particle():
    tracks = load_tracks(the_store(), KEY_OF_THE_TRACKS)
    seen = set()
    for split in SPLITS:
        part = tracks.split(split)
        particles = set(zip(part.event.tolist(), part.particle_in_event.tolist()))
        assert len(particles) == part.number_of_tracks
        assert not (seen & particles)
        seen |= particles


def test_steps_fall_on_the_planes_of_the_track_set():
    tracks = load_tracks(the_store(), KEY_OF_THE_TRACKS, check_the_key=False)
    for number_of_steps in (2, 64, 128, 256):
        on = tracks.planes_a_step_starts_on(number_of_steps)
        planes = tracks.start_planes_mm(number_of_steps)
        assert len(on) == len(planes) == number_of_steps
        assert np.abs(tracks.planes_mm[on] - planes).max() < 1e-9
    with pytest.raises(ValueError):
        tracks.planes_a_step_starts_on(3)
