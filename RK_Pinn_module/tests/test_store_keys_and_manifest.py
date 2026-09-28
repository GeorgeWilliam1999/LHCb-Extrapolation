"""Gate: the store names things by their content and overwrites nothing.

Proves
  * a key depends on the names, the types, the shapes and the values of the
    arrays, and on nothing else;
  * the store refuses to write where something is, and a location must be given;
  * the manifest refuses a row it already lists and a row with wrong columns;
  * a track set whose content no longer gives its key is refused on loading;
  * writing from uncommitted changes is refused, or recorded as not traceable.

These gates use a store in a temporary folder. They do not touch the store of
the project.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from rkpinn.run_record import manifest, provenance
from rkpinn.run_record.store import (
    LENGTH_OF_A_KEY, AlreadyInTheStore, NotInTheStore, Store, key_of_content,
    location_from_the_environment)
from rkpinn.track_data.build_tracks import ARRAYS_OF_A_SPLIT, SPLITS
from rkpinn.track_data.load_tracks import ContentDoesNotGiveTheKey, load_tracks


def some_arrays():
    generator = np.random.default_rng(1)
    arrays = {"planes_mm": np.linspace(0.0, 100.0, 5)}
    for split in SPLITS:
        for name in ARRAYS_OF_A_SPLIT:
            shape = (3, 5, 5) if name == "reference_states_on_planes" else (3, 5)
            arrays["%s.%s" % (split, name)] = generator.random(shape)
    return arrays


def write_track_set(store, arrays, key=None):
    key = key or key_of_content(arrays)
    os.makedirs(store.folder_of_track_set(key))
    np.savez_compressed(store.file_of_tracks(key), **arrays)
    with open(store.description_of_tracks(key), "w") as handle:
        json.dump({"key": key, "settings": {"first_plane_mm": 0.0, "last_plane_mm": 100.0,
                                           "field_map": "v8r1_up"}}, handle)
    return key


def test_a_key_depends_on_the_content_and_nothing_else():
    a = {"x": np.arange(6.0).reshape(2, 3), "y": np.array([1, 2, 3])}
    same = {"y": np.array([1, 2, 3]), "x": np.arange(6.0).reshape(2, 3).copy()}
    assert key_of_content(a) == key_of_content(same)
    assert len(key_of_content(a)) == LENGTH_OF_A_KEY
    changed_value = {"x": a["x"].copy(), "y": a["y"]}
    changed_value["x"][0, 0] = np.nextafter(0.0, 1.0)
    others = (
        changed_value,
        {"x": a["x"].reshape(3, 2), "y": a["y"]},
        {"x": a["x"].astype(np.float32), "y": a["y"]},
        {"z": a["x"], "y": a["y"]},
    )
    keys = {key_of_content(other) for other in others} | {key_of_content(a)}
    assert len(keys) == 5


def test_the_location_of_the_store_must_be_given(monkeypatch):
    monkeypatch.delenv("RKPINN_STORE", raising=False)
    with pytest.raises(NotInTheStore):
        location_from_the_environment()
    monkeypatch.setenv("RKPINN_STORE", "/somewhere")
    assert location_from_the_environment() == "/somewhere"
    with pytest.raises(ValueError):
        Store("")


def test_the_store_refuses_to_overwrite(tmp_path):
    store = Store(str(tmp_path / "store")).create()
    assert os.path.isfile(os.path.join(store.location, "README.md"))
    key = write_track_set(store, some_arrays())
    assert store.holds_tracks(key)
    with pytest.raises(AlreadyInTheStore):
        store.refuse_if_present(store.folder_of_track_set(key))
    store.refuse_if_present(store.folder_of_track_set("000000000000"))
    assert store.file_of_exact_states(key, "test", 64, 2).endswith(
        os.path.join(key, "exact_scheme", "test", "64_steps_2_stages.npz"))


def test_a_track_set_is_loaded_by_its_key(tmp_path):
    store = Store(str(tmp_path / "store")).create()
    arrays = some_arrays()
    key = write_track_set(store, arrays)
    tracks = load_tracks(store, key)
    assert tracks.key == key
    assert tracks.test.number_of_tracks == 3
    assert np.array_equal(tracks.validation.momentum_gev, arrays["validation.momentum_gev"])
    assert np.array_equal(tracks.training.reference_end_state,
                          arrays["training.reference_states_on_planes"][:, -1])
    with pytest.raises(NotInTheStore):
        load_tracks(store, "000000000000")
    with pytest.raises(KeyError):
        tracks.split("train")


def test_content_that_does_not_give_its_key_is_refused(tmp_path):
    store = Store(str(tmp_path / "store")).create()
    write_track_set(store, some_arrays(), key="aaaaaaaaaaaa")
    with pytest.raises(ContentDoesNotGiveTheKey):
        load_tracks(store, "aaaaaaaaaaaa")


def test_the_manifest_lists_a_thing_once(tmp_path):
    store = Store(str(tmp_path / "store")).create()
    row = {column: "x" for column in manifest.COLUMNS_OF_TRACKS}
    row["key"] = "abc"
    manifest.add_track_set(store, row)
    assert [listed["key"] for listed in manifest.list_track_sets(store)] == ["abc"]
    with pytest.raises(manifest.AlreadyListed):
        manifest.add_track_set(store, row)
    with pytest.raises(ValueError):
        manifest.add_track_set(store, {"key": "def"})
    with pytest.raises(ValueError):
        manifest.add_track_set(store, dict(row, key="def", unknown_column=1))
    exact = {column: "x" for column in manifest.COLUMNS_OF_EXACT_STATES}
    manifest.add_exact_states(store, exact)
    manifest.add_exact_states(store, dict(exact, number_of_stages="y"))
    with pytest.raises(manifest.AlreadyListed):
        manifest.add_exact_states(store, exact)
    assert len(manifest.list_exact_states(store)) == 2


def test_uncommitted_changes_are_refused_or_recorded(monkeypatch):
    monkeypatch.setattr(provenance, "uncommitted_changes", lambda: [" M some_file.py"])
    with pytest.raises(provenance.UncommittedChanges):
        provenance.provenance()
    record = provenance.provenance(allow_uncommitted_changes=True)
    assert record["traceable_to_the_commit"] is False
    assert record["uncommitted_changes"] == [" M some_file.py"]
    monkeypatch.setattr(provenance, "uncommitted_changes", lambda: [])
    record = provenance.provenance()
    assert record["traceable_to_the_commit"] is True
    assert len(record["commit"]) == 40 and record["package_version"]
    assert record["machine"] and record["processor"]
