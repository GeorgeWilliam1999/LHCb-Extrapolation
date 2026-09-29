"""The manifest: the lists of what the store holds.

Built so far: the list of track sets and the list of the exact scheme's
states (phase 2), and the list of runs (phase 4). The metrics recomputed from
snapshots are added with the evaluation.

The list of runs says that a run exists and what it is. Where a run is, how
many rounds it has and how it ended, is in the run's own `state.json`.

A list is a CSV file with a fixed set of columns. A row is added when the
thing it names is written. The same thing is never listed twice.
"""
from __future__ import annotations

import csv
import os

from rkpinn.run_record.store import Store

COLUMNS_OF_TRACKS = (
    "key", "created", "number_of_training_tracks", "number_of_validation_tracks",
    "number_of_test_tracks", "number_of_planes", "first_plane_mm", "last_plane_mm",
    "field_map", "hash_of_field_map_file", "hash_of_sample_file",
    "reference_step_length_mm", "package_version", "commit",
    "traceable_to_the_commit", "machine",
)

COLUMNS_OF_EXACT_STATES = (
    "tracks_key", "split", "number_of_steps", "number_of_stages", "step_length_mm",
    "number_of_tracks", "solves_not_converged", "key_of_content", "created",
    "package_version", "commit", "traceable_to_the_commit", "machine", "processor",
)

COLUMNS_OF_RUNS = (
    "key", "created", "tracks_key", "network", "loss", "target", "number_of_steps",
    "number_of_stages", "seed", "package_version", "commit",
    "traceable_to_the_commit", "machine",
)


class AlreadyListed(ValueError):
    """The manifest already lists this."""


def _read(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _add(path: str, columns: tuple, row: dict, identity: tuple) -> None:
    missing = [c for c in columns if c not in row]
    unknown = [c for c in row if c not in columns]
    if missing or unknown:
        raise ValueError("a row of %s must have exactly its columns; missing %r, "
                         "unknown %r" % (os.path.basename(path), missing, unknown))
    for listed in _read(path):
        if all(str(listed[c]) == str(row[c]) for c in identity):
            raise AlreadyListed("%s already lists %r"
                                % (os.path.basename(path), {c: row[c] for c in identity}))
    is_new = not os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def file_of_tracks(store: Store) -> str:
    return os.path.join(store.folder_of_the_manifest, "tracks.csv")


def file_of_exact_states(store: Store) -> str:
    return os.path.join(store.folder_of_the_manifest, "exact_states.csv")


def list_track_sets(store: Store) -> list[dict]:
    return _read(file_of_tracks(store))


def add_track_set(store: Store, row: dict) -> None:
    _add(file_of_tracks(store), COLUMNS_OF_TRACKS, row, identity=("key",))


def list_exact_states(store: Store) -> list[dict]:
    return _read(file_of_exact_states(store))


def add_exact_states(store: Store, row: dict) -> None:
    _add(file_of_exact_states(store), COLUMNS_OF_EXACT_STATES, row,
         identity=("tracks_key", "split", "number_of_steps", "number_of_stages"))


def file_of_runs(store: Store) -> str:
    return os.path.join(store.folder_of_the_manifest, "runs.csv")


def list_runs(store: Store) -> list[dict]:
    return _read(file_of_runs(store))


def add_run(store: Store, row: dict) -> None:
    _add(file_of_runs(store), COLUMNS_OF_RUNS, row, identity=("key",))
