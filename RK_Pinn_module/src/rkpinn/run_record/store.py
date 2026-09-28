"""The store: the one place track sets, exact states and runs are kept.

    <store>/
    ├── README.md
    ├── tracks/<key>/
    │   ├── tracks.npz              the arrays
    │   ├── description.json        what they are and how they were made
    │   └── exact_scheme/<split>/<N>_steps_<q>_stages.npz  and its .json
    ├── runs/<key>/                 written by the trainer (phase 4)
    └── manifest/
        ├── tracks.csv
        └── exact_states.csv

A track set is named by a key computed from its content, so the same content
cannot exist under two names. Nothing in the store is overwritten: writing to
a place that exists is refused.

The location of the store is never a default in the code. It is passed as an
argument, or read from the environment variable RKPINN_STORE.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np

LENGTH_OF_A_KEY = 12
NAME_OF_THE_ENVIRONMENT_VARIABLE = "RKPINN_STORE"

README_OF_THE_STORE = """# The store of the rkpinn package

Track sets, the exact scheme's states and training runs. Written by the
package `rkpinn`; see `RK_Pinn_module/src/rkpinn/run_record/store.py`.

| Folder | Holds |
|---|---|
| `tracks/<key>/` | one track set: `tracks.npz`, `description.json`, and the exact scheme's states solved from it under `exact_scheme/` |
| `runs/<key>/` | one training run |
| `manifest/` | the lists of what is in the store |

A key is computed from the content it names. Nothing here is overwritten or
edited by hand.
"""


class AlreadyInTheStore(FileExistsError):
    """Something was about to be written where something already is."""


class NotInTheStore(FileNotFoundError):
    """A key or a file that the store does not hold was asked for."""


def key_of_content(arrays: dict) -> str:
    """The key of a set of named arrays: the start of the sha256 hash of
    their names, types, shapes and bytes, taken in the order of the names."""
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()[:LENGTH_OF_A_KEY]


def hash_of_file(path: str, chunk_bytes: int = 1 << 20) -> str:
    """The md5 hash of a file."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def location_from_the_environment() -> str:
    """The location of the store, from RKPINN_STORE. Refused if it is not set."""
    location = os.environ.get(NAME_OF_THE_ENVIRONMENT_VARIABLE)
    if not location:
        raise NotInTheStore(
            "the location of the store was not given: pass it, or set the "
            "environment variable %s" % NAME_OF_THE_ENVIRONMENT_VARIABLE)
    return location


class Store:
    """The folders and files of one store."""

    def __init__(self, location: str):
        if not location:
            raise ValueError("a store needs a location")
        self.location = os.path.abspath(location)

    # -- folders ---------------------------------------------------------------
    @property
    def folder_of_tracks(self) -> str:
        return os.path.join(self.location, "tracks")

    @property
    def folder_of_runs(self) -> str:
        return os.path.join(self.location, "runs")

    @property
    def folder_of_the_manifest(self) -> str:
        return os.path.join(self.location, "manifest")

    def create(self) -> "Store":
        """Make the folders of the store if they are not there."""
        for folder in (self.folder_of_tracks, self.folder_of_runs,
                       self.folder_of_the_manifest):
            os.makedirs(folder, exist_ok=True)
        readme = os.path.join(self.location, "README.md")
        if not os.path.exists(readme):
            with open(readme, "w") as handle:
                handle.write(README_OF_THE_STORE)
        return self

    # -- one track set ---------------------------------------------------------
    def folder_of_track_set(self, key: str) -> str:
        return os.path.join(self.folder_of_tracks, key)

    def file_of_tracks(self, key: str) -> str:
        return os.path.join(self.folder_of_track_set(key), "tracks.npz")

    def description_of_tracks(self, key: str) -> str:
        return os.path.join(self.folder_of_track_set(key), "description.json")

    def holds_tracks(self, key: str) -> bool:
        return os.path.isfile(self.file_of_tracks(key))

    # -- the exact scheme's states of one track set ------------------------------
    def file_of_exact_states(self, key: str, split: str, number_of_steps: int,
                             number_of_stages: int) -> str:
        return os.path.join(
            self.folder_of_track_set(key), "exact_scheme", split,
            "%d_steps_%d_stages.npz" % (number_of_steps, number_of_stages))

    def description_of_exact_states(self, key: str, split: str, number_of_steps: int,
                                    number_of_stages: int) -> str:
        return os.path.splitext(self.file_of_exact_states(
            key, split, number_of_steps, number_of_stages))[0] + ".json"

    # -- refusing to overwrite ---------------------------------------------------
    @staticmethod
    def refuse_if_present(path: str) -> None:
        if os.path.exists(path):
            raise AlreadyInTheStore(
                "%s is already in the store; nothing in the store is overwritten" % path)
