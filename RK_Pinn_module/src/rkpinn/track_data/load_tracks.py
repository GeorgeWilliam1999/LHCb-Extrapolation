"""Load a track set from the store by its key.

A run refers to its tracks by key, never by path. The key is computed from
the content, and it is checked when the tracks are loaded: a file whose
content no longer gives its key is refused.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from rkpinn.run_record.store import NotInTheStore, Store, key_of_content
from rkpinn.track_data.build_tracks import ARRAYS_OF_A_SPLIT, SPLITS


class ContentDoesNotGiveTheKey(ValueError):
    """The arrays of a track set no longer hash to the key they are stored under."""


@dataclass(frozen=True)
class TracksOfOneSplit:
    """The tracks of the training, validation or test split. The rows of every
    array are the same particles in the same order."""

    split: str
    start_state: np.ndarray
    reference_states_on_planes: np.ndarray
    reference_state_on_own_fibre_plane: np.ndarray
    true_state_on_own_upstream_plane: np.ndarray
    true_state_on_own_fibre_plane: np.ndarray
    own_upstream_plane_mm: np.ndarray
    own_fibre_plane_mm: np.ndarray
    momentum_gev: np.ndarray
    pseudorapidity: np.ndarray
    particle_type: np.ndarray
    event: np.ndarray
    particle_in_event: np.ndarray

    @property
    def number_of_tracks(self) -> int:
        return len(self.start_state)

    @property
    def reference_end_state(self) -> np.ndarray:
        """The reference state on the last plane of the crossing."""
        return self.reference_states_on_planes[:, -1]


@dataclass(frozen=True)
class TrackSet:
    key: str
    planes_mm: np.ndarray
    description: dict
    training: TracksOfOneSplit
    validation: TracksOfOneSplit
    test: TracksOfOneSplit

    @property
    def first_plane_mm(self) -> float:
        return float(self.description["settings"]["first_plane_mm"])

    @property
    def last_plane_mm(self) -> float:
        return float(self.description["settings"]["last_plane_mm"])

    @property
    def length_mm(self) -> float:
        return self.last_plane_mm - self.first_plane_mm

    @property
    def number_of_steps_between_planes(self) -> int:
        return len(self.planes_mm) - 1

    def split(self, name: str) -> TracksOfOneSplit:
        if name not in SPLITS:
            raise KeyError("the splits are %s, not %r" % (", ".join(SPLITS), name))
        return getattr(self, name)

    def planes_a_step_starts_on(self, number_of_steps: int) -> np.ndarray:
        """The index, among the planes of the track set, of the plane each of
        `number_of_steps` equal steps starts on."""
        finest = self.number_of_steps_between_planes
        if number_of_steps < 1 or finest % number_of_steps:
            raise ValueError(
                "%d steps do not fall on the planes of this track set, which has "
                "%d steps between its planes" % (number_of_steps, finest))
        return np.arange(0, finest, finest // number_of_steps)

    def start_planes_mm(self, number_of_steps: int) -> np.ndarray:
        """z of the plane each step starts on: first plane + k * step length."""
        return (self.first_plane_mm
                + np.arange(number_of_steps) * (self.length_mm / number_of_steps))


def load_tracks(store: Store, key: str, check_the_key: bool = True) -> TrackSet:
    if not store.holds_tracks(key):
        raise NotInTheStore("the store at %s holds no track set with the key %r"
                            % (store.location, key))
    with np.load(store.file_of_tracks(key), allow_pickle=False) as file:
        arrays = {name: file[name] for name in file.files}
    if check_the_key and key_of_content(arrays) != key:
        raise ContentDoesNotGiveTheKey(
            "the track set stored under %r has content with the key %r"
            % (key, key_of_content(arrays)))
    with open(store.description_of_tracks(key)) as handle:
        description = json.load(handle)
    splits = {
        name: TracksOfOneSplit(split=name, **{
            column: arrays["%s.%s" % (name, column)] for column in ARRAYS_OF_A_SPLIT})
        for name in SPLITS}
    return TrackSet(key=key, planes_mm=arrays["planes_mm"], description=description,
                    **splits)
