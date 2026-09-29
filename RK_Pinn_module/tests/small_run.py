"""A small store and a small configuration, for the gates of the training.
Not a gate.

The store is made in a temporary folder and holds the first tracks of each
split of the project's track set, under the key of that content. The
configuration is small so that a run takes seconds. Its values are those of
the gates; they are not defaults of the package.
"""
from __future__ import annotations

import copy
import json
import os

import numpy as np

from rkpinn.run_record.store import Store, key_of_content
from rkpinn.track_data.build_tracks import ARRAYS_OF_A_SPLIT, SPLITS
from rkpinn.track_data.load_tracks import load_tracks
from the_store import KEY_OF_THE_TRACKS, the_store

NUMBER_OF_TRACKS = {"training": 300, "validation": 100, "test": 50}


def small_store(folder) -> tuple:
    """(store, key of its tracks): a store in `folder` with a part of the
    project's tracks."""
    project = load_tracks(the_store(), KEY_OF_THE_TRACKS, check_the_key=False)
    arrays = {"planes_mm": project.planes_mm}
    for split in SPLITS:
        for name in ARRAYS_OF_A_SPLIT:
            arrays["%s.%s" % (split, name)] = getattr(
                project.split(split), name)[:NUMBER_OF_TRACKS[split]]
    key = key_of_content(arrays)
    store = Store(str(folder)).create()
    os.makedirs(store.folder_of_track_set(key))
    np.savez_compressed(store.file_of_tracks(key), **arrays)
    description = copy.deepcopy(project.description)
    description.update(key=key, whole_track_set=False, counts=dict(NUMBER_OF_TRACKS),
                       part_of=KEY_OF_THE_TRACKS)
    with open(store.description_of_tracks(key), "w") as handle:
        json.dump(description, handle, indent=1)
    return store, key


def small_configuration(tracks_key, **changes) -> dict:
    """A run of a stage network on four steps. `changes` replace whole blocks,
    or settings of the training when named `training_<setting>`."""
    configuration = {
        "equation_of_motion": {"system": "lhcb", "field_map": "v8r1_up"},
        "tracks": {"key": tracks_key},
        "steps": {"number_of_steps": 4, "number_of_stages": 2},
        "network": {"kind": "stage_network", "output_form": "direct_states",
                    "width": 16, "depth": 2, "activation": "tanh",
                    "end_state": "summed_from_the_stages",
                    "scale_of_inputs": "spread_of_first_round_states",
                    "scale_of_outputs": "spread_of_first_round_states"},
        "loss": {"name": "pooled", "terms": "stages"},
        "target": {"name": "no_target"},
        "training": {
            "protocol": {"name": "rounds_on_own_predictions", "states_per_round": 400},
            "restarts_per_round": 2,
            "end_a_round_when_stalled": None,
            "optimiser": {"name": "lbfgs_restarts", "iterations_per_restart": 5,
                          "history": 120, "tolerance_of_the_gradient": 1e-13,
                          "tolerance_of_the_change": 1e-16,
                          "line_search": "strong_wolfe",
                          "renew_scale_when_loss_falls_to": 0.1},
            "stopping_rule": {
                "name": "validation_plateau",
                "quantity": "validation_endpoint_error_larger_of_x_and_y_median_micrometres",
                "window_in_rounds": 10, "tolerance": 0.05, "rounds_held": 3},
            "rounds_at_most": 3,
        },
        "seed": 0,
    }
    for name, value in changes.items():
        if name.startswith("training_"):
            configuration["training"][name[len("training_"):]] = value
        else:
            configuration[name] = value
    return configuration


def whole_crossing_configuration(tracks_key, **changes) -> dict:
    """A run of the whole-crossing network on labels."""
    configuration = small_configuration(
        tracks_key,
        steps={"number_of_steps": 1, "number_of_stages": 0},
        network={"kind": "whole_crossing_network", "output_form": "direct_states",
                 "width": 16, "depth": 2, "activation": "tanh",
                 "scale_of_inputs": "spread_of_first_round_states",
                 "scale_of_outputs": "spread_of_target_end_states"},
        loss={"name": "supervised_endpoint", "divisor": "spread_of_target_end_states"},
        target={"name": "reference_end_state"},
        training_protocol={"name": "start_states_of_the_tracks", "states_per_round": 400})
    for name, value in changes.items():
        configuration[name] = value
    return configuration
