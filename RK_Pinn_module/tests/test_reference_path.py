"""Gate: the reference path, the exact states kept with their stages, and a
small run of the path network through the trainer.

Proves
  * `exact_states --keep-stages` writes the stage states beside the plane
    states, and they come back from the store as written (round trip on a
    small store); the plane states are those written without stages;
  * the reference path has the shape (n, steps, stages + 1, 4) and is the
    stored stage states followed by the stored plane state of every step;
  * it refuses a layout the store does not hold, a file written without its
    stage states, and a layout with no stages;
  * the project's store holds the training and validation splits of the
    project's tracks at 64 steps and 2 stages with their stage states, every
    solve converged, and the stage states of the first tracks, solved again
    now, agree with the store;
  * a path network on the reference path trains through the round trainer on
    the start states of the tracks, and the scale `spread_of_target_states`
    is the spread of every labelled state pooled.
"""
from __future__ import annotations

import numpy as np
import pytest

from building_blocks import equations
from rkpinn import registry
from rkpinn.integrators.gauss_legendre_tableau import gauss_legendre_tableau
from rkpinn.losses.pooled import spread_of_components
from rkpinn.predicted_track.predicted_track import track_of_the_exact_scheme
from rkpinn.predicted_track.track_layout import TrackLayout
from rkpinn.run_record.store import NotInTheStore
from rkpinn.track_data.exact_states import (
    load_exact_states, read_exact_states_with_stages, solve_exact_states, write_exact_states)
from rkpinn.track_data.load_tracks import load_tracks
from rkpinn.run_record.configuration import plain
from rkpinn.training import checkpoints, round_trainer
from small_run import small_configuration, small_store
from the_store import KEY_OF_THE_TRACKS, the_store

QUIET = dict(allow_uncommitted_changes=True, report=lambda text: None)
STEPS, STAGES = 4, 2


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    store, key = small_store(tmp_path_factory.mktemp("path_store"))
    for split in ("training", "validation"):
        write_exact_states(store, key, split, STEPS, STAGES, workers=4, keep_stages=True, **QUIET)
    write_exact_states(store, key, "test", STEPS, STAGES, workers=4, **QUIET)
    return store, key, load_tracks(store, key)


def target_of(store, tracks, steps=STEPS, stages=STAGES):
    layout = TrackLayout(tracks.first_plane_mm, tracks.last_plane_mm, steps,
                         gauss_legendre_tableau(stages) if stages else None)
    return registry.component("target", "reference_path")(layout=layout, store=store)


def test_the_stages_come_back_as_written(small):
    store, key, tracks = small
    planes, stages = read_exact_states_with_stages(store, key, "training", STEPS, STAGES)
    again = solve_exact_states(tracks, "training", STEPS, STAGES, workers=1, most_tracks=10,
                               keep_stages=True)
    assert np.array_equal(planes[:10], again[0]) and np.array_equal(stages[:10], again[3])
    without = solve_exact_states(tracks, "training", STEPS, STAGES, workers=1, most_tracks=10)
    assert len(without) == 3 and np.array_equal(without[0], again[0])
    states, _, description = load_exact_states(store, key, "training", STEPS, STAGES)
    assert np.array_equal(states, planes) and "stage_states" in description["arrays"]
    assert stages.shape == (tracks.training.number_of_tracks, STEPS, STAGES, 4)


def test_the_reference_path(small):
    store, key, tracks = small
    target = target_of(store, tracks)
    labels = target.for_states(tracks, "training")
    planes, stages = read_exact_states_with_stages(store, key, "training", STEPS, STAGES)
    assert labels.shape == (tracks.training.number_of_tracks, STEPS, STAGES + 1, 4)
    assert np.array_equal(labels[:, :, :STAGES], stages)
    assert np.array_equal(labels[:, :, STAGES], planes[:, 1:, :4])
    assert target.for_states(tracks, "training") is labels            # read once
    assert target.needs_labels and target.can_be_trained_on and target.needs_layout


def test_what_the_reference_path_refuses(small):
    store, key, tracks = small
    with pytest.raises(NotInTheStore):
        target_of(store, tracks, stages=4).for_states(tracks, "training")
    with pytest.raises(NotInTheStore):
        target_of(store, tracks).for_states(tracks, "test")             # written without stages
    with pytest.raises(ValueError):
        target_of(store, tracks, stages=0)
    with pytest.raises(KeyError):
        target_of(store, tracks).for_states(tracks, "train")


@pytest.mark.parametrize("split", ("training", "validation"))
def test_the_project_store_holds_the_reference_path(split):
    store = the_store()
    tracks = load_tracks(store, KEY_OF_THE_TRACKS, check_the_key=False)
    planes, stages = read_exact_states_with_stages(store, KEY_OF_THE_TRACKS, split, 64, 2)
    _, _, description = load_exact_states(store, KEY_OF_THE_TRACKS, split, 64, 2,
                                          check_the_key=False)
    n = tracks.split(split).number_of_tracks
    assert planes.shape == (n, 65, 5) and stages.shape == (n, 64, 2, 4)
    assert description["solves_not_converged"] == 0
    assert description["provenance"]["traceable_to_the_commit"] is True
    assert np.array_equal(planes[:, 0], tracks.split(split).start_state)
    layout = TrackLayout(tracks.first_plane_mm, tracks.last_plane_mm, 64,
                         gauss_legendre_tableau(2))
    again = track_of_the_exact_scheme(equations()[0], tracks.split(split).start_state[:2], layout)
    assert np.abs(again.stage_states[..., :2] - stages[:2, ..., :2]).max() < 1e-11
    assert np.abs(again.stage_states[..., 2:] - stages[:2, ..., 2:]).max() < 1e-15


def path_configuration(key):
    return small_configuration(
        key,
        network={"kind": "path_network", "output_form": "straight_line_plus_scaled_correction",
                 "width": 8, "depth": 2, "activation": "tanh", "extra_outputs": 4,
                 "scale_of_inputs": "spread_of_first_round_states",
                 "scale_of_outputs": "per_track"},
        loss={"name": "supervised_path", "divisor": "spread_of_target_states",
              "uncertainty": "diagonal"},
        target={"name": "reference_path"},
        training_protocol={"name": "start_states_of_the_tracks", "states_per_round": 400},
        training_rounds_at_most=2)


def test_a_path_network_trains_through_the_trainer(small):
    store, key, tracks = small
    configuration = path_configuration(key)
    run = round_trainer.build_run(configuration, store)
    labels = run.first_round.labels
    assert labels.shape == (tracks.training.number_of_tracks, STEPS, STAGES + 1, 4)
    assert run.constants["loss"]["divisor_per_component"] == \
        spread_of_components(labels.reshape(-1, 4)).tolist()
    state = round_trainer.train(configuration, store, if_the_run_exists="refuse", **QUIET)
    assert state["state"] == checkpoints.AT_ITS_CAP and state["rounds_done"] == 2
    resumed = round_trainer.build_run(configuration, store, recorded_constants=run.constants)
    assert plain(resumed.network.settings()) == run.constants["network"]
