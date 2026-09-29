"""Gate: the training, from one restart to a run that stops and resumes.

Proves
  * the optimiser is that of the frozen trainer: after three restarts the
    weights and the losses are identical, to the last bit, to those of the
    frozen lines run on the frozen loss;
  * the stopping rule is identical to the frozen plateau rule, on random and
    on made series;
  * the protocols give the states they say: round 1 from the reference, later
    rounds from the network's own predictions, and labels in the order of the
    states;
  * a run trains, writes a snapshot after every round, and ends at its cap;
  * a run that is stopped and resumed is identical, to the last bit, to a run
    that was not stopped: weights, losses and validation errors;
  * a second writer is refused; a run that exists is not made again; a
    snapshot is never overwritten;
  * a run at its cap is extended, and its earlier snapshots are not touched;
  * a run ends plateaued when the rule holds, and a round ends early when it
    is set to end when stalled;
  * the constants recorded in the run are the constants the optimiser used;
  * the whole-crossing network trains on labels;
  * no absolute path is written into a run;
  * training from uncommitted changes is refused.

The runs are small and are made in a temporary store. They prove the
mechanics. They say nothing about what a network can reach.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pytest
import torch

from building_blocks import crossing, equations
from frozen_code import frozen_module, frozen_plateau_rule, identical
from rkpinn.losses.pooled import Pooled
from rkpinn.losses.stage_residual import TERMS_STAGES_AND_END_STATE
from rkpinn.predicted_track.predicted_track import END_STATE_PREDICTED
from rkpinn.run_record import manifest, provenance
from rkpinn.run_record.configuration import run_key
from rkpinn.run_record.store import AlreadyInTheStore, NotInTheStore
from rkpinn.training import checkpoints, round_trainer
from rkpinn.training.checkpoints import RunFolder, RunIsLocked
from rkpinn.training.optimiser import LbfgsRestarts
from rkpinn.training.stopping_rule import ValidationPlateau, validation_has_plateaued
from rkpinn.training.training_protocols import (
    FROM_OWN_PREDICTIONS, FROM_THE_REFERENCE, FROM_START_STATES)
from rkpinn.training.validation_error import QUANTITIES, validation_errors
from small_run import small_configuration, small_store, whole_crossing_configuration
from test_losses import AsTheFrozenCodeSeesIt

QUIET = dict(allow_uncommitted_changes=True, report=lambda text: None)
NOT_TIMES = [c for c in checkpoints.COLUMNS_OF_ROUNDS if not c.startswith("seconds")]
NOT_TIMES_OF_RESTARTS = [c for c in checkpoints.COLUMNS_OF_RESTARTS if c != "seconds"]


def train(configuration, store, **how):
    return round_trainer.train(configuration, store, **dict(QUIET, **how))


def hash_of(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


@pytest.fixture(scope="module")
def stores(tmp_path_factory):
    """Two stores with the same small track set."""
    first = small_store(tmp_path_factory.mktemp("first_store"))
    second = small_store(tmp_path_factory.mktemp("second_store"))
    assert first[1] == second[1]
    return first[0], second[0], first[1]


@pytest.fixture(scope="module")
def run_in_one_call(stores):
    """Three rounds, not stopped, in the first store."""
    store, _, tracks_key = stores
    configuration = small_configuration(tracks_key)
    state = train(configuration, store, if_the_run_exists="refuse")
    return configuration, RunFolder(store, run_key(configuration)), state


# ------------------------------------------------------------------ the optimiser
def test_optimiser_is_identical_to_the_frozen_trainer():
    cell = crossing(64, 2)
    states, planes = cell.states(600)
    settings = dict(iterations_per_restart=12, history=120,
                    tolerance_of_the_gradient=1e-13, tolerance_of_the_change=1e-16,
                    line_search="strong_wolfe", renew_scale_when_loss_falls_to=0.1)

    # the package
    ours = cell.network(END_STATE_PREDICTED, seed=2)
    loss = Pooled(equation_of_motion=equations()[1], terms=TERMS_STAGES_AND_END_STATE)
    constants = loss.constants(cell.first_round_states)
    optimiser = LbfgsRestarts(**settings)
    optimiser.start_round(ours.parameters())
    rows = [optimiser.one_restart(
        lambda: loss.value(ours.predict(states, planes), None, constants))
        for _ in range(3)]

    # the lines of the frozen trainer, on the frozen loss
    frozen_model, frozen_reference = frozen_module("model"), frozen_module("reference")
    theirs = cell.network(END_STATE_PREDICTED, seed=2)
    model = AsTheFrozenCodeSeesIt(theirs, cell.spread)
    rates = frozen_model.LHCbRates(frozen_reference.make_field("up"))
    zn = planes[:, None] + torch.as_tensor(cell.tableau.nodes)[None, :] * cell.step_length_mm
    A, b = torch.as_tensor(cell.tableau.stage_matrix), torch.as_tensor(cell.tableau.weights)

    def loss_now():
        return frozen_model.physics_loss(model, rates, states, cell.step_length_mm, zn,
                                         A, b, planes)

    def make_opt():
        return torch.optim.LBFGS(theirs.parameters(), max_iter=12, history_size=120,
                                 tolerance_grad=1e-13, tolerance_change=1e-16,
                                 line_search_fn="strong_wolfe")

    state = {"opt": make_opt(), "factor": None, "factor_loss": None}
    frozen_rows = []
    for _ in range(3):
        with torch.no_grad():
            before = loss_now().item()
        if state["factor"] is None or before < 0.1 * state["factor_loss"]:
            state.update(opt=make_opt(), factor=1.0 / before, factor_loss=before)
        opt, factor = state["opt"], state["factor"]

        def closure():
            opt.zero_grad()
            value = loss_now() * factor
            value.backward()
            return value

        opt.step(closure)
        with torch.no_grad():
            frozen_rows.append((before, loss_now().item(), factor))

    assert [(r["loss_before"], r["loss_after"], r["factor"]) for r in rows] == frozen_rows
    assert rows[0]["factor_renewed"] == 1 and rows[0]["loss_after"] < rows[0]["loss_before"]
    assert sum(r["iterations"] for r in rows) > 0
    for mine, frozen in zip(ours.parameters(), theirs.parameters()):
        assert identical(mine.detach().numpy(), frozen.detach().numpy())


def test_the_factor_is_renewed_when_the_loss_has_fallen():
    parameter = torch.nn.Parameter(torch.tensor([3.0], dtype=torch.float64))
    optimiser = LbfgsRestarts(
        iterations_per_restart=3, history=10, tolerance_of_the_gradient=1e-13,
        tolerance_of_the_change=1e-16, line_search="strong_wolfe",
        renew_scale_when_loss_falls_to=0.1)
    optimiser.start_round([parameter])
    first = optimiser.one_restart(lambda: (parameter ** 4).sum())
    second = optimiser.one_restart(lambda: (parameter ** 4).sum())
    assert first["factor_renewed"] == 1 and first["factor"] == 1.0 / 81.0
    assert second["loss_before"] == first["loss_after"]
    assert second["factor_renewed"] == int(second["loss_before"] < 0.1 * 81.0)
    optimiser.start_round([parameter])
    assert optimiser.one_restart(lambda: (parameter ** 4).sum())["factor_renewed"] == 1


# ---------------------------------------------------------------- the stopping rule
def test_stopping_rule_is_identical_to_the_frozen_rule():
    frozen = frozen_plateau_rule()
    generator = np.random.default_rng(7)
    verdicts = []
    for _ in range(400):
        length = int(generator.integers(1, 60))
        fall = generator.uniform(0.0, 0.03)
        series = 100.0 * np.exp(-fall * np.arange(length)) \
            * (1 + 0.05 * generator.standard_normal(length))
        for window, tolerance, held in ((10, 0.05, 3), (3, 0.02, 2), (1, 0.0, 1)):
            ours = validation_has_plateaued(series, window, tolerance, held)
            theirs = frozen.plateaued_now(series, window=window, tol=tolerance, hold=held)
            assert ours == bool(theirs)
            verdicts.append(ours)
    assert any(verdicts) and not all(verdicts)


def test_stopping_rule_on_made_series():
    rule = ValidationPlateau(quantity=QUANTITIES[2], window_in_rounds=10, tolerance=0.05,
                             rounds_held=3)
    assert not rule.holds(np.full(21, 50.0))            # one round too few
    assert rule.holds(np.full(22, 50.0))
    assert not rule.holds(100.0 * 0.97 ** np.arange(40))   # still falling 3 % a round
    assert rule.holds(np.concatenate([100.0 * 0.9 ** np.arange(10), np.full(25, 30.0)]))
    with pytest.raises(ValueError):
        ValidationPlateau(quantity="the_loss", window_in_rounds=10, tolerance=0.05,
                          rounds_held=3)


def test_validation_errors():
    reference = np.zeros((5, 5))
    predicted = np.zeros((5, 4))
    predicted[:, 0] = [0.001, -0.002, 0.003, -0.004, 0.005]      # mm
    predicted[:, 1] = [0.010, 0.000, -0.001, 0.000, 0.000]
    errors = validation_errors(predicted, reference)
    assert errors[QUANTITIES[0]] == pytest.approx(3.0)           # micrometres
    assert errors[QUANTITIES[1]] == pytest.approx(0.0)
    assert errors[QUANTITIES[2]] == pytest.approx(4.0)


# ------------------------------------------------------------------- the protocols
def test_protocols_give_the_states_they_say(stores):
    store, _, tracks_key = stores
    run = round_trainer.build_run(small_configuration(tracks_key), store)
    tracks, layout = run.tracks, run.layout
    first = run.first_round
    assert first.source == FROM_THE_REFERENCE and first.labels is None
    assert len(first.states) == 400
    assert set(np.unique(first.start_planes_mm)) == set(layout.start_planes_mm)
    on_planes = tracks.training.reference_states_on_planes[
        :, tracks.planes_a_step_starts_on(4)]
    step = np.searchsorted(layout.start_planes_mm, first.start_planes_mm)
    for row in range(0, 400, 37):
        assert (on_planes[:, step[row]] == first.states[row]).all(axis=1).any()

    second = run.protocol.states_of_round(2, 0, tracks, layout, run.network, run.target)
    assert second.source == FROM_OWN_PREDICTIONS
    whole = run.network.whole_track(tracks.training.start_state, layout)
    step = np.searchsorted(layout.start_planes_mm, second.start_planes_mm)
    for row in range(0, 400, 37):
        assert (whole.input_states[:, step[row]] == second.states[row]).all(axis=1).any()
    again = run.protocol.states_of_round(2, 0, tracks, layout, run.network, run.target)
    assert identical(again.states, second.states)
    other = run.protocol.states_of_round(3, 0, tracks, layout, run.network, run.target)
    assert not identical(other.states, second.states)

    crossing_run = round_trainer.build_run(whole_crossing_configuration(tracks_key), store)
    given = crossing_run.first_round
    assert given.source == FROM_START_STATES
    assert identical(given.states, tracks.training.start_state)
    assert identical(given.labels, tracks.training.reference_end_state)
    assert (given.start_planes_mm == layout.first_plane_mm).all()


# ----------------------------------------------------------------------- a run
def test_a_run_trains_and_ends_at_its_cap(run_in_one_call, stores):
    configuration, folder, state = run_in_one_call
    assert state["state"] == checkpoints.AT_ITS_CAP
    assert state["rounds_done"] == 3 and state["restarts_done"] == 6
    assert not folder.is_locked()
    rounds, restarts = folder.rounds(), folder.restarts()
    assert [int(r["round"]) for r in rounds] == [1, 2, 3]
    assert [int(r["restart"]) for r in restarts] == [1, 2, 3, 4, 5, 6]
    assert [r["source_of_states"] for r in rounds] == [
        FROM_THE_REFERENCE, FROM_OWN_PREDICTIONS, FROM_OWN_PREDICTIONS]
    assert float(rounds[0]["loss_last"]) < float(rounds[0]["loss_first"])
    for number in (1, 2, 3):
        assert os.path.isfile(os.path.join(folder.folder_of_snapshot(number), "network.pt"))
        scores = folder.scores_of_round(number)
        assert scores["round"] == number and scores["run_key"] == folder.key
        assert scores["number_of_validation_tracks"] == 100
        for quantity in QUANTITIES:
            assert np.isfinite(scores[quantity]) and scores[quantity] > 0
    listed = manifest.list_runs(stores[0])
    assert [row["key"] for row in listed].count(folder.key) == 1
    record = folder.provenance()
    assert record["tracks_key"] == configuration["tracks"]["key"]
    assert record["hash_of_field_map_file"] == record["hash_of_field_map_file_of_the_tracks"]
    assert len(record["commit"]) == 40 and record["package_version"]


def test_a_run_stopped_and_resumed_is_the_run_not_stopped(run_in_one_call, stores):
    configuration, not_stopped, _ = run_in_one_call
    _, store, _ = stores
    state = train(configuration, store, if_the_run_exists="refuse", rounds_in_this_call=1)
    assert state["state"] == checkpoints.INTERRUPTED and state["rounds_done"] == 1
    state = train(configuration, store, if_the_run_exists="resume", rounds_in_this_call=1)
    assert state["state"] == checkpoints.INTERRUPTED and state["rounds_done"] == 2
    state = train(configuration, store, if_the_run_exists="resume")
    assert state["state"] == checkpoints.AT_ITS_CAP and state["rounds_done"] == 3
    assert [i["asked"] for i in state["invocations"]] == ["refuse", "resume", "resume"]
    assert [i["from_round"] for i in state["invocations"]] == [0, 1, 2]

    resumed = RunFolder(store, run_key(configuration))
    assert resumed.key == not_stopped.key
    for number in (1, 2, 3):
        a, b = not_stopped.weights_of_round(number), resumed.weights_of_round(number)
        assert list(a) == list(b)
        for name in a:
            assert identical(a[name].numpy(), b[name].numpy()), (number, name)
    for a, b in zip(not_stopped.rounds(), resumed.rounds()):
        assert [a[c] for c in NOT_TIMES] == [b[c] for c in NOT_TIMES]
    for a, b in zip(not_stopped.restarts(), resumed.restarts()):
        assert [a[c] for c in NOT_TIMES_OF_RESTARTS] == [b[c] for c in NOT_TIMES_OF_RESTARTS]
    assert len(resumed.restarts()) == len(not_stopped.restarts()) == 6
    assert resumed.constants() == not_stopped.constants()


def test_a_run_that_exists_is_not_made_again(run_in_one_call, stores):
    configuration, folder, _ = run_in_one_call
    before = hash_of(os.path.join(folder.folder_of_snapshot(3), "network.pt"))
    with pytest.raises(AlreadyInTheStore):
        train(configuration, stores[0], if_the_run_exists="refuse")
    with pytest.raises(ValueError, match="has ended"):
        train(configuration, stores[0], if_the_run_exists="resume")
    with pytest.raises(ValueError, match="cap must be higher"):
        train(configuration, stores[0], if_the_run_exists="extend")
    with pytest.raises(ValueError):
        train(configuration, stores[0], if_the_run_exists="overwrite")
    assert hash_of(os.path.join(folder.folder_of_snapshot(3), "network.pt")) == before
    assert folder.state()["rounds_done"] == 3 and not folder.is_locked()


def test_a_run_that_does_not_exist_cannot_be_resumed(stores):
    store, _, tracks_key = stores
    never_made = small_configuration(tracks_key, seed=77)
    for asked in ("resume", "extend"):
        with pytest.raises(NotInTheStore):
            train(never_made, store, if_the_run_exists=asked)
    assert not RunFolder(store, run_key(never_made)).exists()


def test_a_second_writer_is_refused(stores):
    store, _, tracks_key = stores
    configuration = small_configuration(tracks_key, seed=5)
    train(configuration, store, if_the_run_exists="refuse", rounds_in_this_call=1)
    folder = RunFolder(store, run_key(configuration))
    folder.take_lock()                       # a job that holds the run
    try:
        with pytest.raises(RunIsLocked):
            train(configuration, store, if_the_run_exists="resume")
        with pytest.raises(RunIsLocked):
            folder.take_lock()
        assert folder.is_locked(), "the refused writer must not remove the lock it met"
        assert folder.state()["rounds_done"] == 1
    finally:
        folder.clear_lock()
    state = train(configuration, store, if_the_run_exists="resume", rounds_in_this_call=1)
    assert state["rounds_done"] == 2 and not folder.is_locked()


def test_a_snapshot_is_never_overwritten(run_in_one_call):
    _, folder, _ = run_in_one_call
    rows = folder.restarts()[:1]
    with pytest.raises(ValueError, match="cannot follow"):
        folder.write_round(2, folder.weights_of_round(2), {}, rows, folder.rounds()[1])
    state = folder.state()
    state["rounds_done"] = 1                 # as if a job had died while writing round 2
    with open(folder.path("state.json.as_if"), "w") as handle:
        json.dump(state, handle)
    real = folder.path("state.json")
    os.replace(real, real + ".kept")
    os.replace(folder.path("state.json.as_if"), real)
    try:
        with pytest.raises(AlreadyInTheStore):
            folder.write_round(2, folder.weights_of_round(2), {}, rows, folder.rounds()[1])
    finally:
        os.replace(real + ".kept", real)
    assert folder.state()["rounds_done"] == 3


def test_a_run_at_its_cap_is_extended(stores):
    store, _, tracks_key = stores
    configuration = small_configuration(tracks_key, seed=9, training_rounds_at_most=2)
    state = train(configuration, store, if_the_run_exists="refuse")
    assert state["state"] == checkpoints.AT_ITS_CAP and state["rounds_done"] == 2
    folder = RunFolder(store, run_key(configuration))
    before = {n: hash_of(os.path.join(folder.folder_of_snapshot(n), "network.pt"))
              for n in (1, 2)}
    longer = small_configuration(tracks_key, seed=9, training_rounds_at_most=4)
    assert run_key(longer) == folder.key
    state = train(longer, store, if_the_run_exists="extend")
    assert state["state"] == checkpoints.AT_ITS_CAP and state["rounds_done"] == 4
    assert state["rounds_at_most"] == 4
    assert [(e["from_rounds_at_most"], e["to_rounds_at_most"], e["rounds_done"])
            for e in state["extensions"]] == [(2, 4, 2)]
    for number in (1, 2):
        assert hash_of(os.path.join(folder.folder_of_snapshot(number),
                                    "network.pt")) == before[number]
    assert [int(r["round"]) for r in folder.rounds()] == [1, 2, 3, 4]
    assert [int(r["restart"]) for r in folder.restarts()] == list(range(1, 9))


def test_a_run_ends_plateaued_when_the_rule_holds(stores):
    store, _, tracks_key = stores
    configuration = small_configuration(tracks_key, seed=11, training_rounds_at_most=6)
    configuration["training"]["stopping_rule"].update(
        window_in_rounds=1, tolerance=0.99, rounds_held=1)
    state = train(configuration, store, if_the_run_exists="refuse")
    assert state["state"] == checkpoints.PLATEAUED
    assert state["rounds_done"] == 2 == state["plateaued_at_round"]


def test_a_round_ends_early_when_it_is_set_to(stores):
    store, _, tracks_key = stores
    configuration = small_configuration(
        tracks_key, seed=13, training_restarts_per_round=4, training_rounds_at_most=1,
        training_end_a_round_when_stalled={"gain_below": 10.0, "restarts_running": 2})
    train(configuration, store, if_the_run_exists="refuse")
    rounds = RunFolder(store, run_key(configuration)).rounds()
    assert int(rounds[0]["restarts"]) == 2


def test_constants_recorded_are_the_constants_used(run_in_one_call, stores):
    configuration, folder, _ = run_in_one_call
    recorded = folder.constants()
    run = round_trainer.build_run(configuration, stores[0], recorded)
    measured_again = round_trainer.build_run(configuration, stores[0])
    assert recorded == run.constants == measured_again.constants
    spread = run.first_round.states.std(axis=0)
    assert recorded["network"]["scale_of_inputs"] == spread.tolist()
    assert recorded["network"]["scale_of_outputs"] == spread[:4].tolist()
    assert recorded["loss"]["divisor_per_component"] == spread[:4].tolist()
    assert recorded["loss"]["terms"] == configuration["loss"]["terms"]
    assert recorded["network"]["activation"] == "tanh"
    measured = round_trainer.checks_before_training(run)
    assert measured["loss_at_the_exact_solution"] < 1e-18
    run.constants = dict(recorded, network=dict(recorded["network"], width=17))
    with pytest.raises(round_trainer.CheckBeforeTrainingFailed):
        round_trainer.checks_before_training(run)


@pytest.mark.parametrize("loss", (
    {"name": "unweighted", "terms": "stages"},
    {"name": "cost_weighted", "terms": "stages", "momentum_window_gev": [10.0, 50.0],
     "roll_off": 0.6931, "floor": 0.05, "clamp": 5.0,
     "samples_of_the_field_integral": 1024, "lever_arm_is_on": True,
     "track_bend_is_on": True, "momentum_window_is_on": True,
     "reference_bend_mm": "median_of_first_round_states"},
    {"name": "pooled", "terms": "stages_and_end_state"},
))
def test_every_loss_trains_from_a_configuration(stores, loss):
    store, _, tracks_key = stores
    configuration = small_configuration(tracks_key, loss=loss, seed=21,
                                        training_rounds_at_most=1)
    if loss["terms"] == "stages_and_end_state":
        configuration["network"]["end_state"] = "predicted"
    state = train(configuration, store, if_the_run_exists="refuse")
    folder = RunFolder(store, run_key(configuration))
    assert state["rounds_done"] == 1
    assert float(folder.rounds()[0]["loss_last"]) < float(folder.rounds()[0]["loss_first"])
    assert folder.constants()["loss"]["name"] == loss["name"]
    if loss["name"] == "cost_weighted":
        assert folder.constants()["loss"]["momentum_window_gev"] == [10.0, 50.0]
        assert folder.constants()["loss"]["reference_bend_mm"] > 0


def test_the_whole_crossing_network_trains_on_labels(stores):
    store, _, tracks_key = stores
    configuration = whole_crossing_configuration(tracks_key)
    state = train(configuration, store, if_the_run_exists="refuse")
    folder = RunFolder(store, run_key(configuration))
    assert state["state"] == checkpoints.AT_ITS_CAP and state["rounds_done"] == 3
    rounds = folder.rounds()
    assert [r["source_of_states"] for r in rounds] == [FROM_START_STATES] * 3
    assert float(rounds[-1]["loss_last"]) < float(rounds[0]["loss_first"])
    tracks = round_trainer.build_run(configuration, store, folder.constants()).tracks
    constants = folder.constants()
    assert constants["loss"]["divisor_per_component"] \
        == tracks.training.reference_end_state.std(axis=0)[:4].tolist()
    assert constants["network"]["scale_of_outputs"] \
        == constants["loss"]["divisor_per_component"]


def test_no_absolute_path_is_written_into_a_run(run_in_one_call, stores):
    _, folder, _ = run_in_one_call
    for name in ("configuration.yaml", "provenance.json", "constants.json", "state.json",
                 "restarts.csv", "rounds.csv", "snapshots/round_0001/scores.json"):
        with open(folder.path(name)) as handle:
            text = handle.read()
        assert stores[0].location not in text and "/data/" not in text, name


def test_training_from_uncommitted_changes_is_refused(stores, monkeypatch):
    store, _, tracks_key = stores
    configuration = small_configuration(tracks_key, seed=31)
    monkeypatch.setattr(provenance, "uncommitted_changes", lambda: [" M some_file.py"])
    with pytest.raises(provenance.UncommittedChanges):
        round_trainer.train(configuration, store, if_the_run_exists="refuse",
                            report=lambda text: None)
    assert not RunFolder(store, run_key(configuration)).exists()
    state = round_trainer.train(configuration, store, if_the_run_exists="refuse",
                                allow_uncommitted_changes=True, rounds_in_this_call=1,
                                report=lambda text: None)
    folder = RunFolder(store, run_key(configuration))
    assert folder.provenance()["traceable_to_the_commit"] is False
    assert state["invocations"][0]["traceable_to_the_commit"] is False
