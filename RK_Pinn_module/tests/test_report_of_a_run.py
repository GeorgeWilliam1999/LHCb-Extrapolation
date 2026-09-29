"""Gate: the standard report of a run, from one command.

Proves
  * the report of a run with stages holds all ten outputs but the comparison
    with the exact scheme, whose states the small store does not hold, and
    says so;
  * the report of the whole-crossing network leaves out the outputs that need
    stages or a loss without labels, and says which;
  * the endpoint errors in the report are those of the snapshot's own weights,
    computed again, and the validation error of the round is found again from
    the weights;
  * the stage errors of the report pass their check;
  * the endpoint errors are added to the metrics of the manifest, once;
  * a report is never written over another, and a round that does not exist
    is refused;
  * no absolute path is written into a report;
  * the figures are written.

The runs are the small runs of the gates of the training. The numbers of
their reports say nothing about what a network can reach.
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np
import pytest

from rkpinn.evaluation import conventions
from rkpinn.run_record import manifest
from rkpinn.run_record.configuration import run_key
from rkpinn.run_record.report_of_a_run import report_of_a_run
from rkpinn.run_record.store import AlreadyInTheStore
from rkpinn.training import round_trainer
from rkpinn.training.checkpoints import RunFolder
from small_run import small_configuration, small_store, whole_crossing_configuration

QUIET = dict(allow_uncommitted_changes=True, report=lambda text: None)
SETTINGS = dict(split="test", tracks_for_the_stage_errors=4,
                step_length_of_the_carrying_mm=2.0, fraction_for_the_derivative=1e-3,
                states_for_the_loss_shares=200, bins_of_the_histograms=20)

ALL_TABLES = {
    "endpoint_error_per_component", "endpoint_error_magnitudes", "signed_distributions",
    "single_step_error_per_component", "single_step_error_per_step",
    "error_along_the_track", "stage_errors_held_and_carried", "three_references",
    "convergence", "validation_per_round", "where_the_loss_puts_its_weight", "cost"}


def table(folder, name):
    with open(os.path.join(folder, name + ".csv")) as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    store, tracks_key = small_store(tmp_path_factory.mktemp("store_of_reports"))
    # scaled outputs keep the untrained small network inside the field map
    configuration = small_configuration(tracks_key, training_rounds_at_most=2)
    round_trainer.train(configuration, store, if_the_run_exists="refuse", **QUIET)
    key = run_key(configuration)
    where = report_of_a_run(store, key, round_number="last", **SETTINGS, **QUIET)
    return store, key, where


def test_the_report_holds_the_ten_outputs(made):
    store, key, where = made
    assert os.path.basename(os.path.dirname(where)) == "reports"
    assert os.path.basename(where).startswith("round_0002_at_")
    files = set(os.listdir(where))
    assert {name + ".csv" for name in ALL_TABLES} <= files
    with open(os.path.join(where, "summary.json")) as handle:
        summary = json.load(handle)
    assert summary["outputs_left_out"] == []
    assert summary["exact_states_were_in_the_store"] is False
    assert summary["run_key"] == key and summary["round"] == 2
    assert summary["name"] == "4 steps of 1294.5 mm, 2 stages"
    assert summary["settings_of_the_report"]["tracks_for_the_stage_errors"] == 4
    comparisons = {row["comparison"] for row in table(where, "three_references")}
    assert comparisons == {"prediction against reference", "prediction against true state",
                           "reference against true state"}
    assert len(table(where, "error_along_the_track")) == 5
    assert len(table(where, "single_step_error_per_step")) == 4
    assert len(table(where, "validation_per_round")) == 2
    assert len(table(where, "endpoint_error_per_component")) == 4 * 6     # no window
    shares = table(where, "where_the_loss_puts_its_weight")
    assert sum(float(r["share_of_the_loss_percent"]) for r in shares
               if r["kind"] == "momentum band") == pytest.approx(100.0)
    assert sum(float(r["share_of_the_loss_percent"]) for r in shares
               if r["kind"] == "quarter of the crossing") == pytest.approx(100.0)
    cost, = table(where, "cost")
    assert int(cost["applications_per_track"]) == 4 and int(cost["restarts"]) == 4
    assert float(cost["microseconds_to_carry_one_track"]) > 0


def test_the_report_is_of_the_weights_of_the_snapshot(made):
    store, key, where = made
    folder = RunFolder(store, key)
    run = round_trainer.build_run(
        round_trainer.read_configuration(folder.path("configuration.yaml")), store,
        folder.constants())
    run.network.load_state_dict(folder.weights_of_round(2))
    assert round_trainer.validation_of(run) == {
        name: folder.scores_of_round(2)[name] for name in round_trainer.validation_of(run)}
    test = run.tracks.test
    end = run.network.whole_track(test.start_state, run.layout).final_state()
    in_x = np.abs(end[:, 0] - test.reference_end_state[:, 0]) * 1e3
    row, = [r for r in table(where, "endpoint_error_per_component")
            if r["band"] == conventions.ALL_MOMENTA and r["component"] == "x"]
    assert float(row["median_of_absolute"]) == float(np.median(in_x))
    assert int(row["number_of_tracks"]) == 50


def test_the_stage_errors_of_the_report_pass_their_check(made):
    _, _, where = made
    with open(os.path.join(where, "summary.json")) as handle:
        check = json.load(handle)["check_of_the_stage_errors"]
    assert check["number_of_tracks"] == 4
    assert check["sum_against_final_minus_carried_start_largest_micrometres"] < 1e-4
    assert len(table(where, "stage_errors_held_and_carried")) == 4


def test_the_metrics_are_listed_once(made):
    store, key, where = made
    listed = [row for row in manifest.list_metrics(store) if row["run_key"] == key]
    assert len(listed) == 4 * 6 * len(conventions.STATISTICS)
    assert {row["kind_of_error"] for row in listed} == {"endpoint"}
    assert {row["round"] for row in listed} == {"2"}
    with pytest.raises(manifest.AlreadyListed):
        manifest.add_metrics(store, [dict(listed[0])])


def test_a_report_is_not_written_over_another(made):
    store, key, where = made
    before = os.path.getmtime(os.path.join(where, "summary.json"))
    with pytest.raises(AlreadyInTheStore):
        report_of_a_run(store, key, round_number=2, **SETTINGS, **QUIET)
    assert os.path.getmtime(os.path.join(where, "summary.json")) == before
    for wrong in (0, 3):
        with pytest.raises(ValueError, match="has the rounds"):
            report_of_a_run(store, key, round_number=wrong, **SETTINGS, **QUIET)


def test_no_absolute_path_and_the_figures(made):
    store, _, where = made
    for name in os.listdir(where):
        if name.endswith((".csv", ".json")):
            with open(os.path.join(where, name)) as handle:
                text = handle.read()
            assert store.location not in text and "/data/" not in text, name
    figures = {name for name in os.listdir(where) if name.endswith(".png")}
    assert figures == {"error_along_the_track.png", "endpoint_error_by_momentum_band.png",
                       "signed_distributions.png", "convergence.png",
                       "stage_errors_held_and_carried.png"}
    for name in figures:
        assert os.path.getsize(os.path.join(where, name)) > 5000


def test_the_report_of_the_whole_crossing_network_says_what_it_left_out(made):
    store, _, _ = made
    tracks_key = [row["key"] for row in manifest.list_track_sets(store)] or None
    configuration = whole_crossing_configuration(
        RunFolder(store, made[1]).provenance()["tracks_key"])
    configuration["training"]["rounds_at_most"] = 1
    round_trainer.train(configuration, store, if_the_run_exists="refuse", **QUIET)
    where = report_of_a_run(store, run_key(configuration), round_number="last", **SETTINGS,
                            **QUIET)
    with open(os.path.join(where, "summary.json")) as handle:
        summary = json.load(handle)
    assert [text.split(",")[0] for text in summary["outputs_left_out"]] == ["4", "6", "9"]
    assert summary["name"] == "1 step of 5177.8 mm, no stages"
    files = set(os.listdir(where))
    assert "single_step_error_per_step.csv" not in files
    assert "stage_errors_held_and_carried.csv" not in files
    assert {"three_references.csv", "convergence.csv", "cost.csv",
            "error_along_the_track.csv"} <= files
    assert len(table(where, "error_along_the_track")) == 2
    assert tracks_key is None or True
