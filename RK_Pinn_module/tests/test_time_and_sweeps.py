"""Gate: the report with the sweeps, and the time per event.

Proves, on a small store with a small path run and a small chain run:
  * the report of a path run with `with_one_sweep` writes `sweeps.json` and
    `sweeps.md`, with one row for the track alone and one after each of two
    sweeps; the field lookups and Jacobians per track are N q per sweep; after
    two sweeps the track is the exact scheme's to rounding, whatever the
    network gave; the path run's report has no single-step error;
  * the report without the option writes no sweeps and is otherwise made;
  * the time per event gives one row per method, with the chain and the path
    network when their runs are given and without them when not, positive
    times, the lookups each method costs, and finite errors.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from rkpinn.evaluation.time_per_event import time_per_event
from rkpinn.run_record.report_of_a_run import report_of_a_run
from rkpinn.track_data.exact_states import load_exact_states, write_exact_states
from rkpinn.training import round_trainer
from small_run import small_configuration, small_store
from test_reference_path import path_configuration

QUIET = dict(allow_uncommitted_changes=True, report=lambda text: None)
SETTINGS = dict(split="test", tracks_for_the_stage_errors=4,
                step_length_of_the_carrying_mm=2.0, fraction_for_the_derivative=1e-3,
                states_for_the_loss_shares=200, bins_of_the_histograms=20)
STEPS, STAGES = 4, 2


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    store, key = small_store(tmp_path_factory.mktemp("sweep_store"))
    for split in ("training", "validation"):
        write_exact_states(store, key, split, STEPS, STAGES, workers=4, keep_stages=True, **QUIET)
    write_exact_states(store, key, "test", STEPS, STAGES, workers=4, **QUIET)
    path = path_configuration(key)
    path["training"]["rounds_at_most"] = 1
    # the loss of P1; the Gaussian loss of P3 does not yet train stably (2026-11-04)
    path["loss"]["uncertainty"], path["network"]["extra_outputs"] = "none", 0
    chain = small_configuration(key, training_rounds_at_most=1)
    for configuration in (path, chain):
        round_trainer.train(configuration, store, if_the_run_exists="refuse", **QUIET)
    return (store, key, round_trainer.run_key(path), round_trainer.run_key(chain))


def test_the_report_with_the_sweeps(made):
    store, key, path_key, _ = made
    # an untrained path network's states are far from any track, and the exact
    # scheme does not converge from them, so the stage errors are left out here
    where = report_of_a_run(store, path_key, round_number="last", with_one_sweep=True,
                            **dict(SETTINGS, tracks_for_the_stage_errors=0), **QUIET)
    with open(os.path.join(where, "sweeps.json")) as handle:
        sweeps = json.load(handle)
    assert [s["sweeps"] for s in sweeps] == [0, 1, 2]
    assert [s["field_lookups_per_track"] for s in sweeps] == [0, STEPS * STAGES,
                                                              2 * STEPS * STAGES]
    assert [s["jacobians_per_track"] for s in sweeps] == [0, STEPS * STAGES, 2 * STEPS * STAGES]
    exact_scheme = [s["against_the_exact_scheme"]["largest_micrometres"] for s in sweeps]
    # 4 steps of 1.3 m from a network of one round: measured 0.18 um after two
    assert exact_scheme[0] > exact_scheme[1] > exact_scheme[2] and exact_scheme[2] < 1.0
    assert sweeps[0]["largest_residual"] > sweeps[2]["largest_residual"]
    assert os.path.isfile(os.path.join(where, "sweeps.md"))
    with open(os.path.join(where, "summary.json")) as handle:
        summary = json.load(handle)
    assert summary["sweeps"] == sweeps
    exact = load_exact_states(store, key, "test", STEPS, STAGES)[0][:, -1]
    assert np.isfinite(exact).all()


def test_the_report_without_the_sweeps(made):
    store, _, _, chain_key = made
    where = report_of_a_run(store, chain_key, round_number="last", **SETTINGS, **QUIET)
    assert not os.path.exists(os.path.join(where, "sweeps.json"))
    with open(os.path.join(where, "summary.json")) as handle:
        assert json.load(handle)["sweeps"] is None


def test_the_time_per_event(made):
    store, key, path_key, chain_key = made
    common = dict(tracks_key=key, split="test", number_of_steps=STEPS,
                  number_of_stages=STAGES, round_number="last", tracks_per_event=10,
                  events=2, report=lambda text: None)
    rows = {r["method"]: r for r in time_per_event(store, chain_run=chain_key,
                                                   path_run=path_key, **common)}
    assert list(rows) == ["chain network", "path network", "path network + 1 sweep",
                          "Euler + 1 sweep", "Euler + 2 sweeps", "RK4"]
    for row in rows.values():
        assert row["milliseconds_per_event"] > 0 and row["threads"] == 1
        assert abs(row["milliseconds_per_track"] * 10 - row["milliseconds_per_event"]) < 1e-9
    N, q = STEPS, STAGES
    assert rows["chain network"]["field_lookups_per_track"] == N * q   # summed, direct states
    assert rows["path network"]["field_lookups_per_track"] == 16
    assert rows["path network + 1 sweep"]["field_lookups_per_track"] == 16 + N * q
    assert rows["Euler + 2 sweeps"]["field_lookups_per_track"] == N + 2 * N * q
    assert rows["RK4"]["field_lookups_per_track"] == 4 * N
    # at 4 steps of 1.3 m Newton is far from converged, so no order is asserted
    assert all(np.isfinite(r["radial_median_micrometres"]) for r in rows.values())
    alone = time_per_event(store, chain_run=None, path_run=None, **common)
    assert [r["method"] for r in alone] == ["Euler + 1 sweep", "Euler + 2 sweeps", "RK4"]
    with pytest.raises(ValueError):
        time_per_event(store, chain_run=None, path_run=None,
                       **dict(common, tracks_per_event=30))
