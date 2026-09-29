"""The standard report of a run: one command, the same tables and figures
for any run.

It builds the run from the configuration kept in the run, loads the weights
of a snapshot, makes the predictions, and hands states and targets to the
standard report of the evaluation, which knows of no network.

What it does, in order:

  1  carries the tracks of the split across the crossing with the network
  2  applies the network once from the reference state on every start plane
     (the single-step error; only a network with stages)
  3  holds and carries the stage errors of the first tracks of the split
     (only a network with stages)
  4  carries the predicted end states to each particle's own plane of the
     fibre tracker, and reads the exact scheme's states from the store if
     they are there
  5  reads the rounds of the run
  6  takes the loss of every state of a draw from the first round's states
     (only a loss without labels)
  7  measures the time to carry a track

It writes to a new folder inside the run,

    <store>/runs/<key>/reports/round_<round>_at_<commit>/

and adds the endpoint errors to `manifest/metrics.csv`. A report is never
written over another: the folder is named by the round and by the commit the
report was made at.

Every number that shapes the report is an argument with no default.

    PYTHONNOUSERSITE=1 PYTHONPATH=src /data/bfys/gscriven/conda/envs/TE/bin/python \\
        -m rkpinn.run_record.report_of_a_run --store /data/bfys/gscriven/rkpinn_store \\
        --run <key> --round last --split test --tracks-for-the-stage-errors 200 \\
        --step-length-of-the-carrying-mm 1.0 --fraction-for-the-derivative 1e-3 \\
        --states-for-the-loss-shares 8000 --bins-of-the-histograms 60

Gates: tests/test_report_of_a_run.py.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from rkpinn.evaluation import conventions, endpoint_error
from rkpinn.evaluation.error_along_the_track import stage_errors_held_and_carried
from rkpinn.evaluation.standard_report import standard_report, write_figures, write_report
from rkpinn.evaluation.three_references import carry_to_own_planes
from rkpinn.integrators.runge_kutta_sixth_order import REFERENCE_STEP_LENGTH_MM
from rkpinn.losses.cost_weighted import momentum_gev as momentum_from_charge_over_momentum
from rkpinn.predicted_track.predicted_track import PredictedTrack
from rkpinn.run_record import manifest
from rkpinn.run_record.configuration import read_configuration
from rkpinn.run_record.provenance import provenance
from rkpinn.run_record.store import Store
from rkpinn.track_data.exact_states import load_exact_states
from rkpinn.training.checkpoints import RunFolder
from rkpinn.training.round_trainer import build_run

ROWS_AT_A_TIME = 8192


def _separate_steps(run, reference_on_planes):
    """One application from the reference state on every start plane.
    reference_on_planes: shape (n, S + 1, 5)."""
    n, planes = reference_on_planes.shape[:2]
    steps = planes - 1
    start_planes = run.layout.start_planes_mm
    states = reference_on_planes[:, :steps].reshape(-1, 5)
    after = reference_on_planes[:, 1:].reshape(-1, 5)
    step = np.repeat(np.arange(steps)[None, :], n, axis=0).reshape(-1)
    ends = np.empty((len(states), 5))
    with torch.no_grad():
        for i in range(0, len(states), ROWS_AT_A_TIME):
            track = run.network.predict(
                torch.as_tensor(states[i:i + ROWS_AT_A_TIME]),
                torch.as_tensor(np.ascontiguousarray(start_planes[step[i:i + ROWS_AT_A_TIME]])))
            ends[i:i + ROWS_AT_A_TIME] = track.end_state(0).numpy()
    return dict(end_states=ends, reference_after_the_step=after, step_of_the_track=step,
                start_planes_mm=start_planes)


def _first_tracks(track: PredictedTrack, number: int) -> PredictedTrack:
    return PredictedTrack(
        step_length_mm=track.step_length_mm, tableau=track.tableau,
        start_planes_mm=track.start_planes_mm[:number],
        input_states=track.input_states[:number], stage_states=track.stage_states[:number],
        end_states=track.end_states[:number], end_state_was=track.end_state_was,
        layout=track.layout, produced_by=track.produced_by)


def _loss_shares(run, number_of_states):
    states_of_round = run.first_round
    pick = np.linspace(0, len(states_of_round.states) - 1,
                       min(number_of_states, len(states_of_round.states))).astype(int)
    states = states_of_round.states[pick]
    planes = states_of_round.start_planes_mm[pick]
    values = np.empty(len(states))
    with torch.no_grad():
        for i in range(0, len(states), ROWS_AT_A_TIME):
            track = run.network.predict(torch.as_tensor(states[i:i + ROWS_AT_A_TIME]),
                                        torch.as_tensor(planes[i:i + ROWS_AT_A_TIME]))
            terms = run.loss.squared_terms(track, None, run.constants["loss"])
            values[i:i + ROWS_AT_A_TIME] = terms.mean(dim=(1, 2)).numpy()
    step = np.rint((planes - run.layout.first_plane_mm)
                   / run.layout.step_length_mm).astype(int)
    return dict(loss_of_each_state=values,
                momentum_gev=momentum_from_charge_over_momentum(states[:, 4]),
                step_of_the_track=step, number_of_steps=run.layout.number_of_steps)


def report_of_a_run(store: Store, key: str, *, round_number, split,
                    tracks_for_the_stage_errors, step_length_of_the_carrying_mm,
                    fraction_for_the_derivative, states_for_the_loss_shares,
                    bins_of_the_histograms, allow_uncommitted_changes=False,
                    report=print) -> str:
    """Make the standard report of one snapshot of a run. Returns its folder.

    round_number  the round of the snapshot, or "last"
    """
    record = provenance(allow_uncommitted_changes)
    folder = RunFolder(store, key)
    state = folder.state()
    if round_number == "last":
        round_number = state["rounds_done"]
    round_number = int(round_number)
    if not 1 <= round_number <= state["rounds_done"]:
        raise ValueError("the run %s has the rounds 1 to %d, not %r"
                         % (key, state["rounds_done"], round_number))
    where = folder.path("reports", "round_%04d_at_%s" % (round_number, record["commit"][:8]))
    Store.refuse_if_present(where)

    configuration = read_configuration(folder.path("configuration.yaml"))
    run = build_run(configuration, store, folder.constants())
    run.network.load_state_dict(folder.weights_of_round(round_number))
    tracks = run.tracks.split(split)
    layout = run.layout
    steps, stages = layout.number_of_steps, layout.number_of_stages
    window = run.constants["loss"].get("momentum_window_gev")
    name = conventions.name_of_network(steps, stages, layout.step_length_mm)

    started = time.time()
    whole = run.network.whole_track(tracks.start_state, layout)
    seconds_of_carrying = time.time() - started
    states_on_planes = whole.states_on_planes()
    on = np.append(run.tracks.planes_a_step_starts_on(steps),
                   run.tracks.number_of_steps_between_planes)
    reference_on_planes = tracks.reference_states_on_planes[:, on]
    report("%s: carried %d tracks of the %s split" % (key, len(states_on_planes), split))

    single_steps = stage_errors = loss_shares = None
    if stages:
        single_steps = _separate_steps(run, reference_on_planes)
        if tracks_for_the_stage_errors:
            stage_errors = stage_errors_held_and_carried(
                _first_tracks(whole, tracks_for_the_stage_errors), run.equation_of_motion,
                tracks.reference_end_state[:tracks_for_the_stage_errors],
                step_length_of_the_carrying_mm=step_length_of_the_carrying_mm,
                fraction_for_the_derivative=fraction_for_the_derivative)
            report("%s: stage errors of %d tracks held and carried; check %r"
                   % (key, tracks_for_the_stage_errors, stage_errors["check"]))
    if not run.loss.needs_labels and states_for_the_loss_shares:
        loss_shares = _loss_shares(run, states_for_the_loss_shares)

    exact_end_states = None
    if stages and os.path.isfile(store.file_of_exact_states(run.tracks.key, split, steps,
                                                            stages)):
        exact_end_states = load_exact_states(
            store, run.tracks.key, split, steps, stages)[0][:, -1]
    references = dict(
        predicted_on_own_planes=carry_to_own_planes(
            run.equation_of_motion, states_on_planes[:, -1], layout.last_plane_mm,
            tracks.own_fibre_plane_mm,
            step_length_of_the_integrator_mm=REFERENCE_STEP_LENGTH_MM),
        reference_on_own_planes=tracks.reference_state_on_own_fibre_plane,
        true_states_on_own_planes=tracks.true_state_on_own_fibre_plane,
        exact_end_states=exact_end_states)

    rule = configuration["training"]["stopping_rule"]
    rounds = dict(rows=folder.rounds()[:round_number], quantity=rule["quantity"],
                  window_in_rounds=rule["window_in_rounds"], tolerance=rule["tolerance"],
                  rounds_held=rule["rounds_held"])
    restarts = [r for r in folder.restarts() if int(r["round"]) <= round_number]
    cost = {
        "microseconds_to_carry_one_track": 1e6 * seconds_of_carrying / len(states_on_planes),
        "tracks_carried": int(len(states_on_planes)),
        "applications_per_track": int(steps),
        "parameters_of_the_network": int(sum(p.numel() for p in run.network.parameters())),
        "rounds": int(round_number), "restarts": int(len(restarts)),
        "hours_of_training": sum(float(r["seconds"]) for r in restarts) / 3600.0,
        "machine": record["machine"], "threads": int(torch.get_num_threads()),
    }

    made = standard_report(
        name=name, momentum_gev=tracks.momentum_gev, momentum_window_gev=window,
        states_on_planes=states_on_planes, reference_states_on_planes=reference_on_planes,
        planes_mm=layout.planes_mm, single_steps=single_steps, stage_errors=stage_errors,
        references=references, rounds=rounds, loss_shares=loss_shares, cost=cost)
    made["summary"].update(
        run_key=key, round=round_number, split=split, tracks_key=run.tracks.key,
        network=configuration["network"]["kind"], loss=configuration["loss"]["name"],
        exact_states_were_in_the_store=exact_end_states is not None,
        settings_of_the_report=dict(
            tracks_for_the_stage_errors=tracks_for_the_stage_errors,
            step_length_of_the_carrying_mm=step_length_of_the_carrying_mm,
            fraction_for_the_derivative=fraction_for_the_derivative,
            states_for_the_loss_shares=states_for_the_loss_shares,
            bins_of_the_histograms=bins_of_the_histograms),
        provenance=record)

    os.makedirs(folder.path("reports"), exist_ok=True)
    write_report(made, where)
    write_figures(made, where, histograms=endpoint_error.histograms_of_signed_error(
        states_on_planes[:, -1], reference_on_planes[:, -1], bins_of_the_histograms))
    rows = []
    for row in made["tables"]["endpoint_error_per_component"]:
        for statistic in conventions.STATISTICS:
            rows.append(dict(
                run_key=key, round=round_number, split=split, momentum_band=row["band"],
                component=row["component"], unit=row["unit"], statistic=statistic,
                value=row[statistic], number_of_tracks=row["number_of_tracks"],
                kind_of_error="endpoint", created=record["created"],
                package_version=record["package_version"], commit=record["commit"],
                traceable_to_the_commit=record["traceable_to_the_commit"]))
    manifest.add_metrics(store, rows)
    report("%s: report written to reports/%s" % (key, os.path.basename(where)))
    return where


def main(arguments=None):
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    parser = argparse.ArgumentParser(description="Make the standard report of a run.")
    parser.add_argument("--store", required=True)
    parser.add_argument("--run", required=True, help="the key of the run")
    parser.add_argument("--round", required=True, help="the round of the snapshot, or last")
    parser.add_argument("--split", required=True)
    parser.add_argument("--tracks-for-the-stage-errors", type=int, required=True)
    parser.add_argument("--step-length-of-the-carrying-mm", type=float, required=True)
    parser.add_argument("--fraction-for-the-derivative", type=float, required=True)
    parser.add_argument("--states-for-the-loss-shares", type=int, required=True)
    parser.add_argument("--bins-of-the-histograms", type=int, required=True)
    parser.add_argument("--allow-uncommitted-changes", action="store_true")
    a = parser.parse_args(arguments)
    return report_of_a_run(
        Store(a.store), a.run, round_number=a.round, split=a.split,
        tracks_for_the_stage_errors=a.tracks_for_the_stage_errors,
        step_length_of_the_carrying_mm=a.step_length_of_the_carrying_mm,
        fraction_for_the_derivative=a.fraction_for_the_derivative,
        states_for_the_loss_shares=a.states_for_the_loss_shares,
        bins_of_the_histograms=a.bins_of_the_histograms,
        allow_uncommitted_changes=a.allow_uncommitted_changes,
        report=lambda text: print(text, flush=True))


if __name__ == "__main__":
    main()
