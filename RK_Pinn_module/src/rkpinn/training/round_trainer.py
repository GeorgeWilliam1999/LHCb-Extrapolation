"""The round trainer: trains the network a configuration names, in rounds.

It names no loss, no network, no target and no protocol. It takes each from
the registry, by the name the configuration gives.

The life of a run (WORKFLOW.md, section 4):

    a configuration is validated and keyed
    the run is made in the store, or found there
    the lock is taken
    the checks before training are run
    rounds:  the protocol gives the states of the round
             the optimiser makes restarts on them, all of the round's
                 restarts or fewer if the round is set to end when stalled
             the validation tracks are carried across the crossing
             the round is written: snapshot, restarts, row, state
             the stopping rule is applied to the validation errors
    the run ends plateaued, at its cap, or interrupted
    the lock is released

What is measured once, when the run is made, and then read from the run:
the scales of the network and the constants of the loss. They are in
`constants.json`. A resumed run reads them; it does not measure them again.

What the caller must say, with no default: what to do if the run exists.

    refuse   make a new run; if its key is in the store, refuse
    resume   continue a run that was interrupted
    extend   continue a run that ended, up to the cap the configuration now gives

To train, from `RK_Pinn_module`:

    PYTHONNOUSERSITE=1 PYTHONPATH=src /data/bfys/gscriven/conda/envs/TE/bin/python \\
        -m rkpinn.training.round_trainer --store /data/bfys/gscriven/rkpinn_store \\
        --configuration <file.yaml> --if-the-run-exists refuse

The round protocol is ported from `E1_Network_grid/train_network.py`. What is
not ported: the rule that ended a frozen run on the loss, and its confirmation
pass. A run here ends on the validation error or at its cap.
Gates: tests/test_training.py.
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from rkpinn import registry
from rkpinn.equation_of_motion.field_map import hash_of_field_map_file
from rkpinn.integrators.gauss_legendre_tableau import gauss_legendre_tableau
from rkpinn.losses.pooled import spread_of_components
from rkpinn.predicted_track.predicted_track import steps_of_the_exact_scheme
from rkpinn.predicted_track.track_layout import TrackLayout
from rkpinn.run_record import manifest
from rkpinn.run_record.configuration import plain, read_configuration, run_key, validate
from rkpinn.run_record.provenance import provenance
from rkpinn.run_record.store import AlreadyInTheStore, NotInTheStore, Store
from rkpinn.track_data.load_tracks import load_tracks
from rkpinn.training import checkpoints
from rkpinn.training.checkpoints import RunFolder
from rkpinn.training.validation_error import validation_errors

IF_THE_RUN_EXISTS = ("refuse", "resume", "extend")
SPREAD_OF_FIRST_ROUND_STATES = "spread_of_first_round_states"
SPREAD_OF_TARGET_END_STATES = "spread_of_target_end_states"

NUMBER_OF_STATES_IN_THE_CHECK = 8
LARGEST_LOSS_AT_THE_EXACT_SOLUTION = 1e-18


class CheckBeforeTrainingFailed(RuntimeError):
    """A check that is run before training did not pass."""


@dataclass
class RunContext:
    """What the run has built, for the components that are built after it."""

    seed: int
    equation_of_motion: object              # the differentiable form
    layout: TrackLayout
    first_round_states: np.ndarray
    first_round_labels: Optional[np.ndarray]

    def scale(self, value, number):
        """`number` constants, from their name or given outright."""
        if isinstance(value, str):
            if value == SPREAD_OF_FIRST_ROUND_STATES:
                return spread_of_components(self.first_round_states)[:number].tolist()
            if value == SPREAD_OF_TARGET_END_STATES:
                if self.first_round_labels is None:
                    raise ValueError("there are no end states of a target in this run, "
                                     "so their spread cannot be a scale")
                return spread_of_components(self.first_round_labels)[:number].tolist()
            raise ValueError("a scale is %s, %s or numbers, not %r"
                             % (SPREAD_OF_FIRST_ROUND_STATES, SPREAD_OF_TARGET_END_STATES,
                                value))
        numbers = [float(v) for v in value]
        if len(numbers) != number:
            raise ValueError("this scale has %d components, not %d" % (number, len(numbers)))
        return numbers


@dataclass
class Run:
    """Everything a configuration resolves to."""

    configuration: dict
    key: str
    tracks: object
    layout: TrackLayout
    equation_of_motion: object              # numpy
    differentiable_equation: object         # torch
    protocol: object
    target: object
    network: object
    loss: object
    optimiser: object
    stopping_rule: object
    constants: dict                         # {"network": ..., "loss": ...}
    first_round: object


def _without(block, naming):
    return {name: value for name, value in block.items() if name != naming}


def build_run(configuration: dict, store: Store, recorded_constants=None) -> Run:
    """Build everything the configuration names. Writes nothing.

    recorded_constants  the constants of a run that exists; None measures them
    """
    validate(configuration)
    key = run_key(configuration)
    torch.set_num_threads(1)
    tracks = load_tracks(store, configuration["tracks"]["key"])
    field_map_name = configuration["equation_of_motion"]["field_map"]
    if tracks.description["settings"]["field_map"] != field_map_name:
        raise ValueError("the tracks were built with the field map %r and the "
                         "configuration names %r"
                         % (tracks.description["settings"]["field_map"], field_map_name))
    field_map = registry.component("field_map", field_map_name)()
    system = configuration["equation_of_motion"]["system"]
    equation = registry.component("equation_of_motion", system)(field_map)
    differentiable = registry.component(
        "equation_of_motion", system + "_differentiable")(field_map)

    steps = configuration["steps"]
    tableau = (gauss_legendre_tableau(steps["number_of_stages"])
               if steps["number_of_stages"] else None)
    layout = TrackLayout(tracks.first_plane_mm, tracks.last_plane_mm,
                         steps["number_of_steps"], tableau)
    tracks.planes_a_step_starts_on(steps["number_of_steps"])     # refuses steps off the planes

    training = configuration["training"]
    seed = configuration["seed"]
    protocol = registry.component("training_protocol", training["protocol"]["name"]) \
        .from_configuration(_without(training["protocol"], "name"), None)
    target = registry.component("target", configuration["target"]["name"])()
    first_round = protocol.states_of_round(1, seed, tracks, layout, None, target)
    context = RunContext(seed=seed, equation_of_motion=differentiable, layout=layout,
                         first_round_states=first_round.states,
                         first_round_labels=first_round.labels)

    network_block = _without(configuration["network"], "kind")
    if recorded_constants is None:
        network_block["scale_of_inputs"] = context.scale(network_block["scale_of_inputs"], 5)
        network_block["scale_of_outputs"] = context.scale(
            network_block["scale_of_outputs"], 4)
    else:
        network_block["scale_of_inputs"] = recorded_constants["network"]["scale_of_inputs"]
        network_block["scale_of_outputs"] = recorded_constants["network"]["scale_of_outputs"]
    network = registry.component("network", configuration["network"]["kind"]) \
        .from_configuration(network_block, context)

    loss = registry.component("loss", configuration["loss"]["name"]) \
        .from_configuration(_without(configuration["loss"], "name"), context)
    if recorded_constants is None:
        constants = {"network": plain(network.settings()),
                     "loss": plain(loss.constants(first_round.states))}
    else:
        constants = recorded_constants
    optimiser = registry.component("optimiser", training["optimiser"]["name"]) \
        .from_configuration(_without(training["optimiser"], "name"), context)
    stopping_rule = registry.component("stopping_rule", training["stopping_rule"]["name"]) \
        .from_configuration(_without(training["stopping_rule"], "name"), context)
    return Run(configuration=configuration, key=key, tracks=tracks, layout=layout,
               equation_of_motion=equation, differentiable_equation=differentiable,
               protocol=protocol, target=target, network=network, loss=loss,
               optimiser=optimiser,
               stopping_rule=stopping_rule, constants=constants, first_round=first_round)


def checks_before_training(run: Run) -> dict:
    """The checks of the chosen network, loss and tracks. A mistake then costs
    seconds, not farm hours. Returns what was measured."""
    measured = {}
    for name, parameter in run.network.named_parameters():
        if parameter.dtype != torch.float64:
            raise CheckBeforeTrainingFailed("the parameter %s is not in double precision" % name)
    recorded, used = run.constants["network"], plain(run.network.settings())
    if recorded != used:
        raise CheckBeforeTrainingFailed(
            "the settings of the network are not those recorded in the run")
    if not run.loss.needs_labels:
        pick = np.linspace(0, len(run.first_round.states) - 1,
                           NUMBER_OF_STATES_IN_THE_CHECK).astype(int)
        exact = steps_of_the_exact_scheme(
            run.equation_of_motion, run.first_round.states[pick],
            run.first_round.start_planes_mm[pick], run.layout.step_length_mm,
            run.layout.tableau)
        with torch.no_grad():
            value = float(run.loss.value(exact, None, run.constants["loss"]))
        measured["loss_at_the_exact_solution"] = value
        if not value < LARGEST_LOSS_AT_THE_EXACT_SOLUTION:
            raise CheckBeforeTrainingFailed(
                "the loss at the exact scheme's stages is %r, not zero to rounding" % value)
    with torch.no_grad():
        first = float(loss_of_a_round(run, run.first_round)())
    measured["loss_before_training_on_first_round_states"] = first
    if not np.isfinite(first):
        raise CheckBeforeTrainingFailed("the loss on the first round's states is %r" % first)
    return measured


def loss_of_a_round(run: Run, states_of_the_round):
    """The function that gives the loss of the run on the states of a round."""
    states = torch.as_tensor(states_of_the_round.states)
    start_planes = torch.as_tensor(states_of_the_round.start_planes_mm)
    labels = (None if states_of_the_round.labels is None
              else torch.as_tensor(states_of_the_round.labels))
    network, loss, constants = run.network, run.loss, run.constants["loss"]

    def loss_now():
        return loss.value(network.predict(states, start_planes), labels, constants)

    return loss_now


def validation_of(run: Run) -> dict:
    validation = run.tracks.validation
    track = run.network.whole_track(validation.start_state, run.layout)
    return validation_errors(np.asarray(track.final_state()), validation.reference_end_state)


def _open_run(run: Run, folder: RunFolder, store: Store, if_the_run_exists: str,
              record: dict):
    """Make the run or find it, as the caller asked. Returns its state."""
    cap = run.configuration["training"]["rounds_at_most"]
    if not folder.exists():
        if if_the_run_exists != "refuse":
            raise NotInTheStore(
                "there is no run %s to %s" % (run.key, if_the_run_exists))
        tracks_record = run.tracks.description
        run_record = dict(record, run_key=run.key, tracks_key=run.tracks.key,
                          hash_of_field_map_file=hash_of_field_map_file(
                              run.configuration["equation_of_motion"]["field_map"]),
                          hash_of_field_map_file_of_the_tracks=tracks_record[
                              "field_map"]["hash_of_file"])
        folder.create(run.configuration, run_record, run.constants, cap)
        manifest.add_run(store, {
            "key": run.key, "created": record["created"], "tracks_key": run.tracks.key,
            "network": run.configuration["network"]["kind"],
            "loss": run.configuration["loss"]["name"],
            "target": run.configuration["target"]["name"],
            "number_of_steps": run.configuration["steps"]["number_of_steps"],
            "number_of_stages": run.configuration["steps"]["number_of_stages"],
            "seed": run.configuration["seed"],
            "package_version": record["package_version"], "commit": record["commit"],
            "traceable_to_the_commit": record["traceable_to_the_commit"],
            "machine": record["machine"]})
        return folder.state()
    if if_the_run_exists == "refuse":
        raise AlreadyInTheStore(
            "the run %s already exists. Resume it, extend it, or change the "
            "configuration" % run.key)
    state = folder.state()
    ended = state["state"] in (checkpoints.PLATEAUED, checkpoints.AT_ITS_CAP)
    if if_the_run_exists == "resume":
        if ended:
            raise ValueError("the run %s has ended, %s; it can be extended, not resumed"
                             % (run.key, state["state"]))
        if cap != state["rounds_at_most"]:
            raise ValueError("the run %s has a cap of %d rounds and the configuration "
                             "gives %d; a cap is changed by extending the run"
                             % (run.key, state["rounds_at_most"], cap))
    else:
        if not ended:
            raise ValueError("the run %s has not ended; it can be resumed, not extended"
                             % run.key)
        if cap <= state["rounds_done"]:
            raise ValueError("the run %s has %d rounds; to extend it the cap must be "
                             "higher, and the configuration gives %d"
                             % (run.key, state["rounds_done"], cap))
        state["extensions"].append({
            "from_rounds_at_most": state["rounds_at_most"], "to_rounds_at_most": cap,
            "rounds_done": state["rounds_done"], "state_before": state["state"],
            "when": record["created"], "commit": record["commit"]})
        state["rounds_at_most"] = cap
    return state


def train(configuration: dict, store: Store, *, if_the_run_exists: str,
          allow_uncommitted_changes: bool = False, rounds_in_this_call=None,
          report=print) -> dict:
    """Train the run the configuration names. Returns the state of the run.

    rounds_in_this_call  stop after this many rounds in this call, leaving the
                         run interrupted; None trains until the run ends
    """
    if if_the_run_exists not in IF_THE_RUN_EXISTS:
        raise ValueError("if the run exists: %s, not %r"
                         % (", ".join(IF_THE_RUN_EXISTS), if_the_run_exists))
    validate(configuration)
    record = provenance(allow_uncommitted_changes)          # refuses before any work
    key = run_key(configuration)
    folder = RunFolder(store, key)
    recorded = folder.constants() if folder.exists() else None
    run = build_run(configuration, store, recorded)
    state = _open_run(run, folder, store, if_the_run_exists, record)

    folder.take_lock()
    try:
        state["state"] = checkpoints.TRAINING
        state["invocations"].append({
            "started": record["created"], "commit": record["commit"],
            "package_version": record["package_version"],
            "traceable_to_the_commit": record["traceable_to_the_commit"],
            "machine": record["machine"], "asked": if_the_run_exists,
            "from_round": state["rounds_done"]})
        folder.write_state(state)
        if state["rounds_done"]:
            run.network.load_state_dict(folder.weights_of_round(state["rounds_done"]))
        measured = checks_before_training(run)
        report("run %s: checks before training passed %r" % (key, measured))

        training = run.configuration["training"]
        stall = training["end_a_round_when_stalled"]
        quantity = run.stopping_rule.quantity
        rounds_here = 0
        while True:
            state = folder.state()
            if state["rounds_done"] >= state["rounds_at_most"]:
                state["state"] = checkpoints.AT_ITS_CAP
                break
            if rounds_in_this_call is not None and rounds_here >= rounds_in_this_call:
                state["state"] = checkpoints.INTERRUPTED
                break
            number = state["rounds_done"] + 1
            states_of_the_round = (run.first_round if number == 1 else
                                   run.protocol.states_of_round(
                                       number, run.configuration["seed"], run.tracks,
                                       run.layout, run.network, run.target))
            loss_now = loss_of_a_round(run, states_of_the_round)
            run.optimiser.start_round(run.network.parameters())
            rows, stalled = [], 0
            for in_round in range(1, training["restarts_per_round"] + 1):
                row = run.optimiser.one_restart(loss_now)
                row.update(restart=state["restarts_done"] + len(rows) + 1, round=number,
                           restart_in_round=in_round)
                rows.append(row)
                report("  run %s round %d restart %d: loss %.4e -> %.4e (gain %+.2e), "
                       "%d iterations, %.0f s"
                       % (key, number, in_round, row["loss_before"], row["loss_after"],
                          row["gain"], row["iterations"], row["seconds"]))
                if stall is not None:
                    stalled = 0 if row["gain"] >= stall["gain_below"] else stalled + 1
                    if stalled >= stall["restarts_running"]:
                        break
            started = time.time()
            errors = validation_of(run)
            seconds_of_validation = time.time() - started
            row_of_round = dict(
                round=number, source_of_states=states_of_the_round.source,
                number_of_states=int(len(states_of_the_round.states)), restarts=len(rows),
                loss_first=rows[0]["loss_before"], loss_last=rows[-1]["loss_after"],
                seconds_of_training=round(sum(r["seconds"] for r in rows), 1),
                seconds_of_validation=round(seconds_of_validation, 1), **errors)
            scores = dict(row_of_round, run_key=key, number_of_validation_tracks=int(
                run.tracks.validation.number_of_tracks))
            folder.write_round(number, run.network.state_dict(), scores, rows, row_of_round)
            rounds_here += 1
            report("run %s ROUND %d done (%s): %d restarts, loss %.4e -> %.4e, "
                   "validation %.1f um"
                   % (key, number, states_of_the_round.source, len(rows),
                      row_of_round["loss_first"], row_of_round["loss_last"],
                      errors[quantity]))
            history = [float(r[quantity]) for r in folder.rounds()]
            if run.stopping_rule.holds(history):
                state = folder.state()
                state["state"] = checkpoints.PLATEAUED
                state["plateaued_at_round"] = number
                break
        folder.write_state(state)
        report("run %s: %s after %d rounds" % (key, state["state"], state["rounds_done"]))
        return state
    except BaseException:
        if folder.exists():
            state = folder.state()
            if state["state"] == checkpoints.TRAINING:
                state["state"] = checkpoints.INTERRUPTED
                folder.write_state(state)
        raise
    finally:
        folder.release_lock()


def main(arguments=None):
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    parser = argparse.ArgumentParser(description="Train the run a configuration names.")
    parser.add_argument("--store", required=True)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--if-the-run-exists", required=True, choices=IF_THE_RUN_EXISTS)
    parser.add_argument("--rounds-in-this-call", type=int, default=None,
                        help="stop after this many rounds, leaving the run interrupted")
    parser.add_argument("--allow-uncommitted-changes", action="store_true",
                        help="train although the package has uncommitted changes; the "
                             "run is then recorded as not traceable")
    a = parser.parse_args(arguments)
    return train(read_configuration(a.configuration), Store(a.store),
                 if_the_run_exists=a.if_the_run_exists,
                 allow_uncommitted_changes=a.allow_uncommitted_changes,
                 rounds_in_this_call=a.rounds_in_this_call,
                 report=lambda text: print(text, flush=True))


if __name__ == "__main__":
    main()
