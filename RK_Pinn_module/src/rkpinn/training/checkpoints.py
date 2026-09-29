"""The folder of a run: its snapshots, its records and its lock.

    <store>/runs/<key>/
    ├── configuration.yaml      the configuration, as the run was made from it
    ├── provenance.json         commit, package version, tracks key, field map hash
    ├── constants.json          the scales of the network and the constants of
    │                           the loss, measured once, on the first round's states
    ├── state.json              where the run is
    ├── restarts.csv            one row per restart
    ├── rounds.csv              one row per round
    ├── snapshots/round_0001/   network.pt and scores.json; never overwritten
    └── LOCK                    present while a job writes to the run

What is on disk always describes whole rounds. A round is written when it
ends: its snapshot, then its restarts, then its row, then the state. A job
that is interrupted inside a round leaves nothing of that round, and the run
resumes from the last snapshot. Every round starts with a new optimiser and
draws its states from the seed and the number of the round, so a resumed run
repeats the interrupted round exactly.

One job writes to a run at a time. The lock is a file made in one step that
fails if the file is there. A lock left by a job that died is cleared by
hand, with `clear_lock`, after checking that no job holds it.

No absolute path is written into a run.
"""
from __future__ import annotations

import csv
import json
import os
import platform
import time

import torch

from rkpinn.run_record.configuration import write_configuration
from rkpinn.run_record.store import AlreadyInTheStore, NotInTheStore, Store

TRAINING = "training"
INTERRUPTED = "interrupted"
PLATEAUED = "plateaued"
AT_ITS_CAP = "at its cap"
STATES = (TRAINING, INTERRUPTED, PLATEAUED, AT_ITS_CAP)

COLUMNS_OF_RESTARTS = (
    "restart", "round", "restart_in_round", "loss_before", "loss_after", "gain",
    "factor", "factor_renewed", "iterations", "evaluations", "stopped_early", "seconds")

COLUMNS_OF_ROUNDS = (
    "round", "source_of_states", "number_of_states", "restarts", "loss_first",
    "loss_last", "validation_endpoint_error_in_x_median_micrometres",
    "validation_endpoint_error_in_y_median_micrometres",
    "validation_endpoint_error_larger_of_x_and_y_median_micrometres",
    "seconds_of_training", "seconds_of_validation")


class RunIsLocked(RuntimeError):
    """Another job is writing to the run, or left its lock behind."""


def _append(path, columns, rows):
    is_new = not os.path.exists(path)
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if is_new:
            writer.writeheader()
        for row in rows:
            if set(row) != set(columns):
                raise ValueError("a row of %s must have exactly its columns"
                                 % os.path.basename(path))
            writer.writerow(row)


def _read(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path, content):
    temporary = path + ".writing"
    with open(temporary, "w") as handle:
        json.dump(content, handle, indent=1)
    os.replace(temporary, path)


def _read_json(path):
    with open(path) as handle:
        return json.load(handle)


class RunFolder:
    def __init__(self, store: Store, key: str):
        self.store, self.key = store, key
        self.folder = os.path.join(store.folder_of_runs, key)

    def path(self, *parts) -> str:
        return os.path.join(self.folder, *parts)

    def exists(self) -> bool:
        return os.path.isfile(self.path("state.json"))

    # -- making the run ------------------------------------------------------------
    def create(self, configuration, provenance, constants, rounds_at_most) -> None:
        Store.refuse_if_present(self.folder)
        os.makedirs(self.path("snapshots"))
        write_configuration(configuration, self.path("configuration.yaml"))
        _write_json(self.path("provenance.json"), provenance)
        _write_json(self.path("constants.json"), constants)
        _write_json(self.path("state.json"), {
            "state": TRAINING, "rounds_done": 0, "restarts_done": 0,
            "rounds_at_most": int(rounds_at_most), "extensions": [], "invocations": []})

    # -- the lock -------------------------------------------------------------------
    def take_lock(self) -> None:
        try:
            handle = os.open(self.path("LOCK"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            held = ""
            try:
                with open(self.path("LOCK")) as lock:
                    held = lock.read().strip()
            except OSError:
                pass
            raise RunIsLocked(
                "the run %s is locked: %s. Wait, or clear the lock after checking "
                "that no job holds it" % (self.key, held or "by an unknown job")) from None
        with os.fdopen(handle, "w") as lock:
            json.dump({"machine": platform.node(), "process": os.getpid(),
                       "taken": time.strftime("%Y-%m-%d %H:%M:%S")}, lock)

    def release_lock(self) -> None:
        try:
            os.remove(self.path("LOCK"))
        except FileNotFoundError:
            pass

    def is_locked(self) -> bool:
        return os.path.exists(self.path("LOCK"))

    def clear_lock(self) -> None:
        """Remove a lock left behind. Only after checking that no job holds it."""
        self.release_lock()

    # -- reading ---------------------------------------------------------------------
    def state(self) -> dict:
        if not self.exists():
            raise NotInTheStore("the store holds no run with the key %r" % self.key)
        return _read_json(self.path("state.json"))

    def constants(self) -> dict:
        return _read_json(self.path("constants.json"))

    def provenance(self) -> dict:
        return _read_json(self.path("provenance.json"))

    def restarts(self) -> list:
        return _read(self.path("restarts.csv"))

    def rounds(self) -> list:
        return _read(self.path("rounds.csv"))

    def folder_of_snapshot(self, round_number: int) -> str:
        return self.path("snapshots", "round_%04d" % round_number)

    def weights_of_round(self, round_number: int) -> dict:
        return torch.load(os.path.join(self.folder_of_snapshot(round_number), "network.pt"),
                          weights_only=True)

    def scores_of_round(self, round_number: int) -> dict:
        return _read_json(os.path.join(self.folder_of_snapshot(round_number), "scores.json"))

    # -- writing ---------------------------------------------------------------------
    def write_state(self, state: dict) -> None:
        if state["state"] not in STATES:
            raise ValueError("a run is %s, not %r" % (", ".join(STATES), state["state"]))
        _write_json(self.path("state.json"), state)

    def write_round(self, round_number, weights, scores, rows_of_restarts,
                    row_of_round) -> None:
        """Write a round that has ended. Its snapshot is never overwritten."""
        state = self.state()
        if round_number != state["rounds_done"] + 1:
            raise ValueError("round %d cannot follow round %d"
                             % (round_number, state["rounds_done"]))
        final = self.folder_of_snapshot(round_number)
        if os.path.exists(final):
            raise AlreadyInTheStore(
                "the snapshot of round %d of the run %s is already there, although "
                "the state of the run says %d rounds are done. A job died while "
                "writing the round; look at the run before anything is changed"
                % (round_number, self.key, state["rounds_done"]))
        writing = final + ".writing"
        os.makedirs(writing, exist_ok=True)
        torch.save(weights, os.path.join(writing, "network.pt"))
        _write_json(os.path.join(writing, "scores.json"), scores)
        os.replace(writing, final)
        _append(self.path("restarts.csv"), COLUMNS_OF_RESTARTS, rows_of_restarts)
        _append(self.path("rounds.csv"), COLUMNS_OF_ROUNDS, [row_of_round])
        state["rounds_done"] = round_number
        state["restarts_done"] += len(rows_of_restarts)
        self.write_state(state)
