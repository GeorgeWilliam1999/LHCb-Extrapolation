"""Gate: a run survives what the farm does to a job.

The farm stops a job at its time limit with a signal and starts it again. A
job can also be killed outright, and then it cannot clean up.

Proves
  * a job that is stopped by the signal of the farm releases its lock and
    leaves the run interrupted, with the rounds it had finished;
  * the run is then resumed, and ends identical, to the last bit, to a run
    that was never stopped;
  * a lock that is being touched is never cleared: a second writer is refused
    whatever it says about locks left behind;
  * a lock that was not touched for longer than the caller says is cleared,
    the run resumes, and its state records the clearing;
  * without that setting a lock left behind is not cleared;
  * the key of a configuration is printed by the command the farm files use.

The command is run as the farm runs it: as a process of its own.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

import pytest

from rkpinn.run_record.configuration import run_key, write_configuration
from rkpinn.training import checkpoints, round_trainer
from rkpinn.training.checkpoints import RunFolder, RunIsLocked
from frozen_code import identical
from small_run import small_configuration, small_store

MODULE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUIET = dict(allow_uncommitted_changes=True, report=lambda text: None)


def environment():
    env = dict(os.environ, PYTHONNOUSERSITE="1", OMP_NUM_THREADS="1",
               PYTHONPATH=os.path.join(MODULE, "src"))
    return env


def command(store, file, asked, *more):
    return [sys.executable, "-m", "rkpinn.training.round_trainer", "--store",
            store.location, "--configuration", file, "--if-the-run-exists", asked,
            "--allow-uncommitted-changes"] + list(more)


@pytest.fixture(scope="module")
def stores(tmp_path_factory):
    first = small_store(tmp_path_factory.mktemp("store_stopped"))
    second = small_store(tmp_path_factory.mktemp("store_not_stopped"))
    return first[0], second[0], first[1]


def test_the_key_is_printed_by_the_command(stores, tmp_path):
    configuration = small_configuration(stores[2])
    file = str(tmp_path / "run.yaml")
    write_configuration(configuration, file)
    printed = subprocess.run(
        [sys.executable, "-m", "rkpinn.run_record.configuration", file],
        env=environment(), capture_output=True, text=True, check=True).stdout.strip()
    assert printed == run_key(configuration)


def test_a_job_stopped_by_the_farm_is_resumed_to_the_same_run(stores, tmp_path):
    stopped_store, other_store, tracks_key = stores
    # long restarts, so that the signal arrives inside a round
    configuration = small_configuration(
        tracks_key, seed=41, training_rounds_at_most=3, training_restarts_per_round=3)
    configuration["training"]["optimiser"]["iterations_per_restart"] = 40
    configuration["training"]["protocol"]["states_per_round"] = 1200
    file = str(tmp_path / "run.yaml")
    write_configuration(configuration, file)
    folder = RunFolder(stopped_store, run_key(configuration))

    job = subprocess.Popen(command(stopped_store, file, "refuse"), env=environment(),
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    waited = 0.0
    while waited < 300 and not (folder.exists() and folder.state()["rounds_done"] >= 1):
        time.sleep(0.2)
        waited += 0.2
        assert job.poll() is None, "the job ended before it could be stopped:\n%s" % (
            job.stdout.read())
    assert folder.is_locked()
    job.send_signal(signal.SIGTERM)
    job.wait(timeout=120)
    assert job.returncode == 128 + signal.SIGTERM
    state = folder.state()
    assert state["state"] == checkpoints.INTERRUPTED
    assert not folder.is_locked(), "a job stopped by the farm must release its lock"
    done = state["rounds_done"]
    assert 1 <= done < 3, "the job was to be stopped inside the run, after a round"
    assert len(folder.rounds()) == done
    assert len(folder.restarts()) == 3 * done

    ended = subprocess.run(command(stopped_store, file, "resume"), env=environment(),
                           capture_output=True, text=True)
    assert ended.returncode == 0, ended.stdout + ended.stderr
    assert folder.state()["state"] == checkpoints.AT_ITS_CAP

    round_trainer.train(configuration, other_store, if_the_run_exists="refuse", **QUIET)
    not_stopped = RunFolder(other_store, run_key(configuration))
    for number in (1, 2, 3):
        a, b = not_stopped.weights_of_round(number), folder.weights_of_round(number)
        for name in a:
            assert identical(a[name].numpy(), b[name].numpy()), (number, name)
    losses = [(r["loss_before"], r["loss_after"]) for r in folder.restarts()]
    assert losses == [(r["loss_before"], r["loss_after"]) for r in not_stopped.restarts()]


def test_a_lock_that_is_touched_is_never_cleared(stores):
    store, _, tracks_key = stores
    configuration = small_configuration(tracks_key, seed=43)
    round_trainer.train(configuration, store, if_the_run_exists="refuse",
                        rounds_in_this_call=1, **QUIET)
    folder = RunFolder(store, run_key(configuration))
    folder.take_lock()                      # a job that is alive
    try:
        assert folder.minutes_since_the_lock_was_touched() < 1.0
        for setting in (None, 30.0):
            with pytest.raises(RunIsLocked):
                round_trainer.train(configuration, store, if_the_run_exists="resume",
                                    a_lock_is_left_behind_after_minutes=setting, **QUIET)
        assert folder.is_locked()
        old = time.time() - 45 * 60
        os.utime(folder.path("LOCK"), (old, old))
        folder.touch_lock()                 # the job makes a restart
        assert folder.minutes_since_the_lock_was_touched() < 1.0
        with pytest.raises(RunIsLocked):
            round_trainer.train(configuration, store, if_the_run_exists="resume",
                                a_lock_is_left_behind_after_minutes=30.0, **QUIET)
    finally:
        folder.clear_lock()


def test_a_lock_left_behind_is_cleared_when_the_caller_says(stores):
    store, _, tracks_key = stores
    configuration = small_configuration(tracks_key, seed=47)
    round_trainer.train(configuration, store, if_the_run_exists="refuse",
                        rounds_in_this_call=1, **QUIET)
    folder = RunFolder(store, run_key(configuration))
    folder.take_lock()                      # a job that was killed outright
    old = time.time() - 45 * 60
    os.utime(folder.path("LOCK"), (old, old))
    assert 44 < folder.minutes_since_the_lock_was_touched() < 46

    with pytest.raises(RunIsLocked):        # without the setting: never cleared
        round_trainer.train(configuration, store, if_the_run_exists="resume", **QUIET)
    with pytest.raises(RunIsLocked):        # not yet old enough
        round_trainer.train(configuration, store, if_the_run_exists="resume",
                            a_lock_is_left_behind_after_minutes=60.0, **QUIET)
    assert folder.is_locked() and folder.state()["rounds_done"] == 1

    state = round_trainer.train(configuration, store, if_the_run_exists="resume",
                                a_lock_is_left_behind_after_minutes=30.0, **QUIET)
    assert state["state"] == checkpoints.AT_ITS_CAP and state["rounds_done"] == 3
    assert not folder.is_locked()
    cleared, = state["locks_cleared"]
    assert 44 < cleared["minutes_since_it_was_touched"] < 46
    assert "process" in cleared["lock_of"]
