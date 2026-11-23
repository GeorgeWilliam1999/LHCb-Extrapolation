"""The checks before training and the cost of one loss and gradient, for one
configuration, on one thread (as A1 was timed in stage 0).

    OMP_NUM_THREADS=1 PYTHONNOUSERSITE=1 PYTHONPATH=../../../../RK_Pinn_module/src \
        nice /data/bfys/gscriven/conda/envs/TE/bin/python time_and_check.py <configuration> [<A1 key>]

Builds the run as the trainer does (nothing is written to the store), runs the
checks before training, times REPEATS loss-and-gradient evaluations on the
first round's states, and prints one JSON line with the peak memory. With a
key of a trained run, it also gives the share the drift term would have under
that run's weights at its latest saved round.
"""
from __future__ import annotations

import json
import resource
import sys
import time

import torch

from rkpinn.run_record.configuration import read_configuration, run_key
from rkpinn.run_record.store import Store
from rkpinn.training.checkpoints import RunFolder
from rkpinn.training.round_trainer import build_run, checks_before_training, loss_of_a_round

STORE = "/data/bfys/gscriven/rkpinn_store"
REPEATS = 3


def main(path, trained=None):
    torch.set_num_threads(1)
    store = Store(STORE)
    configuration = read_configuration(path)
    started = time.time()
    run = build_run(configuration, store)
    built = time.time() - started
    checks = checks_before_training(run)
    loss = loss_of_a_round(run, run.first_round)
    seconds = []
    for _ in range(REPEATS):
        run.network.zero_grad()
        started = time.time()
        value = loss()
        value.backward()
        seconds.append(time.time() - started)
    out = dict(configuration=path, key=run_key(configuration), seconds_to_build=built,
               checks=checks, seconds_per_loss_and_gradient=seconds,
               peak_memory_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)
    constants = run.constants["loss"]
    for name in ("weight_of_the_drift_term", "residual_at_round_one", "drift_at_round_one"):
        if name in constants:
            out[name] = constants[name]
    if trained and "weight_of_the_drift_term" in constants:
        folder = RunFolder(store, trained)
        number = int(folder.state()["rounds_done"])
        run.network.load_state_dict(folder.weights_of_round(number))
        with torch.no_grad():
            track = run.network.predict(torch.as_tensor(run.first_round.states))
            residual = float(run.loss.residual_part(track, constants))
            drift = float(run.loss.drift(track, constants))
        w = constants["weight_of_the_drift_term"]
        out["under_the_trained_run"] = dict(run=trained, round=number, residual=residual,
                                            drift=drift, share=w * drift / (residual + w * drift))
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
