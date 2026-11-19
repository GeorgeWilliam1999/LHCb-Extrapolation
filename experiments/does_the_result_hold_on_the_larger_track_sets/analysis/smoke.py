"""Smoke check of the configurations, before any farm job. Writes nothing to the store.

For each configuration: build the run (the checks before training included),
time one loss and its gradient on the first round's states on one thread, and
take 30 iterations of L-BFGS in memory to see the loss fall.

    PYTHONNOUSERSITE=1 PYTHONPATH=../../../RK_Pinn_module/src OMP_NUM_THREADS=1 \
        /data/bfys/gscriven/conda/envs/TE/bin/python smoke.py ../configurations/<file> ...
"""
from __future__ import annotations

import sys
import time

import torch

from rkpinn.run_record.configuration import read_configuration
from rkpinn.run_record.store import Store
from rkpinn.training import round_trainer

STORE = "/data/bfys/gscriven/rkpinn_store"


def main(files):
    store = Store(STORE)
    for file in files:
        run = round_trainer.build_run(read_configuration(file), store)
        measured = round_trainer.checks_before_training(run)
        loss_now = round_trainer.loss_of_a_round(run, run.first_round)
        parameters = list(run.network.parameters())
        times = []
        for _ in range(3):
            t = time.time()
            for p in parameters:
                p.grad = None
            loss = loss_now()
            loss.backward()
            times.append(time.time() - t)
        optimiser = torch.optim.LBFGS(parameters, max_iter=30, history_size=120,
                                      line_search_fn="strong_wolfe",
                                      tolerance_grad=1e-13, tolerance_change=1e-16)
        evaluations = [0]

        def closure():
            optimiser.zero_grad()
            value = loss_now()
            value.backward()
            evaluations[0] += 1
            return value

        first = float(loss_now().detach())
        t = time.time()
        optimiser.step(closure)
        took = time.time() - t
        last = float(loss_now().detach())
        constants = {k: v for k, v in run.constants["loss"].items()
                     if k in ("weight_of_the_carried_term", "pooled_at_round_one",
                              "carried_at_round_one", "first_round_states_are_whole_tracks")}
        print("%s\n  key %s, %d states, loss at the exact scheme %.2e\n"
              "  one loss and gradient: %.2f s; 30 iterations: %.0f s (%d evaluations, "
              "%.2f s each)\n  loss %.4e -> %.4e\n  %s"
              % (file.split("/")[-1], run.key, len(run.first_round.states),
                 measured["loss_at_the_exact_solution"], min(times), took, evaluations[0],
                 took / evaluations[0], first, last, constants), flush=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main(sys.argv[1:])
