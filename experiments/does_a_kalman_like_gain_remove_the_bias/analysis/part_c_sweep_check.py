"""Part C check: is the trained C2 correction one fixed-point sweep?

A fixed-point sweep of the stage equations moves the stages by dY = -r
(Y' = S0 + dz A F). This script compares the correction dY of each trained C2
run with -r, on one step from the reference state on 20,000 test start
planes, and scores the sweep itself as an extrapolator from the same frozen
networks (the bias study's C.2 in the form of this package).

    part_c_sweep_check.py run      writes results/c_sweep_check.csv
"""
from __future__ import annotations

import sys

import numpy as np
import torch

from common import (NUMBER_OF_STEPS, START_PLANES_MM, chain, endpoint_scores, network_of,
                    on_planes, setup_threads, split, write_rows)
from rkpinn.networks.collect_and_sum import collect_and_sum, rates_at_the_stages

C2 = [("straight line global", "6c881486c499"), ("direct states", "1eaf33293506"),
      ("straight line per-track", "b38147788963"),
      ("direct states, per-track correction scale", "cac1a385740d")]
STATES = 20000


def sweep_step(stage_network):
    """One step: the stage network, one fixed-point sweep, the end summed."""
    def step(states, planes):
        S0, z0 = torch.as_tensor(states), torch.as_tensor(planes.copy())
        with torch.no_grad():
            stages, rates, residual = network_first_call(stage_network, S0, z0)
            second = stages - residual
            p = z0[:, None] + stage_network.nodes[None, :] * stage_network.step_length_mm
            f = rates_at_the_stages(stage_network.equation_of_motion, S0, second, p)
            return collect_and_sum(S0, f, stage_network.step_length_mm,
                                   stage_network.weights_of_the_tableau).numpy()
    return step


def network_first_call(network, S0, z0):
    from rkpinn.networks.stage_network_of_a_run import first_call
    return first_call(network, S0, z0)


def run():
    setup_threads()
    ref = on_planes("test")
    states = ref[:, :-1].reshape(-1, 5)
    planes = np.broadcast_to(START_PLANES_MM, (len(ref), NUMBER_OF_STEPS)).reshape(-1)
    pick = np.random.default_rng(0).choice(len(states), STATES, replace=False)
    S0, z0 = torch.as_tensor(states[pick]), torch.as_tensor(planes[pick].copy())
    test = split("test")
    rows = []
    for name, key in C2:
        network = network_of(key)
        with torch.no_grad():
            stages, rates, residual = network.first_call(S0, z0)
            dY = network.correction(S0, z0, residual)
        rel = ((dY + residual).norm(dim=(1, 2)) / residual.norm(dim=(1, 2))).numpy()
        sweep = chain(sweep_step(network.stage_network), test.start_state)
        row, _ = endpoint_scores(sweep[:, -1], prefix="sweep_")
        rows.append(dict(c2_run=name, run_key=key, states=STATES,
                         median_of_norm_of_correction_plus_r_over_norm_of_r=float(np.median(rel)),
                         percentile_90_of_it=float(np.percentile(rel, 90)), **row))
        print(rows[-1], flush=True)
    write_rows("c_sweep_check.csv", rows)


if __name__ == "__main__":
    {"run": run}[sys.argv[1]]()
