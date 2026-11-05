"""Part C check: why does a per-track scale of the correction fail?

The learned gain must give the matrix W = diag(1/s_out) K diag(s_r) (C1a,
4 x 8) and the corrector W = -diag(1/s_out) M^-1 diag(s_r) (C2, 8 x 8), where
K and M are those of integrators/linearised_gain.py at the frozen network's
stages, s_r the rms of the residual recorded by the Part C run, and s_out the
scale of the correction. This script measures, on one step from the reference
state on every start plane of the test tracks, how much each entry of the
target W varies over the states, for two scales:

  fixed      the four numbers of the frozen network's output form (the global
             straight line: the median per-track scale; direct states: the
             spread of the states)
  per track  the per-track scale of output_forms.py at each state

Spread of an entry: log10 of (99th / 1st percentile of |W_dm|) over the
states. Only the entries that carry the correction count: those whose median
|W_dm| is at least 1/100 of the largest median of their row d (r / s_r is of
order one, so these give nearly all of each component of the correction). The table gives the median and the largest spread over those entries,
and the same for log10 of the per-track scale itself.

    part_c_scale_check.py run      writes results/c_scale_check.csv
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

from common import (DIFFERENTIABLE, NUMBER_OF_STEPS, START_PLANES_MM, STEP_MM, TABLEAU,
                    network_of, on_planes, setup_threads, store, write_rows)
from rkpinn.integrators.linearised_gain import gain_matrix, rates_and_jacobian, stage_map
from rkpinn.networks.output_forms import per_track_scale

# frozen network, its Part C run (C1a, whose constants hold s_r), and its fixed scale
FROZEN = [("straight line global", "9cef9c160b7d", "c6a7d5d15caa"),
          ("direct states", "49e5e2450eaa", "d22c95bb773d"),
          ("straight line per-track", "25dfa17ef168", "d986d2239b65")]
STATES = 20000


def recorded(key):
    with open(os.path.join(store.folder_of_runs, key, "constants.json")) as handle:
        return json.load(handle)["network"]


def spreads(W, keep):
    a = np.abs(W.reshape(len(W), -1))[:, keep]
    lo, hi = np.percentile(a, 1, axis=0), np.percentile(a, 99, axis=0)
    s = np.log10(np.maximum(hi, 1e-300) / np.maximum(lo, 1e-300))
    return float(np.median(s)), float(np.max(s))


def run():
    setup_threads()
    ref = on_planes("test")
    states = ref[:, :-1].reshape(-1, 5)
    planes = np.broadcast_to(START_PLANES_MM, (len(ref), NUMBER_OF_STEPS)).reshape(-1)
    pick = np.random.default_rng(0).choice(len(states), STATES, replace=False)
    S0, z0 = torch.as_tensor(states[pick]), torch.as_tensor(planes[pick].copy())
    A = torch.as_tensor(TABLEAU.stage_matrix)
    b = torch.as_tensor(TABLEAU.weights)
    c = torch.as_tensor(TABLEAU.nodes)
    rows = []
    for name, frozen, part_c in FROZEN:
        network = network_of(frozen)
        with torch.no_grad():
            Y = network(S0, z0)[:, :2]
        f, J = rates_and_jacobian(DIFFERENTIABLE, Y, S0[:, 4:5], z0[:, None] + c[None, :] * STEP_MM)
        with torch.no_grad():
            K = gain_matrix(J, STEP_MM, A, b)                          # (n, 4, 8)
            Minv = torch.linalg.inv(stage_map(J, STEP_MM, A))          # (n, 8, 8)
            s_r = torch.as_tensor(recorded(part_c)["scale_of_the_residual_and_rates"]["residual"])
            per = per_track_scale(S0, z0, STEP_MM, DIFFERENTIABLE.field)          # (n, 4)
            form = network.output_form
            fixed = (form.scale_of_outputs if hasattr(form, "scale_of_outputs")
                     else torch.as_tensor(np.median(per.numpy(), axis=0)))
        for scale_name, s_out in (("fixed", fixed.expand(len(S0), 4)), ("per track", per)):
            W1 = (K * s_r[None, None, :] / s_out[:, :, None]).numpy()
            W2 = (-Minv * s_r[None, None, :] / s_out.repeat(1, 2)[:, :, None]).numpy()
            row = dict(frozen_network=name, scale=scale_name, states=STATES)
            for label, W in (("c1a", W1), ("c2", W2)):
                med = np.median(np.abs(W), axis=0)                      # (rows, 8)
                keep = (med >= 1e-2 * med.max(axis=1, keepdims=True)).reshape(-1)
                row[label + "_entries_kept"] = int(keep.sum())
                row[label + "_median_spread_decades"], row[label + "_largest_spread_decades"] = \
                    spreads(W, keep)
            ls = np.log10(s_out.numpy())
            row["scale_x_spread_decades"] = float(np.percentile(ls[:, 0], 99) - np.percentile(ls[:, 0], 1))
            row["scale_tx_spread_decades"] = float(np.percentile(ls[:, 2], 99) - np.percentile(ls[:, 2], 1))
            row["fraction_of_states_at_the_floor_of_the_scale"] = float(
                (s_out[:, 0] <= 1.0000001e-9).double().mean())
            rows.append(row)
            print(row, flush=True)
    write_rows("c_scale_check.csv", rows)


if __name__ == "__main__":
    {"run": run}[sys.argv[1]]()
