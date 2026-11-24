"""A check of the copy of Allen's adaptive Runge-Kutta-Nystrom integrator.

LHCb checks that integrator of Allen against the HLT2 integrator
TrackSTEPExtrapolator in a Moore test (Moore v60r3,
Hlt/Moore/tests/options/allen_reco/compare_hlt1_hlt2_extrapolator_states.py,
with CompareExtrapolatorStates of Allen): both carry the grid of 1,331 states
of the Rec test on the v8r1 map, and every state must agree within

    x, y:    |difference| <= 0.01 mm + 1 % of the larger value
    tx, ty:  |difference| <= 5e-5   + 1 % of the larger value

We cannot read the states of Allen out of that test. But our copy of
TrackSTEPExtrapolator prints the same numbers as the real tool, so the same
comparison between our two copies says whether our copy of the Allen
integrator behaves as the real one must: it passes when every state agrees
within the limits of the LHCb test.

    allen_copy_against_the_step_copy.py     writes results/allen_copy_against_the_step_copy.csv
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastex.compiled import run  # noqa: E402
from rec_copy_against_the_rec_reference import Z1, Z2, grid_of_the_test  # noqa: E402


def main():
    states = grid_of_the_test()
    rows = []
    for map_name in ("v8r1_down", "v8r1_up"):
        step, step_counts = run("rec_step", states, Z1, Z2, map_name=map_name, reader="rec")
        allen, allen_counts = run("allen_adaptive_nystrom", states, Z1, Z2, map_name=map_name,
                                  reader="allen", precision="float")
        d = np.abs(allen[:, :4] - step[:, :4])
        larger = np.maximum(np.abs(allen[:, :4]), np.abs(step[:, :4]))
        limits = np.array([1e-2, 1e-2, 5e-5, 5e-5])[None, :] + 1e-2 * larger
        differs = (d > limits).any(axis=1)
        rows.append(dict(map=map_name, states=len(states), states_outside_the_limits=int(differs.sum()),
                         largest_difference_of_position_mm=float(d[:, :2].max()),
                         median_difference_of_position_um=float(np.median(d[:, :2].max(axis=1)) * 1e3),
                         largest_difference_of_slope=float(d[:, 2:].max()),
                         field_reads_mean_step=float(step_counts[:, 0].mean()),
                         field_reads_mean_allen=float(allen_counts[:, 0].mean())))
        print(rows[-1], flush=True)
    with open(os.path.join(HERE, "results", "allen_copy_against_the_step_copy.csv"), "w",
              newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
