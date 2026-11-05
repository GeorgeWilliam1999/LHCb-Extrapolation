"""Part C: the eight farm runs of the learned gain (C1a) and the
error-conditioned stage network (C2), scored as Part B scores its variants.

For each run at its last round, and for the three frozen networks it is built
on:

  endpoint     the test tracks chained over the 64 steps; radial median at
               7826 mm per momentum band, and the mean signed error in x and y
  single step  one step from the reference state on each start plane, 92,928
               steps; radial median and rms, rms of the slopes, and the median
               lag-1 autocorrelation along the track (common.py)

The standard report of each run is made beside this, by
rkpinn.run_record.report_of_a_run; this script adds the signed means and the
lag-1, which the report does not hold, in the form of the Part B tables.

    part_c.py run       writes results/c_endpoint.csv, results/c_single_step.csv
"""
from __future__ import annotations

import sys

import numpy as np

from common import (RUNS, SHORT, chain, endpoint_scores, last_round, network_of,
                    plain_step, setup_threads, single_step_errors, split, write_rows)
from part_b import single_step_row

PART_C = [
    ("C1a", "straight line per-track", "d986d2239b65"),
    ("C1a", "straight line global", "c6a7d5d15caa"),
    ("C1a", "direct states", "d22c95bb773d"),
    ("C1a", "direct states, per-track correction scale", "1006d1956c3c"),
    ("C2", "straight line per-track", "b38147788963"),
    ("C2", "straight line global", "6c881486c499"),
    ("C2", "direct states", "1eaf33293506"),
    ("C2", "direct states, per-track correction scale", "cac1a385740d"),
]


def run():
    setup_threads()
    test = split("test")
    rows_end, rows_step = [], []
    models = [("frozen", SHORT[key], key) for _, key in RUNS] + PART_C
    for variant, frozen, key in models:
        network = network_of(key)
        states = chain(plain_step(network), test.start_state)
        row, _ = endpoint_scores(states[:, -1])
        common = dict(variant=variant, frozen_network=frozen, run_key=key,
                      round=last_round(key))
        rows_end.append(dict(common, **row))
        e = single_step_errors(network, "test")
        rows_step.append(dict(common, **single_step_row(e)))
        print("%-6s %-45s %s: endpoint %.4f um, single step %.6f um, lag-1 x %+.2f"
              % (variant, frozen, key, row["endpoint_radial_median_um"],
                 rows_step[-1]["single_step_radial_median_um"],
                 rows_step[-1]["lag1_x_median"]), flush=True)
    write_rows("c_endpoint.csv", rows_end)
    write_rows("c_single_step.csv", rows_step)


if __name__ == "__main__":
    {"run": run}[sys.argv[1]]()
