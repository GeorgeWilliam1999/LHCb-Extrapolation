"""The bias-study scores of the Part B runs, beside those of their baselines.

    part_b_scores.py [round]    each run at its last finished round (or the
                                round given), and the two baselines at their
                                last round; writes results/b_part_b_scores.csv

Per run: the test endpoint radial median and its value per momentum band, the
mean signed endpoint error in x and y with its standard error, the lag-1
autocorrelation of the signed single-step errors (median over tracks), and the
pass criteria of Part B (direct states <= 2.85 um, straight line <= 0.92 um, no
band more than 2x worse than the baseline).

The standard report of a run (with output 11, resolution_per_momentum_band) is
made by the package: python -m rkpinn.run_record.report_of_a_run (see its
README); this script adds the bias-study numbers only.
"""
from __future__ import annotations

import os
import sys

from common import (BANDS, STORE, chain, endpoint_scores, lag_one_scores, network_of,
                    plain_step, setup_threads, single_step_errors, split, write_rows)

RUNS_OF_PART_B = [
    ("direct states", "B.3 residual draws", "5cb2aeb6778b", "49e5e2450eaa", 2.85),
    ("direct states", "B.4 bias term", "6aba11cc7475", "49e5e2450eaa", 2.85),
    ("straight line, per-track scale", "B.3 residual draws", "8a0f3b8f422f", "25dfa17ef168",
     0.92),
    ("straight line, per-track scale", "B.4 bias term", "f0f219df805e", "25dfa17ef168", 0.92),
]


def finished_rounds(key):
    folder = os.path.join(STORE, "runs", key, "snapshots")
    if not os.path.isdir(folder):
        return []
    return sorted(int(n.split("_")[1]) for n in os.listdir(folder)
                  if os.path.exists(os.path.join(folder, n, "scores.json")))


def scores(key, round_number):
    net = network_of(key, [round_number])
    row, _ = endpoint_scores(chain(plain_step(net), split("test").start_state)[:, -1])
    row.update(lag_one_scores(single_step_errors(net, "test")))
    return row


def main(asked_round=None):
    setup_threads()
    baselines = {}
    rows = []
    for form, item, key, baseline, target in RUNS_OF_PART_B:
        if baseline not in baselines:
            last = finished_rounds(baseline)[-1]
            baselines[baseline] = dict(scores(baseline, last), round=last)
            rows.append(dict(form=form, item="baseline", run_key=baseline,
                             **baselines[baseline]))
        rounds = finished_rounds(key)
        if not rounds:
            print(key, "has no finished round", flush=True)
            continue
        r = rounds[-1] if asked_round is None else int(asked_round)
        row = dict(form=form, item=item, run_key=key, round=r, **scores(key, r))
        base = baselines[baseline]
        worst = max(row["median_um_" + b[0]] / base["median_um_" + b[0]] for b in BANDS)
        row.update(target_um=target,
                   passes_target=bool(row["endpoint_radial_median_um"] <= target),
                   worst_band_ratio_to_baseline=worst,
                   no_band_more_than_2x_worse=bool(worst <= 2.0))
        rows.append(row)
        print(row, flush=True)
    columns = []
    for row in rows:
        columns += [c for c in row if c not in columns]
    write_rows("b_part_b_scores.csv", [{c: row.get(c, "") for c in columns} for row in rows])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
