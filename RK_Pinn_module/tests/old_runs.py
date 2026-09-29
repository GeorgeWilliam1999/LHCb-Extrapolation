"""The old runs and the tables of the second mini-paper, for the gates of the
evaluation. Not a gate. Everything here is only read.

The old runs are the three pooled-loss and the three cost-weighted runs the
paper compares. Each keeps the states its network gave on every plane, for
the validation and the test tracks.
"""
from __future__ import annotations

import csv
import os

import numpy as np

from frozen_code import PROJECT_ROOT, SINGLE_NETWORK_FOLDER, needed_file

TABLES_OF_THE_PAPER = os.path.join(PROJECT_ROOT, "Self_chained_paper", "results")

# (number of steps, number of stages)
CELLS_OF_THE_PAPER = ((64, 2), (128, 8), (256, 16))
# the name the paper gives the loss -> the folder of its runs
FOLDERS_OF_RUNS = {
    "pooled": os.path.join(SINGLE_NETWORK_FOLDER, "Block_E_single_network_chain",
                           "E1_Network_grid", "results"),
    "reweighted": os.path.join(SINGLE_NETWORK_FOLDER, "Block_F_reweighted_loss",
                               "F1_Training", "results", "full"),
}
OLD_RUNS = [(steps, stages, loss) for steps, stages in CELLS_OF_THE_PAPER
            for loss in ("pooled", "reweighted")]

WINDOW_OF_THE_PAPER_GEV = (10.0, 50.0)

# the names the paper gives -> the names of the package
BAND_OF_THE_PAPER = {
    "all": "all momenta", "<3": "below 3 GeV", "3-8": "3 to 8 GeV",
    "8-20": "8 to 20 GeV", "20-50": "20 to 50 GeV", ">50": "above 50 GeV",
    "10-50 (loss window)": "10 to 50 GeV (window of the loss)",
}
COMPONENT_OF_THE_PAPER = {"x": "x", "y": "y", "tx": "slope in x", "ty": "slope in y"}
STATISTIC_OF_THE_PAPER = {
    "med_abs": "median_of_absolute", "p95_abs": "percentile_95_of_absolute",
    "signed_med": "signed_median", "hw68": "half_width_68", "rms": "root_mean_square"}


def folder_of_run(number_of_steps, number_of_stages, loss) -> str:
    return needed_file(os.path.join(
        FOLDERS_OF_RUNS[loss], "N%03d_q%02d" % (number_of_steps, number_of_stages)))


def states_of_run(number_of_steps, number_of_stages, loss, split="test"):
    """The states the old network gave on every plane, shape (n, N + 1, 5)."""
    path = needed_file(os.path.join(
        folder_of_run(number_of_steps, number_of_stages, loss), "chain_states.npz"))
    with np.load(path) as file:
        return np.asarray(file["%s_states" % split])


def table_of_the_paper(name, number_of_steps=None, number_of_stages=None, loss=None):
    """The rows of one table of the paper, those of one run if one is named."""
    with open(needed_file(os.path.join(TABLES_OF_THE_PAPER, name + ".csv"))) as handle:
        rows = list(csv.DictReader(handle))
    if number_of_steps is not None:
        rows = [r for r in rows if int(r["N"]) == number_of_steps
                and int(r["q"]) == number_of_stages and (loss is None or r["loss"] == loss)]
    return rows


def rows_kept_once(rows, column):
    """The rows with a repeated number kept once, the last one written, in
    order. Two farm jobs wrote to one of the old runs at the same time, so its
    files hold repeated rounds. The paper reads them this way."""
    seen = {}
    for row in rows:
        seen[int(float(row[column]))] = row
    return [seen[number] for number in sorted(seen)]


def rounds_of_run(number_of_steps, number_of_stages, loss):
    path = needed_file(os.path.join(
        folder_of_run(number_of_steps, number_of_stages, loss), "rounds.csv"))
    with open(path) as handle:
        return rows_kept_once(list(csv.DictReader(handle)), "round")
