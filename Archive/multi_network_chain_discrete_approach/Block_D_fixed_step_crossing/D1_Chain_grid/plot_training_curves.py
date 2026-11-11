#!/usr/bin/env python
"""D1 - the continuation study: endpoint error against L-BFGS restart, for the
chains retrained with the 0.1 percent stall rule and the medians logged after
every restart (`--stall-tol 0.001 --log-medians`, results/tight_stall/).

    figures/training_curves.png    loss and test endpoint median against
                                   restart, one panel per leg, with the point
                                   where the 1 percent rule of the grid run
                                   stopped marked
    results/training_curves.csv    the same numbers
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
import numpy as np                                    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TIGHT = os.path.join(HERE, "results", "tight_stall")
GRID = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")
INK2 = "#5f5e5a"


def main():
    runs = sorted(glob.glob(os.path.join(TIGHT, "N*_q*", "leg*_errors.csv")))
    if not runs:
        print("no continuation records yet")
        return
    rows_out = []
    fig, axes = plt.subplots(len(runs), 2, figsize=(12, 3.2 * len(runs)), squeeze=False)
    for ax_row, p in zip(axes, runs):
        m = re.search(r"N(\d+)_q(\d+)[/\\]leg(\d+)_errors", p)
        N, q, k = int(m.group(1)), int(m.group(2)), int(m.group(3))
        with open(p) as f:
            rows = list(csv.DictReader(f))
        n = np.array([int(r["outer"]) for r in rows])
        loss = np.array([float(r["loss"]) for r in rows])
        te = np.array([float(r["test_endpoint_med_um"]) for r in rows])
        va = np.array([float(r["val_endpoint_med_um"]) for r in rows])
        tr = np.array([float(r["train_endpoint_med_um"]) for r in rows])
        grid_rec = os.path.join(GRID, "N%03d_q%02d" % (N, q), "leg%03d.json" % k)
        stop = None
        if os.path.exists(grid_rec):
            with open(grid_rec) as f:
                g = json.load(f)
            stop = (g["restarts"], g["test"]["endpoint_med_um"])
        ax = ax_row[0]
        ax.plot(n, loss, "-", color="#2a78d6", lw=1.5)
        ax.set_yscale("log"); ax.set_ylabel("physics loss"); ax.set_title("N = %d, q = %d, leg %d" % (N, q, k))
        ax = ax_row[1]
        ax.plot(n, tr, "-", color="#1baf7a", lw=1.2, label="train")
        ax.plot(n, va, "-", color="#eda100", lw=1.2, label="val")
        ax.plot(n, te, "-", color="#eb6834", lw=1.8, label="test")
        if stop:
            ax.axvline(stop[0], color=INK2, ls="--", lw=1)
            ax.text(stop[0], max(te), " grid run stopped here\n (%.0f µm)" % stop[1], fontsize=8, va="top")
        ax.set_yscale("log"); ax.set_ylabel("endpoint median  [µm]"); ax.legend(fontsize=8)
        for r in rows:
            rows_out.append(dict(N=N, q=q, leg=k, **r))
    for ax in axes[-1]:
        ax.set_xlabel("L-BFGS restart (200 iterations each)")
    fig.suptitle("Training past the 1 percent stall: does the error keep falling with the loss?")
    fig.tight_layout()
    os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(os.path.join(FIGURES, "training_curves.png"), dpi=140)
    with open(os.path.join(GRID, "training_curves.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader(); w.writerows(rows_out)
    print("wrote figures/training_curves.png (%d legs)" % len(runs))


if __name__ == "__main__":
    main()
