#!/usr/bin/env python
"""D1 - the continuation study: endpoint error against training, restart by
restart, for the chains rerun with the 0.1 % stall rule and the 400-restart
cap (results/tight_stall/). Reads the <tag>_errors.csv the shared trainer
writes with --log-medians; computes nothing.

    figures/training_curves.png      loss and the three splits' endpoint
                                     medians against restart, one panel per leg
    results/tight_stall/summary.csv  where the 1 % rule stopped, where the
                                     0.1 % rule stopped, and the error at each
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TS = os.path.join(HERE, "results", "tight_stall")
BASE = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
INK, INK2, GRID = "#1a1a19", "#5f5e5a", "#e3e2dc"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False})


def main():
    files = sorted(glob.glob(os.path.join(TS, "N*_q*", "leg*_errors.csv")))
    if not files:
        print("no error logs yet")
        return
    rows_out = []
    fig, axes = plt.subplots(len(files), 2, figsize=(13, 3.4 * len(files)), squeeze=False)
    for (axl, axe), p in zip(axes, files):
        m = re.search(r"N(\d+)_q(\d+)/leg(\d+)", p)
        N, q, k = int(m.group(1)), int(m.group(2)), int(m.group(3))
        rows = list(csv.DictReader(open(p)))
        n = np.array([int(r["outer"]) for r in rows]) + 1
        loss = np.array([float(r["loss"]) for r in rows])
        tr = np.array([float(r["train_endpoint_med_um"]) for r in rows])
        va = np.array([float(r["val_endpoint_med_um"]) for r in rows])
        te = np.array([float(r["test_endpoint_med_um"]) for r in rows])
        conf = np.array([r["phase"] == "confirm" for r in rows])
        base_rec = os.path.join(BASE, "N%03d_q%02d" % (N, q), "leg%03d.json" % k)
        base = json.load(open(base_rec)) if os.path.exists(base_rec) else None
        axl.plot(n, loss, "-", color="#2a78d6", lw=1.5)
        axl.set_yscale("log"); axl.set_xscale("log")
        axl.set_ylabel("physics loss"); axl.set_title("N = %d, q = %d, leg %d: loss" % (N, q, k))
        axe.plot(n, tr, "-", color="#2a78d6", lw=1.4, label="train")
        axe.plot(n, va, "-", color="#eb6834", lw=1.4, label="validation")
        axe.plot(n, te, "-", color="#1baf7a", lw=1.4, label="test")
        if conf.any():
            axe.plot(n[conf], te[conf], "o", ms=3, color=INK, label="confirmation restarts")
        if base:
            axe.axvline(base["restarts"], color=INK2, ls="--", lw=1)
            axe.text(base["restarts"], te.max(), " 1 % rule stopped here\n (%.0f µm)" % base["test"]["endpoint_med_um"],
                     fontsize=8, va="top", color=INK2)
        axe.set_yscale("log"); axe.set_xscale("log")
        axe.set_ylabel("endpoint median  [µm]"); axe.set_title("endpoint error against training")
        axe.legend(fontsize=8)
        for ax in (axl, axe):
            ax.set_xlabel("L-BFGS restart (200 iterations each)")
        rows_out.append({"N": N, "q": q, "leg": k,
                         "restarts_1pct": base["restarts"] if base else "",
                         "test_med_um_1pct": base["test"]["endpoint_med_um"] if base else "",
                         "restarts_0p1pct": int(n[-1]),
                         "test_med_um_0p1pct": float(te[-1]),
                         "test_med_um_best_any_restart": float(te.min()),
                         "restart_of_best": int(n[np.argmin(te)]),
                         "loss_1pct": float(loss[min(len(loss), base["restarts"]) - 1]) if base else "",
                         "loss_0p1pct": float(loss[-1]),
                         "train_over_val_at_end": float(tr[-1] / va[-1])})
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "training_curves.png"), dpi=140)
    with open(os.path.join(TS, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader(); w.writerows(rows_out)
    for r in rows_out:
        print(r)


if __name__ == "__main__":
    main()
