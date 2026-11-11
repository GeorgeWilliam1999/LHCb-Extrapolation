#!/usr/bin/env python
"""E3 - how training converged: the loss, and the error it was meant to buy.

Two curves per network, against the same axis of L-BFGS restarts:

  * the training loss after every restart (`history.csv`). Within a round it
    falls steadily; at every round boundary the training states are redrawn
    from the network's own tracks, and the loss steps back up;
  * the validation error at the SciFi plane after every round (`rounds.csv`),
    which is what the loss is supposed to buy.

The pair is the evidence for stopping: the round-end loss and the validation
error both flatten long before training was stopped, so the last hundreds of
restarts moved neither.

Run:     PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python convergence.py
Outputs: figures/convergence_grid.png, figures/convergence_summary.png,
         results/convergence.csv
"""
from __future__ import annotations

import csv
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
E1 = os.path.join(HERE, "..", "E1_Network_grid", "results")


def series(run):
    h = list(csv.DictReader(open(os.path.join(run, "history.csv"))))
    rr = list(csv.DictReader(open(os.path.join(run, "rounds.csv"))))
    restart = np.array([int(x["restart"]) for x in h])
    loss = np.array([float(x["loss_after"]) for x in h])
    rounds = np.array([int(x["round"]) for x in h])
    cum = np.cumsum([int(x["restarts"]) for x in rr])
    val = np.array([float(x["val_z1_pos_med_um"]) for x in rr])
    loss_end = np.array([float(x["loss_last"]) for x in rr])
    return restart, loss, rounds, cum, val, loss_end


def main():
    runs = sorted(glob.glob(os.path.join(E1, "N[0-9][0-9][0-9]_q[0-9][0-9]")))
    data, rows = {}, []
    for run in runs:
        rec = json.load(open(os.path.join(run, "record.json")))
        N, q = rec["N"], rec["q"]
        data[(N, q)] = series(run)
        restart, loss, _, cum, val, loss_end = data[(N, q)]
        last = min(8, len(val))
        rows.append(dict(N=N, q=q, restarts=len(restart), rounds=len(val),
                         loss_first=float(loss[0]), loss_final=float(loss[-1]),
                         loss_end_first=float(loss_end[0]), loss_end_last=float(loss_end[-1]),
                         loss_end_last8_ratio=float(loss_end[-last] / loss_end[-1]),
                         val_first=float(val[0]), val_final=float(val[-1]),
                         val_last8_mean=float(val[-last:].mean()), val_last8_std=float(val[-last:].std()),
                         val_best=float(val.min()), val_best_round=int(np.argmin(val) + 1),
                         test_final=rec["test"]["vs_rk6_endpoint"]["pos_med_um"]))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "convergence.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Ns = sorted({k[0] for k in data})
    qs = sorted({k[1] for k in data})
    fig, ax = plt.subplots(len(Ns), len(qs), figsize=(4.2 * len(qs), 3.0 * len(Ns)), sharex=False)
    for r, N in enumerate(Ns):
        for c, q in enumerate(qs):
            a = ax[r, c]
            restart, loss, rounds, cum, val, loss_end = data[(N, q)]
            a.plot(restart, loss, "-", color="#1f77b4", lw=0.9)
            a.plot(cum - 1, loss_end, "o", color="#08306b", ms=3, label="round end")
            a.set_yscale("log")
            a.set_ylabel("training loss", color="#1f77b4", fontsize=8)
            a.tick_params(axis="y", labelcolor="#1f77b4", labelsize=7)
            a.tick_params(axis="x", labelsize=7)
            b = a.twinx()
            b.plot(cum - 1, val, "-s", color="#d62728", ms=3, lw=1.2)
            b.set_ylabel("val error at z1 [µm]", color="#d62728", fontsize=8)
            b.tick_params(axis="y", labelcolor="#d62728", labelsize=7)
            a.set_title("N = %d, q = %d" % (N, q), fontsize=9)
            if r == len(Ns) - 1:
                a.set_xlabel("L-BFGS restart", fontsize=8)
    fig.suptitle("Block E: training loss (blue, left) and validation error at the SciFi plane "
                 "(red, right) against restart", fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "figures", "convergence_grid.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    cmap = plt.get_cmap("viridis")
    for i, (k, v) in enumerate(sorted(data.items())):
        restart, loss, rounds, cum, val, loss_end = v
        col = cmap(i / (len(data) - 1))
        ax[0].plot(cum, loss_end, "-o", ms=3, color=col, label="N=%d q=%d" % k)
        ax[1].plot(cum, val, "-o", ms=3, color=col)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("L-BFGS restarts")
    ax[0].set_ylabel("loss at the end of each round")
    ax[0].set_title("The loss on freshly drawn states stops falling", fontsize=10)
    ax[0].legend(fontsize=6, ncol=2)
    ax[1].set_xlabel("L-BFGS restarts")
    ax[1].set_ylabel("validation error at z1 [µm]")
    ax[1].set_yscale("log")
    ax[1].set_title("and the error it buys flattens with it", fontsize=10)
    for a in ax:
        a.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "convergence_summary.png"), dpi=130)
    print("wrote figures/convergence_grid.png, convergence_summary.png, results/convergence.csv")
    print("  N   q | restarts rounds | loss end first -> last | val first -> last (best) | last-8 val")
    for x in rows:
        print("%4d %3d | %8d %6d | %9.2e -> %.2e | %6.0f -> %5.0f (%4.0f at round %2d) | %5.0f +- %.0f"
              % (x["N"], x["q"], x["restarts"], x["rounds"], x["loss_end_first"], x["loss_end_last"],
                 x["val_first"], x["val_final"], x["val_best"], x["val_best_round"],
                 x["val_last8_mean"], x["val_last8_std"]))


if __name__ == "__main__":
    main()
