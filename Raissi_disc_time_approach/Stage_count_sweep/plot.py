#!/usr/bin/env python
"""The two figures of the stage-count sweep.

    PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python plot.py

Reads only what `aggregate.py` wrote plus the per-restart history CSVs the
shared `train.py` appends; it never re-scores a model.

    figures/error_vs_stages.png
        left  - median endpoint error against the number of stages, log y:
                the physics-loss network, its supervised data twin, and the
                exact scheme solved directly by a root-finder (the ceiling the
                network is trying to reach), with the straight-line
                do-nothing error as a dashed reference. Error bars are the
                spread over the three seeds, not an uncertainty.
        right - the same for the stage states rather than the endpoint: how
                well the interior of the step is reproduced, which only the
                network is asked for.

    figures/convergence.png
        loss against L-BFGS restart for every run, one panel per q, physics
        and data twin distinguished. The marker shows where the confirmation
        pass took over from the first stall.
"""
from __future__ import annotations

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")

Q_VALUES = (2, 4, 8, 16)
SEEDS = (0, 1, 2)
STYLE = {
    "physics": dict(color="#1f77b4", marker="o", label="physics loss"),
    "data": dict(color="#d62728", marker="s", label="data twin"),
}


def read_csv(path):
    with open(path) as f:
        rows = [r for r in csv.DictReader(
            line for line in f if not line.startswith("#"))]
    return rows


def num(x):
    return float(x) if x not in ("", None) else None


# ------------------------------------------------------ error vs stages -----
def figure_error_vs_stages(out_png):
    rows = read_csv(os.path.join(RESULTS, "error_vs_stages.csv"))
    rows = sorted(rows, key=lambda r: int(r["q"]))
    qs = [int(r["q"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

    # ---- left: the endpoint
    ax = axes[0]
    for mode, st in STYLE.items():
        x, y, lo, hi = [], [], [], []
        for r in rows:
            med = num(r["%s_endpoint_med_um" % mode])
            if med is None:
                continue
            x.append(int(r["q"]))
            y.append(med)
            lo.append(med - num(r["%s_endpoint_min_um" % mode]))
            hi.append(num(r["%s_endpoint_max_um" % mode]) - med)
        if x:
            ax.errorbar(x, y, yerr=[lo, hi], capsize=4, lw=1.8, ms=6, **st)

    sx = [int(r["q"]) for r in rows if num(r["scheme_endpoint_med_um"])]
    sy = [num(r["scheme_endpoint_med_um"]) for r in rows
          if num(r["scheme_endpoint_med_um"])]
    if sx:
        ax.plot(sx, sy, color="#2ca02c", marker="^", lw=1.8, ms=7,
                label="exact scheme (leg B)")

    straight = next((num(r["straight_med_um"]) for r in rows
                     if num(r["straight_med_um"])), None)
    if straight:
        ax.axhline(straight, ls="--", color="0.45", lw=1.3)
        ax.text(qs[0], straight * 1.15, "straight line, %.0f um" % straight,
                color="0.35", fontsize=8, va="bottom")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(qs)
    ax.set_xticklabels([str(q) for q in qs])
    ax.set_xlabel("stages q")
    ax.set_ylabel("median endpoint error / um  (test split)")
    ax.set_title("Endpoint across the magnet")
    ax.grid(alpha=0.3, which="both")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=9)

    # ---- right: the stage states
    ax = axes[1]
    for mode, st in STYLE.items():
        x = [int(r["q"]) for r in rows if num(r["%s_stage_med_um" % mode])]
        y = [num(r["%s_stage_med_um" % mode]) for r in rows
             if num(r["%s_stage_med_um" % mode])]
        if x:
            ax.plot(x, y, lw=1.8, ms=6, **st)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(qs)
    ax.set_xticklabels([str(q) for q in qs])
    ax.set_xlabel("stages q")
    ax.set_ylabel("median stage-state error / um  (test split)")
    ax.set_title("Interior stage states")
    ax.grid(alpha=0.3, which="both")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=9)

    fig.suptitle("Stage-count sweep on the frozen magnet crossing "
                 "(median over 3 seeds, bars = seed spread)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print("wrote %s" % out_png)


# --------------------------------------------------------- convergence ------
def figure_convergence(out_png):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=False)
    any_run = False
    for ax, q in zip(axes.ravel(), Q_VALUES):
        for mode, st in STYLE.items():
            for seed in SEEDS:
                path = os.path.join(
                    RESULTS, "q%02d_%s_s%d_history.csv" % (q, mode, seed))
                if not os.path.exists(path):
                    continue
                rows = read_csv(path)
                if not rows:
                    continue
                any_run = True
                loss = [float(r["loss"]) for r in rows]
                ax.plot(range(len(loss)), loss, lw=1.1, alpha=0.85,
                        color=st["color"],
                        label=st["label"] if seed == 0 else None)
                # where the confirmation pass begins
                first_conf = next((i for i, r in enumerate(rows)
                                   if r.get("phase") == "confirm"), None)
                if first_conf is not None:
                    ax.plot([first_conf], [loss[first_conf]], marker="|",
                            ms=11, color=st["color"])
        ax.set_yscale("log")
        ax.set_title("q = %d" % q)
        ax.set_xlabel("L-BFGS restart")
        ax.set_ylabel("training loss")
        ax.grid(alpha=0.3, which="both")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    fig.suptitle("Training loss per restart, all runs "
                 "(tick = start of the confirmation pass)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print("wrote %s%s" % (out_png, "" if any_run else "  (no histories yet)"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()
    os.makedirs(FIGURES, exist_ok=True)
    figure_error_vs_stages(os.path.join(FIGURES, "error_vs_stages.png"))
    figure_convergence(os.path.join(FIGURES, "convergence.png"))


if __name__ == "__main__":
    main()
