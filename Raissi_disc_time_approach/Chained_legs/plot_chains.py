#!/usr/bin/env python
"""The three figures of A3b. Loads the result tables only; computes no physics.

    figures/error_vs_chained_legs.png  median endpoint error against the number
        of legs already walked, one thin line per seed, the seed median as a
        band, the selected seed bold and the data twin dashed; one panel per
        architecture.
    figures/error_growth_examples.png  a few particles drawn out: the network's
        path and the reference path in x-z and in y-z.
    figures/leg_d_reproduction.png  the downstream leg walked as two steps
        against the same leg taken in one giant step, with the exact-scheme
        ceilings at q = 8 and q = 16 and the straight line.
"""
from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")
# the exact scheme on leg D, ../Simple_first_pass/results/scheme_error_vs_q.csv
CEIL_D_Q8_UM = 739.2
CEIL_D_Q16_UM = 92.5


def read(name):
    with open(os.path.join(RESULTS, name)) as f:
        return list(csv.DictReader(f))


def main():
    os.makedirs(FIGURES, exist_ok=True)
    summ = read("chain_summary.csv")
    sel = read("selection.csv")
    selected = {(r["width"], r["mode"]): r["network"] for r in sel
                if r["selected"] == "True"}
    widths = sorted({r["width"] for r in summ}, key=int)

    # --------------------------------------------- error vs chained legs ----
    fig, axes = plt.subplots(1, len(widths), figsize=(6.2 * len(widths), 5),
                             sharey=True, squeeze=False)
    for i, w in enumerate(widths):
        ax = axes[0, i]
        rows = [r for r in summ if r["width"] == w and r["split"] == "test"]
        for mode, colour, ls in (("physics", "#1f77b4", "-"),
                                 ("data", "#d62728", "--")):
            tags = sorted({r["network"] for r in rows if r["mode"] == mode})
            curves = []
            for t in tags:
                rr = sorted([r for r in rows if r["network"] == t],
                            key=lambda r: int(r["step"]))
                x = [int(r["step"]) + 1 for r in rr]
                y = [float(r["end_med_um"]) for r in rr]
                curves.append((t, x, y))
                ax.plot(x, y, ls, color=colour, lw=0.9, alpha=0.45)
            if curves:
                L = min(len(c[2]) for c in curves)
                arr = np.array([c[2][:L] for c in curves])
                ax.fill_between(curves[0][1][:L], arr.min(axis=0),
                                arr.max(axis=0), color=colour, alpha=0.12)
                ax.plot(curves[0][1][:L], np.median(arr, axis=0), ls,
                        color=colour, lw=2.2,
                        label="%s, seed median (%d seeds)" % (mode, len(curves)))
            best = selected.get((w, mode))
            for t, x, y in curves:
                if t == best:
                    ax.plot(x, y, ls, color="black", lw=2.4,
                            label="selected: %s" % t)
        ax.set_yscale("log")
        ax.set_xlabel("legs walked")
        ax.set_title("4x%s" % w)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
        if i == 0:
            ax.set_ylabel("median endpoint error [um]")
    fig.suptitle("A3b  chained along real particle paths (test split): error "
                 "against the number of legs walked", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(FIGURES, "error_vs_chained_legs.png"), dpi=140)
    plt.close(fig)
    print("wrote figures/error_vs_chained_legs.png")

    # -------------------------------------------------- growth examples -----
    ex = np.load(os.path.join(RESULTS, "example_paths.npz"))
    best = selected.get((widths[-1], "physics")) or selected.get((widths[0], "physics"))
    pred = ex["pred_%s" % best]
    ref = ex["ex_ref"]
    planes = ex["ex_planes"]
    S0 = ex["ex_S0"]
    P = ex["ex_P"]
    n = len(P)
    fig, axes = plt.subplots(2, n, figsize=(3.1 * n, 6), sharex="col")
    for i in range(n):
        z = planes[i]
        for row, comp, name in ((0, 0, "x"), (1, 1, "y")):
            ax = axes[row, i]
            ax.plot(z, np.concatenate([[S0[i, comp]], ref[i, :, comp]]), "o-",
                    color="0.25", ms=4, lw=1.4, label="reference (RK4)")
            ax.plot(z, np.concatenate([[S0[i, comp]], pred[i, :, comp]]), "s--",
                    color="#1f77b4", ms=4, lw=1.4, label="network, chained")
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_ylabel("%s [mm]" % name)
            if row == 1:
                ax.set_xlabel("z [mm]")
            if row == 0:
                ax.set_title("p = %.1f GeV" % P[i], fontsize=9)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("A3b  chained paths against the reference, network %s" % best,
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(FIGURES, "error_growth_examples.png"), dpi=140)
    plt.close(fig)
    print("wrote figures/error_growth_examples.png")

    # ------------------------------------------------ leg D reproduction ----
    d = read("leg_d_reproduction.csv")
    fig, ax = plt.subplots(figsize=(9, 5))
    groups, labels = [], []
    for w in widths:
        for mode in ("physics", "data"):
            for variant, marker in (("composite_two_steps", "o"),
                                    ("single_giant_step", "x")):
                v = [float(r["med_um"]) for r in d if r["width"] == w
                     and r["mode"] == mode and r["variant"] == variant]
                if v:
                    groups.append((v, marker,
                                   "#1f77b4" if mode == "physics" else "#d62728"))
                    labels.append("4x%s %s\n%s" % (w, mode,
                                                   "2 steps" if variant.startswith("comp")
                                                   else "1 step"))
    for i, (v, marker, colour) in enumerate(groups):
        ax.plot(np.full(len(v), i) + np.linspace(-0.14, 0.14, len(v)), v,
                marker, ms=6, color=colour)
        ax.plot([i - 0.28, i + 0.28], [np.median(v)] * 2, "-", lw=2.4,
                color=colour)
    straight = float(d[0]["straight_med_um"])
    ax.axhline(CEIL_D_Q8_UM, ls=":", color="green", lw=1.6,
               label="exact scheme, leg D, q=8 (%.0f um)" % CEIL_D_Q8_UM)
    ax.axhline(CEIL_D_Q16_UM, ls="-.", color="darkgreen", lw=1.4,
               label="exact scheme, leg D, q=16 (%.0f um)" % CEIL_D_Q16_UM)
    ax.axhline(straight, ls="--", color="0.4", lw=1.2,
               label="straight line (%.0f um)" % straight)
    ax.set_yscale("log")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("median error against the stored D-leg label [um]")
    ax.set_title("A3b  the downstream leg: two composite steps vs one giant step")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "leg_d_reproduction.png"), dpi=140)
    plt.close(fig)
    print("wrote figures/leg_d_reproduction.png")


if __name__ == "__main__":
    main()
