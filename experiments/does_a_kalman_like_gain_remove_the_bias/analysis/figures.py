"""The figures of the note, from the tables in results/. No computation.

    figures.py      writes figures/b_endpoint_by_band.png, figures/b_low_momentum_track.png
"""
from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS, FIGURES = os.path.join(HERE, "results"), os.path.join(HERE, "figures")
BANDS = ["< 3 GeV", "3-8 GeV", "8-20 GeV", "20-50 GeV", "> 50 GeV"]
BEFORE, AFTER, EXACT, TEXT, MUTED = "#eb6834", "#2a78d6", "#52514e", "#0b0b0b", "#c3c2b7"
SOURCES = ["straight line per-track", "straight line global", "direct states",
           "straight line (no network)", "explicit Euler (no network)"]
# The exact 2-stage Gauss-Legendre scheme, 64 steps (per-step bias study, c_comparators.csv).
EXACT_BY_BAND = [0.66, 0.25, 0.085, 0.035, 0.012]


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color=MUTED, linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT, labelsize=8)


def endpoint_by_band():
    rows = list(csv.DictReader(open(os.path.join(RESULTS, "b_endpoint.csv"))))
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.6), sharey=True)
    x = np.arange(len(BANDS))
    for ax, source in zip(axes, SOURCES):
        get = {r["variant"]: r for r in rows if r["source"] == source}
        before = [float(get["plain"]["median_um_" + b]) for b in BANDS]
        after = [float(get["B1 corrected, chained"]["median_um_" + b]) for b in BANDS]
        ax.bar(x - 0.2, before, 0.38, color=BEFORE, label="no correction")
        ax.bar(x + 0.2, after, 0.38, color=AFTER, label="gain, chained (B1)")
        ax.plot(x, EXACT_BY_BAND, "_", color=EXACT, markersize=18, markeredgewidth=2,
                label="exact scheme")
        ax.set_yscale("log")
        ax.set_xticks(x, ["<3", "3-8", "8-20", "20-50", ">50"])
        ax.set_xlabel("momentum band (GeV)", fontsize=8, color=TEXT)
        ax.set_title(source, fontsize=9, color=TEXT)
        style(ax)
    axes[0].set_ylabel("endpoint radial median (um)", fontsize=9, color=TEXT)
    axes[0].legend(fontsize=7, frameon=False, loc="upper right")
    fig.suptitle("Endpoint error at z = 7826 mm, 1,452 test tracks, 64 steps", fontsize=10,
                 color=TEXT)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "b_endpoint_by_band.png"), dpi=150)


def low_momentum_track():
    t = np.load(os.path.join(RESULTS, "b_low_momentum_track.npz"))
    z = t["z_start_mm"] + 80.903 / 2
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for ax, source in zip(axes, SOURCES[:3]):
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.plot(z, t[source + " | plain"], color=BEFORE, linewidth=2, marker="o", markersize=3,
                label="no correction")
        ax.plot(z, t[source + " | gain"], color=AFTER, linewidth=2, marker="o", markersize=3,
                label="with the gain")
        ax.plot(z, t["exact"], color=EXACT, linewidth=1, linestyle="--", label="exact scheme")
        ax.set_title(source, fontsize=9, color=TEXT)
        ax.set_xlabel("z of the step centre (mm)", fontsize=8, color=TEXT)
        style(ax)
    axes[0].set_ylabel("signed single-step error in x (um)", fontsize=9, color=TEXT)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("One test track, p = %.2f GeV (median track below 3 GeV): each step from the "
                 "reference state" % float(t["momentum_gev"]), fontsize=10, color=TEXT)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "b_low_momentum_track.png"), dpi=150)


if __name__ == "__main__":
    os.makedirs(FIGURES, exist_ok=True)
    endpoint_by_band()
    low_momentum_track()
