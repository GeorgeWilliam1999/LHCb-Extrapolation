"""The figures of the results note, from the tables in results/ and the cache.

    figures.py      writes figures/along_the_track.png, figures/a5_share_by_z.png,
                    figures/a7_toy_fit.png, figures/c_variants.png
"""
from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                           # noqa: E402
import numpy as np                                                        # noqa: E402

from common import (FIGURES, RESULTS, RUNS, SHORT, START_PLANES_MM, BANDS,   # noqa: E402
                    bend_sign_of_steps, cached_single_step_errors, split)
from part_a import exact_errors                                          # noqa: E402

COLOURS = {"25dfa17ef168": "#1f77b4", "9cef9c160b7d": "#ff7f0e", "49e5e2450eaa": "#2ca02c"}


def read(file):
    with open(os.path.join(RESULTS, file)) as handle:
        return list(csv.DictReader(handle))


def save(fig, name):
    os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(os.path.join(FIGURES, name), dpi=120)
    plt.close(fig)
    print("written: figures/%s" % name)


def along_the_track():
    """e_x along the bend against z for six test tracks of 3-8 GeV, each run and the
    exact scheme: the error wanders slowly along a track, with a sign of its own."""
    test = split("test")
    pick = np.flatnonzero((test.momentum_gev > 3) & (test.momentum_gev < 8))[:6]
    bend = bend_sign_of_steps("test")
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    for ax, t in zip(axes.ravel(), pick):
        for _, key in RUNS:
            e = cached_single_step_errors(key, "test")
            ax.plot(START_PLANES_MM, e[t, :, 0] * bend[t], color=COLOURS[key], label=SHORT[key])
        ax.plot(START_PLANES_MM, exact_errors("test")[t, :, 0] * bend[t], "k:", label="exact scheme")
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_title("test track %d, p = %.1f GeV" % (t, test.momentum_gev[t]))
    for ax in axes[1]:
        ax.set_xlabel("z of the start of the step (mm)")
    for ax in axes[:, 0]:
        ax.set_ylabel("e_x along the bend (um)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Single-step error in x along the bend, along six tracks")
    fig.tight_layout()
    save(fig, "along_the_track.png")


def share_by_z():
    rows = read("a5_endpoint_share_by_z.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, comp in zip(axes, ("x", "y")):
        for _, key in RUNS:
            r = [x for x in rows if x["run_key"] == key]
            centres = [0.5 * (float(x["z_from_mm"]) + float(x["z_to_mm"])) for x in r]
            ax.plot(centres, [float(x["share_of_endpoint_variance_%s" % comp]) for x in r],
                    "o-", color=COLOURS[key], label=SHORT[key])
        ax.axhline(1 / 8, color="grey", ls="--", lw=0.8, label="even share (1/8)")
        ax.set_xlabel("z bin of 8 steps (mm)")
        ax.set_title("share of the variance of the endpoint error in %s" % comp)
    axes[0].set_ylabel("share")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    save(fig, "a5_share_by_z.png")


def toy_fit():
    rows = [r for r in read("a7_toy_fit_by_band.csv") if r["band"] != "all"]
    labels = [b[0] for b in BANDS]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(labels))
    for j, (_, key) in enumerate(RUNS):
        r = {row["band"]: row for row in rows if row["run_key"] == key}
        ratio = [float(r[b]["mean_bias_over_resolution"]) for b in labels]
        width = [float(r[b]["pull_width_network"]) for b in labels]
        axes[0].bar(x + (j - 1) * 0.27, ratio, 0.27, color=COLOURS[key], label=SHORT[key])
        axes[1].bar(x + (j - 1) * 0.27, width, 0.27, color=COLOURS[key], label=SHORT[key])
    for s in (0.1, -0.1):
        axes[0].axhline(s, color="red", ls="--", lw=0.8)
    axes[0].set_ylabel("mean bias dp/p / momentum resolution")
    axes[1].axhline(1.0, color="black", lw=0.8)
    axes[1].set_ylabel("pull width of q/p, network fit")
    for ax in axes:
        ax.set_xticks(x, labels)
        ax.set_xlabel("momentum band")
    axes[0].legend(fontsize=8)
    fig.suptitle("Toy UT + SciFi fit: the network against the reference (hits only, no scattering)")
    fig.tight_layout()
    save(fig, "a7_toy_fit.png")


def variants():
    fig, ax = plt.subplots(figsize=(12, 5.5))
    names = []
    for j, (_, key) in enumerate(RUNS):
        file = "c_variants_%s.csv" % key
        if not os.path.exists(os.path.join(RESULTS, file)):
            continue
        rows = read(file)
        names = [r["variant"] for r in rows]
        y = np.arange(len(rows))
        ax.barh(y + (j - 1) * 0.27, [float(r["endpoint_radial_median_um"]) for r in rows], 0.27,
                color=COLOURS[key], label=SHORT[key])
    comparators = {r["series"]: float(r["endpoint_radial_median_um"]) for r in read("c_comparators.csv")}
    for name, value in comparators.items():
        ax.axvline(value, ls="--", color="black" if "RK4" in name else "grey", lw=0.8)
        ax.text(value, len(names) - 0.4, " " + name.split(" (")[0], fontsize=7, rotation=90,
                va="top")
    ax.set_yticks(np.arange(len(names)), names)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("test endpoint radial median (um)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    save(fig, "c_variants.png")


if __name__ == "__main__":
    along_the_track()
    share_by_z()
    if os.path.exists(os.path.join(RESULTS, "a7_toy_fit_by_band.csv")):
        toy_fit()
    variants()
