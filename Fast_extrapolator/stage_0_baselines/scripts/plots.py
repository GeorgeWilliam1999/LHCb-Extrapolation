"""The figures of stage 0, from the tables in results/.

    figures/time_against_error.png        compiled time a track against the median error,
                                          every method, the full test split
    figures/cost_of_a_network_call.png    the time of one network call against its width,
                                          beside one field read and one LHCb crossing
    figures/error_by_momentum.png         the median error in the five momentum bands for
                                          the LHCb methods and the best present networks

    plots.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS, FIGURES = os.path.join(HERE, "results"), os.path.join(HERE, "figures")
COLOURS = {"LHCb": "#d62728", "classical": "#555555", "network": "#1f77b4"}
BANDS = [("0_3", "below 3"), ("3_8", "3 to 8"), ("8_20", "8 to 20"), ("20_50", "20 to 50"),
         ("above_50", "above 50")]


def time_against_error(m):
    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    # one point a method: the fastest setting of each (single precision and fast tanh where it has them)
    best = m.sort_values("ns_lowest").groupby("label", as_index=False).first()
    for family, rows in best.groupby("family"):
        ax.scatter(rows["ns_lowest"], rows["radial_median_um"], s=46, color=COLOURS[family],
                   edgecolor="k", linewidth=0.4, label=family, zorder=3)
    rk4 = best[best["label"].str.startswith("RK4") & best["label"].str.endswith("linear map")]
    rk4 = rk4.sort_values("field_reads_mean")
    ax.plot(rk4["ns_lowest"], rk4["radial_median_um"], color="#999999", lw=0.9, ls="--", zorder=2)
    notes = {
        "HLT2 adaptive Nystrom (TrackSTEPExtrapolator, tolerance 5 um)": "HLT2 adaptive Nystrom",
        "HLT2 adaptive Cash-Karp (TrackRungeKuttaExtrapolator, tolerance 1 um)": "HLT2 adaptive Cash-Karp",
        "HLT1 parametrised step (polynomials, no field read)": "HLT1 parametrised step",
        "HLT1 T-track integrator (fixed Nystrom, 500 mm, 1 read a step)": "HLT1 T-track integrator",
        "HLT1 adaptive Nystrom (Allen, tolerance 5 um)": "HLT1 adaptive Nystrom (Allen)",
        "Whole-crossing network, 2x256, linear map": "whole-crossing network, 2x256",
        "Physics layer (1 sample), 16 steps, 2x128, linear map": "physics layer, 16 steps",
        "Physics layer (1 sample), 32 steps, 2x128, cubic map": "physics layer, 32 steps",
        "Physics layer (1 sample), 64 steps, 2x32, linear map": "physics layer, 64 steps, 2x32",
        "Chain, straight line, per-track scale (1 sample), 8 steps, 2x128, linear map": "chain, 8 steps",
        "Chain, straight line, per-track scale (1 sample), 16 steps, 2x128, linear map": "chain, 16 steps",
        "Chain, straight line, per-track scale (1 sample), 64 steps, 2x128, linear map": "chain, 64 steps",
        "RK4, 8 steps, linear map": "RK4, 8 steps", "RK4, 16 steps, linear map": "RK4, 16 steps",
        "RK4, 32 steps, linear map": "RK4, 32 steps", "RK4, 4 steps, linear map": "RK4, 4 steps",
    }
    offsets = {"HLT1 adaptive Nystrom (Allen, tolerance 5 um)": (-128, -13),
               "HLT2 adaptive Nystrom (TrackSTEPExtrapolator, tolerance 5 um)": (7, 4)}
    for _, r in best.iterrows():
        if r["label"] in notes:
            ax.annotate(notes[r["label"]], (r["ns_lowest"], r["radial_median_um"]),
                        textcoords="offset points", xytext=offsets.get(r["label"], (6, 5)),
                        fontsize=7.5)
    ax.axhline(1.0, color="#2ca02c", lw=1.0, ls=":")
    ax.text(0.99, 1.12, "target A: median 1 um", color="#2ca02c", fontsize=8, ha="right",
            transform=ax.get_yaxis_transform())
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("compiled time for one track (ns), one thread, lowest of 15 passes")
    ax.set_ylabel("median radial error on the last plane (um), 13,447 test tracks")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "time_against_error.png"), dpi=150)
    plt.close(fig)


def cost_of_a_network_call(mlp, field, m):
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    styles = {("float", "tanh from the vector exponential"): ("#1f77b4", "-", "single precision, fast tanh"),
              ("double", "tanh from the vector exponential"): ("#1f77b4", "--", "double precision, fast tanh"),
              ("float", "tanh of the C library"): ("#ff7f0e", "-", "single precision, tanh of the C library"),
              ("float", "none"): ("#999999", ":", "single precision, layers only")}
    two = mlp[mlp["depth"] == 2]
    for (precision, activation), (colour, style, label) in styles.items():
        rows = two[(two["precision"] == precision) & (two["activation"] == activation)].sort_values("width")
        ax.plot(rows["width"], rows["ns_lowest"], color=colour, ls=style, marker="o", ms=4, label=label)
    read = field[(field["reader"] == "rec") & (field["precision"] == "float")
                 & (field["pattern"] == "along the tracks")]["ns_lowest"].iloc[0]
    ax.axhline(read, color="#d62728", lw=0.9, ls=":")
    ax.text(8.2, read * 1.12, "one field read of LHCb (%.0f ns)" % read, color="#d62728", fontsize=8)
    for label, name in (("HLT2 adaptive Nystrom (TrackSTEPExtrapolator, tolerance 5 um)",
                         "one crossing of the HLT2 adaptive Nystrom"),
                        ("HLT2 adaptive Cash-Karp (TrackRungeKuttaExtrapolator, tolerance 1 um)",
                         "one crossing of the HLT2 adaptive Cash-Karp"),
                        ("HLT1 parametrised step (polynomials, no field read)",
                         "the HLT1 parametrised step")):
        time = m[m["label"] == label]["ns_lowest"].iloc[0]
        ax.axhline(time, color="#d62728", lw=0.9)
        ax.text(8.2, time * 1.08, "%s (%.0f ns)" % (name, time), color="#d62728", fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([8, 16, 32, 64, 128, 256, 512])
    ax.set_xticklabels(["8", "16", "32", "64", "128", "256", "512"])
    ax.set_xlabel("width of the two hidden layers (6 inputs, 8 outputs)")
    ax.set_ylabel("time of one network call (ns)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "cost_of_a_network_call.png"), dpi=150)
    plt.close(fig)


def error_by_momentum(m):
    chosen = [("HLT2 adaptive Nystrom (TrackSTEPExtrapolator, tolerance 5 um)", "double", "HLT2 adaptive Nystrom (56 reads)", "#d62728"),
              ("HLT2 adaptive Cash-Karp (TrackRungeKuttaExtrapolator, tolerance 1 um)", "double", "HLT2 adaptive Cash-Karp (43 reads)", "#e377c2"),
              ("HLT1 parametrised step (polynomials, no field read)", "float", "HLT1 parametrised step (0 reads)", "#ff9896"),
              ("RK4, 8 steps, linear map", "double", "RK4, 8 steps (32 reads)", "#555555"),
              ("RK4, 16 steps, linear map", "double", "RK4, 16 steps (64 reads)", "#999999"),
              ("Whole-crossing network, 2x256, linear map", "double", "whole-crossing network (0 reads)", "#aec7e8"),
              ("Physics layer (1 sample), 16 steps, 2x128, linear map", "double", "physics layer, 16 steps (80 reads)", "#1f77b4")]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    width = 0.115
    for k, (label, precision, name, colour) in enumerate(chosen):
        row = m[(m["label"] == label) & (m["precision"] == precision)].iloc[0]
        values = [row["radial_median_um_%s_gev" % key] for key, _ in BANDS]
        ax.bar(np.arange(len(BANDS)) + (k - 3) * width, values, width, color=colour, label=name,
               edgecolor="k", linewidth=0.3)
    ax.axhline(3.0, color="#2ca02c", lw=1.0, ls=":")
    ax.text(4.45, 3.4, "target A: 3 um in every band", color="#2ca02c", fontsize=8, ha="right")
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(BANDS)))
    ax.set_xticklabels([name + " GeV" for _, name in BANDS])
    ax.set_ylabel("median radial error on the last plane (um)")
    ax.grid(True, axis="y", which="both", alpha=0.25)
    ax.legend(fontsize=8.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "error_by_momentum.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    methods = pd.read_csv(os.path.join(RESULTS, "methods.csv"))
    methods = methods[~methods["label"].str.contains("flat sample only|transport matrix")]
    time_against_error(methods)
    cost_of_a_network_call(pd.read_csv(os.path.join(RESULTS, "micro_mlp.csv")),
                           pd.read_csv(os.path.join(RESULTS, "micro_field.csv")), methods)
    error_by_momentum(methods)
