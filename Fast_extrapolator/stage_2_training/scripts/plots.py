"""The figure of the preparation of stage 2, from the tables in results/.

    figures/what_a_smooth_correction_reaches.png
        for each start: its median error alone, and with the polynomial
        correction of degree 7 (what_must_a_correction_learn.py), on the
        validation tracks; one panel a reference

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
LHCB_MEDIAN_UM = {"cubic": 25.9, "linear": 10.1}     # stage 1: the adaptive Nystrom integrator of LHCb


def main():
    size = pd.read_csv(os.path.join(HERE, "results", "what_a_correction_must_learn.csv"))
    probe = pd.read_csv(os.path.join(HERE, "results", "polynomial_probe.csv"))
    probe = probe[(probe["sample"] == "validation") & (probe["degree"] == 7)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharey=True,
                             gridspec_kw=dict(width_ratios=[9, 4]))
    for ax, reference in zip(axes, ("cubic", "linear")):
        rows = size[size["reference"] == reference].reset_index(drop=True)
        x = np.arange(len(rows))
        alone = rows["radial_median_um"].to_numpy()
        corrected = np.array([probe[(probe["reference"] == reference) & (probe["start"] == s)]
                              ["radial_median_um"].iloc[0] for s in rows["start"]])
        tail = np.array([probe[(probe["reference"] == reference) & (probe["start"] == s)]
                         ["radial_95_um"].iloc[0] for s in rows["start"]])
        ax.bar(x - 0.2, alone, 0.4, color="#9e9e9e", label="the start alone, median")
        ax.bar(x + 0.2, corrected, 0.4, color="#1f77b4", label="with a smooth correction, median")
        ax.scatter(x + 0.2, tail, marker="_", s=260, color="#0b3c5d", lw=2,
                   label="with a smooth correction, 95 % of the tracks")
        for i, (a, c) in enumerate(zip(alone, corrected)):
            ax.text(i - 0.2, min(a, 2.2e3) * 1.12, "%.3g" % a if a < 1e4 else "%.0f mm" % (a / 1e3),
                    ha="center", fontsize=7.5, color="#555555")
            ax.text(i + 0.2, c * 1.12, "%.2g" % c, ha="center", fontsize=7.5, color="#1f77b4")
        ax.axhline(1.0, color="#2ca02c", lw=1.0, ls=":")
        ax.text(0.99, 1.1, "target A: median 1 um", color="#2ca02c", fontsize=8, ha="right",
                transform=ax.get_yaxis_transform())
        ax.axhline(LHCB_MEDIAN_UM[reference], color="#d62728", lw=1.0, ls="--")
        ax.text(0.99, LHCB_MEDIAN_UM[reference] * 1.1, "LHCb adaptive Nystrom: %g um"
                % LHCB_MEDIAN_UM[reference], color="#d62728", fontsize=8, ha="right",
                transform=ax.get_yaxis_transform())
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("the straight line", "straight line\n(no field read)")
                            .replace("Nystrom, ", "").replace(" equal steps, ", " steps\n")
                            .replace(" reading", " read") + ("\n%d reads" % r if r else "")
                            for s, r in zip(rows["start"], rows["field_reads"])], fontsize=8)
        ax.set_yscale("log")
        ax.set_ylim(0.5, 3e3)
        ax.set_title("reference: the %s reading of the map%s"
                     % (reference, " (this project)" if reference == "cubic" else " (LHCb)"), fontsize=10)
        ax.grid(True, which="both", axis="y", alpha=0.25)
    axes[0].set_ylabel("radial error on the last plane (um), 13,684 validation tracks")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("What a smooth correction of the start state reaches (a polynomial of degree 7, "
                 "no network)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "what_a_smooth_correction_reaches.png"), dpi=150)


if __name__ == "__main__":
    main()
