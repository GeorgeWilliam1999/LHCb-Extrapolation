#!/usr/bin/env python
"""Figures for the exact-scheme scan: the ceiling curves and the cost.
Run:  /data/bfys/gscriven/conda/envs/TE/bin/python plot_scheme.py"""
import csv
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

SURFACE, TEXT1, TEXT2 = "#fcfcfb", "#0b0b0b", "#52514e"
BLUE, GREEN, MAGENTA, YELLOW, NEUTRAL = "#2a78d6", "#008300", "#e87ba4", "#eda100", "#c9c8c2"
LEG_COLOR = {"A": BLUE, "B": GREEN, "C": MAGENTA, "D": YELLOW}
LEG_LABEL = {"A": "A vertex fetch", "B": "B cross-magnet (one step!)",
             "C": "C plane-to-plane", "D": "D downstream->PV"}
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": TEXT1, "axes.edgecolor": TEXT2, "axes.labelcolor": TEXT1,
    "xtick.color": TEXT2, "ytick.color": TEXT2, "axes.grid": True,
    "grid.color": "#e7e6e1", "grid.linewidth": 0.6, "axes.axisbelow": True, "font.size": 10.5,
})

with open(os.path.join(HERE, "results", "scheme_error_vs_q.csv")) as f:
    agg = list(csv.DictReader(f))
with open(os.path.join(HERE, "results", "scheme_scan.csv")) as f:
    scan = list(csv.DictReader(f))

fig, ax = plt.subplots(1, 3, figsize=(14, 4.6))

for leg in "ABCD":
    qs = [int(a["q"]) for a in agg if a["leg"] == leg]
    med = [float(a["median_err_mm"]) * 1e3 for a in agg if a["leg"] == leg]   # um
    worst = [float(a["worst_err_mm"]) * 1e3 for a in agg if a["leg"] == leg]
    ax[0].plot(qs, med, marker="o", ms=5, lw=2, color=LEG_COLOR[leg], label=LEG_LABEL[leg])
    ax[0].plot(qs, worst, marker="o", ms=3, lw=1, ls="--", color=LEG_COLOR[leg], alpha=0.5)
ax[0].axhline(30, color=NEUTRAL, lw=1)
ax[0].text(2.1, 34, "~30 um: the C0-field floor", fontsize=8.5, color=TEXT2)
ax[0].set_xscale("log", base=2); ax[0].set_yscale("log")
ax[0].set_xticks([2, 4, 8, 16, 32]); ax[0].set_xticklabels([2, 4, 8, 16, 32])
ax[0].set_xlabel("stages q"); ax[0].set_ylabel("endpoint error vs fp64 RK4 [um]")
ax[0].legend(fontsize=8)
ax[0].set_title("the ceiling: exact scheme error (solid median, dashed worst)", fontsize=10.5)

# error vs momentum at q=8, per leg
for leg in "ABCD":
    rows = [r for r in scan if r["leg"] == leg and r["q"] == "8" and r["converged"] == "1"]
    p = np.array([float(r["p_GeV"]) for r in rows])
    e = np.array([max(float(r["err_x_mm"]), float(r["err_y_mm"])) * 1e3 for r in rows])
    o = np.argsort(p)
    ax[1].plot(p[o], np.maximum(e[o], 1e-6), marker=".", ms=4, lw=1,
               color=LEG_COLOR[leg], alpha=0.8)
ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set_xlabel("|p| [GeV]"); ax[1].set_ylabel("endpoint error at q = 8 [um]")
ax[1].set_title("who is hard: soft tracks on long legs", fontsize=10.5)

# cost: function evaluations vs q
for leg in "ABCD":
    qs = [int(a["q"]) for a in agg if a["leg"] == leg]
    nev = [int(a["median_nev"]) for a in agg if a["leg"] == leg]
    ax[2].plot(qs, nev, marker="o", ms=5, lw=2, color=LEG_COLOR[leg])
ax[2].set_xscale("log", base=2)
ax[2].set_xticks([2, 4, 8, 16, 32]); ax[2].set_xticklabels([2, 4, 8, 16, 32])
ax[2].set_xlabel("stages q"); ax[2].set_ylabel("median residual evaluations per solve")
ax[2].set_title("root-finder cost (what the network replaces)", fontsize=10.5)

fig.suptitle(
    "Step 1 — the exact Gauss-Legendre scheme on real LHCb legs: 640/640 solves converged, "
    "one implicit step crosses the magnet at ~30 um", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "scheme_error_vs_q.png"), dpi=150)
print("wrote", os.path.join(FIG, "scheme_error_vs_q.png"))
