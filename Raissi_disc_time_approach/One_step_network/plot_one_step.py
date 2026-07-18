#!/usr/bin/env python
"""Figures for the first one-step network attempt.
Run:  /data/bfys/gscriven/conda/envs/TE/bin/python plot_one_step.py"""
import csv
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES, FIG = os.path.join(HERE, "results"), os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

SURFACE, TEXT1, TEXT2 = "#fcfcfb", "#0b0b0b", "#52514e"
BLUE, GREEN, MAGENTA, YELLOW, NEUTRAL = "#2a78d6", "#008300", "#e87ba4", "#eda100", "#c9c8c2"
MODE_COLOR = {"physics": BLUE, "data": GREEN}
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": TEXT1, "axes.edgecolor": TEXT2, "axes.labelcolor": TEXT1,
    "xtick.color": TEXT2, "ytick.color": TEXT2, "axes.grid": True,
    "grid.color": "#e7e6e1", "grid.linewidth": 0.6, "axes.axisbelow": True, "font.size": 10.5,
})

with open(os.path.join(RES, "summary.csv")) as f:
    summary = list(csv.DictReader(f))
preds = np.load(os.path.join(RES, "predictions.npz"))
P = preds["test_P"]

CEILING_UM = 29.0   # exact-scheme median on B legs at q=8 (Simple_first_pass)
STRAIGHT_UM = float(summary[0]["straight_med_um"])

fig, ax = plt.subplots(1, 3, figsize=(14, 4.6))

# (1) endpoint error distributions per mode (best seed by median)
best = {}
for mode in ("physics", "data"):
    rows = [s for s in summary if s["mode"] == mode]
    best[mode] = min(rows, key=lambda s: float(s["endpoint_med_um"]))
bins = np.logspace(0, np.log10(max(STRAIGHT_UM * 3, 1e5)), 50)
for mode in ("physics", "data"):
    e = preds["%s_seed%s_end_err_um" % (mode, best[mode]["seed"])]
    ax[0].hist(np.clip(e, 1, bins[-1]), bins=bins, histtype="step", lw=2,
               color=MODE_COLOR[mode],
               label="%s (median %.0f um)" % (mode, float(best[mode]["endpoint_med_um"])))
ax[0].axvline(CEILING_UM, color=TEXT1, lw=1.5, ls=":")
ax[0].text(CEILING_UM * 1.15, 0.95, "exact-scheme\nceiling 29 um",
           transform=ax[0].get_xaxis_transform(), fontsize=8, color=TEXT2, va="top")
ax[0].axvline(STRAIGHT_UM, color=NEUTRAL, lw=1.5)
ax[0].text(STRAIGHT_UM * 1.15, 0.75, "straight line",
           transform=ax[0].get_xaxis_transform(), fontsize=8, color=TEXT2, va="top")
ax[0].set_xscale("log")
ax[0].set_xlabel("endpoint error vs fp64 RK4 [um]"); ax[0].set_ylabel("test states")
ax[0].legend(fontsize=8.5)
ax[0].set_title("held-out endpoint error (best seed per mode)", fontsize=10.5)

# (2) error vs momentum
pg = np.logspace(0, np.log10(200), 16)
for mode in ("physics", "data"):
    e = preds["%s_seed%s_end_err_um" % (mode, best[mode]["seed"])]
    med = [np.median(e[(P >= a) & (P < b_)]) if ((P >= a) & (P < b_)).any() else np.nan
           for a, b_ in zip(pg[:-1], pg[1:])]
    ax[1].plot(np.sqrt(pg[:-1] * pg[1:]), med, marker="o", ms=4, lw=2,
               color=MODE_COLOR[mode], label=mode)
ax[1].axhline(CEILING_UM, color=TEXT1, lw=1.5, ls=":")
ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set_xlabel("|p| [GeV]"); ax[1].set_ylabel("median endpoint error [um]")
ax[1].legend(fontsize=8.5)
ax[1].set_title("error vs momentum", fontsize=10.5)

# (3) all seeds: endpoint median + stage median
xpos, lbl = [], []
for i, s in enumerate(summary):
    color = MODE_COLOR[s["mode"]]
    ax[2].plot(i, float(s["endpoint_med_um"]), marker="o", ms=8, color=color)
    ax[2].plot(i, float(s["stage_med_um"]), marker="s", ms=5, color=color, alpha=0.5)
    xpos.append(i); lbl.append("%s\ns%s" % (s["mode"][:4], s["seed"]))
ax[2].axhline(CEILING_UM, color=TEXT1, lw=1.5, ls=":")
ax[2].set_xticks(xpos); ax[2].set_xticklabels(lbl, fontsize=8)
ax[2].set_yscale("log"); ax[2].set_ylabel("median error [um]")
ax[2].set_title("all runs (o endpoint, squares stages)", fontsize=10.5)

fig.suptitle(
    "Step 2 first attempt - one implicit q=8 step across the magnet, "
    "physics-only vs data twin (test split, n=%s)" % summary[0]["n"], fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "one_step_results.png"), dpi=150)
print("wrote", os.path.join(FIG, "one_step_results.png"))
