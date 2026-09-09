#!/usr/bin/env python
"""Figures for the one-step network, trained to stall (continue_training.py).
Compares against the budget-6 first attempt snapshot in results/budget6/.
Run:  /data/bfys/gscriven/conda/envs/TE/bin/python plot_one_step.py"""
import csv
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES, FIG = os.path.join(HERE, "results"), os.path.join(HERE, "figures")
B6 = os.path.join(RES, "budget6")
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


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


summary = read_csv(os.path.join(RES, "summary.csv"))
hist = read_csv(os.path.join(RES, "histories.csv"))
preds = np.load(os.path.join(RES, "predictions.npz"))
P = preds["test_P"]
b6_summary = read_csv(os.path.join(B6, "summary.csv"))
b6_preds = np.load(os.path.join(B6, "predictions.npz"))

CEILING_UM = 29.0   # exact-scheme median on B legs at q=8 (S1_Simple_first_pass)
STRAIGHT_UM = float(summary[0]["straight_med_um"])

fig, ax = plt.subplots(2, 2, figsize=(13.5, 9.2))
ax = ax.ravel()

# (1) endpoint error distributions per mode (best seed by median), vs budget-6
best, b6_best = {}, {}
for mode in ("physics", "data"):
    best[mode] = min((s for s in summary if s["mode"] == mode),
                     key=lambda s: float(s["endpoint_med_um"]))
    b6_best[mode] = min((s for s in b6_summary if s["mode"] == mode),
                        key=lambda s: float(s["endpoint_med_um"]))
bins = np.logspace(0, np.log10(max(STRAIGHT_UM * 3, 1e5)), 50)
for mode in ("physics", "data"):
    e6 = b6_preds["%s_seed%s_end_err_um" % (mode, b6_best[mode]["seed"])]
    ax[0].hist(np.clip(e6, 1, bins[-1]), bins=bins, histtype="step", lw=1.2,
               ls="--", color=MODE_COLOR[mode], alpha=0.55,
               label="%s, 6-restart cap (med %.0f um)"
               % (mode, float(b6_best[mode]["endpoint_med_um"])))
for mode in ("physics", "data"):
    e = preds["%s_seed%s_end_err_um" % (mode, best[mode]["seed"])]
    ax[0].hist(np.clip(e, 1, bins[-1]), bins=bins, histtype="step", lw=2,
               color=MODE_COLOR[mode],
               label="%s, to stall (med %.0f um)"
               % (mode, float(best[mode]["endpoint_med_um"])))
ax[0].axvline(CEILING_UM, color=TEXT1, lw=1.5, ls=":")
ax[0].text(CEILING_UM * 1.15, 0.95, "exact-scheme\nceiling 29 um",
           transform=ax[0].get_xaxis_transform(), fontsize=8, color=TEXT2, va="top")
ax[0].axvline(STRAIGHT_UM, color=NEUTRAL, lw=1.5)
ax[0].text(STRAIGHT_UM * 1.15, 0.75, "straight line",
           transform=ax[0].get_xaxis_transform(), fontsize=8, color=TEXT2, va="top")
ax[0].set_xscale("log")
ax[0].set_xlabel("endpoint error vs fp64 RK4 [um]"); ax[0].set_ylabel("test states")
ax[0].legend(fontsize=8)
ax[0].set_title("held-out endpoint error (best seed per mode)", fontsize=10.5)

# (2) training loss vs restart — the convergence evidence
for mode in ("physics", "data"):
    for seed in ("0", "1", "2"):
        rows = [r for r in hist if r["mode"] == mode and r["seed"] == seed]
        rows.sort(key=lambda r: int(r["outer"]))
        ax[1].plot([int(r["outer"]) for r in rows], [float(r["loss"]) for r in rows],
                   color=MODE_COLOR[mode], lw=1.6, alpha=0.8,
                   label=mode if seed == "0" else None)
ax[1].axvline(5.5, color=TEXT2, lw=1.2, ls="--")
ax[1].text(5.7, 0.96, "first-attempt cap\n(6 restarts)",
           transform=ax[1].get_xaxis_transform(), fontsize=8, color=TEXT2, va="top")
ax[1].set_yscale("log")
ax[1].set_xlabel("L-BFGS restart"); ax[1].set_ylabel("training loss")
ax[1].legend(fontsize=8.5)
ax[1].set_title("loss histories: all six runs, to stall", fontsize=10.5)

# (3) error vs momentum
pg = np.logspace(0, np.log10(200), 16)
for mode in ("physics", "data"):
    e = preds["%s_seed%s_end_err_um" % (mode, best[mode]["seed"])]
    med = [np.median(e[(P >= a) & (P < b_)]) if ((P >= a) & (P < b_)).any() else np.nan
           for a, b_ in zip(pg[:-1], pg[1:])]
    ax[2].plot(np.sqrt(pg[:-1] * pg[1:]), med, marker="o", ms=4, lw=2,
               color=MODE_COLOR[mode], label=mode)
ax[2].axhline(CEILING_UM, color=TEXT1, lw=1.5, ls=":")
ax[2].set_xscale("log"); ax[2].set_yscale("log")
ax[2].set_xlabel("|p| [GeV]"); ax[2].set_ylabel("median endpoint error [um]")
ax[2].legend(fontsize=8.5)
ax[2].set_title("error vs momentum (to stall)", fontsize=10.5)

# (4) all seeds: endpoint + stage medians, stalled vs budget-6
xpos, lbl = [], []
for i, s in enumerate(summary):
    color = MODE_COLOR[s["mode"]]
    s6 = next(r for r in b6_summary
              if r["mode"] == s["mode"] and r["seed"] == s["seed"])
    ax[3].plot(i, float(s6["endpoint_med_um"]), marker="o", ms=8, mfc="none",
               color=color, alpha=0.55)
    ax[3].plot(i, float(s["endpoint_med_um"]), marker="o", ms=8, color=color)
    ax[3].plot(i, float(s["stage_med_um"]), marker="s", ms=5, color=color, alpha=0.5)
    xpos.append(i); lbl.append("%s\ns%s" % (s["mode"][:4], s["seed"]))
ax[3].axhline(CEILING_UM, color=TEXT1, lw=1.5, ls=":")
ax[3].set_xticks(xpos); ax[3].set_xticklabels(lbl, fontsize=8)
ax[3].set_yscale("log"); ax[3].set_ylabel("median error [um]")
ax[3].set_title("all runs (o endpoint: open = 6-restart cap, filled = to stall; "
                "squares = stages)", fontsize=9.5)

fig.suptitle(
    "Step 2 - one implicit q=8 step across the magnet, trained to stall: "
    "physics-only vs data twin (test split, n=%s)" % summary[0]["n"], fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "one_step_results.png"), dpi=150)
print("wrote", os.path.join(FIG, "one_step_results.png"))
