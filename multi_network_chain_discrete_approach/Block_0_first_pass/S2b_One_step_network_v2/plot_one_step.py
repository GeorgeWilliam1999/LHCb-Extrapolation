#!/usr/bin/env python
"""Figures for the v2 one-step network (official-sample data), trained to stall.

Six panels: the converged result against v1, the loss histories, the momentum
dependence, the per-run comparison, and the fiducial story — what the ~1% of
trajectories leaving the field map did to the physics loss before they were cut.

Run:  /data/bfys/gscriven/conda/envs/TE/bin/python plot_one_step.py
"""
import csv
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES, FIG = os.path.join(HERE, "results"), os.path.join(HERE, "figures")
RAW = os.path.join(HERE, "results_nofiducial")
V1RES = os.path.join(HERE, "..", "S2_One_step_network", "results")
os.makedirs(FIG, exist_ok=True)

SURFACE, TEXT1, TEXT2 = "#fcfcfb", "#0b0b0b", "#52514e"
BLUE, GREEN, MAGENTA, NEUTRAL = "#2a78d6", "#008300", "#e87ba4", "#c9c8c2"
MODE_COLOR = {"physics": BLUE, "data": GREEN}
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": TEXT1, "axes.edgecolor": TEXT2, "axes.labelcolor": TEXT1,
    "xtick.color": TEXT2, "ytick.color": TEXT2, "axes.grid": True,
    "grid.color": "#e7e6e1", "grid.linewidth": 0.6, "axes.axisbelow": True, "font.size": 10.5,
})


def read_csv(p):
    with open(p) as f:
        return list(csv.DictReader(f))


summary = read_csv(os.path.join(RES, "summary.csv"))
hist = read_csv(os.path.join(RES, "histories.csv"))
preds = np.load(os.path.join(RES, "predictions.npz"))
P = preds["test_P"]
v1_summary = read_csv(os.path.join(V1RES, "summary.csv"))
v1_preds = np.load(os.path.join(V1RES, "predictions.npz"))
raw_summary = read_csv(os.path.join(RAW, "summary.csv"))

CEILING_UM = 29.0
STRAIGHT_UM = float(summary[0]["straight_med_um"])

fig, ax = plt.subplots(2, 3, figsize=(18.5, 9.4))
ax = ax.ravel()

best = {m: min((s for s in summary if s["mode"] == m),
               key=lambda s: float(s["endpoint_med_um"])) for m in ("physics", "data")}
v1_best = {m: min((s for s in v1_summary if s["mode"] == m),
                  key=lambda s: float(s["endpoint_med_um"])) for m in ("physics", "data")}

# (1) endpoint error distributions, v2 (solid) vs v1 (dashed)
bins = np.logspace(0, np.log10(max(STRAIGHT_UM * 3, 1e5)), 50)
for m in ("physics", "data"):
    e1 = v1_preds["%s_seed%s_end_err_um" % (m, v1_best[m]["seed"])]
    ax[0].hist(np.clip(e1, 1, bins[-1]), bins=bins, histtype="step", lw=1.2, ls="--",
               color=MODE_COLOR[m], alpha=0.55, density=True,
               label="%s, v1 our sample (%.0f um)" % (m, float(v1_best[m]["endpoint_med_um"])))
for m in ("physics", "data"):
    e = preds["%s_seed%s_end_err_um" % (m, best[m]["seed"])]
    ax[0].hist(np.clip(e, 1, bins[-1]), bins=bins, histtype="step", lw=2,
               color=MODE_COLOR[m], density=True,
               label="%s, v2 official (%.0f um)" % (m, float(best[m]["endpoint_med_um"])))
ax[0].axvline(CEILING_UM, color=TEXT1, lw=1.5, ls=":")
ax[0].text(CEILING_UM * 1.15, 0.96, "exact-scheme\nceiling 29 um",
           transform=ax[0].get_xaxis_transform(), fontsize=8, color=TEXT2, va="top")
ax[0].axvline(STRAIGHT_UM, color=NEUTRAL, lw=1.5)
ax[0].text(STRAIGHT_UM * 1.2, 0.7, "straight\nline", transform=ax[0].get_xaxis_transform(),
           fontsize=8, color=TEXT2, va="top")
ax[0].set_xscale("log"); ax[0].set_xlabel("endpoint error vs fp64 RK4 [um]")
ax[0].set_ylabel("density"); ax[0].legend(fontsize=7.5)
ax[0].set_title("held-out endpoint error (best seed per mode)", fontsize=10.5)

# (2) loss histories
for m in ("physics", "data"):
    for sd in ("0", "1", "2"):
        rows = sorted((r for r in hist if r["mode"] == m and r["seed"] == sd),
                      key=lambda r: int(r["outer"]))
        ax[1].plot([int(r["outer"]) for r in rows], [float(r["loss"]) for r in rows],
                   color=MODE_COLOR[m], lw=1.6, alpha=0.85, label=m if sd == "0" else None)
ax[1].set_yscale("log"); ax[1].set_xlabel("L-BFGS restart"); ax[1].set_ylabel("training loss")
ax[1].legend(fontsize=8.5); ax[1].set_title("loss histories, all six v2 runs to stall", fontsize=10.5)

# (3) error vs momentum
pg = np.logspace(0, np.log10(200), 16)
for m in ("physics", "data"):
    e = preds["%s_seed%s_end_err_um" % (m, best[m]["seed"])]
    med = [np.median(e[(P >= a) & (P < b_)]) if ((P >= a) & (P < b_)).any() else np.nan
           for a, b_ in zip(pg[:-1], pg[1:])]
    ax[2].plot(np.sqrt(pg[:-1] * pg[1:]), med, "o-", ms=4, lw=2, color=MODE_COLOR[m], label=m)
ax[2].axhline(CEILING_UM, color=TEXT1, lw=1.5, ls=":")
ax[2].set_xscale("log"); ax[2].set_yscale("log")
ax[2].set_xlabel("|p| [GeV]"); ax[2].set_ylabel("median endpoint error [um]")
ax[2].legend(fontsize=8.5); ax[2].set_title("error vs momentum (v2)", fontsize=10.5)

# (4) per-run medians, v2 filled vs v1 open
xpos, lbl = [], []
for i, s in enumerate(summary):
    col = MODE_COLOR[s["mode"]]
    s1 = next(r for r in v1_summary if r["mode"] == s["mode"] and r["seed"] == s["seed"])
    ax[3].plot(i, float(s1["endpoint_med_um"]), "o", ms=8, mfc="none", color=col, alpha=0.55)
    ax[3].plot(i, float(s["endpoint_med_um"]), "o", ms=8, color=col)
    ax[3].plot(i, float(s["stage_med_um"]), "s", ms=5, color=col, alpha=0.5)
    xpos.append(i); lbl.append("%s\ns%s" % (s["mode"][:4], s["seed"]))
ax[3].axhline(CEILING_UM, color=TEXT1, lw=1.5, ls=":")
ax[3].set_xticks(xpos); ax[3].set_xticklabels(lbl, fontsize=8)
ax[3].set_yscale("log"); ax[3].set_ylabel("median error [um]")
ax[3].set_title("per run (open = v1, filled = v2; squares = v2 stages)", fontsize=9.5)

# (5) the fiducial story
groups = [("v1\n(our sample)", v1_summary), ("v2 raw\n(out-of-map kept)", raw_summary),
          ("v2 fiducial\n(in-domain only)", summary)]
w = 0.36
for gi, (name, rows) in enumerate(groups):
    for mi, m in enumerate(("physics", "data")):
        vals = [float(r["endpoint_med_um"]) for r in rows if r["mode"] == m]
        ax[4].bar(gi + (mi - 0.5) * w, np.median(vals), width=w, color=MODE_COLOR[m],
                  alpha=0.85, label=m if gi == 0 else None)
        ax[4].plot([gi + (mi - 0.5) * w] * len(vals), vals, "o", ms=4, color=TEXT1, alpha=0.6)
ax[4].set_xticks(range(len(groups)))
ax[4].set_xticklabels([g[0] for g in groups], fontsize=8.5)
ax[4].set_ylabel("median endpoint error [um]"); ax[4].legend(fontsize=8.5)
ax[4].set_title("the fiducial effect (bars = seed median, dots = seeds)", fontsize=10)

# (6) what was cut: the trajectories that leave the field map
raw = np.load(os.path.join(RAW, "frozen_leg_data.npz"))
ref, zout = raw["train_ref"], raw["zout"]
XMAX = 4000.0
out = (np.abs(ref[:, :, 0]) > XMAX).any(axis=1) | (np.abs(ref[:, :, 1]) > XMAX).any(axis=1)
rng = np.random.default_rng(0)
for i in rng.choice(np.where(~out)[0], 120, replace=False):
    ax[5].plot(zout, ref[i, :, 0], color=NEUTRAL, lw=0.7, alpha=0.6)
for i in np.where(out)[0]:
    ax[5].plot(zout, ref[i, :, 0], color=MAGENTA, lw=1.4, alpha=0.9)
ax[5].axhline(XMAX, color=TEXT1, lw=1.5, ls="--")
ax[5].axhline(-XMAX, color=TEXT1, lw=1.5, ls="--")
ax[5].text(zout[0], XMAX * 1.02, "field-map edge |x| = 4 m", fontsize=8, color=TEXT2)
ax[5].set_xlabel("z [mm]"); ax[5].set_ylabel("x [mm]")
ax[5].set_title("the %d cut trajectories (pink) leave the map;\n120 kept ones in grey"
                % out.sum(), fontsize=10)

fig.suptitle(
    "One implicit q=8 step across the magnet on OFFICIAL-sample data (v2), trained to stall: "
    "physics-only vs data twin (test split, n=%s)" % summary[0]["n"], fontsize=12.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "one_step_results_v2.png"), dpi=140)
print("wrote", os.path.join(FIG, "one_step_results_v2.png"))
