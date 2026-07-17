#!/usr/bin/env python
"""Characterise the training sample against the simulated events it came from.

Reads : results/states.npz, training_v1/train_mb100_v1.npz,
        results/g2_label_vs_next_hit.npz, truth_mb100/particles.csv
Writes: figures/{population,legs,label_gates}.png + results/*.csv

The three claims a supervisor needs, made visual:
  1. POPULATION — the training states inherit the kinematics of real simulated
     crossings (p, eta, slopes, origins), with the only bias being detector
     acceptance itself (shown explicitly as coverage vs eta).
  2. GEOMETRY — the legs are the geometry the extrapolator actually serves
     (vertex fetches, cross-magnet transfers, plane-to-plane steps).
  3. LABELS — field-only RK4 labels are self-consistent to nm (closure) and
     track the true Geant4 trajectories to a few um at high momentum; the
     residual grows toward low p exactly like multiple scattering (the material
     effects deliberately excluded from label scope).

Run:  /data/bfys/gscriven/conda/envs/TE/bin/python characterise.py
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES, FIG = os.path.join(HERE, "results"), os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

# palette (validated dataviz reference, light mode)
SURFACE, TEXT1, TEXT2 = "#fcfcfb", "#0b0b0b", "#52514e"
BLUE, GREEN, MAGENTA, YELLOW, NEUTRAL = "#2a78d6", "#008300", "#e87ba4", "#eda100", "#c9c8c2"
LEG_COLOR = {0: BLUE, 1: GREEN, 2: MAGENTA, 3: YELLOW}
LEG_NAME = {0: "A vertex fetch", 1: "B cross-magnet", 2: "C plane-to-plane", 3: "D downstream->PV"}
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": TEXT1, "axes.edgecolor": TEXT2, "axes.labelcolor": TEXT1,
    "xtick.color": TEXT2, "ytick.color": TEXT2, "axes.grid": True,
    "grid.color": "#e7e6e1", "grid.linewidth": 0.6, "axes.axisbelow": True, "font.size": 10.5,
})

st = np.load(os.path.join(RES, "states.npz"))
tr = np.load(os.path.join(HERE, "training_v1", "train_mb100_v1.npz"))
g2 = np.load(os.path.join(RES, "g2_label_vs_next_hit.npz"))
truth = pd.read_csv(os.path.join(HERE, "truth_mb100", "particles.csv"))
truth = truth[truth["charge3"].fillna(0) != 0]
truth["p_GeV"] = np.sqrt(truth.px**2 + truth.py**2 + truth.pz**2) / 1000.0
tpt = np.hypot(truth.px, truth.py)
truth["eta"] = np.arcsinh(np.where(tpt > 0, truth.pz / np.where(tpt > 0, tpt, 1), np.inf))

# unique harvested particles (first state of each)
pk = st["evt"].astype(np.int64) * 10_000_000 + st["mc_key"]
_, first = np.unique(pk, return_index=True)
hp, heta = st["p_GeV"][first], st["eta"][first]
hq = st["q"][first]
apid = np.abs(st["pid"][first])
horr = np.hypot(st["origin_x"][first], st["origin_y"][first])
core = (apid != 11) & (horr < 10.0) & (heta > 1.8) & (heta < 5.2)

X, LEG, P = tr["X"], tr["LEG"], tr["P"]

# ------------------------------------------------------------- fig 1: population
fig, ax = plt.subplots(2, 3, figsize=(14, 7.5))
bins = np.logspace(np.log10(0.2), np.log10(300), 50)
ax[0, 0].hist(truth.p_GeV.clip(0.2, 300), bins=bins, density=True, color=NEUTRAL,
              label="all charged truth")
ax[0, 0].hist(np.clip(hp, 0.2, 300), bins=bins, density=True, histtype="step", lw=2,
              color=BLUE, label="harvested (has tracker states)")
ax[0, 0].hist(np.clip(hp[core], 0.2, 300), bins=bins, density=True, histtype="step", lw=2,
              color=GREEN, label="track-like core (%.0f%%)" % (100 * core.mean()))
ax[0, 0].set_xscale("log"); ax[0, 0].set_xlabel("|p| [GeV]"); ax[0, 0].set_ylabel("norm.")
ax[0, 0].legend(fontsize=8.5); ax[0, 0].set_title("momentum spectrum", fontsize=11)

be = np.linspace(-8, 8, 65)
ax[0, 1].hist(np.clip(truth.eta, -8, 8), bins=be, density=True, color=NEUTRAL)
ax[0, 1].hist(np.clip(heta, -8, 8), bins=be, density=True, histtype="step", lw=2, color=BLUE)
ax[0, 1].axvspan(2, 5, color=YELLOW, alpha=0.15, lw=0)
ax[0, 1].text(3.5, 0.95, "LHCb acceptance", transform=ax[0, 1].get_xaxis_transform(),
              ha="center", va="top", fontsize=8.5, color=TEXT2)
ax[0, 1].set_xlabel(r"$\eta$"); ax[0, 1].set_title("pseudorapidity", fontsize=11)

# coverage vs eta (the acceptance bias, shown honestly)
hcnt, _ = np.histogram(np.clip(heta, -8, 8), bins=be)
tcnt, _ = np.histogram(np.clip(truth.eta, -8, 8), bins=be)
cov = np.divide(hcnt, tcnt, out=np.zeros_like(hcnt, float), where=tcnt > 0)
ctr = 0.5 * (be[1:] + be[:-1])
ax[0, 2].step(ctr, cov, where="mid", color=BLUE, lw=2)
ax[0, 2].axvspan(2, 5, color=YELLOW, alpha=0.15, lw=0)
ax[0, 2].set_xlabel(r"$\eta$"); ax[0, 2].set_ylabel("harvested / truth")
ax[0, 2].set_title("coverage vs eta (= acceptance)", fontsize=11)

ax[1, 0].hist(st["tx"], bins=np.linspace(-1, 1, 80), color=BLUE)
ax[1, 0].set_xlabel("tx at plane crossings"); ax[1, 0].set_title("slope tx", fontsize=11)
ax[1, 1].hist(st["z"] / 1000.0, bins=80, color=BLUE)
ax[1, 1].set_xlabel("z of harvested states [m]"); ax[1, 1].set_title("where states live", fontsize=11)

pid_ct = pd.Series(apid[core]).value_counts().head(6)
names = {211: "pi", 321: "K", 2212: "p", 11: "e", 13: "mu", 3222: "Sigma", 3112: "Sigma-"}
ax[1, 2].bar(range(len(pid_ct)), pid_ct.values, color=BLUE, width=0.65)
ax[1, 2].set_xticks(range(len(pid_ct)))
ax[1, 2].set_xticklabels([names.get(int(k), str(int(k))) for k in pid_ct.index], fontsize=9)
qplus_core = float((hq[core] > 0).mean())
ax[1, 2].set_title(
    "species, track-like core (q+ frac %.3f)" % qplus_core, fontsize=11)
fig.suptitle(
    "Training population vs simulated-event truth — 100 minbias crossings, 2024 conditions "
    "(%d charged truth, %d harvested particles, %d states)"
    % (len(truth), len(first), len(pk)), fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "population.png"), dpi=150); plt.close(fig)

# ------------------------------------------------------------------ fig 2: legs
fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))
for t in range(4):
    m = LEG == t
    ax[0].hist(np.abs(X[m, 6] - X[m, 5]), bins=np.logspace(0, 4, 45),
               histtype="step", lw=2, color=LEG_COLOR[t], label="%s (%d)" % (LEG_NAME[t], m.sum()))
ax[0].set_xscale("log"); ax[0].set_xlabel("|dz| [mm]"); ax[0].set_ylabel("rows")
ax[0].legend(fontsize=8); ax[0].set_title("leg length by type", fontsize=11)

sub = np.random.default_rng(0).choice(len(X), size=min(20000, len(X)), replace=False)
for t in range(4):
    m = LEG[sub] == t
    ax[1].plot(X[sub][m, 5] / 1000, X[sub][m, 6] / 1000, ls="none", marker=".", ms=2,
               color=LEG_COLOR[t], alpha=0.5)
ax[1].plot([-0.5, 10], [-0.5, 10], color=NEUTRAL, lw=1)
ax[1].set_xlabel("z0 [m]"); ax[1].set_ylabel("z1 [m]")
ax[1].set_title("leg geometry (below diagonal = backward)", fontsize=11)

m = LEG == 1
bend = np.abs(tr["Y"][m, 0] - (X[m, 0] + X[m, 2] * (X[m, 6] - X[m, 5])))
ax[2].plot(P[m], bend, ls="none", marker=".", ms=2, color=GREEN, alpha=0.35)
pg = np.logspace(0, np.log10(200), 18)
med = [np.median(bend[(P[m] >= a) & (P[m] < b)]) if ((P[m] >= a) & (P[m] < b)).any() else np.nan
       for a, b in zip(pg[:-1], pg[1:])]
ax[2].plot(np.sqrt(pg[:-1] * pg[1:]), med, color=TEXT1, lw=2, label="median (~1/p)")
ax[2].set_xscale("log"); ax[2].set_yscale("log")
ax[2].set_xlabel("|p| [GeV]"); ax[2].set_ylabel("|x_true - x_straight| [mm]")
ax[2].legend(fontsize=8.5); ax[2].set_title("magnet bend, cross-magnet legs", fontsize=11)
fig.suptitle("Leg geometry of the training set (%d rows)" % len(X), fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "legs.png"), dpi=150); plt.close(fig)

# ----------------------------------------------------------- fig 3: label gates
fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))
r, p = g2["resid_mm"] * 1000.0, g2["p_GeV"]  # um
bins = np.logspace(-1, 4, 55)
ax[0].hist(np.clip(r, 0.1, 1e4), bins=bins, color=BLUE)
ax[0].hist(np.clip(r[p > 5], 0.1, 1e4), bins=bins, histtype="step", lw=2, color=GREEN,
           label="p > 5 GeV (median %.1f um)" % np.median(r[p > 5]))
ax[0].set_xscale("log"); ax[0].set_xlabel("|label - true next hit| [um]")
ax[0].legend(fontsize=8.5)
ax[0].set_title("labels vs Geant4 truth (plane-to-plane)", fontsize=11)

pg = np.logspace(0, np.log10(200), 22)
med = [np.median(r[(p >= a) & (p < b)]) if ((p >= a) & (p < b)).any() else np.nan
       for a, b in zip(pg[:-1], pg[1:])]
ctr_p = np.sqrt(pg[:-1] * pg[1:])
ax[1].plot(ctr_p, med, color=BLUE, lw=2, marker="o", ms=4, label="median residual")
ax[1].plot(ctr_p, med[np.nanargmin(np.abs(ctr_p - 10))] * (10.0 / ctr_p), color=NEUTRAL,
           lw=1.5, ls="--", label="1/p scaling (multiple scattering)")
ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set_xlabel("|p| [GeV]"); ax[1].set_ylabel("median residual [um]")
ax[1].legend(fontsize=8.5)
ax[1].set_title("residual = material effects (out of label scope)", fontsize=11)

with open(os.path.join(RES, "gates.json")) as f:
    gates = json.load(f)
ax[2].axis("off")
lines = [
    "Integrity gates (results/gates.json)",
    "",
    "G1 re-propagation closure (fwd+back):",
    "    median %.1e mm, worst %.1e mm" % (
        gates["G1_reprop_closure_mm"]["median"], gates["G1_reprop_closure_mm"]["worst"]),
    "G2 label vs true next hit:",
    "    median %.1f um; p>5 GeV %.1f um; p<2 GeV %.1f um" % (
        gates["G2_label_vs_next_hit_mm"]["median"] * 1e3,
        gates["G2_label_vs_next_hit_mm"]["median_p_gt_5GeV"] * 1e3,
        gates["G2_label_vs_next_hit_mm"]["median_p_lt_2GeV"] * 1e3),
    "G3 step convergence 5 mm vs 1 mm:",
    "    median %.1e mm, worst %.1e mm" % (
        gates["G3_step_convergence_mm_5vs1"]["median"],
        gates["G3_step_convergence_mm_5vs1"]["worst"]),
    "G4 qop passthrough exact: %s" % gates["G4_qop_passthrough_exact"],
]
ax[2].text(0.02, 0.95, "\n".join(lines), transform=ax[2].transAxes, va="top",
           family="monospace", fontsize=9.5, color=TEXT1)
fig.suptitle("Label integrity", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "label_gates.png"), dpi=150); plt.close(fig)

# ------------------------------------------------------------------ result CSVs
qs = [0.05, 0.25, 0.5, 0.75, 0.95]
pd.DataFrame({
    "quantile": qs,
    "truth_p_GeV": np.quantile(truth.p_GeV, qs),
    "harvested_p_GeV": np.quantile(hp, qs),
    "truth_eta": np.quantile(truth.eta[np.isfinite(truth.eta)], qs),
    "harvested_eta": np.quantile(heta, qs),
}).to_csv(os.path.join(RES, "population_stats.csv"), index=False)
pd.DataFrame({
    "leg": [LEG_NAME[t] for t in range(4)],
    "rows": [(LEG == t).sum() for t in range(4)],
    "median_abs_dz_mm": [float(np.median(np.abs(X[LEG == t, 6] - X[LEG == t, 5]))) for t in range(4)],
    "median_p_GeV": [float(np.median(P[LEG == t])) for t in range(4)],
}).to_csv(os.path.join(RES, "legs_summary.csv"), index=False)
pd.DataFrame({"eta_bin_centre": ctr, "coverage": cov}).to_csv(
    os.path.join(RES, "coverage_vs_eta.csv"), index=False)

print("figures ->", FIG)
print("results ->", RES)
