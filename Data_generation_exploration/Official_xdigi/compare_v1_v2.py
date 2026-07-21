#!/usr/bin/env python
"""Compare the v1 (our GaussMB100) and v2 (official XDIGI) state populations.

The point: v2's provenance is centrally-produced simulation end to end; if the
two populations agree, that also retroactively validates our local generation.
Distributions are density-normalised (different event counts).

Run:  /data/bfys/gscriven/conda/envs/TE/bin/python compare_v1_v2.py
Outputs: figures/v1_vs_v2_population.png, results/compare_v1_v2.json
"""
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(HERE, "..", "Data", "results", "states.npz")
V2 = os.path.join(HERE, "results", "states.npz")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

SURFACE, TEXT1, TEXT2 = "#fcfcfb", "#0b0b0b", "#52514e"
BLUE, GREEN = "#2a78d6", "#008300"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": TEXT1, "axes.edgecolor": TEXT2, "axes.labelcolor": TEXT1,
    "xtick.color": TEXT2, "ytick.color": TEXT2, "axes.grid": True,
    "grid.color": "#e7e6e1", "grid.linewidth": 0.6, "axes.axisbelow": True, "font.size": 10.5,
})

d1 = np.load(V1, allow_pickle=True)
d2 = np.load(V2, allow_pickle=True)
n_evt1, n_evt2 = len(np.unique(d1["evt"])), len(np.unique(d2["evt"]))
print("v1: %d states from %d events | v2: %d states from %d events"
      % (len(d1["x"]), n_evt1, len(d2["x"]), n_evt2))

PANELS = [
    ("p_GeV", "|p| [GeV]", np.logspace(-1, np.log10(300), 60), "log"),
    ("eta", "pseudorapidity", np.linspace(0, 8, 60), "linear"),
    ("z", "state z [mm]", np.linspace(-400, 10000, 60), "linear"),
    ("x", "state x [mm]", np.linspace(-3000, 3000, 60), "linear"),
    ("tx", "slope tx", np.linspace(-0.6, 0.6, 60), "linear"),
    ("origin_z", "particle origin z [mm]", np.linspace(-400, 1000, 60), "linear"),
]

fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2))
stats = {}
for ax, (key, label, bins, xscale) in zip(axes.ravel(), PANELS):
    a, b = d1[key], d2[key]
    ax.hist(np.clip(a, bins[0], bins[-1]), bins=bins, density=True, histtype="step",
            lw=2, color=BLUE, label="v1 ours (GaussMB100)")
    ax.hist(np.clip(b, bins[0], bins[-1]), bins=bins, density=True, histtype="step",
            lw=2, color=GREEN, label="v2 official (XDIGI)")
    if xscale == "log":
        ax.set_xscale("log")
    ax.set_xlabel(label)
    ax.set_ylabel("density")
    # a scale-free distance between the two samples: max CDF gap (KS statistic)
    lo, hi = bins[0], bins[-1]
    aa = np.clip(a, lo, hi); bb = np.clip(b, lo, hi)
    grid = np.linspace(lo, hi, 400)
    ks = float(np.abs(
        np.searchsorted(np.sort(aa), grid) / len(aa)
        - np.searchsorted(np.sort(bb), grid) / len(bb)).max())
    stats[key] = {"ks": ks, "median_v1": float(np.median(a)), "median_v2": float(np.median(b))}
    ax.set_title("%s   (KS distance %.3f)" % (key, ks), fontsize=10)
axes[0, 0].legend(fontsize=8.5)
fig.suptitle(
    "State populations: our generated sample (v1, %d events) vs the official "
    "TestFileDB sample (v2, %d events)" % (n_evt1, n_evt2), fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "v1_vs_v2_population.png"), dpi=150)
print("wrote", os.path.join(FIG, "v1_vs_v2_population.png"))

out = {"n_states": {"v1": int(len(d1["x"])), "v2": int(len(d2["x"]))},
       "n_events": {"v1": n_evt1, "v2": n_evt2}, "per_variable": stats}
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
with open(os.path.join(HERE, "results", "compare_v1_v2.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps({k: round(v["ks"], 3) for k, v in stats.items()}, indent=1))
