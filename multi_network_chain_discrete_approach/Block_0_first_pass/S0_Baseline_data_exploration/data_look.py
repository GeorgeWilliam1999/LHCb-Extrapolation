#!/usr/bin/env python
"""Step-0 data look: the numbers the discrete-time loss needs, measured from
the event-derived training set.

1. SCALES — per state component and per leg type: the population spread of the
   inputs (sigma_pop) and the spread of the true correction over the straight
   line (sigma_corr). The q+1 reconstruction residuals of the paper's loss mix
   components spanning mm, radians and 1/MeV; dividing each component's
   residual by a fixed scale makes the loss dimensionless and balanced (the
   van der Pol state was O(1), so its loss needed none of this).
   -> results/scales.csv + the recommendation in results/scales.json

2. FIELD ALONG LEGS — By sampled along straight-line paths of real legs, and
   the bending integral per leg type. Shows where the stiffness lives and why
   cross-magnet legs are the hard case. -> figures/field_along_legs.png

Run:  /data/bfys/gscriven/conda/envs/TE/bin/python data_look.py
"""
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from reference_card import FIELD, load_training  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES, FIG = os.path.join(HERE, "results"), os.path.join(HERE, "figures")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

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

COMP = ["x_mm", "y_mm", "tx", "ty", "qop"]

d = load_training(split="train")
X, Y, LEG = d["X"].astype(np.float64), d["Y"].astype(np.float64), d["LEG"]
dz = X[:, 6] - X[:, 5]
straight = X[:, :5].copy()
straight[:, 0] += X[:, 2] * dz
straight[:, 1] += X[:, 3] * dz

rows = []
for t in range(4):
    m = LEG == t
    for ci, cname in enumerate(COMP):
        rows.append({
            "leg": LEG_NAME[t], "component": cname,
            "sigma_pop": float(np.std(X[m, ci])),
            "sigma_corr": float(np.std(Y[m, ci] - straight[m, ci])),
            "median_abs_corr": float(np.median(np.abs(Y[m, ci] - straight[m, ci]))),
        })
overall = {
    "leg": "ALL", "component": None,
    "sigma_pop": [float(np.std(X[:, i])) for i in range(5)],
    "sigma_corr": [float(np.std(Y[:, i] - straight[:, i])) for i in range(5)],
}

import csv  # noqa: E402

with open(os.path.join(RES, "scales.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

scales = {
    "recommendation": (
        "normalise each reconstruction-residual component by sigma_pop of the "
        "training population per component (x,y: mm-scale spread; tx,ty: slope "
        "spread; qop: its own spread). sigma_corr documents the physical "
        "correction scale per leg for reading results; not used in the loss."
    ),
    "sigma_pop_all_legs": {c: overall["sigma_pop"][i] for i, c in enumerate(COMP)},
    "sigma_corr_all_legs": {c: overall["sigma_corr"][i] for i, c in enumerate(COMP)},
    "per_leg_csv": "results/scales.csv",
}
with open(os.path.join(RES, "scales.json"), "w") as f:
    json.dump(scales, f, indent=1)
print(json.dumps(scales["sigma_pop_all_legs"], indent=1))

# ---------------------------------------------------------- field along legs
rng = np.random.default_rng(3)
fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))

mB = np.where((LEG == 1) & (dz > 0))[0]
sel = rng.choice(mB, size=min(300, len(mB)), replace=False)
zg = np.linspace(2600, 7900, 200)
prof = np.empty((len(sel), len(zg)))
for i, r in enumerate(sel):
    frac = (zg - X[r, 5]) / (X[r, 6] - X[r, 5])
    xs = X[r, 0] + X[r, 2] * (zg - X[r, 5])
    ys = X[r, 1] + X[r, 3] * (zg - X[r, 5])
    _, By, _ = FIELD(xs, ys, zg)
    prof[i] = By
med = np.median(prof, axis=0)
lo, hi = np.percentile(prof, [10, 90], axis=0)
ax[0].fill_between(zg / 1000, lo, hi, color=GREEN, alpha=0.25, lw=0, label="10-90% band")
ax[0].plot(zg / 1000, med, color=GREEN, lw=2, label="median By")
ax[0].set_xlabel("z [m]"); ax[0].set_ylabel("By [field units]")
ax[0].legend(fontsize=8.5)
ax[0].set_title("By along 300 real cross-magnet legs", fontsize=11)

# bending integral |integral By dz| per leg type (coarse, along straight line)
for t in range(4):
    m = np.where(LEG == t)[0]
    s = rng.choice(m, size=min(400, len(m)), replace=False)
    integ = []
    for r in s:
        zz = np.linspace(X[r, 5], X[r, 6], 60)
        xs = X[r, 0] + X[r, 2] * (zz - X[r, 5])
        ys = X[r, 1] + X[r, 3] * (zz - X[r, 5])
        _, By, _ = FIELD(xs, ys, zz)
        integ.append(abs(np.trapz(By, zz)))
    integ = np.array(integ)
    integ = integ[integ > 0]
    ax[1].hist(integ, bins=np.logspace(-2, 4, 40), histtype="step", lw=2,
               color=LEG_COLOR[t], label=LEG_NAME[t])
ax[1].set_xscale("log"); ax[1].set_xlabel("|integral By dz|  [field units * mm]")
ax[1].set_ylabel("legs"); ax[1].legend(fontsize=8)
ax[1].set_title("bending integral per leg type", fontsize=11)

# correction scale vs momentum (B legs): the thing the network must learn
mBf = (LEG == 1)
corr = np.abs(Y[mBf, 0] - straight[mBf, 0])
P = d["P"][mBf]
ax[2].plot(P, np.maximum(corr, 1e-4), ls="none", marker=".", ms=2, color=GREEN, alpha=0.3)
pg = np.logspace(0, np.log10(200), 16)
medc = [np.median(corr[(P >= a) & (P < b)]) if ((P >= a) & (P < b)).any() else np.nan
        for a, b in zip(pg[:-1], pg[1:])]
ax[2].plot(np.sqrt(pg[:-1] * pg[1:]), medc, color=TEXT1, lw=2)
ax[2].set_xscale("log"); ax[2].set_yscale("log")
ax[2].set_xlabel("|p| [GeV]"); ax[2].set_ylabel("|x correction| [mm]")
ax[2].set_title("what the net must learn (B legs)", fontsize=11)

fig.suptitle("Step-0 data look: field and correction scales over the event-derived legs", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "field_along_legs.png"), dpi=150)
print("wrote", os.path.join(FIG, "field_along_legs.png"))
