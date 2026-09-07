#!/usr/bin/env python
"""C0.5 - what the dataset is made of.

Six panels, all read from `results/magnet_tracks_v3.npz` and its meta json;
nothing is recomputed. Momentum and pseudorapidity are the population the
selection let through; |dz| per stratum and the start plane z0 are what the
draw did with it; the last two panels are the direction and split bookkeeping
that the counts table in the README quotes.

Run:
    PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python plot_population.py

Output: figures/dataset_population.png
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"

import argparse

import numpy as np

import use_shared                                            # noqa: F401
from _shared.prepare import STRATUM_NAMES

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")
SPLITS = ("train", "val", "test")
SCOL = {"train": "#4c72b0", "val": "#dd8452", "test": "#55a868"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=os.path.join(RESULTS,
                                                  "magnet_tracks_v3.npz"))
    ap.add_argument("--out", default=os.path.join(FIGURES,
                                                  "dataset_population.png"))
    a = ap.parse_args()
    os.makedirs(FIGURES, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(a.npz)
    X, P, ETA = d["X"], d["P"], d["ETA"]
    STRAT, SPLIT, DIR = d["STRATUM"], d["SPLIT"], d["DIRECTION"]
    z0, dz = X[:, 5], X[:, 6]
    names = [str(s) for s in d["stratum_names"]] if "stratum_names" in d.files \
        else list(STRATUM_NAMES)
    ns = len(names)

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9))

    ax = axes[0, 0]
    bins = np.geomspace(1.0, 200.0, 60)
    for s, sp in enumerate(SPLITS):
        ax.hist(P[SPLIT == s], bins=bins, histtype="step", lw=1.6,
                color=SCOL[sp], label="%s (%d)" % (sp, int((SPLIT == s).sum())))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("truth momentum  p  [GeV]")
    ax.set_ylabel("rows")
    ax.set_title("Momentum")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for s, sp in enumerate(SPLITS):
        ax.hist(ETA[SPLIT == s], bins=np.linspace(2.0, 5.0, 60),
                histtype="step", lw=1.6, color=SCOL[sp], label=sp)
    ax.set_xlabel("pseudorapidity  eta")
    ax.set_ylabel("rows")
    ax.set_title("Pseudorapidity  (the 2 < eta < 5 cut)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[0, 2]
    bins = np.geomspace(max(1e-3, np.abs(dz).min() * 0.9),
                        np.abs(dz).max() * 1.1, 90)
    cmap = plt.get_cmap("turbo")
    for i, nm in enumerate(names):
        m = STRAT == i
        if not m.any():
            continue
        ax.hist(np.abs(dz[m]), bins=bins, histtype="step", lw=1.7,
                color=cmap(i / max(1, ns - 1)),
                label="%s  (%d)" % (nm, int(m.sum())))
    ax.set_xscale("log")
    ax.set_xlabel("|dz|  [mm]")
    ax.set_ylabel("rows")
    ax.set_title("Step length, by stratum")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5)

    ax = axes[1, 0]
    for i, nm in enumerate(names):
        m = STRAT == i
        if not m.any():
            continue
        ax.hist(z0[m], bins=80, histtype="step", lw=1.5,
                color=cmap(i / max(1, ns - 1)), label=nm)
    ax.set_xlabel("start plane  z0  [mm]")
    ax.set_ylabel("rows")
    ax.set_title("Where the step starts\n(dense states every 10 mm along the "
                 "RK6 path)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5)

    ax = axes[1, 1]
    xs = np.arange(ns)
    w = 0.38
    fwd = [int(((STRAT == i) & (DIR > 0)).sum()) for i in range(ns)]
    bwd = [int(((STRAT == i) & (DIR < 0)).sum()) for i in range(ns)]
    ax.bar(xs - w / 2, fwd, w, color="#4c72b0", label="dz > 0")
    ax.bar(xs + w / 2, bwd, w, color="#c44e52", label="dz < 0")
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("rows")
    ax.set_title("Direction split")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)

    ax = axes[1, 2]
    bottom = np.zeros(ns)
    for s, sp in enumerate(SPLITS):
        v = np.array([int(((STRAT == i) & (SPLIT == s)).sum())
                      for i in range(ns)], dtype=float)
        ax.bar(xs, v, 0.6, bottom=bottom, color=SCOL[sp], label=sp)
        bottom += v
    for i in range(ns):
        ax.text(i, bottom[i] + 120, "%d" % int(bottom[i]), ha="center",
                fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("rows")
    ax.set_title("Rows per stratum, by split (60/20/20 by particle)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)

    fig.suptitle("magnet_tracks_v3: %d rows, %s field, RK6 %g mm reference"
                 % (len(X), str(d["field"]) if "field" in d.files else "?",
                    float(d["rk6_step_mm"]) if "rk6_step_mm" in d.files
                    else float("nan")), fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(a.out, dpi=140)
    plt.close(fig)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
