#!/usr/bin/env python
"""The three figures of A3a. Loads the csv tables only; computes no physics.

    figures/error_by_leg_and_momentum.png  leg type x momentum band; every
        physics seed a point, the seed median a bar, the data twin beside it,
        with the straight-line baseline and the exact-scheme ceiling drawn in.
    figures/stage_errors.png  the error at each of the eight Gauss nodes inside
        the leg (and at its endpoint), per leg type.
    figures/frozen_vs_general.png  the leg-B error of the one-for-all network
        against the frozen-leg baseline (One_step_network_v2) at 4x50.
"""
from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.environ.get("A3_RESULTS") or os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")
LEGS = ("A", "B", "C")
LEG_TITLE = {"A": "A  vertex fetch", "B": "B  cross-magnet",
             "C": "C  plane-to-plane"}
COLS = ("all", "1-5GeV", "5-20GeV", "20-200GeV")
# the converged frozen-leg baseline, One_step_network_v2/results/summary.csv
V2_PHYSICS = [182.65, 177.39, 235.44]
V2_DATA = [191.75, 162.44, 207.53]


def read(name):
    with open(os.path.join(RESULTS, name)) as f:
        return list(csv.DictReader(f))


def f(x):
    return float(x) if x not in ("", None) else float("nan")


def sel(rows, **kw):
    out = rows
    for k, v in kw.items():
        out = [r for r in out if r[k] == str(v)]
    return out


def main():
    os.makedirs(FIGURES, exist_ok=True)
    by_leg = [r for r in read("by_leg.csv") if r["split"] == "test"]
    # the architectures are whatever is in the table, so a later wave of runs
    # at a new width is picked up without touching this script
    widths = sorted({int(r["width"]) for r in by_leg})
    depth = int(by_leg[0]["depth"])
    ARCHS = [(w, "%dx%d" % (depth, w)) for w in widths]

    # -------------------------------------------- error by leg and momentum --
    fig, axes = plt.subplots(len(LEGS), len(COLS), figsize=(15, 9.5),
                             sharey="row")
    for i, leg in enumerate(LEGS):
        for jx, band in enumerate(COLS):
            ax = axes[i, jx]
            rows = [r for r in by_leg if r["leg"] == leg and r["band"] == band]
            if not rows:
                ax.set_axis_off()
                continue
            for k, (w, wname) in enumerate(ARCHS):
                ph = [f(r["endpoint_med_um"]) for r in rows
                      if r["mode"] == "physics" and int(r["width"]) == w]
                dt = [f(r["endpoint_med_um"]) for r in rows
                      if r["mode"] == "data" and int(r["width"]) == w]
                x = k * 1.0
                if ph:
                    ax.plot(np.full(len(ph), x - 0.13)
                            + np.linspace(-0.05, 0.05, len(ph)), ph, "o",
                            ms=4, color="#1f77b4",
                            label="physics seeds" if (i == 0 and jx == 0 and k == 0) else None)
                    ax.plot([x - 0.24, x - 0.02], [np.median(ph)] * 2, "-",
                            lw=2.5, color="#1f77b4")
                if dt:
                    ax.plot(np.full(len(dt), x + 0.13)
                            + np.linspace(-0.04, 0.04, len(dt)), dt, "s",
                            ms=4, color="#d62728",
                            label="data twin" if (i == 0 and jx == 0 and k == 0) else None)
                    ax.plot([x + 0.02, x + 0.24], [np.median(dt)] * 2, "-",
                            lw=2.5, color="#d62728")
            straight = np.median([f(r["straight_med_um"]) for r in rows])
            # prefer the ceiling measured on this experiment's own states;
            # fall back to the published stratified-sample value
            ceil_pub = f(rows[0]["ceiling_band_um"])
            ceil_own = (f(rows[0]["ceiling_own_um"])
                        if "ceiling_own_um" in rows[0] else float("nan"))
            ceil = ceil_own if np.isfinite(ceil_own) else ceil_pub
            ceil_name = ("exact scheme q=8, this population"
                         if np.isfinite(ceil_own) else "exact scheme q=8")
            ax.axhline(straight, ls="--", color="0.4", lw=1.2,
                       label="straight line" if (i == 0 and jx == 0) else None)
            ax.set_yscale("log")
            # Keep the panel readable. On legs A and C the q=8 exact scheme is
            # accurate to a nanometre or better, so drawing it as a line would
            # stretch the axis over eight decades; it is written in instead.
            vals = [f(r["endpoint_med_um"]) for r in rows] + [straight]
            vals = [v for v in vals if np.isfinite(v) and v > 0]
            bottom = min(vals) / 5.0 if vals else None
            if np.isfinite(ceil) and ceil > 0 and (bottom is None or ceil >= bottom):
                ax.axhline(ceil, ls=":", color="green", lw=1.6,
                           label=ceil_name if (i == 0 and jx == 0) else None)
            elif np.isfinite(ceil):
                ax.text(0.02, 0.04, "%s: %.3g um" % (ceil_name, ceil),
                        transform=ax.transAxes, fontsize=7, color="green")
            if bottom is not None:
                ax.set_ylim(bottom=bottom)
            ax.set_xticks([0, 1])
            ax.set_xticklabels([a[1] for a in ARCHS])
            ax.set_xlim(-0.6, 1.6)
            ax.grid(alpha=0.3, which="both")
            if i == 0:
                ax.set_title(band if band != "all" else "all momenta")
            if jx == 0:
                ax.set_ylabel("%s\nendpoint error [um]" % LEG_TITLE[leg])
    axes[0, 0].legend(fontsize=8, loc="lower right")
    fig.suptitle("A3a  one network for legs A, B and C: test-split endpoint "
                 "error by leg type and momentum", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(FIGURES, "error_by_leg_and_momentum.png"), dpi=140)
    plt.close(fig)
    print("wrote figures/error_by_leg_and_momentum.png")

    # ------------------------------------------------------- stage errors ----
    st = read("stage_errors.csv")
    fig, axes = plt.subplots(1, len(LEGS), figsize=(14, 4.4))
    for i, leg in enumerate(LEGS):
        ax = axes[i]
        rows = [r for r in st if r["leg"] == leg]
        for w, wname in ARCHS:
            for mode, colour, ls in (("physics", "#1f77b4", "-"),
                                     ("data", "#d62728", "--")):
                tags = sorted({r["tag"] for r in rows
                               if r["mode"] == mode and int(r["width"]) == w})
                first = True
                for t in tags:
                    rr = sorted([r for r in rows if r["tag"] == t],
                                key=lambda r: int(r["node"]))
                    ax.plot([int(r["node"]) for r in rr],
                            [f(r["med_um"]) for r in rr], ls, color=colour,
                            lw=2.0 if w == 100 else 1.0, alpha=0.55,
                            label=("%s %s" % (mode, wname)) if first else None)
                    first = False
        ax.set_yscale("log")
        ax.set_xlabel("stage node inside the leg  (8 = the endpoint)")
        ax.set_title(LEG_TITLE[leg])
        ax.grid(alpha=0.3, which="both")
        if i == 0:
            ax.set_ylabel("median position error [um]")
            ax.legend(fontsize=7)
    fig.suptitle("A3a  error along the step: the eight Gauss nodes and the "
                 "endpoint (test split)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(FIGURES, "stage_errors.png"), dpi=140)
    plt.close(fig)
    print("wrote figures/stage_errors.png")

    # -------------------------------------------------- frozen vs general ----
    rowsB = [r for r in by_leg if r["leg"] == "B" and r["band"] == "all"]
    fig, ax = plt.subplots(figsize=(8, 5))
    groups = [
        ("frozen leg %s\nphysics (v2)" % ARCHS[0][1], V2_PHYSICS, "#1f77b4"),
        ("frozen leg %s\ndata twin (v2)" % ARCHS[0][1], V2_DATA, "#d62728"),
    ]
    for w, wname in ARCHS:
        for mode, colour in (("physics", "#1f77b4"), ("data", "#d62728")):
            groups.append(("general %s\n%s" % (wname, mode),
                           [f(r["endpoint_med_um"]) for r in rowsB
                            if r["mode"] == mode and int(r["width"]) == w],
                           colour))
    for i, (name, vals, colour) in enumerate(groups):
        if not vals:
            continue
        ax.plot(np.full(len(vals), i) + np.linspace(-0.12, 0.12, len(vals)),
                vals, "o", ms=6, color=colour)
        ax.plot([i - 0.25, i + 0.25], [np.median(vals)] * 2, "-", lw=2.5,
                color=colour)
    ceilB = f(rowsB[0]["ceiling_leg_um"]) if rowsB else float("nan")
    ax.axhline(ceilB, ls=":", color="green", lw=1.6,
               label="exact scheme ceiling, leg B, q=8 (%.0f um)" % ceilB)
    ax.set_yscale("log")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=8)
    ax.set_ylabel("leg-B endpoint median error [um]")
    ax.set_title("A3a  the cross-magnet leg: one frozen leg vs one network for "
                 "all three leg types")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "frozen_vs_general.png"), dpi=140)
    plt.close(fig)
    print("wrote figures/frozen_vs_general.png")


if __name__ == "__main__":
    main()
