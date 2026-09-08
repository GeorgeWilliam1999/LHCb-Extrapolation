#!/usr/bin/env python
"""C5.3 - the figure, drawn from `results/hit_table.csv` alone.

Four panels, all of them about one line: the **material floor**, i.e. the fine
reference's own miss against the particle's real next hit. It is the part of the
step that is multiple scattering and energy loss, and no field-only
extrapolator - scheme, network or exact solve - can predict it.

1. the median error against the real hit, per |dz| stratum, with the floor as
   the thick black line and the straight line as the null step;
2. the same divided by the floor, per momentum band: the excess the network
   costs over a perfect field propagation;
3. each predictor's error against the **fine reference's own answer** - the
   field-only error - with the floor drawn over it. Where a curve is below the
   floor line, a comparison against hits cannot tell that predictor from the
   reference;
4. the cross-magnet column alone, by momentum band.

    PYTHONNOUSERSITE=1 python plot.py
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import argparse
import csv

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")

FLOOR = "fine reference (material floor)"
STRAIGHT = "straight line"
SCHEME = "exact scheme q=8"
NET = "network (best val seed)"
BANDS = ("1-5 GeV", "5-20 GeV", "20-200 GeV")


def read_table(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            r["n"] = int(r["n"])
            for k in ("median_abs_dz_mm", "vs_hit_med_um", "vs_hit_p95_um",
                      "vs_reference_med_um", "vs_reference_p95_um"):
                r[k] = float(r[k])
            rows.append(r)
    return rows


def label_of(r):
    if r["predictor"] == NET:
        return "%s %s" % (r["arch"], "physics" if r["mode"] == "physics"
                          else "twin")
    return r["predictor"]


def series(rows, strata, key, band="all", predictor=None, arch=None, mode=None):
    out = []
    for s in strata:
        v = [r for r in rows if r["stratum"] == s and r["p_band"] == band
             and r["direction"] == "all" and r["predictor"] == predictor
             and (arch is None or r["arch"] == arch)
             and (mode is None or r["mode"] == mode)]
        out.append(v[0][key] if v else np.nan)
    return np.array(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--table", default=os.path.join(RESULTS, "hit_table.csv"))
    ap.add_argument("--figures", default=FIGURES)
    a = ap.parse_args(argv)
    os.makedirs(a.figures, exist_ok=True)

    rows = read_table(a.table)
    strata = []
    for r in rows:
        if r["stratum"] not in strata and r["stratum"] != "all":
            strata.append(r["stratum"])
    x = np.array([np.median([r["median_abs_dz_mm"] for r in rows
                             if r["stratum"] == s]) for s in strata])
    nets = sorted({(r["arch"], r["mode"]) for r in rows
                   if r["predictor"] == NET})
    colours = plt.cm.plasma(np.linspace(0, 0.85, max(len(nets), 1)))

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.4))

    # ------------------------------------------------------------- panel 1 --
    ax = axes[0][0]
    floor = series(rows, strata, "vs_hit_med_um", predictor=FLOOR)
    ax.plot(x, floor, "k-", lw=3.0, marker="o", ms=5,
            label="fine reference = material floor")
    ax.plot(x, series(rows, strata, "vs_hit_med_um", predictor=SCHEME),
            "k--", lw=1.6, marker="s", ms=4, label="exact scheme q = 8")
    ax.plot(x, series(rows, strata, "vs_hit_med_um", predictor=STRAIGHT),
            color="0.55", lw=1.6, ls=":", marker="^", ms=4,
            label="straight line")
    for i, (arch, mode) in enumerate(nets):
        ax.plot(x, series(rows, strata, "vs_hit_med_um", predictor=NET,
                          arch=arch, mode=mode),
                "-o", color=colours[i], lw=1.3, ms=3.5,
                label="%s %s" % (arch, "physics" if mode == "physics"
                                 else "twin"))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("|dz|  (mm, stratum median)")
    ax.set_ylabel("median error against the real next hit  (um)")
    ax.set_title("1. against the simulated hit", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=6.0, ncol=2, frameon=False)

    # ------------------------------------------------------------- panel 2 --
    # Colour is the momentum band, line style the arm; the four architectures
    # of one (band, arm) are drawn in the same colour and style, so the spread
    # of a bundle is the architecture spread.
    ax = axes[0][1]
    band_colour = {"1-5 GeV": "tab:blue", "5-20 GeV": "tab:orange",
                   "20-200 GeV": "tab:green"}
    for band in BANDS:
        fl = series(rows, strata, "vs_hit_med_um", band=band, predictor=FLOOR)
        for arch, mode in nets:
            y = series(rows, strata, "vs_hit_med_um", band=band,
                       predictor=NET, arch=arch, mode=mode) / fl
            ax.plot(x, y, "-" if mode == "physics" else "--",
                    color=band_colour[band], lw=1.2, marker="o", ms=3.0,
                    alpha=0.85)
    for band in BANDS:
        ax.plot([], [], "-", color=band_colour[band], lw=1.2,
                label="%s, physics" % band)
        ax.plot([], [], "--", color=band_colour[band], lw=1.2,
                label="%s, twin" % band)
    ax.axhline(1.0, color="k", lw=2.2, label="the material floor")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("|dz|  (mm, stratum median)")
    ax.set_ylabel("network median / material floor")
    ax.set_title("2. the excess over the floor, by momentum band\n"
                 "(four architectures per bundle)", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=6.5, ncol=2, frameon=False, loc="upper left")

    # ------------------------------------------------------------- panel 3 --
    ax = axes[1][0]
    ax.plot(x, floor, "k-", lw=3.0, marker="o", ms=5,
            label="material floor (the resolution of this test)")
    ax.plot(x, series(rows, strata, "vs_reference_med_um", predictor=SCHEME),
            "k--", lw=1.6, marker="s", ms=4, label="exact scheme q = 8")
    ax.plot(x, series(rows, strata, "vs_reference_med_um", predictor=STRAIGHT),
            color="0.55", lw=1.6, ls=":", marker="^", ms=4,
            label="straight line")
    for i, (arch, mode) in enumerate(nets):
        ax.plot(x, series(rows, strata, "vs_reference_med_um", predictor=NET,
                          arch=arch, mode=mode),
                "-o", color=colours[i], lw=1.3, ms=3.5,
                label="%s %s" % (arch, "physics" if mode == "physics"
                                 else "twin"))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("|dz|  (mm, stratum median)")
    ax.set_ylabel("median error against the fine reference  (um)")
    ax.set_title("3. the field-only error, and where it disappears under "
                 "the floor", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=6.0, ncol=2, frameon=False)

    # ------------------------------------------------------------- panel 4 --
    ax = axes[1][1]
    st = strata[-1]
    names = [FLOOR, SCHEME] + ["%s|%s" % (an, mn) for an, mn in nets]
    width = 0.26
    for bi, band in enumerate(BANDS):
        vals = []
        for nm in names:
            if "|" in nm:
                an, mn = nm.split("|")
                v = series(rows, [st], "vs_hit_med_um", band=band,
                           predictor=NET, arch=an, mode=mn)[0]
            else:
                v = series(rows, [st], "vs_hit_med_um", band=band,
                           predictor=nm)[0]
            vals.append(v)
        ax.bar(np.arange(len(names)) + (bi - 1) * width, vals, width,
               label=band)
    ax.set_yscale("log")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("|", " ").replace("data", "twin")
                        .replace("fine reference (material floor)", "floor")
                        .replace("exact scheme q=8", "scheme q=8")
                        for n in names], rotation=35, ha="right", fontsize=6.5)
    ax.set_ylabel("median error against the real next hit  (um)")
    ax.set_title("4. the %s column, by momentum band" % st, fontsize=10)
    ax.grid(alpha=0.3, axis="y", which="both")
    ax.legend(fontsize=7, frameon=False)

    fig.suptitle("C5 steps that end on a real simulated hit: the material "
                 "floor, the exact scheme and the grid networks", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(a.figures, "hit_comparison.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("wrote %s" % os.path.relpath(path, HERE))


if __name__ == "__main__":
    main()
