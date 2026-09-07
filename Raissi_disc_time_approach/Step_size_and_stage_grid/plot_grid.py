#!/usr/bin/env python
"""C3.5 - the grid's three figures, drawn from `results/error_vs_dz_q.csv` only.

Nothing here recomputes anything: `aggregate_grid.py` writes the long table and
this script reads it, so a figure can never disagree with the csv it is drawn
from. Every cell is the **median over the seeds that confirmed**; if no seed of
a cell confirmed, the median over all its seeds is used instead and the cell is
marked (a dot in the heat map, a hollow marker in the curves), because a
silently-filtered grid is worse than a marked one.

The test split is drawn by default (`--split val` for the other).

1. `figures/heatmap_w<W>_d<D>.png` - one file per architecture. Rows are q,
   columns are the six |dz| strata, colour is log10 of the endpoint median in
   um, with the physics arm and the data twin side by side on one shared colour
   scale so the two can be read against each other. The bottom row of each
   panel is the straight line on the same rows - the thing every cell has to
   beat before it is worth anything.

2. `figures/error_vs_dz.png` - the same numbers as curves: one panel per
   architecture, endpoint median against the stratum's median |dz|, one line
   per q, the straight line in black.

3. `figures/error_vs_q.png` - the cut the other way: one panel per stratum,
   endpoint median against q, one line per architecture.

    PYTHONNOUSERSITE=1 python plot_grid.py
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import argparse
import csv
import json
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")
STRATA = ("0.05-0.2 mm", "0.5-2 mm", "5-20 mm", "50-200 mm", "500-2000 mm",
          "full crossing")
MODES = ("physics", "data")
MODE_TITLE = {"physics": "physics loss", "data": "data twin"}


def read_long(path, split="test", direction="all"):
    with open(path) as f:
        rows = [r for r in csv.DictReader(f)
                if r["split"] == split and r["direction"] == direction]
    for r in rows:
        for k in ("width", "depth", "q", "seed", "n"):
            r[k] = int(r[k])
        for k in ("median", "p95", "slope", "rho", "straight_line"):
            r[k] = float(r[k]) if r[k] not in ("", "None") else float("nan")
        r["converged"] = str(r["converged"]).lower() in ("true", "1")
    return rows


def cells(rows):
    """(width, depth, mode, q, stratum) -> (median, straight, confirmed, n_seeds)."""
    bucket = defaultdict(list)
    for r in rows:
        bucket[(r["width"], r["depth"], r["mode"], r["q"], r["stratum"])].append(r)
    out = {}
    for key, rs in bucket.items():
        ok = [r for r in rs if r["converged"]]
        use, confirmed = (ok, True) if ok else (rs, False)
        out[key] = (float(np.median([r["median"] for r in use])),
                    float(np.median([r["straight_line"] for r in use])),
                    confirmed, len(use))
    return out


def dz_medians():
    """The median |dz| of each stratum, from the C3.1 check if it is there."""
    path = os.path.join(RESULTS, "scale_check.json")
    fallback = {"0.05-0.2 mm": 0.1, "0.5-2 mm": 1.0, "5-20 mm": 10.0,
                "50-200 mm": 100.0, "500-2000 mm": 1000.0,
                "full crossing": 5200.0}
    try:
        with open(path) as f:
            check = json.load(f)
        return {k: v["abs_dz_mm"]["median"]
                for k, v in check["per_stratum"].items()}
    except (OSError, ValueError, KeyError):
        return fallback


# ------------------------------------------------------------- the heat map --
def heatmaps(cell, split, out_dir):
    archs = sorted({(w, d) for (w, d, _, _, _) in cell})
    qs = sorted({q for (_, _, _, q, _) in cell})
    if not archs or not qs:
        return []
    finite = [v[0] for v in cell.values() if np.isfinite(v[0]) and v[0] > 0]
    finite += [v[1] for v in cell.values() if np.isfinite(v[1]) and v[1] > 0]
    norm = Normalize(np.log10(min(finite)), np.log10(max(finite)))
    written = []
    for (w, d) in archs:
        height = max(4.0, 1.6 + 0.34 * (len(qs) + 2))
        fig, axes = plt.subplots(1, len(MODES),
                                 figsize=(7.2 * len(MODES), height),
                                 squeeze=False)
        for col, mode in enumerate(MODES):
            ax = axes[0][col]
            M = np.full((len(qs) + 1, len(STRATA)), np.nan)
            marks = []
            for i, q in enumerate(qs):
                for j, s in enumerate(STRATA):
                    v = cell.get((w, d, mode, q, s))
                    if v is None:
                        continue
                    M[i, j] = np.log10(v[0]) if v[0] > 0 else np.nan
                    M[len(qs), j] = np.log10(v[1]) if v[1] > 0 else np.nan
                    if not v[2]:
                        marks.append((j, i))
            im = ax.imshow(M, aspect="auto", cmap="viridis", norm=norm,
                           origin="upper")
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    if np.isfinite(M[i, j]):
                        ax.text(j, i, "%.3g" % (10 ** M[i, j]), ha="center",
                                va="center", fontsize=6.5,
                                color="w" if M[i, j] < norm.vmin +
                                0.55 * (norm.vmax - norm.vmin) else "k")
            for j, i in marks:
                ax.plot(j, i, ".", color="r", ms=3)
            ax.set_xticks(range(len(STRATA)))
            ax.set_xticklabels(STRATA, rotation=30, ha="right", fontsize=7)
            ax.set_yticks(range(len(qs) + 1))
            ax.set_yticklabels(["q = %d" % q for q in qs] + ["straight line"],
                               fontsize=7)
            ax.set_title("%s  -  %d x %d" % (MODE_TITLE[mode], d, w), fontsize=10)
            fig.colorbar(im, ax=ax, label="log10( endpoint median / um )")
        fig.suptitle("C3 grid: endpoint median error, %s split, depth %d width "
                     "%d\n(red dot: no seed of that cell confirmed)"
                     % (split, d, w), fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        path = os.path.join(out_dir, "heatmap_w%d_d%d.png" % (w, d))
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(path)
    return written


# --------------------------------------------------------------- the curves --
def error_vs_dz(cell, split, out_dir):
    archs = sorted({(w, d) for (w, d, _, _, _) in cell})
    qs = sorted({q for (_, _, _, q, _) in cell})
    if not archs:
        return None
    dzm = dz_medians()
    x = [dzm.get(s, np.nan) for s in STRATA]
    ncol = min(4, len(archs))
    nrow = int(np.ceil(len(archs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.8 * nrow),
                             squeeze=False)
    colours = plt.cm.plasma(np.linspace(0, 0.88, max(len(qs), 1)))
    for k, (w, d) in enumerate(archs):
        ax = axes[k // ncol][k % ncol]
        for iq, q in enumerate(qs):
            for mode, ls in (("physics", "-"), ("data", "--")):
                y, xs, hollow = [], [], []
                for j, s in enumerate(STRATA):
                    v = cell.get((w, d, mode, q, s))
                    if v is None:
                        continue
                    xs.append(x[j])
                    y.append(v[0])
                    hollow.append(not v[2])
                if not y:
                    continue
                ax.plot(xs, y, ls, color=colours[iq], lw=1.3, ms=3.5,
                        marker="o", mfc="none" if any(hollow) else colours[iq],
                        label="q = %d, %s" % (q, mode) if k == 0 else None)
        ys, xs = [], []
        for j, s in enumerate(STRATA):
            for key in cell:
                if key[0] == w and key[1] == d and key[4] == s:
                    xs.append(x[j])
                    ys.append(cell[key][1])
                    break
        if ys:
            ax.plot(xs, ys, "k-", lw=2.0, label="straight line" if k == 0 else None)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("|dz|  (mm, stratum median)")
        ax.set_ylabel("endpoint median error  (um)")
        ax.set_title("depth %d, width %d" % (d, w), fontsize=10)
        ax.grid(alpha=0.3, which="both")
    for k in range(len(archs), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(6, len(labels)),
                   fontsize=6.5, frameon=False)
    fig.suptitle("C3 grid: error against step length, %s split" % split,
                 fontsize=12)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    path = os.path.join(out_dir, "error_vs_dz.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def error_vs_q(cell, split, out_dir):
    archs = sorted({(w, d) for (w, d, _, _, _) in cell})
    qs = sorted({q for (_, _, _, q, _) in cell})
    present = [s for s in STRATA if any(k[4] == s for k in cell)]
    if not present:
        return None
    ncol = min(3, len(present))
    nrow = int(np.ceil(len(present) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.8 * ncol, 3.8 * nrow),
                             squeeze=False)
    colours = plt.cm.viridis(np.linspace(0, 0.88, max(len(archs), 1)))
    for k, s in enumerate(present):
        ax = axes[k // ncol][k % ncol]
        straight = None
        for ia, (w, d) in enumerate(archs):
            for mode, ls in (("physics", "-"), ("data", "--")):
                xs, y = [], []
                for q in qs:
                    v = cell.get((w, d, mode, q, s))
                    if v is None:
                        continue
                    xs.append(q)
                    y.append(v[0])
                    straight = v[1]
                if y:
                    ax.plot(xs, y, ls, color=colours[ia], lw=1.3, marker="o",
                            ms=3.5,
                            label="%dx%d %s" % (d, w, mode) if k == 0 else None)
        if straight is not None:
            ax.axhline(straight, color="k", lw=2.0,
                       label="straight line" if k == 0 else None)
        ax.set_yscale("log")
        ax.set_xlabel("q  (Gauss-Legendre stages)")
        ax.set_ylabel("endpoint median error  (um)")
        ax.set_title(s, fontsize=10)
        ax.grid(alpha=0.3, which="both")
    for k in range(len(present), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center",
                   ncol=min(6, len(labels)), fontsize=6.5, frameon=False)
    fig.suptitle("C3 grid: error against the number of stages, %s split" % split,
                 fontsize=12)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    path = os.path.join(out_dir, "error_vs_q.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--table", default=os.path.join(RESULTS,
                                                    "error_vs_dz_q.csv"))
    ap.add_argument("--split", default="test", choices=("val", "test"))
    ap.add_argument("--direction", default="all",
                    choices=("all", "forward", "backward"))
    ap.add_argument("--figures", default=FIGURES)
    a = ap.parse_args(argv)

    os.makedirs(a.figures, exist_ok=True)
    rows = read_long(a.table, split=a.split, direction=a.direction)
    if not rows:
        print("no rows in %s for split %s" % (a.table, a.split))
        return
    cell = cells(rows)
    written = heatmaps(cell, a.split, a.figures)
    for p in (error_vs_dz(cell, a.split, a.figures),
              error_vs_q(cell, a.split, a.figures)):
        if p:
            written.append(p)
    for p in written:
        print("wrote %s" % os.path.relpath(p, HERE))


if __name__ == "__main__":
    main()
