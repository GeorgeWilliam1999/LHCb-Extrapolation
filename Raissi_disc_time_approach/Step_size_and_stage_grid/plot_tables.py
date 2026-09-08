#!/usr/bin/env python
"""C4.2 - the six figures of the error(dz, q) reading, from the C4 csvs only.

Nothing here recomputes a score. `tables.py` writes `results/table_cells.csv`
and the eight `results/reading_*.csv`; this script reads them, so a figure can
never disagree with the table it is drawn from.

| figure | what it shows |
|---|---|
| `figures/heatmap_{depth}x{width}.png` | one file per architecture: rows q, columns the six \|dz\| strata, the physics arm and the data twin side by side on one log colour scale, with the exact scheme's own rows and the straight line drawn underneath. A cell with fewer than three seeds is hatched. |
| `figures/error_vs_dz.png` | error against \|dz\| on log-log, one curve per q, each point the best of the twelve architectures at that stratum; the straight line solid and the q = 8 exact scheme dashed |
| `figures/error_vs_q.png` | error against q, one curve per stratum, at 4x64 and 8x256 |
| `figures/physics_over_twin.png` | the ratio map: physics median / twin median, per cell, per architecture |
| `figures/cost_vs_error.png` | parameters and forward-pass multiply-adds against error, at 50-200 mm and the full crossing, with the cost-error front |
| `figures/best_architecture_per_cell.png` | which architecture wins each (q, stratum) cell |

**A name clash worth knowing about.** `plot_grid.py` (C3.5) also writes
`figures/error_vs_dz.png` and `figures/error_vs_q.png`, with the C3 content -
one panel per architecture rather than the C4 distillation. The two scripts
overwrite each other on those two names; the committed versions are this
script's, so run `plot_tables.py` last.

    PYTHONNOUSERSITE=1 python plot_tables.py
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import argparse
import csv

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.colors import Normalize, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Rectangle             # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")
SCHEME = os.path.join(HERE, "..", "Exact_scheme_table", "results")

STRATA = ("0.05-0.2 mm", "0.5-2 mm", "5-20 mm", "50-200 mm", "500-2000 mm",
          "full crossing")
MODES = ("physics", "data")
MODE_TITLE = {"physics": "physics loss", "data": "data twin"}
QS = tuple(range(2, 21, 2))
SHOWCASE = ("4x64", "8x256")
COST_STRATA = ("50-200 mm", "full crossing")


# --------------------------------------------------------------- the tables --
def read_cells(path, split="test", direction="all"):
    """(arch, mode, q, stratum) -> the cell dict, for one split and direction."""
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["split"] != split or r["direction"] != direction:
                continue
            for k in ("width", "depth", "q", "n_seeds", "n_confirmed",
                      "n_parameters", "multiply_adds", "n_rows"):
                r[k] = int(r[k])
            for k in ("median_um", "min_um", "max_um", "p95_um", "slope_mrad",
                      "rho", "straight_um", "abs_dz_mm"):
                r[k] = float(r[k])
            r["pending"] = str(r["pending"]).lower() in ("true", "1")
            out[(r["arch"], r["mode"], r["q"], r["stratum"])] = r
    return out


def read_scheme():
    out = {}
    with open(os.path.join(SCHEME, "scheme_table.csv")) as f:
        for r in csv.DictReader(f):
            if r["direction"] == "both":
                out[(int(r["q"]), r["stratum"])] = float(r["endpoint_med_um"])
    return out


def archs_of(cells):
    a = sorted({k[0] for k in cells},
               key=lambda s: (int(s.split("x")[0]), int(s.split("x")[1])))
    return a


def annotate(ax, M, norm, fontsize=6.0):
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, "%.3g" % (10 ** M[i, j]), ha="center",
                        va="center", fontsize=fontsize,
                        color="w" if M[i, j] < norm.vmin
                        + 0.55 * (norm.vmax - norm.vmin) else "k")


# ------------------------------------------------------------- 1. heat maps --
def heatmaps(cells, scheme, split, out_dir):
    """Rows q, columns stratum, the two arms side by side; references beneath."""
    finite = [c["median_um"] for c in cells.values() if c["median_um"] > 0]
    finite += [v for v in scheme.values() if v > 0]
    finite += [c["straight_um"] for c in cells.values() if c["straight_um"] > 0]
    norm = Normalize(np.log10(min(finite)), np.log10(max(finite)))
    nrow = len(QS) + len(QS) + 1
    written = []
    for arch in archs_of(cells):
        fig, axes = plt.subplots(1, 2, figsize=(8.0, 1.2 + 0.30 * nrow),
                                 squeeze=False)
        for col, mode in enumerate(MODES):
            ax = axes[0][col]
            M = np.full((nrow, len(STRATA)), np.nan)
            thin = []
            for i, q in enumerate(QS):
                for j, s in enumerate(STRATA):
                    c = cells.get((arch, mode, q, s))
                    if c is None or c["median_um"] <= 0:
                        continue
                    M[i, j] = np.log10(c["median_um"])
                    if c["pending"]:
                        thin.append((j, i))
            for i, q in enumerate(QS):
                for j, s in enumerate(STRATA):
                    v = scheme.get((q, s))
                    if v and v > 0:
                        M[len(QS) + i, j] = np.log10(v)
            for j, s in enumerate(STRATA):
                any_c = next((cells[k] for k in cells if k[3] == s), None)
                if any_c and any_c["straight_um"] > 0:
                    M[-1, j] = np.log10(any_c["straight_um"])
            im = ax.imshow(M, aspect="auto", cmap="viridis", norm=norm)
            annotate(ax, M, norm)
            for j, i in thin:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       hatch="///", edgecolor="r", lw=0.8))
            ax.axhline(len(QS) - 0.5, color="w", lw=1.6)
            ax.axhline(2 * len(QS) - 0.5, color="w", lw=1.6)
            ax.set_xticks(range(len(STRATA)))
            ax.set_xticklabels(STRATA, rotation=35, ha="right", fontsize=6.5)
            ax.set_yticks(range(nrow))
            ax.set_yticklabels(["q = %d" % q for q in QS]
                               + ["scheme q = %d" % q for q in QS]
                               + ["straight line"], fontsize=6.0)
            ax.set_title("%s  -  %s" % (MODE_TITLE[mode], arch), fontsize=9)
        fig.subplots_adjust(wspace=0.45)
        fig.colorbar(im, ax=axes[0].tolist(),
                     label="log10( endpoint median / um )", fraction=0.05)
        fig.suptitle("C4 error(dz, q), %s split, architecture %s\n"
                     "network cells above, the exact scheme and the straight "
                     "line beneath; red hatch = fewer than three seeds"
                     % (split, arch), fontsize=10)
        path = os.path.join(out_dir, "heatmap_%s.png" % arch)
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    return written


# --------------------------------------------------------- 2. error vs |dz| --
def error_vs_dz(cells, scheme, split, out_dir):
    """One curve per q; each point is the best architecture at that stratum."""
    archs = archs_of(cells)
    x = [cells[(archs[0], "physics", 2, s)]["abs_dz_mm"] for s in STRATA]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), squeeze=False)
    colours = plt.cm.plasma(np.linspace(0, 0.88, len(QS)))
    for col, mode in enumerate(MODES):
        ax = axes[0][col]
        for iq, q in enumerate(QS):
            y = []
            for s in STRATA:
                vals = [cells[(a, mode, q, s)]["median_um"] for a in archs
                        if (a, mode, q, s) in cells]
                y.append(min(vals) if vals else np.nan)
            ax.plot(x, y, "-o", color=colours[iq], lw=1.3, ms=3.5,
                    label="q = %d" % q)
        straight = [cells[(archs[0], mode, 2, s)]["straight_um"] for s in STRATA]
        ax.plot(x, straight, "k-", lw=2.2, label="straight line")
        ax.plot(x, [scheme[(8, s)] for s in STRATA], "k--", lw=1.6,
                label="exact scheme, q = 8")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("|dz|  (mm, stratum median)")
        ax.set_ylabel("endpoint median error  (um)")
        ax.set_title("%s - best of the twelve architectures" % MODE_TITLE[mode],
                     fontsize=10)
        ax.grid(alpha=0.3, which="both")
    axes[0][0].legend(fontsize=6.5, ncol=2, frameon=False, loc="upper left")
    fig.suptitle("C4.2 error against step length, %s split" % split, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(out_dir, "error_vs_dz.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# ------------------------------------------------------------ 3. error vs q --
def error_vs_q(cells, scheme, split, out_dir):
    """One curve per stratum, at the two showcase architectures."""
    fig, axes = plt.subplots(len(SHOWCASE), 2, figsize=(11.5, 8.0),
                             squeeze=False)
    colours = plt.cm.viridis(np.linspace(0, 0.88, len(STRATA)))
    for r, arch in enumerate(SHOWCASE):
        for col, mode in enumerate(MODES):
            ax = axes[r][col]
            for js, s in enumerate(STRATA):
                y = [cells[(arch, mode, q, s)]["median_um"] for q in QS
                     if (arch, mode, q, s) in cells]
                qq = [q for q in QS if (arch, mode, q, s) in cells]
                if not y:
                    continue
                ax.plot(qq, y, "-o", color=colours[js], lw=1.4, ms=4,
                        label=s if (r == 0 and col == 0) else None)
                ax.axhline(cells[(arch, mode, qq[0], s)]["straight_um"],
                           color=colours[js], ls=":", lw=1.0)
                ax.plot(QS, [scheme[(q, s)] for q in QS], "--",
                        color=colours[js], lw=1.0)
            ax.set_yscale("log")
            ax.set_xlabel("q  (Gauss-Legendre stages)")
            ax.set_ylabel("endpoint median error  (um)")
            ax.set_title("%s  -  %s" % (arch, MODE_TITLE[mode]), fontsize=10)
            ax.grid(alpha=0.3, which="both")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=7,
               frameon=False)
    fig.suptitle("C4.2 error against the number of stages, %s split\n"
                 "solid: the network. dotted: the straight line on the same "
                 "rows. dashed: the exact scheme at that q." % split,
                 fontsize=11)
    fig.tight_layout(rect=[0, 0.06, 1, 0.92])
    path = os.path.join(out_dir, "error_vs_q.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# ------------------------------------------------------ 4. physics vs twin --
def physics_over_twin(cells, split, out_dir):
    archs = archs_of(cells)
    ncol = 4
    nrow = int(np.ceil(len(archs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.1 * nrow),
                             squeeze=False)
    ratios = []
    for arch in archs:
        for q in QS:
            for s in STRATA:
                p = cells.get((arch, "physics", q, s))
                t = cells.get((arch, "data", q, s))
                if p and t:
                    ratios.append(np.log10(p["median_um"] / t["median_um"]))
    lim = float(np.max(np.abs(ratios)))
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    for k, arch in enumerate(archs):
        ax = axes[k // ncol][k % ncol]
        M = np.full((len(QS), len(STRATA)), np.nan)
        for i, q in enumerate(QS):
            for j, s in enumerate(STRATA):
                p = cells.get((arch, "physics", q, s))
                t = cells.get((arch, "data", q, s))
                if p and t:
                    M[i, j] = np.log10(p["median_um"] / t["median_um"])
        im = ax.imshow(M, aspect="auto", cmap="coolwarm", norm=norm)
        for i in range(len(QS)):
            for j in range(len(STRATA)):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, "%.2g" % (10 ** M[i, j]), ha="center",
                            va="center", fontsize=5.5)
        ax.set_xticks(range(len(STRATA)))
        ax.set_xticklabels(STRATA, rotation=35, ha="right", fontsize=6)
        ax.set_yticks(range(len(QS)))
        ax.set_yticklabels(["q = %d" % q for q in QS], fontsize=6)
        ax.set_title(arch, fontsize=9)
    for k in range(len(archs), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.subplots_adjust(hspace=0.85, wspace=0.30)
    fig.colorbar(im, ax=axes.ravel().tolist(),
                 label="log10( physics median / twin median )", fraction=0.02)
    fig.suptitle("C4.2 the label-free loss against supervision, %s split\n"
                 "blue: the physics loss is ahead.  red: the twin is ahead."
                 % split, fontsize=11)
    path = os.path.join(out_dir, "physics_over_twin.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


# ------------------------------------------------------------ 5. cost curve --
def cost_vs_error(cells, split, out_dir):
    fig, axes = plt.subplots(len(COST_STRATA), 2, figsize=(11.5, 8.0),
                             squeeze=False)
    for r, s in enumerate(COST_STRATA):
        for col, xkey in enumerate(("n_parameters", "multiply_adds")):
            ax = axes[r][col]
            for mode, colour in (("physics", "tab:blue"), ("data", "tab:red")):
                pts = [(c[xkey], c["median_um"]) for k, c in cells.items()
                       if k[1] == mode and k[3] == s]
                pts.sort()
                ax.plot([p[0] for p in pts], [p[1] for p in pts], "o",
                        color=colour, ms=3, alpha=0.45,
                        label=MODE_TITLE[mode])
                best = np.inf
                front = []
                for x, y in pts:
                    if y < best:
                        best = y
                        front.append((x, y))
                ax.plot([p[0] for p in front], [p[1] for p in front], "-",
                        color=colour, lw=1.8,
                        label="%s, cost-error front" % MODE_TITLE[mode])
            straight = next(c["straight_um"] for k, c in cells.items()
                            if k[3] == s)
            ax.axhline(straight, color="k", lw=1.8, label="straight line")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("parameters" if col == 0
                          else "multiply-adds per forward pass")
            ax.set_ylabel("endpoint median error  (um)")
            ax.set_title("%s" % s, fontsize=10)
            ax.grid(alpha=0.3, which="both")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=7.5,
               frameon=False)
    fig.suptitle("C4.2 cost against error, %s split\n"
                 "every (architecture, q) cell; the line is the best error at "
                 "or below each cost" % split, fontsize=11)
    fig.tight_layout(rect=[0, 0.06, 1, 0.92])
    path = os.path.join(out_dir, "cost_vs_error.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# --------------------------------------------------- 6. the winning arch map --
def best_architecture_per_cell(cells, split, out_dir):
    archs = archs_of(cells)
    lut = {a: i for i, a in enumerate(archs)}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), squeeze=False)
    for col, mode in enumerate(MODES):
        ax = axes[0][col]
        M = np.full((len(QS), len(STRATA)), np.nan)
        for i, q in enumerate(QS):
            for j, s in enumerate(STRATA):
                cand = [(cells[(a, mode, q, s)]["median_um"], a) for a in archs
                        if (a, mode, q, s) in cells]
                if not cand:
                    continue
                cand.sort()
                M[i, j] = lut[cand[0][1]]
                ax.text(j, i, "%s\n%.3g" % (cand[0][1], cand[0][0]),
                        ha="center", va="center", fontsize=5.5)
        im = ax.imshow(M, aspect="auto", cmap="tab20",
                       norm=Normalize(-0.5, len(archs) - 0.5))
        ax.set_xticks(range(len(STRATA)))
        ax.set_xticklabels(STRATA, rotation=35, ha="right", fontsize=6.5)
        ax.set_yticks(range(len(QS)))
        ax.set_yticklabels(["q = %d" % q for q in QS], fontsize=7)
        ax.set_title(MODE_TITLE[mode], fontsize=10)
    cb = fig.colorbar(im, ax=axes[0].tolist(), ticks=range(len(archs)),
                      fraction=0.04)
    cb.ax.set_yticklabels(archs, fontsize=7)
    fig.suptitle("C4.2 which architecture wins each cell, %s split\n"
                 "(depth x width, and the endpoint median it reaches, in um)"
                 % split, fontsize=11)
    path = os.path.join(out_dir, "best_architecture_per_cell.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


# -------------------------------------------------------------------- main --
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", default="test", choices=("val", "test"))
    ap.add_argument("--figures", default=FIGURES)
    a = ap.parse_args(argv)

    os.makedirs(a.figures, exist_ok=True)
    cells = read_cells(os.path.join(RESULTS, "table_cells.csv"), split=a.split)
    scheme = read_scheme()
    written = heatmaps(cells, scheme, a.split, a.figures)
    written.append(error_vs_dz(cells, scheme, a.split, a.figures))
    written.append(error_vs_q(cells, scheme, a.split, a.figures))
    written.append(physics_over_twin(cells, a.split, a.figures))
    written.append(cost_vs_error(cells, a.split, a.figures))
    written.append(best_architecture_per_cell(cells, a.split, a.figures))
    for p in written:
        print("wrote %s" % os.path.relpath(p, HERE))


if __name__ == "__main__":
    main()
