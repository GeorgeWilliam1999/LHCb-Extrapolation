#!/usr/bin/env python
"""C6.4b - the figures, drawn from the C6 csvs alone.

| figure | what it is |
|---|---|
| `figures/chain_growth_<D>x<W>.png` | the cumulative max(\|dx\|, \|dy\|) error against the fraction of the crossing walked, one curve per step length, forward and backward panels, physics arm and data twin. From `results/chain_growth.csv`. |
| `figures/chained_heatmap_<D>x<W>.png` | rows q, columns the six step lengths, physics arm and data twin side by side on one log colour scale, with the single-step full crossing, the straight line and the material floor beneath. From `results/chained_components.csv`. |
| `figures/mini_fig2_<tag>.png` | the mini paper's Figure 2 layout, redrawn for this experiment (see the README's mapping table). |
| `figures/mini_fig3_<tag>.png` | the mini paper's Figure 3 layout, redrawn for this experiment. |
| `figures/components_<tag>.png` | histograms of dx, dy in µm and dtx, dty in mrad at the far plane, the single-step full crossing and every chained column, median and p95 marked. |
| `figures/best_vs_worst_components.png` | the twelve ranked networks' per-component medians side by side. |

The twelve tags the last four are drawn for are the best three and the worst
three of each arm in `results/ranking_extremes.csv`.

    PYTHONNOUSERSITE=1 python plot.py
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import use_shared                                        # noqa: F401
import chain as C

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")
ARCHS = [(d, w) for d in (2, 4, 8) for w in (32, 64, 128, 256)]
QS = tuple(range(2, 21, 2))
COLUMN_NAMES = [c[0] for c in C.COLUMNS]
NOMINAL = {c[0]: (c[1] if c[1] is not None else np.nan) for c in C.COLUMNS}
ARM_LABEL = {"physics": "physics loss", "data": "data twin"}
ARM_COLOUR = {"physics": "tab:blue", "data": "tab:red"}
COLUMN_COLOUR = dict(zip(COLUMN_NAMES,
                         plt.get_cmap("viridis")(np.linspace(0, 0.9,
                                                             len(COLUMN_NAMES)))))


GRID = os.path.join(os.path.dirname(HERE), "Step_size_and_stage_grid")


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def straight_by_stratum():
    """The straight line's own median per training stratum, from the grid."""
    out = {}
    for r in read_csv(os.path.join(GRID, "results", "table_cells.csv")):
        if r["split"] == "test" and r["direction"] == "all":
            out[r["stratum"]] = float(r["straight_um"])
    return out


STRAIGHT_BY_STRATUM = {}


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


# ---------------------------------------------------------- the growth curves --
def growth_figures(growth, out_dir):
    by = defaultdict(lambda: defaultdict(list))
    for r in growth:
        t = r["tag"]
        w, d, q, mode, seed = C.parse_tag(t)
        by[(d, w, mode, r["column"], r["direction"])][
            round(f(r["checkpoint_z_fraction"]), 4)].append(f(r["median_abs"]))
    made = []
    for (d, w) in ARCHS:
        panels = [(mode, direction) for mode in ("physics", "data")
                  for direction in ("forward", "backward")]
        if not any(by.get((d, w, m, col, dr))
                   for m, dr in panels for col in COLUMN_NAMES):
            continue
        fig, ax = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
        for k, (mode, direction) in enumerate(panels):
            a = ax[k // 2][k % 2]
            for col in COLUMN_NAMES:
                pts = by.get((d, w, mode, col, direction), {})
                if not pts:
                    continue
                x = np.array(sorted(pts))
                y = np.array([np.median(pts[v]) for v in x])
                a.plot(x, y, "-o", ms=2.5, lw=1.4, color=COLUMN_COLOUR[col],
                       label=col)
            a.set_yscale("log")
            a.set_title("%s, %s" % (ARM_LABEL[mode], direction), fontsize=10)
            a.grid(alpha=0.3, which="both")
            if k // 2 == 1:
                a.set_xlabel("fraction of the crossing walked")
            if k % 2 == 0:
                a.set_ylabel("median max(|dx|, |dy|) / um")
        ax[0][0].legend(fontsize=7, title="step length", title_fontsize=7,
                        ncol=2)
        fig.suptitle("%d x %d - how the error grows along the chain "
                     "(median over q and seeds)" % (d, w))
        fig.tight_layout()
        p = os.path.join(out_dir, "chain_growth_%dx%d.png" % (d, w))
        fig.savefig(p, dpi=130)
        plt.close(fig)
        made.append(p)
    return made


# ------------------------------------------------------------- the heat maps --
def heatmaps(chained, refs, out_dir):
    by = defaultdict(list)
    for r in chained:
        if (r["direction"] != "all" or r["component"] != "max_xy"
                or r["reference"] != "fine"):
            continue
        by[(int(r["depth"]), int(r["width"]), r["mode"], int(r["q"]),
            r["column"])].append(f(r["median_abs"]))
    ref = {}
    for r in refs:
        if (r["direction"] == "all" and r["component"] == "max_xy"
                and r["reference"] == "fine"):
            ref[(r["predictor"], r["column"])] = f(r["median_abs"])
    made = []
    for (d, w) in ARCHS:
        grids = {}
        for mode in ("physics", "data"):
            g = np.full((len(QS), len(COLUMN_NAMES)), np.nan)
            for i, q in enumerate(QS):
                for j, col in enumerate(COLUMN_NAMES):
                    v = by.get((d, w, mode, q, col))
                    if v:
                        g[i, j] = np.median(v)
            grids[mode] = g
        if np.all(np.isnan(grids["physics"])) and np.all(np.isnan(grids["data"])):
            continue
        extra = np.array([[ref.get(("straight line", c), np.nan)
                           for c in COLUMN_NAMES],
                          [ref.get(("material floor", c), np.nan)
                           for c in COLUMN_NAMES]])
        allv = np.concatenate([grids["physics"].ravel(), grids["data"].ravel(),
                               extra.ravel()])
        allv = allv[np.isfinite(allv) & (allv > 0)]
        vmin, vmax = allv.min(), allv.max()
        fig, ax = plt.subplots(2, 2, figsize=(13, 7),
                               gridspec_kw={"height_ratios": [len(QS), 2]})
        norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
        for k, mode in enumerate(("physics", "data")):
            im = ax[0][k].imshow(grids[mode], aspect="auto", origin="lower",
                                 cmap="viridis", norm=norm)
            ax[0][k].set_yticks(range(len(QS)))
            ax[0][k].set_yticklabels(["q=%d" % q for q in QS], fontsize=7)
            ax[0][k].set_xticks(range(len(COLUMN_NAMES)))
            ax[0][k].set_xticklabels(COLUMN_NAMES, rotation=30, ha="right",
                                     fontsize=7)
            ax[0][k].set_title(ARM_LABEL[mode], fontsize=10)
            for i in range(len(QS)):
                for j in range(len(COLUMN_NAMES)):
                    if np.isfinite(grids[mode][i, j]):
                        ax[0][k].text(j, i, "%.3g" % grids[mode][i, j],
                                      ha="center", va="center", fontsize=5.5,
                                      color="w")
            ax[1][k].imshow(extra, aspect="auto", origin="lower",
                            cmap="viridis", norm=norm)
            ax[1][k].set_yticks([0, 1])
            ax[1][k].set_yticklabels(["straight line", "material floor"],
                                     fontsize=7)
            ax[1][k].set_xticks(range(len(COLUMN_NAMES)))
            ax[1][k].set_xticklabels(COLUMN_NAMES, rotation=30, ha="right",
                                     fontsize=7)
            for i in range(2):
                for j in range(len(COLUMN_NAMES)):
                    if np.isfinite(extra[i, j]):
                        ax[1][k].text(j, i, "%.4g" % extra[i, j], ha="center",
                                      va="center", fontsize=5.5, color="w")
        fig.colorbar(im, ax=ax, label="chained far-plane median "
                                      "max(|dx|, |dy|) / um", shrink=0.8)
        fig.suptitle("%d x %d - the whole crossing walked at every step length"
                     % (d, w))
        p = os.path.join(out_dir, "chained_heatmap_%dx%d.png" % (d, w))
        fig.savefig(p, dpi=130, bbox_inches="tight")
        plt.close(fig)
        made.append(p)
    return made


# ------------------------------------- the mini paper's Figure 3, redrawn ----
def mini_fig3(tag, chained, single, refs, out_dir):
    """Error against the swept variable, with the twin and the ceilings on it.

    The mini paper's Figure 3 is the median endpoint error against the number
    of stages, log vertical axis, with the supervised twin, the exact scheme
    and the straight-line baseline on the same axes and bars for the seed
    spread. Here the swept variable is the **step length**, and the two curves
    are this network's single step and the same network chained across the
    whole crossing at that step length.
    """
    w, d, q, mode, seed = C.parse_tag(tag)
    twin_tag = "w%d_d%d_q%02d_%s_s%d" % (w, d, q,
                                         "data" if mode == "physics"
                                         else "physics", seed)
    ch = defaultdict(dict)
    for r in chained:
        if (r["direction"] == "all" and r["component"] == "max_xy"
                and r["reference"] == "fine"):
            ch[r["tag"]][r["column"]] = f(r["median_abs"])
    ss = defaultdict(dict)
    for r in single:
        if r["direction"] == "all" and r["component"] == "max_xy":
            ss[r["tag"]][r["stratum"]] = f(r["median_abs"])
    ref = {(r["predictor"], r["column"]): f(r["median_abs"]) for r in refs
           if r["direction"] == "all" and r["component"] == "max_xy"
           and r["reference"] == "fine"}
    strata_dz = {"0.05-0.2 mm": 0.1006, "0.5-2 mm": 1.004, "5-20 mm": 10.14,
                 "50-200 mm": 99.77, "500-2000 mm": 998.7,
                 "full crossing": 5175.0}

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for a, (title, lab) in zip(ax, (("Chained across the whole magnet",
                                     "chained far-plane error"),
                                    ("One step of that length", "single step"))):
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xlabel("step length |dz| / mm")
        a.set_ylabel("median max(|dx|, |dy|) / um")
        a.set_title(title, fontsize=11)
        a.grid(alpha=0.3, which="both")
    for t, colour, name in ((tag, ARM_COLOUR[mode], ARM_LABEL[mode]),
                            (twin_tag, ARM_COLOUR["data" if mode == "physics"
                                                  else "physics"],
                             ARM_LABEL["data" if mode == "physics"
                                       else "physics"])):
        x = [NOMINAL[c] if np.isfinite(NOMINAL[c]) else 5175.0
             for c in COLUMN_NAMES if c in ch.get(t, {})]
        y = [ch[t][c] for c in COLUMN_NAMES if c in ch.get(t, {})]
        if x:
            ax[0].plot(x, y, "-o", color=colour, label="%s, %s" % (name, t))
        xs = [strata_dz[s] for s in strata_dz if s in ss.get(t, {})]
        ys = [ss[t][s] for s in strata_dz if s in ss.get(t, {})]
        if xs:
            ax[1].plot(xs, ys, "-s", color=colour, label="%s, %s" % (name, t))
    sl = ref.get(("straight line", "full crossing"), np.nan)
    mf = ref.get(("material floor", "full crossing"), np.nan)
    ax[0].axhline(sl, ls="--", color="grey",
                  label="straight line across the magnet, %.0f um" % sl)
    # the single-step panel's null is the straight line *of that step length*,
    # which is a curve, not a level: it is what the grid records per stratum.
    have = [s for s in strata_dz if s in STRAIGHT_BY_STRATUM]
    ax[1].plot([strata_dz[s] for s in have],
               [STRAIGHT_BY_STRATUM[s] for s in have],
               "--", color="grey", label="straight line of that step length")
    for a in ax:
        a.axhline(mf, ls="-.", color="tab:green",
                  label="material floor, %.0f um" % mf)
        a.axhline(C.REFERENCE_FLOOR_UM, ls=":", color="k",
                  label="fine reference floor, 5e-5 um")
        a.legend(fontsize=7)
    fig.suptitle("%s - error against the step length, chained and single "
                 "(the mini paper's Figure 3 layout)" % tag, fontsize=11)
    fig.tight_layout()
    p = os.path.join(out_dir, "mini_fig3_%s.png" % tag)
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


# ------------------------------------- the mini paper's Figure 2, redrawn ----
def mini_fig2(tag, chains_dir, refs, ranking, out_dir):
    """The error distribution, the per-column summary, and the cost front.

    The mini paper's Figure 2 is a multi-panel summary of one working point:
    the distribution of held-out endpoint errors on a log axis with the ceiling
    and the straight line marked, the run's own histories, and a per-run
    scatter. Here: the distribution of this network's chained far-plane error
    at every step length, the same per column as medians with p95 bars, and -
    the panel that carries Figure 4's cost-versus-error content - where this
    network sits on the cost-error front of all 720.
    """
    path = os.path.join(chains_dir, tag + ".npz")
    if not os.path.exists(path):
        return None
    d = np.load(path)
    ref = {(r["predictor"], r["column"]): f(r["median_abs"]) for r in refs
           if r["direction"] == "all" and r["component"] == "max_xy"
           and r["reference"] == "fine"}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    bins = np.logspace(-1, 7, 60)
    meds, p95s, names = [], [], []
    for col in COLUMN_NAMES:
        key = "far_fine_" + C.COLUMN_KEYS[col]
        if key not in d.files:
            continue
        mx = np.abs(d[key][:, :2]).max(axis=1) * 1e3
        ax[0].hist(mx, bins=bins, histtype="step", lw=1.3,
                   color=COLUMN_COLOUR[col], label=col,
                   weights=np.full(len(mx), 1.0 / len(mx)))
        meds.append(np.median(mx))
        p95s.append(np.quantile(mx, 0.95))
        names.append(col)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("chained far-plane max(|dx|, |dy|) / um")
    ax[0].set_ylabel("fraction of tracks per bin")
    ax[0].axvline(ref.get(("straight line", "full crossing"), np.nan),
                  color="grey", ls="--")
    ax[0].axvline(ref.get(("material floor", "full crossing"), np.nan),
                  color="tab:green", ls="-.")
    ax[0].legend(fontsize=7)
    ax[0].set_title("the distribution, per step length", fontsize=10)

    x = np.arange(len(names))
    ax[1].errorbar(x, meds, yerr=[np.zeros(len(names)),
                                  np.array(p95s) - np.array(meds)],
                   fmt="o", capsize=3, color="tab:blue")
    ax[1].set_yscale("log")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax[1].axhline(ref.get(("straight line", "full crossing"), np.nan),
                  color="grey", ls="--", label="straight line")
    ax[1].axhline(ref.get(("material floor", "full crossing"), np.nan),
                  color="tab:green", ls="-.", label="material floor")
    ax[1].set_ylabel("median (bar to p95) / um")
    ax[1].set_title("median and tail, per step length", fontsize=10)
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3, which="both")

    for mode, colour in ARM_COLOUR.items():
        pts = [(r["multiply_adds"], r["best_median_um"]) for r in ranking
               if r["mode"] == mode and r["multiply_adds"] > 0]
        if pts:
            ax[2].scatter(*zip(*pts), s=6, alpha=0.35, color=colour,
                          label=ARM_LABEL[mode])
    me = [r for r in ranking if r["tag"] == tag]
    if me:
        ax[2].scatter([me[0]["multiply_adds"]], [me[0]["best_median_um"]],
                      s=110, marker="*", color="k", zorder=5,
                      label="%s (rank %d)" % (tag, me[0]["rank"]))
    ax[2].set_xscale("log")
    ax[2].set_yscale("log")
    ax[2].set_xlabel("forward-pass multiply-adds")
    ax[2].set_ylabel("best chained far-plane median / um")
    ax[2].set_title("where it sits on the cost-error front", fontsize=10)
    ax[2].legend(fontsize=7)
    ax[2].grid(alpha=0.3, which="both")

    fig.suptitle("%s - the chained crossing summarised "
                 "(the mini paper's Figure 2 layout)" % tag, fontsize=11)
    fig.tight_layout()
    p = os.path.join(out_dir, "mini_fig2_%s.png" % tag)
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


# ---------------------------------------------------- the component histograms --
def components_figure(tag, chains_dir, single_dir, out_dir):
    cpath = os.path.join(chains_dir, tag + ".npz")
    if not os.path.exists(cpath):
        return None
    d = np.load(cpath)
    labels = ("dx / um", "dy / um", "dtx / mrad", "dty / mrad")
    fig, ax = plt.subplots(4, 1, figsize=(10, 12))
    series = []
    spath = os.path.join(single_dir, tag + ".npz")
    if os.path.exists(spath):
        s = np.load(spath)
        m = np.asarray(s["STRATUM"]) == 5
        series.append(("single step, full crossing",
                       np.asarray(s["d"])[m] * 1e3, "k"))
    for col in COLUMN_NAMES:
        key = "far_fine_" + C.COLUMN_KEYS[col]
        if key not in d.files:
            continue
        series.append(("chained, " + col, d[key] * 1e3, COLUMN_COLOUR[col]))
    for j in range(4):
        a = ax[j]
        for name, e, colour in series:
            v = np.asarray(e)[:, j]
            v = v[np.isfinite(v)]
            if not len(v):
                continue
            lim = np.quantile(np.abs(v), 0.99) * 1.2 + 1e-12
            a.hist(v, bins=np.linspace(-lim, lim, 80), histtype="step",
                   lw=1.2, color=colour, density=True, label=name)
            a.axvline(np.median(v), color=colour, lw=0.8, ls="--")
            a.axvline(np.quantile(np.abs(v), 0.95), color=colour, lw=0.6,
                      ls=":")
        a.set_xlabel(labels[j])
        a.set_ylabel("density")
        a.set_yscale("log")
        a.grid(alpha=0.3)
        if j == 0:
            a.legend(fontsize=7)
    fig.suptitle("%s - the far-plane error, component by component "
                 "(dashed = median of the component, dotted = p95 of |.|)"
                 % tag, fontsize=11)
    fig.tight_layout()
    p = os.path.join(out_dir, "components_%s.png" % tag)
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def best_vs_worst(extremes, chained, out_dir):
    tags = [r["tag"] for r in extremes]
    by = defaultdict(dict)
    for r in chained:
        if r["direction"] != "all" or r["reference"] != "fine":
            continue
        if r["tag"] in tags:
            by[(r["tag"], r["column"])][r["component"]] = f(r["median_abs"])
    best_col = {r["tag"]: r["best_column"] for r in extremes}
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    comps = [("x", "y", "max_xy"), ("tx", "ty")]
    units = ("um", "mrad")
    x = np.arange(len(tags))
    for k, group in enumerate(comps):
        for i, comp in enumerate(group):
            vals = [by.get((t, best_col[t]), {}).get(comp, np.nan)
                    for t in tags]
            ax[k].bar(x + (i - (len(group) - 1) / 2) * 0.8 / len(group),
                      vals, width=0.8 / len(group), label=comp)
        ax[k].set_yscale("log")
        ax[k].set_xticks(x)
        ax[k].set_xticklabels(
            ["%s\n%s %s" % (r["tag"], r["arm"], r["group"]) for r in extremes],
            rotation=90, fontsize=6)
        ax[k].set_ylabel("median |error| / %s" % units[k])
        ax[k].legend(fontsize=8)
        ax[k].grid(alpha=0.3, axis="y", which="both")
    fig.suptitle("the twelve ranked networks, per component, each at its own "
                 "best step length", fontsize=11)
    fig.tight_layout()
    p = os.path.join(out_dir, "best_vs_worst_components.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=RESULTS)
    ap.add_argument("--figures", default=FIGURES)
    a = ap.parse_args(argv)
    os.makedirs(a.figures, exist_ok=True)
    STRAIGHT_BY_STRATUM.update(straight_by_stratum())

    chained = read_csv(os.path.join(a.results, "chained_components.csv"))
    single = read_csv(os.path.join(a.results, "single_step_components.csv"))
    refs = read_csv(os.path.join(a.results, "reference_rows.csv"))
    growth = read_csv(os.path.join(a.results, "chain_growth.csv"))
    ranking = [dict(r, rank=int(r["rank"]),
                    best_median_um=f(r["best_median_um"]),
                    multiply_adds=int(r["multiply_adds"]))
               for r in read_csv(os.path.join(a.results, "ranking.csv"))]
    extremes = read_csv(os.path.join(a.results, "ranking_extremes.csv"))

    made = growth_figures(growth, a.figures)
    made += heatmaps(chained, refs, a.figures)
    chains_dir = os.path.join(a.results, "chains")
    single_dir = os.path.join(a.results, "single_step")
    for r in extremes:
        for p in (mini_fig3(r["tag"], chained, single, refs, a.figures),
                  mini_fig2(r["tag"], chains_dir, refs, ranking, a.figures),
                  components_figure(r["tag"], chains_dir, single_dir,
                                    a.figures)):
            if p:
                made.append(p)
    if extremes:
        made.append(best_vs_worst(extremes, chained, a.figures))
    print("%d figures written to %s" % (len(made), a.figures))
    for p in made:
        print("   " + os.path.relpath(p, HERE))


if __name__ == "__main__":
    main()
