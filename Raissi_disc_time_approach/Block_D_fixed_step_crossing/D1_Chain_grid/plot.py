#!/usr/bin/env python
"""D1 - the figures, from the CSV tables of aggregate.py alone.

    figures/heatmap_vs_rk6.png        log10 median endpoint error (test, vs the
                                      RK6 truth) over N x q, one-hue ramp
    figures/error_vs_q.png            error against stage count, one line per N,
                                      the exact scheme dashed, the straight line
                                      and the material floor as reference lines
    figures/error_vs_N.png            error against step count at chosen q
    figures/growth_along_crossing.png the chain's error plane by plane, one
                                      panel per N, q on a one-hue ramp
    figures/own_step_vs_inherited.png each leg's own-step error against the
                                      chain's error at its plane
    figures/vs_real_scifi.png         the best chain against the real SciFi
                                      state, per momentum band, with the floor
    figures/components_vs_q.png       x, y, tx, ty separately against q, one
                                      line per N, the exact scheme dashed
    figures/components_growth.png     x, y, tx, ty separately along the
                                      crossing, one panel per component, best
                                      q per N

Colours: the five step counts in a fixed categorical order (blue, orange, aqua,
yellow, magenta), magnitudes on a single-hue blue ramp; every series carries a
direct label or a legend entry, and text stays in ink, never in series colour.
"""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")
D2 = os.path.join(HERE, "..", "D2_Comparators", "results")
D0_META = os.path.join(HERE, "..", "D0_Crossing_dataset", "results", "crossing_particles_meta.json")
N_VALUES = (1, 4, 16, 64, 128)
QS = tuple(range(1, 21))
CAT = {1: "#2a78d6", 4: "#eb6834", 16: "#1baf7a", 64: "#eda100", 128: "#e87ba4"}
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
INK, INK2, GRID = "#1a1a19", "#5f5e5a", "#e3e2dc"

plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False})


def read(name):
    p = os.path.join(RESULTS, name)
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def main():
    os.makedirs(FIGURES, exist_ok=True)
    table = read("chain_table.csv")
    growth = read("growth.csv")
    per_leg = read("per_leg.csv")
    bands = read("by_p_band.csv")
    if not table:
        print("no chain_table.csv yet; run aggregate.py")
        return
    med = defaultdict(dict)                        # comparator -> (N,q) -> median
    for r in table:
        if r["split"] == "test":
            med[r["comparator"]][(int(r["N"]), int(r["q"]))] = fnum(r["pos_med_um"])
    rk6 = med["vs_rk6_endpoint"]
    exact = med.get("exact_scheme_vs_rk6_endpoint", {})
    straight = np.nanmedian([v for v in med["straight_line_vs_rk6_endpoint"].values()])
    floor = np.nanmedian([v for v in med["rk6_truth_vs_real_scifi_state"].values()])
    twin = None
    tp = os.path.join(D2, "twin", "twin_scores.json")
    if os.path.exists(tp):
        with open(tp) as f:
            twin = json.load(f)["test"]["vs_rk6_endpoint"]["pos_med_um"]

    # 1. the heat map
    M = np.full((len(N_VALUES), len(QS)), np.nan)
    for i, N in enumerate(N_VALUES):
        for j, q in enumerate(QS):
            M[i, j] = rk6.get((N, q), np.nan)
    fig, ax = plt.subplots(figsize=(12, 4.2))
    finite = M[np.isfinite(M)]
    if finite.size:
        im = ax.imshow(M, cmap=BLUES, norm=LogNorm(vmin=max(finite.min(), 1e-3),
                                                   vmax=finite.max()), aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, "%.3g" % M[i, j], ha="center", va="center", fontsize=7,
                            color="white" if M[i, j] > np.sqrt(finite.min() * finite.max()) else INK)
        cb = fig.colorbar(im, ax=ax, pad=0.02)
        cb.set_label("median endpoint error vs RK6  [µm]")
    ax.set_xticks(range(len(QS)))
    ax.set_xticklabels(QS)
    ax.set_yticks(range(len(N_VALUES)))
    ax.set_yticklabels(["N = %d" % N for N in N_VALUES])
    ax.set_xlabel("stages q")
    ax.grid(False)
    ax.set_title("Chained crossing, test split: median endpoint error against the RK6 truth")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "heatmap_vs_rk6.png"), dpi=140)
    plt.close(fig)

    # 2. error vs q
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for N in N_VALUES:
        ys = [rk6.get((N, q), np.nan) for q in QS]
        ax.plot(QS, ys, "-o", ms=4, lw=1.8, color=CAT[N], label="N = %d networks" % N)
        ye = [exact.get((N, q), np.nan) for q in QS]
        if np.isfinite(ye).any():
            ax.plot(QS, ye, "--", lw=1.2, color=CAT[N], alpha=0.8)
    ax.axhline(straight, color=INK2, lw=1, ls=":")
    ax.text(QS[-1], straight, " straight line", va="center", fontsize=8, color=INK2)
    ax.axhline(floor, color=INK2, lw=1, ls="-.")
    ax.text(QS[-1], floor, " material floor", va="center", fontsize=8, color=INK2)
    if twin:
        ax.axhline(twin, color=INK, lw=1, ls="--")
        ax.text(QS[-1], twin, " supervised twin", va="center", fontsize=8, color=INK)
    ax.set_yscale("log")
    ax.set_xlabel("stages q  (solid: networks; dashed: the exact scheme at the same N and q)")
    ax.set_ylabel("median endpoint error vs RK6  [µm], test")
    ax.set_xticks(QS)
    ax.legend(loc="upper right")
    ax.set_title("Error at z1 against stage count, for each number of steps")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "error_vs_q.png"), dpi=140)
    plt.close(fig)

    # 3. error vs N at chosen q, q on the one-hue ramp
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for j, q in enumerate((1, 2, 4, 8, 12, 16, 20)):
        ys = [rk6.get((N, q), np.nan) for N in N_VALUES]
        ax.plot(N_VALUES, ys, "-o", ms=4, lw=1.6, color=BLUES(0.25 + 0.75 * j / 6),
                label="q = %d" % q)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(N_VALUES)
    ax.set_xticklabels(["%d\n(%.0f mm)" % (N, 5177.8 / N) for N in N_VALUES])
    ax.set_xlabel("steps N across the crossing (step length)")
    ax.set_ylabel("median endpoint error vs RK6  [µm], test")
    ax.legend()
    ax.set_title("Error at z1 against the number of steps")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "error_vs_N.png"), dpi=140)
    plt.close(fig)

    # 4. growth along the crossing
    fig, axes = plt.subplots(1, len(N_VALUES), figsize=(17, 4), sharey=True)
    g = defaultdict(list)
    for r in growth:
        if r["split"] == "test":
            g[(int(r["N"]), int(r["q"]))].append((int(r["plane"]), fnum(r["z_mm"]),
                                                 fnum(r["pos_med_um"])))
    for ax, N in zip(axes, N_VALUES):
        for q in QS:
            pts = sorted(g.get((N, q), []))
            if not pts:
                continue
            ax.plot([p[1] for p in pts], [max(p[2], 1e-4) for p in pts], "-", lw=1.2,
                    color=BLUES(0.2 + 0.8 * (q - 1) / 19))
        ax.set_yscale("log")
        ax.set_title("N = %d" % N)
        ax.set_xlabel("z  [mm]")
    axes[0].set_ylabel("median error vs RK6 truth  [µm]")
    sm = plt.cm.ScalarMappable(cmap=BLUES, norm=plt.Normalize(1, 20))
    fig.colorbar(sm, ax=axes, pad=0.01, label="stages q")
    fig.suptitle("The chain's error plane by plane along the crossing (test split)")
    fig.savefig(os.path.join(FIGURES, "growth_along_crossing.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)

    # 5. own-step vs inherited
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for N in N_VALUES:
        xs = [fnum(r["own_step_test_endpoint_med_um"]) for r in per_leg if int(r["N"]) == N]
        ys = [fnum(r["chain_test_pos_med_um"]) for r in per_leg if int(r["N"]) == N]
        ax.plot(xs, ys, "o", ms=3.5, alpha=0.7, color=CAT[N], label="N = %d" % N)
    lim = [1e-3, 1e5]
    ax.plot(lim, lim, "-", lw=0.8, color=INK2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("the leg's own-step error (network vs RK6 of its own inputs)  [µm]")
    ax.set_ylabel("the chain's error at that leg's end plane  [µm]")
    ax.legend()
    ax.set_title("What each leg adds against what it inherits")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "own_step_vs_inherited.png"), dpi=140)
    plt.close(fig)

    # 6. the best chain against the real SciFi state, per momentum band
    if bands and rk6:
        best = min(rk6, key=lambda k: rk6[k] if np.isfinite(rk6[k]) else np.inf)
        rows = [r for r in bands if int(r["N"]) == best[0] and int(r["q"]) == best[1]
                and r["split"] == "test"]
        names = sorted({r["p_band"] for r in rows}, key=lambda s: float(s.split("-")[0]))
        x = np.arange(len(names))
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for off, comp, lab, col in ((-0.3, "vs_rk6_endpoint", "network vs RK6", "#2a78d6"),
                                    (0.0, "vs_real_scifi_state", "network vs real SciFi state", "#eb6834"),
                                    (0.3, "rk6_truth_vs_real_scifi_state", "RK6 truth vs real SciFi state (material floor)", "#1baf7a")):
            ys = [next((fnum(r["pos_med_um"]) for r in rows if r["comparator"] == comp
                        and r["p_band"] == nm), np.nan) for nm in names]
            ax.bar(x + off, ys, width=0.28, color=col, label=lab)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_xlabel("momentum band")
        ax.set_ylabel("median endpoint error  [µm]")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title("Best chain (N = %d, q = %d) against both truths, per momentum band" % best)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES, "vs_real_scifi.png"), dpi=140)
        plt.close(fig)
    # 7. the components against q
    comps = read("components.csv")
    if comps:
        cm = defaultdict(dict)
        for r in comps:
            if r["split"] == "test":
                cm[(r["comparator"], r["component"])][(int(r["N"]), int(r["q"]))] = fnum(r["med"])
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        for ax, (name, unit) in zip(axes.ravel(), (("x", "µm"), ("y", "µm"), ("tx", "mrad"), ("ty", "mrad"))):
            for N in N_VALUES:
                ys = [cm[("vs_rk6_endpoint", name)].get((N, q), np.nan) for q in QS]
                ax.plot(QS, ys, "-o", ms=3.5, lw=1.6, color=CAT[N], label="N = %d" % N)
                ye = [cm.get(("exact_scheme_vs_rk6_endpoint", name), {}).get((N, q), np.nan) for q in QS]
                if np.isfinite(ye).any():
                    ax.plot(QS, ye, "--", lw=1.1, color=CAT[N], alpha=0.8)
            ax.set_yscale("log")
            ax.set_title("%s: median |Δ%s| at z1 vs RK6  [%s]" % (name, name, unit))
            ax.set_xticks(QS)
        for ax in axes[1]:
            ax.set_xlabel("stages q  (solid: networks; dashed: exact scheme)")
        axes[0][0].legend(loc="upper right")
        fig.suptitle("Each component separately (test split); q/p is carried through unchanged")
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES, "components_vs_q.png"), dpi=140)
        plt.close(fig)

        # 8. the components along the crossing, best q per N
        g2 = defaultdict(list)
        for r in growth:
            if r["split"] == "test":
                g2[(int(r["N"]), int(r["q"]))].append(r)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        for ax, (name, unit) in zip(axes.ravel(), (("x", "um"), ("y", "um"), ("tx", "mrad"), ("ty", "mrad"))):
            key = "%s_med_%s" % (name, unit)
            for N in N_VALUES:
                cands = {q: cm[("vs_rk6_endpoint", name)].get((N, q), np.nan) for q in QS}
                cands = {q: v for q, v in cands.items() if np.isfinite(v)}
                if not cands:
                    continue
                qb = min(cands, key=cands.get)
                pts = sorted(g2.get((N, qb), []), key=lambda r: int(r["plane"]))
                if pts and pts[0].get(key, "") != "":
                    ax.plot([fnum(r["z_mm"]) for r in pts], [max(fnum(r[key]), 1e-5) for r in pts],
                            "-", lw=1.6, color=CAT[N], label="N = %d (q = %d)" % (N, qb))
            ax.set_yscale("log")
            ax.set_title("%s: median |Δ%s| along the crossing  [%s]" % (name, name, unit.replace("um", "µm")))
        for ax in axes[1]:
            ax.set_xlabel("z  [mm]")
        axes[0][0].legend(fontsize=8)
        fig.suptitle("Each component along the crossing, at each N's best q (test split)")
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES, "components_growth.png"), dpi=140)
        plt.close(fig)
    print("figures written to", FIGURES)


if __name__ == "__main__":
    main()
