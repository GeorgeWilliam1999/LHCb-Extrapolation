#!/usr/bin/env python
"""C2.3 - the two figures, drawn from the csv tables only.

`figures/scheme_error_vs_dz_and_q.png`

  1. endpoint error against step length, log-log, one curve per stage count q,
     the straight line (magnet off) dashed and the reference's own floor of
     5e-5 micron marked. The step length on the x axis is the median |dz| of
     the stratum.
  2. endpoint error against q, one curve per stratum. This is where the
     classical order 2q is visible - or not.

`figures/scheme_convergence_fraction.png`

  the fraction of states the root-finder solved to the 1e-9 residual
  tolerance, as a q x stratum map and as curves against q, plus the median
  number of residual evaluations one solve cost.

Run:
    PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python plot.py
"""
from __future__ import annotations

import csv
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")

STRATA = ("0.05-0.2 mm", "0.5-2 mm", "5-20 mm", "50-200 mm", "500-2000 mm",
          "full crossing")

# The fine reference's own floor (../C1_Fine_reference/README.md, C1.4).
REFERENCE_FLOOR_UM = 5e-5


def read(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in list(r.items()):
            if k in ("stratum", "direction"):
                continue
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                r[k] = np.nan
    return rows


def main():
    os.makedirs(FIGURES, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    S = [r for r in read(os.path.join(RESULTS, "scheme_table.csv"))
         if r["direction"] == "both"]
    L = [r for r in read(os.path.join(RESULTS, "straight_line_table.csv"))
         if r["direction"] == "both"]
    qs = sorted({int(r["q"]) for r in S})
    by = {(int(r["q"]), r["stratum"]): r for r in S}
    line = {r["stratum"]: r for r in L}
    dzmed = {s: line[s]["median_abs_dz_mm"] for s in STRATA}

    cmap = plt.get_cmap("viridis")
    qcol = {q: cmap(i / max(1, len(qs) - 1)) for i, q in enumerate(qs)}
    scol = {s: plt.get_cmap("plasma")(i / (len(STRATA) - 1))
            for i, s in enumerate(STRATA)}

    # ---------------- figure 1 -------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.4))

    x = [dzmed[s] for s in STRATA]
    for q in qs:
        y = [by[(q, s)]["endpoint_med_um"] for s in STRATA]
        ax[0].plot(x, y, "o-", ms=4, lw=1.4, color=qcol[q], label="q = %d" % q)
    ax[0].plot(x, [line[s]["endpoint_med_um"] for s in STRATA], "k--o", ms=5,
               lw=1.8, label="straight line (magnet off)")
    ax[0].axhline(REFERENCE_FLOOR_UM, color="crimson", ls=":", lw=1.5)
    ax[0].text(x[0], REFERENCE_FLOOR_UM * 1.6,
               "fine reference floor on a full crossing, 5e-5 $\\mu$m",
               color="crimson", fontsize=8, va="bottom")
    # guide slopes: what the classical local order 2q+1 would look like at
    # q = 2, and the effective order the kinked map actually delivers
    xg = np.array([120.0, 5200.0])
    for power, style, label in ((5.0, "-.", "slope 5 = classical 2q+1 at q = 2"),
                                (3.2, (0, (1, 1)), "slope 3.2 = measured")):
        yg = by[(2, "50-200 mm")]["endpoint_med_um"] * (xg / xg[0]) ** power
        ax[0].plot(xg, yg, color="0.45", ls=style, lw=1.2, label=label)
    ax[0].axvline(100.0, color="0.6", ls="--", lw=0.9)
    ax[0].text(88.0, 3e6, "one 100 mm\nfield cell", color="0.35", fontsize=8,
               ha="right", va="top")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("step length |dz| [mm]  (median of the stratum)")
    ax[0].set_ylabel("endpoint error vs RK6 reference [$\\mu$m], median")
    ax[0].set_title("The exact scheme's own error against step length")
    ax[0].grid(alpha=0.3, which="both")
    ax[0].legend(fontsize=6.5, ncol=2, loc="upper left")

    for s in STRATA:
        y = [by[(q, s)]["endpoint_med_um"] for q in qs]
        ax[1].plot(qs, y, "o-", ms=4, lw=1.4, color=scol[s],
                   label="%s  (|dz| ~ %.3g mm)" % (s, dzmed[s]))
        ax[1].axhline(line[s]["endpoint_med_um"], color=scol[s], ls=":", lw=0.8)
    ax[1].axhline(REFERENCE_FLOOR_UM, color="crimson", ls=":", lw=1.5)
    ax[1].set_yscale("log")
    ax[1].set_xticks(qs)
    ax[1].set_xlabel("stage count q   (classical order 2q)")
    ax[1].set_ylabel("endpoint error vs RK6 reference [$\\mu$m], median")
    ax[1].set_title("Against stage count; dotted = that stratum's straight line")
    ax[1].grid(alpha=0.3, which="both")
    ax[1].legend(fontsize=7, loc="center left")

    fig.suptitle("Exact Gauss-Legendre scheme, no network: %d test states per "
                 "cell, v8r1.up, vs the 0.1 mm RK6 reference"
                 % int(by[(qs[0], STRATA[0])]["n"]), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p1 = os.path.join(FIGURES, "scheme_error_vs_dz_and_q.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print("wrote", os.path.relpath(p1, HERE))

    # ---------------- figure 2 -------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.8))

    M = np.array([[by[(q, s)]["converged_frac"] for s in STRATA] for q in qs])
    im = ax[0].imshow(M, aspect="auto", origin="lower", vmin=min(0.9, M.min()),
                      vmax=1.0, cmap="RdYlGn")
    ax[0].set_xticks(range(len(STRATA)))
    ax[0].set_xticklabels(STRATA, rotation=35, ha="right", fontsize=8)
    ax[0].set_yticks(range(len(qs)))
    ax[0].set_yticklabels(qs)
    ax[0].set_ylabel("stage count q")
    ax[0].set_title("Converged fraction (residual < 1e-9 mm / mrad)")
    for i in range(len(qs)):
        for j in range(len(STRATA)):
            ax[0].text(j, i, "%.3f" % M[i, j], ha="center", va="center",
                       fontsize=6.5)
    fig.colorbar(im, ax=ax[0])

    for j, s in enumerate(STRATA):
        ax[1].plot(qs, M[:, j], "o-", ms=4, color=scol[s], label=s)
    ax[1].set_xticks(qs)
    ax[1].set_ylim(-0.02, 1.05)
    ax[1].set_xlabel("stage count q")
    ax[1].set_ylabel("converged fraction")
    ax[1].set_title("The same, as curves")
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=7, loc="center left")

    for j, s in enumerate(STRATA):
        ax[2].plot(qs, [by[(q, s)]["median_n_eval"] for q in qs], "o-", ms=4,
                   color=scol[s], label=s)
    ax[2].plot(qs, [4 * q + 5 for q in qs], "k--", lw=1.2,
               label="4q + 5  (one jacobian + a few steps)")
    ax[2].set_xticks(qs)
    ax[2].set_xlabel("stage count q")
    ax[2].set_ylabel("residual evaluations per solve, median")
    ax[2].set_title("What one solve costs")
    ax[2].grid(alpha=0.3)
    ax[2].legend(fontsize=7, loc="upper left")

    fig.tight_layout()
    p2 = os.path.join(FIGURES, "scheme_convergence_fraction.png")
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print("wrote", os.path.relpath(p2, HERE))


if __name__ == "__main__":
    main()
