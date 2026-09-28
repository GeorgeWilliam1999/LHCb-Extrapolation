#!/usr/bin/env python
"""Figure 3 - the exact-collocation ceiling, and how far above it the networks sit.

WHAT THE FIGURE SHOWS

  Both panels carry the same sixteen (N, q) points twice.  The LINES are the
  exact collocation scheme chained N times and solved without a network: the
  radial endpoint error a network of that step count and stage count could
  reach if its residual were driven to zero.  The MARKERS are the trained
  pooled-loss networks at the checkpoint of 18 September 2026, scored the same
  way on the same 1,452 test tracks.  A thin grey connector joins each network
  to its own ceiling, and the ratio is printed for the extreme cases.

  (a) against the stage count q, one line per step count N.
  (b) against the step length dz, one line per stage count q.  dz is fixed by
      N (dz = L / N with L = 5177.8 mm), so this is the same sixteen points
      transposed.

  The point of the figure is the vertical distance: for every chain of 64 steps
  or more the trained network sits two to six orders of magnitude above the
  scheme it is approximating, so nothing the networks do is limited by the
  collocation order.

  The exact scheme has its own stored JSON only for N = 2 and N = 256; the
  N = 64 and N = 128 ceilings come from E3_Analysis/results/error_qdz_chain.csv
  alone.  That distinction is carried in the companion CSV as the column
  `exact_json_present`; it is not drawn on the figure, because the radial
  median plotted here is taken from the CSV for all sixteen points and is
  therefore the same quantity throughout.

INPUT FILES (all opened read-only)

  ../results/tab_exact_ceiling.csv
      N, q, dz_mm, exact_radial_med_um_from_csv, n, exact_json_present.
  ../results/paper_numbers.json
      pooled_grid_snapshot.per_run: radial_med_um of each of the sixteen
      18 September networks (and exact_radial_med_um, cross-checked here
      against the CSV);
      scheme.exact_ceiling.what for the provenance note printed on the run.

OUTPUT

  ../figures/fig_ceiling.png
  ../results/fig03_ceiling.csv   N, q, dz, ceiling, network, ratio, flag

RUN

  cd Self_chained_paper/scripts
  PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python fig03_ceiling.py
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"

import sys                                            # noqa: E402
sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in ("", ".", HERE):
    while _p in sys.path:
        sys.path.remove(_p)

import importlib.util                                 # noqa: E402

import numpy as np                                    # noqa: E402
import matplotlib                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402

_spec = importlib.util.spec_from_file_location("sc_paper_common",
                                               os.path.join(HERE, "common.py"))
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

FIGURES = os.path.join(C.PAPER, "figures")
PNG = os.path.join(FIGURES, "fig_ceiling.png")
CSV = os.path.join(C.RESULTS, "fig03_ceiling.csv")

NS = (2, 64, 128, 256)
QS = (2, 4, 8, 16)
N_COLOUR = {2: C.TEXT1, 64: C.BLUE, 128: C.GREEN, 256: C.MAGENTA}
Q_COLOUR = {2: C.TEXT1, 4: C.BLUE, 8: C.GREEN, 16: C.MAGENTA}
N_MARKER = {2: "s", 64: "o", 128: "^", 256: "D"}
Q_MARKER = {2: "s", 4: "o", 8: "^", 16: "D"}


def style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150,
        "figure.facecolor": C.SURFACE, "axes.facecolor": C.SURFACE,
        "savefig.facecolor": C.SURFACE, "savefig.edgecolor": C.SURFACE,
        "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9,
        "axes.edgecolor": C.TEXT2, "axes.labelcolor": C.TEXT1,
        "text.color": C.TEXT1, "xtick.color": C.TEXT2, "ytick.color": C.TEXT2,
        "axes.linewidth": 0.8, "legend.frameon": False,
        "grid.color": C.NEUTRAL, "grid.alpha": 0.6, "grid.linewidth": 0.6,
    })


def main():
    style()
    J = C.read_json(os.path.join(C.RESULTS, "paper_numbers.json"))
    pooled = J["pooled_grid_snapshot"]["per_run"]
    ceil_rows = C.read_csv(os.path.join(C.RESULTS, "tab_exact_ceiling.csv"))

    D = {}
    for r in ceil_rows:
        N, q = int(r["N"]), int(r["q"])
        p = pooled[C.run_key(N, q)]
        ceiling = float(r["exact_radial_med_um_from_csv"])
        # the same ceiling is carried in the pooled block; they must agree
        assert abs(p["exact_radial_med_um"] - ceiling) < 1e-9 * max(1.0, ceiling)
        D[(N, q)] = dict(N=N, q=q, dz_mm=float(r["dz_mm"]),
                         n=int(r["n"]),
                         exact_json_present=int(r["exact_json_present"]),
                         ceiling_radial_med_um=ceiling,
                         network_radial_med_um=float(p["radial_med_um"]),
                         network_over_ceiling=float(p["radial_med_um"]) / ceiling)

    fig, axes = plt.subplots(1, 2, figsize=(6.3, 4.1))

    # ------------------------------------------------------------ (a) vs q --
    ax = axes[0]
    for i, N in enumerate(NS):
        jit = 2.0 ** (0.075 * (i - 1.5))
        xs = [q * jit for q in QS]
        y_c = [D[(N, q)]["ceiling_radial_med_um"] for q in QS]
        y_n = [D[(N, q)]["network_radial_med_um"] for q in QS]
        ax.plot(xs, y_c, "-", lw=1.5, color=N_COLOUR[N], zorder=3,
                label=r"$N$ = %d,  $\Delta z$ = %.1f mm" % (N, D[(N, 2)]["dz_mm"]))
        ax.plot(xs, y_n, N_MARKER[N], ms=5.5, mfc=N_COLOUR[N], mec=C.SURFACE,
                mew=0.6, zorder=4)
        for x, a, b in zip(xs, y_c, y_n):
            ax.plot([x, x], [a, b], "-", lw=0.8, color=C.NEUTRAL, zorder=2)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(QS)
    ax.set_xticklabels([str(q) for q in QS])
    ax.set_xlim(1.6, 20)
    ax.set_xlabel("stage count  q")
    ax.set_ylabel(r"endpoint error, radial median  ($\mu$m)")
    ax.grid(True, which="both")
    ax.set_title("(a)  against stage count", loc="left", pad=4)
    ax.legend(loc="upper right", handlelength=1.8, labelspacing=0.25)

    # ----------------------------------------------------------- (b) vs dz --
    ax = axes[1]
    dzs = [D[(N, 2)]["dz_mm"] for N in NS]
    for i, q in enumerate(QS):
        jit = 10.0 ** (0.022 * (i - 1.5))
        xs = [dz * jit for dz in dzs]
        y_c = [D[(N, q)]["ceiling_radial_med_um"] for N in NS]
        y_n = [D[(N, q)]["network_radial_med_um"] for N in NS]
        ax.plot(xs, y_c, "-", lw=1.5, color=Q_COLOUR[q], zorder=3,
                label="q = %d" % q)
        ax.plot(xs, y_n, Q_MARKER[q], ms=5.5, mfc=Q_COLOUR[q], mec=C.SURFACE,
                mew=0.6, zorder=4)
        for x, a, b in zip(xs, y_c, y_n):
            ax.plot([x, x], [a, b], "-", lw=0.8, color=C.NEUTRAL, zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"step length  $\Delta z$  (mm)")
    ax.grid(True, which="both")
    ax.set_title("(b)  against step length", loc="left", pad=4)
    ax.legend(loc="upper left", handlelength=1.8, labelspacing=0.25)

    lo = min(min(v["ceiling_radial_med_um"] for v in D.values()),
             min(v["network_radial_med_um"] for v in D.values()))
    hi = max(max(v["ceiling_radial_med_um"] for v in D.values()),
             max(v["network_radial_med_um"] for v in D.values()))
    for ax in axes:
        ax.set_ylim(0.1 * lo, 300.0 * hi)
    axes[1].tick_params(labelleft=False)

    # ---- the message: the vertical gap, called out where it is largest ------
    worst = max(D.values(), key=lambda v: v["network_over_ceiling"])
    best64 = D[(64, 2)]
    for v, frac, fy in ((best64, 0.27, 0.16), (worst, 0.74, 0.055)):
        ymid = np.sqrt(v["ceiling_radial_med_um"] * v["network_radial_med_um"])
        axes[0].annotate(r"$\times$%s  ($N$=%d, q=%d)"
                         % (_fmt_ratio(v["network_over_ceiling"]), v["N"], v["q"]),
                         xy=(v["q"], ymid), xytext=(frac, fy),
                         textcoords="axes fraction", ha="center", va="bottom",
                         fontsize=8.5, color=C.TEXT2,
                         arrowprops=dict(arrowstyle="-", lw=0.7, color=C.TEXT2,
                                         shrinkA=2, shrinkB=2))

    fig.text(0.5, 0.008,
             "markers: trained network, 18 Sept snapshot      "
             "lines: exact collocation scheme, same (N, q)",
             ha="center", va="bottom", fontsize=8, color=C.TEXT2)

    fig.subplots_adjust(left=0.135, right=0.985, top=0.935, bottom=0.155,
                        wspace=0.06)
    os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(PNG, dpi=150, facecolor=C.SURFACE)
    plt.close(fig)

    rows = [D[(N, q)] for N in NS for q in QS]
    C.write_csv(CSV, rows, ["N", "q", "dz_mm", "n", "exact_json_present",
                            "ceiling_radial_med_um", "network_radial_med_um",
                            "network_over_ceiling"])
    print("wrote %s" % PNG)
    print("wrote %s" % CSV)
    print("network / ceiling: min %.3g (N=%d,q=%d), max %.3g (N=%d,q=%d)"
          % (min(v["network_over_ceiling"] for v in D.values()),
             min(D.values(), key=lambda v: v["network_over_ceiling"])["N"],
             min(D.values(), key=lambda v: v["network_over_ceiling"])["q"],
             worst["network_over_ceiling"], worst["N"], worst["q"]))
    print("exact JSON present for N = 2 and N = 256 only; the other eight rows "
          "take the ceiling from error_qdz_chain.csv (flagged in the CSV).")


def _fmt_ratio(r):
    if r >= 1e4:
        e = int(np.floor(np.log10(r)))
        return r"10$^{%d}$" % e
    if r >= 100:
        return "%d" % round(r, -1)
    return "%.3g" % r


if __name__ == "__main__":
    main()
