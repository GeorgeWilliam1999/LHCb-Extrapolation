#!/usr/bin/env python
"""Figure 5 - three references for the same crossing, and which one matters.

A network's endpoint can be compared with three different things, and the
figure puts all three side by side in every momentum band:

  network vs RK6      the field-only order-6 reference at z1: what the network
                      was trained to reproduce;
  network vs true     the particle's real Geant4 state on its own first SciFi
                      plane, the network endpoint carried there with RK6;
  RK6 vs true         the same reference carried to the same plane against the
                      same real state: the material floor, which no field-only
                      extrapolator can go below.

WHAT THE FIGURE SHOWS

  (a) the position error, max metric max(|dx|, |dy|), median per band, in
      micrometres on a logarithmic axis.
  (b) the same for the slope, max(|dtx|, |dty|), median per band.  Slopes are
      DIMENSIONLESS, so the axis is a plain scientific-notation number.

  Filled markers are the three cost-weighted (reweighted-loss) networks, hollow
  markers the three pooled-loss networks extended to their own plateau; the
  marker shape names the network.  `RK6 vs true` does not depend on the
  network, so it is drawn once per band as a single green marker.  The five
  momentum bands are drawn as a partition; the 10-50 GeV loss window overlaps
  two of them and is therefore drawn apart, on a hatched background.  Every
  band prints its n.

  The message is the vertical arrangement: `network vs true` and `RK6 vs true`
  sit on top of one another in every band, while `network vs RK6` is more than
  an order of magnitude below both.  The distance the network still has to the
  reference is small compared with the distance the reference itself has to the
  truth, so closing it further buys nothing against the real state.

INPUT FILES (all opened read-only)

  ../results/paper_numbers.json
      against_true_state.per_run[<loss> N=<N>,q=<q>][<band>] with the three
      blocks nn_vs_rk6 / nn_vs_true / rk6_vs_true, each carrying n,
      pos_max_med_um, pos_max_p95_um and slope_max_med (dimensionless), plus
      the two ratios nn_true_over_rk6_true and rk6_true_over_nn_rk6;
      against_true_state.rk6_vs_true, the network-independent copy, which this
      script asserts is identical to the per-run one.
  ../results/tab_against_true_state.csv
      the same numbers as a table; every plotted value is cross-checked
      against it before the figure is drawn.

OUTPUT

  ../figures/fig_three_references.png
  ../results/fig05_three_references.csv   every plotted value, long format

RUN

  cd Self_chained_paper/scripts
  PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python fig05_three_references.py
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
from matplotlib.lines import Line2D                   # noqa: E402
from matplotlib.ticker import LogFormatterSciNotation  # noqa: E402

_spec = importlib.util.spec_from_file_location("sc_paper_common",
                                               os.path.join(HERE, "common.py"))
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

FIGURES = os.path.join(C.PAPER, "figures")
PNG = os.path.join(FIGURES, "fig_three_references.png")
CSV = os.path.join(C.RESULTS, "fig05_three_references.csv")

BANDS = list(C.BAND_LABELS) + [C.LOSS_WINDOW_LABEL]
PAIR_MARKER = {(64, 2): "o", (128, 8): "s", (256, 16): "^"}
CMP = (("nn_vs_rk6", "network vs RK6", C.BLUE, -0.26),
       ("nn_vs_true", "network vs Geant4 true", C.MAGENTA, 0.0),
       ("rk6_vs_true", "RK6 vs Geant4 true", C.GREEN, 0.26))
JITTER = {(64, 2): -0.065, (128, 8): 0.0, (256, 16): 0.065}


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


def run_name(loss, N, q):
    return "%s N=%d,q=%d" % (loss, N, q)


def main():
    style()
    J = C.read_json(os.path.join(C.RESULTS, "paper_numbers.json"))
    A = J["against_true_state"]
    per_run = A["per_run"]
    tab = C.read_csv(os.path.join(C.RESULTS, "tab_against_true_state.csv"))
    TAB = {(int(r["N"]), int(r["q"]), r["loss"], r["band"]): r for r in tab}

    # the material floor is a property of the test set, not of a network
    for run, blocks in per_run.items():
        for band, v in blocks.items():
            assert abs(v["rk6_vs_true"]["pos_max_med_um"]
                       - A["rk6_vs_true"][band]["pos_max_med_um"]) < 1e-9

    out = []
    fig, axes = plt.subplots(2, 1, figsize=(6.3, 6.9), sharex=True)
    xs = np.arange(len(BANDS), dtype=float)

    for ax, key, ylab, panel in (
            (axes[0], "pos_max_med_um",
             r"position error, max metric, median  ($\mu$m)", "a"),
            (axes[1], "slope_max_med",
             "slope error, max metric, median\n(dimensionless)", "b")):
        # the loss window sits apart from the partition: hatch its slot
        ax.axvspan(len(BANDS) - 1.5, len(BANDS) - 0.5, facecolor="none",
                   edgecolor=C.NEUTRAL, hatch="///", lw=0.0, zorder=0)
        ax.axvline(len(BANDS) - 1.5, color=C.TEXT2, lw=0.8, ls=":", zorder=1)
        for bi, band in enumerate(BANDS):
            for cname, clabel, colour, dx in CMP:
                if cname == "rk6_vs_true":
                    v = A["rk6_vs_true"][band][key]
                    ax.plot([bi + dx], [v], "D", ms=6.5, mfc=colour,
                            mec=C.SURFACE, mew=0.7, zorder=5)
                    out.append(dict(panel=panel, band=band, comparison=clabel,
                                    loss="", N="", q="",
                                    n=A["rk6_vs_true"][band]["n"], value=v,
                                    unit="um" if panel == "a" else "dimensionless"))
                    continue
                for (N, q), mk in PAIR_MARKER.items():
                    for loss, filled in (("reweighted", True), ("pooled", False)):
                        blk = per_run[run_name(loss, N, q)][band]
                        v = blk[cname][key]
                        ref = TAB[(N, q, loss, band)]["%s_%s" % (cname, key)]
                        assert abs(float(ref) - v) <= 1e-9 * max(1.0, abs(v))
                        ax.plot([bi + dx + JITTER[(N, q)]], [v], mk, ms=5.5,
                                mfc=colour if filled else "none",
                                mec=colour if filled else colour,
                                mew=0.7 if filled else 1.1, zorder=4)
                        out.append(dict(panel=panel, band=band,
                                        comparison=clabel, loss=loss, N=N, q=q,
                                        n=blk[cname]["n"], value=v,
                                        unit="um" if panel == "a"
                                        else "dimensionless"))
        ax.set_yscale("log")
        ax.set_ylabel(ylab)
        ax.grid(True, axis="y", which="both")
        ax.set_xlim(-0.6, len(BANDS) - 0.4)
        ax.set_title("(%s)  %s" % (panel, "position" if panel == "a" else "slope"),
                     loc="left", pad=4)

    axes[1].yaxis.set_major_formatter(LogFormatterSciNotation())
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(["<3", "3-8", "8-20", "20-50", ">50", "10-50\nwindow"])
    axes[1].set_xlabel("momentum band  (GeV)")

    # ---- every band prints its n ------------------------------------------
    for ax in axes:
        top = ax.get_ylim()[1]
        ax.set_ylim(ax.get_ylim()[0], top * 10 ** 1.15)
    for bi, band in enumerate(BANDS):
        n = A["rk6_vs_true"][band]["n"]
        for ax in axes:
            ax.annotate("n = %d" % n, xy=(bi, 0.985), xycoords=("data",
                                                                "axes fraction"),
                        ha="center", va="top", fontsize=8.5, color=C.TEXT2)

    # ---- the message -------------------------------------------------------
    rat = [per_run[run_name(l, N, q)]["all"]["rk6_true_over_nn_rk6"]
           for l in ("pooled", "reweighted") for (N, q) in PAIR_MARKER]
    coin = [per_run[run_name(l, N, q)]["all"]["nn_true_over_rk6_true"]
            for l in ("pooled", "reweighted") for (N, q) in PAIR_MARKER]
    axes[0].annotate("over all momenta: network vs true is %.0f-%.0f %% of "
                     "RK6 vs true,\nwhile network vs RK6 is %.0f-%.0f times "
                     "below both" % (100 * min(coin), 100 * max(coin),
                                     min(rat), max(rat)),
                     xy=(0.015, 0.905), xycoords="axes fraction", ha="left",
                     va="top", fontsize=8.5, color=C.TEXT1)

    handles = [Line2D([], [], ls="none", marker="o", ms=6.5, mfc=c, mec=C.SURFACE,
                      mew=0.7, label=lab) for _, lab, c, _ in CMP]
    handles += [Line2D([], [], ls="none", marker=mk, ms=5.5, mfc=C.TEXT2,
                       mec=C.SURFACE, mew=0.7,
                       label=r"$N$ = %d, q = %d, $\Delta z$ = %.1f mm"
                             % (N, q, C.dz_mm(N)))
                for (N, q), mk in PAIR_MARKER.items()]
    handles += [Line2D([], [], ls="none", marker="o", ms=5.5, mfc=C.TEXT2,
                       mec=C.SURFACE, mew=0.7, label="cost-weighted loss"),
                Line2D([], [], ls="none", marker="o", ms=5.5, mfc="none",
                       mec=C.TEXT2, mew=1.1, label="pooled loss (extended)"),
                Line2D([], [], ls="none", marker="none", label=" ")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=8.5, handletextpad=0.5, columnspacing=1.6,
               labelspacing=0.32, bbox_to_anchor=(0.52, 0.002))

    fig.subplots_adjust(left=0.145, right=0.985, top=0.965, bottom=0.205,
                        hspace=0.13)
    os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(PNG, dpi=150, facecolor=C.SURFACE)
    plt.close(fig)

    C.write_csv(CSV, out, ["panel", "band", "comparison", "loss", "N", "q",
                           "n", "value", "unit"])
    print("wrote %s" % PNG)
    print("wrote %s  (%d rows)" % (CSV, len(out)))
    print("all momenta: nn-vs-true / rk6-vs-true = %.4f to %.4f; "
          "rk6-vs-true / nn-vs-rk6 = %.1f to %.1f"
          % (min(coin), max(coin), min(rat), max(rat)))
    print("band counts: %s" % {b: A["rk6_vs_true"][b]["n"] for b in BANDS})


if __name__ == "__main__":
    main()
