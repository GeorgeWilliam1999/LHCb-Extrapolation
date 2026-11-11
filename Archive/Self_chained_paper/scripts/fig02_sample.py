#!/usr/bin/env python
"""Figure 2 - what the track sample is made of.

WHAT THE FIGURE SHOWS

  (a) the momentum spectrum of each split (training 11,567, validation 1,463,
      test 1,452 tracks), normalised so the three shapes can be compared, on a
      logarithmic momentum axis from 1.5 to 200 GeV.  The four interior band
      edges (3, 8, 20, 50 GeV) are drawn as vertical lines, each band is
      labelled with its test-set count, and the 10-50 GeV loss window is drawn
      as a light shaded span.
  (b) the composition of the test split: the four particle types that survive
      the non-electron cut, and the charge balance.  Counts and percentages are
      printed on the bars.
  (c) the starting footprint of the test tracks: (x, y) at z0 = 2648.2 mm, as a
      hexbin with a logarithmic colour scale.
  (d) the pseudorapidity distribution of the test split, with the 2 < eta < 5
      selection drawn.

  Every count on the figure is asserted against `dataset` in paper_numbers.json
  before the figure is drawn.

INPUT FILES (all opened read-only)

  ../results/paper_numbers.json
      dataset.splits.{train,val,test}: n, band_counts, pid_composition,
      n_charge_plus / n_charge_minus, eta and momentum ranges;
      dataset.geometry.z0_mm.
  ../../single_network_chain_discrete_approach/Block_E_single_network_chain/
      E0_Track_dataset/results/tracks.npz
      <split>_P, <split>_ETA, <split>_PID, <split>_S0 (x, y at z0 and the q/p
      that carries the charge).

OUTPUT

  ../figures/fig_sample.png
  ../results/fig02_sample.csv   the plotted histogram bins, counts and shares

RUN

  cd Self_chained_paper/scripts
  PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python fig02_sample.py
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
from matplotlib.colors import LogNorm                 # noqa: E402

_spec = importlib.util.spec_from_file_location("sc_paper_common",
                                               os.path.join(HERE, "common.py"))
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

FIGURES = os.path.join(C.PAPER, "figures")
PNG = os.path.join(FIGURES, "fig_sample.png")
CSV = os.path.join(C.RESULTS, "fig02_sample.csv")

P_LO, P_HI = 1.5, 200.0
PID_ORDER = ("muon", "pion", "kaon", "proton")
SPLIT_COLOUR = {"train": C.BLUE, "val": C.GREEN, "test": C.MAGENTA}


def style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150,
        "figure.facecolor": C.SURFACE, "axes.facecolor": C.SURFACE,
        "savefig.facecolor": C.SURFACE, "savefig.edgecolor": C.SURFACE,
        "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9.5,
        "axes.edgecolor": C.TEXT2, "axes.labelcolor": C.TEXT1,
        "text.color": C.TEXT1, "xtick.color": C.TEXT2, "ytick.color": C.TEXT2,
        "axes.linewidth": 0.8, "legend.frameon": False,
        "grid.color": C.NEUTRAL, "grid.alpha": 0.6, "grid.linewidth": 0.6,
    })


def main():
    style()
    J = C.read_json(os.path.join(C.RESULTS, "paper_numbers.json"))
    DS = J["dataset"]["splits"]
    z0 = J["dataset"]["geometry"]["z0_mm"]

    A = {s: C.split_arrays(s) for s in C.SPLITS}
    rows = []

    # ---- every count on the figure has to match paper_numbers.json ---------
    for s in C.SPLITS:
        assert A[s]["P"].size == DS[s]["n"], s
        bc = C.band_counts(A[s]["P"])
        assert bc == DS[s]["band_counts"], (s, bc, DS[s]["band_counts"])
    qtest = np.sign(A["test"]["S0"][:, 4])
    assert int((qtest > 0).sum()) == DS["test"]["n_charge_plus"]
    assert int((qtest < 0).sum()) == DS["test"]["n_charge_minus"]

    fig = plt.figure(figsize=(6.3, 7.0))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 1.0, 0.92],
                          width_ratios=[1.0, 1.0], hspace=0.58, wspace=0.46)
    axa = fig.add_subplot(gs[0, :])
    axb = fig.add_subplot(gs[1, 0])
    axc = fig.add_subplot(gs[1, 1])
    axd = fig.add_subplot(gs[2, :])

    # ------------------------------------------------- (a) momentum spectrum --
    edges = np.logspace(np.log10(P_LO), np.log10(P_HI), 61)
    axa.axvspan(C.LOSS_WINDOW[0], C.LOSS_WINDOW[1], color=C.NEUTRAL,
                alpha=0.45, lw=0, zorder=0)
    for s in C.SPLITS:
        h, _ = np.histogram(A[s]["P"], bins=edges)
        dens = h / h.sum() / np.diff(np.log10(edges))
        axa.step(edges[:-1], dens, where="post", lw=1.5,
                 color=SPLIT_COLOUR[s], label="%s  (n = %s)" % (s, f"{h.sum():,}"))
        for k in range(len(h)):
            rows.append(dict(panel="a", series=s, key="bin_%02d" % k,
                             x_lo=edges[k], x_hi=edges[k + 1],
                             value=int(h[k]), unit="tracks"))
    axa.set_xscale("log")
    axa.set_xlim(P_LO, P_HI)
    axa.set_ylim(0, 1.85 * axa.get_ylim()[1])
    axa.set_xlabel("momentum  p  (GeV)")
    axa.set_ylabel("tracks per decade\n(split normalised)")
    axa.grid(True, which="both", axis="x")
    axa.grid(True, axis="y")
    for e in C.BAND_EDGES[1:-1]:
        axa.axvline(e, color=C.TEXT2, lw=0.9, ls=":", zorder=1)
    bc_test = DS["test"]["band_counts"]
    centres = [np.sqrt(P_LO * 3.0), np.sqrt(3 * 8.0), np.sqrt(8 * 20.0),
               np.sqrt(20 * 50.0), np.sqrt(50 * P_HI)]
    ytop = axa.get_ylim()[1]
    for cx, lab in zip(centres, C.BAND_LABELS):
        axa.annotate("%s GeV\nn = %d" % (lab, bc_test[lab]), xy=(cx, 0.745 * ytop),
                     ha="center", va="top", fontsize=8.5, color=C.TEXT2)
        rows.append(dict(panel="a", series="test band count", key=lab,
                         x_lo="", x_hi="", value=bc_test[lab], unit="tracks"))
    axa.annotate("10-50 GeV\nloss window", xy=(np.sqrt(10 * 50.0), 0.02 * ytop),
                 ha="center", va="bottom", fontsize=8.5, color=C.TEXT2,
                 bbox=dict(fc=C.SURFACE, ec="none", alpha=0.8, pad=1.0))
    rows.append(dict(panel="a", series="test band count",
                     key=C.LOSS_WINDOW_LABEL, x_lo=10.0, x_hi=50.0,
                     value=bc_test[C.LOSS_WINDOW_LABEL], unit="tracks"))
    axa.legend(loc="upper center", ncol=3, borderaxespad=0.2,
               handlelength=1.6, columnspacing=1.4)
    axa.set_title("(a)  momentum spectrum per split", loc="left", pad=4)

    # ----------------------------------- (b) composition and charge balance --
    pid = DS["test"]["pid_composition"]
    n_test = DS["test"]["n"]
    labels = list(PID_ORDER) + ["q+", "q-"]
    vals = [pid[k]["n"] for k in PID_ORDER] + [DS["test"]["n_charge_plus"],
                                               DS["test"]["n_charge_minus"]]
    cols = [C.BLUE] * 4 + [C.GREEN] * 2
    xs = np.array([0, 1.25, 2.5, 3.75, 5.6, 7.1])
    axb.bar(xs, vals, 0.85, color=cols, lw=0)
    for x, v in zip(xs, vals):
        axb.annotate("%d\n%.1f%%" % (v, 100.0 * v / n_test), xy=(x, v),
                     xytext=(0, 2), textcoords="offset points", ha="center",
                     va="bottom", fontsize=8, color=C.TEXT2)
    for lab, v in zip(labels, vals):
        rows.append(dict(panel="b", series="test split", key=lab, x_lo="",
                         x_hi="", value=v, unit="tracks"))
    axb.set_xticks(xs)
    axb.set_xticklabels(labels, fontsize=8.5, rotation=30, ha="right")
    axb.set_xlim(-0.85, 7.95)
    axb.set_ylim(0, 1.45 * max(vals))
    axb.set_ylabel("test tracks")
    axb.grid(True, axis="y")
    axb.set_title("(b)  composition, test  (n = %d)" % n_test, loc="left", pad=4)

    # ------------------------------------------ (c) starting footprint at z0 --
    x0 = A["test"]["S0"][:, 0]
    y0 = A["test"]["S0"][:, 1]
    hb = axc.hexbin(x0, y0, gridsize=28, bins="log", mincnt=1, linewidths=0.0,
                    cmap="Blues")
    cb = fig.colorbar(hb, ax=axc, pad=0.035, fraction=0.06)
    cb.set_label("tracks per cell", fontsize=9)
    cb.ax.tick_params(labelsize=8.5)
    axc.set_xlabel("x at $z_0$  (mm)")
    axc.set_ylabel("y at $z_0$  (mm)")
    axc.set_title("(c)  footprint at $z_0$ = %.1f mm" % z0, loc="left", pad=4)
    axc.grid(True, alpha=0.3)
    for nm, v in (("x_min", x0.min()), ("x_max", x0.max()),
                  ("x_median", np.median(x0)), ("y_min", y0.min()),
                  ("y_max", y0.max()), ("y_median", np.median(y0))):
        rows.append(dict(panel="c", series="test S0", key=nm, x_lo="", x_hi="",
                         value=float(v), unit="mm"))

    # ------------------------------------------------------------- (d) eta --
    eta = A["test"]["ETA"]
    ebins = np.linspace(1.9, 5.1, 65)
    h, _ = np.histogram(eta, bins=ebins)
    axd.step(ebins[:-1], h, where="post", lw=1.5, color=C.MAGENTA)
    axd.fill_between(ebins[:-1], 0, h, step="post", color=C.MAGENTA, alpha=0.18,
                     lw=0)
    for e in (2.0, 5.0):
        axd.axvline(e, color=C.TEXT2, lw=1.0, ls="--")
    axd.axvspan(1.9, 2.0, color=C.NEUTRAL, alpha=0.5, lw=0)
    axd.axvspan(5.0, 5.1, color=C.NEUTRAL, alpha=0.5, lw=0)
    axd.set_xlim(1.9, 5.1)
    axd.set_xlabel(r"pseudorapidity  $\eta$  (at the particle's origin)")
    axd.set_ylabel("test tracks")
    axd.grid(True)
    axd.annotate(r"selection  $2 < \eta < 5$" "\n"
                 r"measured %.3f to %.3f" % (DS["test"]["eta_min"],
                                             DS["test"]["eta_max"]),
                 xy=(0.985, 0.92), xycoords="axes fraction", ha="right",
                 va="top", fontsize=9, color=C.TEXT2)
    axd.set_title("(d)  pseudorapidity, test split", loc="left", pad=4)
    for k in range(len(h)):
        rows.append(dict(panel="d", series="test", key="bin_%02d" % k,
                         x_lo=ebins[k], x_hi=ebins[k + 1], value=int(h[k]),
                         unit="tracks"))

    fig.subplots_adjust(left=0.115, right=0.955, top=0.955, bottom=0.068)
    os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(PNG, dpi=150, facecolor=C.SURFACE)
    plt.close(fig)

    C.write_csv(CSV, rows, ["panel", "series", "key", "x_lo", "x_hi", "value",
                            "unit"])
    print("wrote %s" % PNG)
    print("wrote %s  (%d rows)" % (CSV, len(rows)))
    print("counts checked against paper_numbers.json dataset: "
          "train %d, val %d, test %d; test bands %s"
          % (DS["train"]["n"], DS["val"]["n"], DS["test"]["n"], bc_test))


if __name__ == "__main__":
    main()
