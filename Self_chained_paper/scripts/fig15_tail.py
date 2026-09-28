#!/usr/bin/env python
"""Figure 15 - the tail: where the large endpoint errors are.

WHAT IT SHOWS
  (a) the 95th percentile of the radial endpoint deviation from the RK6
      reference at z1 = 7826.0 mm, per momentum band, for the six runs
      (log y, micrometres).  Grouped bars: the three networks under the pooled
      loss (grey, darkening with N) beside the same three under the
      cost-weighted loss (blue).  The five bands partition the test set; the
      10-50 GeV loss window is the separate hatched group at the right.  n is
      printed per band.
  (b) the same six runs' fraction of test tracks whose radial endpoint error
      exceeds 1 mm, per band, in per cent (linear axis; a bar can be zero).
  (c) where the tail tracks start: the radial endpoint error against the
      track's |x| on the last UT plane z0 = 2648.2 mm, for the N = 64, q = 2,
      dz = 80.9 mm network under the cost-weighted loss, 3-8 GeV.  The cloud is
      a hexbin, the tracks beyond 1 mm are circled, and the median starting |x|
      of the tail and of the rest are drawn as vertical lines and annotated.
  (d) how concentrated the error is: the share of the summed radial endpoint
      error carried by the worst 5 % of test tracks, for all six runs (73 of
      1,452 tracks).  A flat distribution would give 5 %.

  Positions are micrometres (the starting |x| of panel (c) is millimetres, as
  it is a detector coordinate, not an error); no slopes appear in this figure.

COLOUR CONVENTION (identical in figures 10-15)
  pooled loss        grey, hollow markers, dashed line
  cost-weighted loss blue (#2a78d6), filled markers, solid line

INPUTS READ (all read-only)
  scripts/common.py                       paths, palette, bands, deltas()
  results/paper_numbers.json              group `headline` (p95_um and
                                          frac_above_1mm per band, asserted
                                          against the recomputation) and group
                                          `near_5gev` (the two median starting
                                          |x| values of panel (c))
  <S>/Block_E_single_network_chain/E1_Network_grid/results/N*/chain_states.npz
  <S>/Block_F_reweighted_loss/F1_Training/results/full/N*/chain_states.npz
  <S>/Block_E_single_network_chain/E0_Track_dataset/results/tracks.npz
                                          test_truth, test_P, test_S0

OUTPUTS
  figures/fig_tail.png
  results/fig15_tail.csv

  <S> = LHCb_Extrapolation_Project/single_network_chain_discrete_approach

Run:
  PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python fig15_tail.py
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"

import sys                                   # noqa: E402
sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in ("", ".", HERE):                   # numbers.py shadows stdlib `numbers`
    while _p in sys.path:
        sys.path.remove(_p)

import csv                                   # noqa: E402
import importlib.util                        # noqa: E402

import numpy as np                           # noqa: E402
import matplotlib                            # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402
from matplotlib.lines import Line2D          # noqa: E402
from matplotlib.patches import Patch         # noqa: E402

_spec = importlib.util.spec_from_file_location("sc_paper_common",
                                               os.path.join(HERE, "common.py"))
C = importlib.util.module_from_spec(_spec)
sys.modules["sc_paper_common"] = C
_spec.loader.exec_module(C)

POOLED_SHADES = ("#d6d5d0", "#9a9994", "#52514e")      # N = 64, 128, 256
WEIGHTED_SHADES = ("#a8c9ee", "#2a78d6", "#134a86")
POOLED_LINE = C.TEXT2
WEIGHTED_LINE = C.BLUE
COLOUR_NOTE = ("colour convention: pooled loss = grey, hollow markers, dashed; "
               "cost-weighted loss = blue #2a78d6, filled markers, solid; "
               "within a loss the shade darkens with N")

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150,
    "figure.facecolor": C.SURFACE, "savefig.facecolor": C.SURFACE,
    "axes.facecolor": C.SURFACE, "axes.edgecolor": C.TEXT2,
    "axes.labelcolor": C.TEXT1, "text.color": C.TEXT1,
    "xtick.color": C.TEXT2, "ytick.color": C.TEXT2,
    "font.size": 9, "axes.labelsize": 8.4, "axes.titlesize": 8.8,
    "xtick.labelsize": 7.4, "ytick.labelsize": 7.4, "legend.fontsize": 6.8,
    "axes.linewidth": 0.8, "grid.color": C.NEUTRAL, "grid.linewidth": 0.5,
    "hatch.linewidth": 0.6,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})

WORST_FRACTION = 0.05


def net_label(N, q):
    return "N = %d, q = %d, Δz = %.1f mm" % (N, q, C.dz_mm(N))


def main():
    a = C.split_arrays("test")
    P, truth, S0 = a["P"], a["truth"], a["S0"]
    start_abs_x_mm = np.abs(S0[:, 0])
    J = C.read_json(os.path.join(C.RESULTS, "paper_numbers.json"))
    pairs = J["headline"]["pairs"]

    R = {}
    for N, q in C.PAIRS:
        for loss, run in (("pooled", C.pooled_run(N, q)),
                          ("reweighted", C.reweighted_run(N, q))):
            st = C.load_chain_states(run)
            R[(N, q, loss)] = C.deltas(st[:, -1], truth[:, C.N_MAX])["radial_um"]

    bands = list(C.BAND_LABELS) + [C.LOSS_WINDOW_LABEL]
    band_ticks = list(C.BAND_LABELS) + ["10-50\nloss window"]
    masks = dict(C.band_masks(P))

    fig = plt.figure(figsize=(8.6, 5.7), layout="constrained")
    gs = fig.add_gridspec(2, 2)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    rows = []

    # ---------------------------------------------- (a) p95 and (b) > 1 mm --
    width = 0.13
    offs = {("pooled", 0): -2.5, ("pooled", 1): -1.5, ("pooled", 2): -0.5,
            ("reweighted", 0): 0.5, ("reweighted", 1): 1.5, ("reweighted", 2): 2.5}
    hi_a, hi_b = 0.0, 0.0
    for bi, band in enumerate(bands):
        m = masks[band]
        for loss in ("pooled", "reweighted"):
            for ni, (N, q) in enumerate(C.PAIRS):
                v = R[(N, q, loss)][m]
                p95 = C.quantile(v, 0.95)
                frac = float((v > 1000.0).mean())
                ref = pairs[C.run_key(N, q)][loss]["endpoint"]["by_band"][band]
                assert abs(p95 - ref["p95_um"]) <= 1e-9 * max(1.0, ref["p95_um"])
                assert abs(frac - ref["frac_above_1mm"]) <= 1e-12
                x = bi + offs[(loss, ni)] * width
                col = (POOLED_SHADES if loss == "pooled" else WEIGHTED_SHADES)[ni]
                hatch = "///" if band == C.LOSS_WINDOW_LABEL else None
                ax_a.bar(x, p95, width=width * 0.92, color=col, edgecolor=C.TEXT2,
                         lw=0.35, hatch=hatch, zorder=3)
                ax_b.bar(x, 100.0 * frac, width=width * 0.92, color=col,
                         edgecolor=C.TEXT2, lw=0.35, hatch=hatch, zorder=3)
                hi_a, hi_b = max(hi_a, p95), max(hi_b, 100.0 * frac)
                rows.append(dict(panel="a,b: per band", band_GeV=band, n=int(m.sum()),
                                 loss=("pooled" if loss == "pooled" else "cost-weighted"),
                                 network=net_label(N, q), N=N, q=q,
                                 p95_radial_um=p95, frac_beyond_1mm_pct=100.0 * frac,
                                 start_abs_x_med_mm="", worst5pct_share_of_total=""))

    for ax, hi, ylab, logy in ((ax_a, hi_a, "p95 radial error at $z_1$ [µm]", True),
                               (ax_b, hi_b, "test tracks beyond 1 mm [%]", False)):
        ax.set_xticks(range(len(bands)))
        ax.set_xticklabels(band_ticks, fontsize=7.0)
        ax.set_xlim(-0.55, len(bands) - 0.35)
        ax.axvline(len(C.BAND_LABELS) - 0.5, color=C.TEXT2, lw=0.7, alpha=0.8)
        ax.set_ylabel(ylab, fontsize=8.0)
        ax.set_xlabel("momentum band [GeV]", fontsize=7.8)
        ax.grid(True, axis="y", alpha=0.45)
        ax.set_axisbelow(True)
        if logy:
            ax.set_yscale("log")
            ax.set_ylim(top=hi * 6.0)
        else:
            ax.set_ylim(0, hi * 1.45)
        top = ax.get_ylim()[1]
        for bi, band in enumerate(bands):
            n = int(masks[band].sum())
            ax.text(bi, (top / 1.6 if logy else top * 0.965), "n = %d" % n,
                    ha="center", va="top", fontsize=6.4, color=C.TEXT2)
    ax_a.set_title("(a) the 95th percentile per band", fontsize=8.6)
    ax_b.set_title("(b) how often the endpoint misses by more than 1 mm", fontsize=8.6)

    # ------------------------------------ (c) where the tail tracks start --
    N, Q = 64, 2
    band38 = (P >= 3.0) & (P < 8.0)
    r = R[(N, Q, "reweighted")]
    tail = band38 & (r > 1000.0)
    rest = band38 & ~(r > 1000.0)
    near = J["near_5gev"]["bands"]["3-8"]["reweighted"]
    med_tail = float(np.median(start_abs_x_mm[tail]))
    med_rest = float(np.median(start_abs_x_mm[rest]))
    assert abs(med_tail - near["start_abs_x_med_mm_beyond_1mm"]) <= 1e-9 * max(1.0, med_tail)
    assert abs(med_rest - near["start_abs_x_med_mm_rest"]) <= 1e-9 * max(1.0, med_rest)
    assert int(tail.sum()) == int(near["n_beyond_1mm"])

    hb = ax_c.hexbin(start_abs_x_mm[band38], r[band38], yscale="log", gridsize=(26, 22),
                     mincnt=1, cmap="Blues", linewidths=0.0, zorder=2)
    ax_c.plot(start_abs_x_mm[tail], r[tail], "o", ms=4.2, mfc="none", mew=0.8,
              color="#c0397a", zorder=4, label="beyond 1 mm (n = %d)" % int(tail.sum()))
    ax_c.axhline(1000.0, color=C.TEXT1, lw=0.7, ls=":", zorder=3)
    ax_c.axvline(med_tail, color="#c0397a", lw=1.2, ls="-", zorder=5)
    ax_c.axvline(med_rest, color=WEIGHTED_LINE, lw=1.2, ls="--", zorder=5)
    ax_c.set_yscale("log")
    ax_c.set_xlabel("track's $|x|$ on the last UT plane, $z_0$ [mm]", fontsize=8.0)
    ax_c.set_ylabel("endpoint radial error at $z_1$ [µm]", fontsize=8.0)
    ax_c.set_title("(c) %s, cost-weighted loss, 3-8 GeV (n = %d)"
                   % (net_label(N, Q), int(band38.sum())), fontsize=8.0)
    ax_c.grid(True, alpha=0.4)
    ax_c.set_axisbelow(True)
    ax_c.text(0.98, 0.04,
              "median starting $|x|$\n  beyond 1 mm  %.1f mm\n  the rest  %.1f mm\n"
              "  ratio  %.2f" % (med_tail, med_rest, med_tail / med_rest),
              transform=ax_c.transAxes, va="bottom", ha="right", fontsize=6.4,
              color=C.TEXT1, linespacing=1.25,
              bbox=dict(facecolor=C.SURFACE, edgecolor=C.NEUTRAL, linewidth=0.5,
                        boxstyle="round,pad=0.28", alpha=0.95))
    leg = ax_c.legend(loc="upper left", frameon=True, framealpha=0.95, handlelength=1.2,
                      borderpad=0.3, fontsize=6.6)
    leg.get_frame().set_edgecolor(C.NEUTRAL)
    cb = fig.colorbar(hb, ax=ax_c, fraction=0.042, pad=0.02)
    cb.set_label("tracks per cell", fontsize=6.8)
    cb.ax.tick_params(labelsize=6.4)
    cb.outline.set_edgecolor(C.NEUTRAL)
    rows.append(dict(panel="c: starting |x|, N = 64, q = 2, cost-weighted, 3-8 GeV",
                     band_GeV="3-8", n=int(band38.sum()), loss="cost-weighted",
                     network=net_label(N, Q), N=N, q=Q, p95_radial_um="",
                     frac_beyond_1mm_pct=100.0 * float(tail.sum()) / float(band38.sum()),
                     start_abs_x_med_mm="beyond 1 mm %.4f ; rest %.4f" % (med_tail, med_rest),
                     worst5pct_share_of_total=""))

    # --------------------------- (d) the worst 5 % share of the total error --
    labels, shares, cols = [], [], []
    k = int(np.ceil(WORST_FRACTION * len(P)))
    for N2, q2 in C.PAIRS:
        for loss in ("pooled", "reweighted"):
            v = np.sort(R[(N2, q2, loss)])[::-1]
            share = float(v[:k].sum() / v.sum())
            tag = "pooled" if loss == "pooled" else "cost-weighted"
            labels.append("N = %d, q = %d\n%s" % (N2, q2, tag))
            shares.append(100.0 * share)
            cols.append(POOLED_LINE if loss == "pooled" else WEIGHTED_LINE)
            rows.append(dict(panel="d: worst %d tracks of %d" % (k, len(P)),
                             band_GeV="all", n=int(len(P)), loss=tag,
                             network=net_label(N2, q2), N=N2, q=q2, p95_radial_um="",
                             frac_beyond_1mm_pct="", start_abs_x_med_mm="",
                             worst5pct_share_of_total=share))
    y = np.arange(len(labels))
    ax_d.barh(y, shares, height=0.6, color=cols, edgecolor=C.TEXT2, lw=0.35, zorder=3)
    for yy, v in zip(y, shares):
        ax_d.text(v + 0.8, yy, "%.0f %%" % v, va="center", ha="left", fontsize=6.6)
    ax_d.axvline(100.0 * WORST_FRACTION, color=C.TEXT1, lw=0.9, ls=":", zorder=4)
    ax_d.text(100.0 * WORST_FRACTION + 0.8, -0.72, "5 %: a flat distribution",
              fontsize=6.4, color=C.TEXT1, ha="left", va="center")
    ax_d.set_yticks(y)
    ax_d.set_yticklabels(labels, fontsize=6.6)
    ax_d.invert_yaxis()
    ax_d.set_xlim(0, max(shares) * 1.32)
    ax_d.set_ylim(len(labels) - 0.4, -1.1)
    ax_d.set_xlabel("share of the summed radial endpoint error carried by\n"
                    "the worst 5 %% of test tracks (%d of %d) [%%]" % (k, len(P)),
                    fontsize=7.6)
    ax_d.set_title("(d) how concentrated the error is", fontsize=8.6)
    ax_d.grid(True, axis="x", alpha=0.45)
    ax_d.set_axisbelow(True)

    handles = ([Patch(facecolor=POOLED_SHADES[i], edgecolor=C.TEXT2, lw=0.35,
                      label="pooled, " + net_label(Nn, qq))
                for i, (Nn, qq) in enumerate(C.PAIRS)]
               + [Patch(facecolor=WEIGHTED_SHADES[i], edgecolor=C.TEXT2, lw=0.35,
                        label="cost-weighted, " + net_label(Nn, qq))
                  for i, (Nn, qq) in enumerate(C.PAIRS)]
               + [Line2D([], [], color=C.TEXT2, lw=0, label=" "),
                  Patch(facecolor="none", edgecolor=C.TEXT2, lw=0.35, hatch="///",
                        label="10-50 GeV loss window (hatched)")])
    leg = fig.legend(handles=handles, loc="outside upper center", ncol=4, frameon=True,
                     framealpha=0.95, handlelength=1.5, columnspacing=1.0, borderpad=0.35)
    leg.get_frame().set_edgecolor(C.NEUTRAL)

    out_png = os.path.join(C.PAPER, "figures", "fig_tail.png")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)

    fields = ["panel", "band_GeV", "n", "loss", "network", "N", "q", "p95_radial_um",
              "frac_beyond_1mm_pct", "start_abs_x_med_mm", "worst5pct_share_of_total"]
    out_csv = os.path.join(C.RESULTS, "fig15_tail.csv")
    with open(out_csv, "w", newline="") as f:
        f.write("# figure 15, fig_tail.png: the tail of the endpoint error.  Positions in\n")
        f.write("# micrometres; the starting |x| of panel (c) is a detector coordinate in mm.\n")
        f.write("# p95_radial_um and frac_beyond_1mm_pct are asserted equal to\n")
        f.write("# results/paper_numbers.json group `headline`; the two median starting |x|\n")
        f.write("# values of panel (c) to group `near_5gev`.  worst5pct_share_of_total is\n")
        f.write("# computed in this script (it is not in paper_numbers.json): the summed\n")
        f.write("# radial endpoint error of the 73 worst of the 1,452 test tracks divided by\n")
        f.write("# the sum over all of them.\n")
        f.write("# %s.\n" % COLOUR_NOTE)
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("wrote %s" % out_png)
    print("wrote %s" % out_csv)


if __name__ == "__main__":
    main()
