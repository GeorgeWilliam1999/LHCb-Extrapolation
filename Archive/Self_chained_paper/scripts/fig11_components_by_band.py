#!/usr/bin/env python
"""Figure 11 - the four state components of the endpoint error, band by band.

WHAT IT SHOWS
  Four panels, one per component of the state at z1 = 7826.0 mm: x and y in
  micrometres, and the dimensionless slopes t_x = dx/dz and t_y = dy/dz (the
  axes are logarithmic, so the slope decades read as scientific notation; no
  quantity here is scaled by 1e3 and the word mrad appears nowhere).  In each
  panel, grouped bars give the median |deviation| of the six runs in each
  momentum band: the three networks under the pooled loss (grey, light to dark
  with N) beside the same three under the cost-weighted loss (blue, light to
  dark with N).  The five bands partition the test set; the 10-50 GeV loss
  window is a separate, hatched group at the right, and overlaps two of the
  five on purpose.  The number of test tracks is printed above each group.

  On top of every bar a small triangle marks the SIGNED median at the same
  |value|: pointing up when the signed median is positive, down when negative.
  The x panel is the point of the figure: under the pooled loss the signed
  median is large and negative in every band (a systematic bend deficit),
  under the cost-weighted loss it is far smaller.

COLOUR CONVENTION (identical in figures 10-15)
  pooled loss        grey, hollow markers, dashed line
  cost-weighted loss blue (#2a78d6), filled markers, solid line

INPUTS READ (all read-only)
  scripts/common.py                    paths, palette, band labels, dz_mm()
  results/paper_numbers.json           group `headline`, per pair and loss:
                                       endpoint.by_band[band][{x,y}_med_abs_um,
                                       {x,y}_signed_med_um,
                                       {tx,ty}_med_abs_slope,
                                       {tx,ty}_signed_med_slope, n]
  Nothing else: every number plotted is taken from the JSON, which was computed
  by scripts/numbers.py from the six runs' chain_states.npz and tracks.npz.

OUTPUTS
  figures/fig_components_by_band.png
  results/fig11_components_by_band.csv

Run:
  PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python fig11_components_by_band.py
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
COLOUR_NOTE = ("colour convention: pooled loss = grey, hollow markers, dashed; "
               "cost-weighted loss = blue #2a78d6, filled markers, solid; "
               "within a loss the shade darkens with N")

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150,
    "figure.facecolor": C.SURFACE, "savefig.facecolor": C.SURFACE,
    "axes.facecolor": C.SURFACE, "axes.edgecolor": C.TEXT2,
    "axes.labelcolor": C.TEXT1, "text.color": C.TEXT1,
    "xtick.color": C.TEXT2, "ytick.color": C.TEXT2,
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.4,
    "axes.linewidth": 0.8, "grid.color": C.NEUTRAL, "grid.linewidth": 0.5,
    "hatch.linewidth": 0.6,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})

PANELS = (("x", "um", "$|\\Delta x|$ at $z_1$ [µm]"),
          ("y", "um", "$|\\Delta y|$ at $z_1$ [µm]"),
          ("tx", "slope", "$|\\Delta t_x|$ at $z_1$ (dimensionless)"),
          ("ty", "slope", "$|\\Delta t_y|$ at $z_1$ (dimensionless)"))


def net_label(N, q):
    return "N = %d, q = %d, Δz = %.1f mm" % (N, q, C.dz_mm(N))


def main():
    J = C.read_json(os.path.join(C.RESULTS, "paper_numbers.json"))
    pairs = J["headline"]["pairs"]
    bands = list(C.BAND_LABELS) + [C.LOSS_WINDOW_LABEL]
    band_ticks = list(C.BAND_LABELS) + ["10-50\nloss window"]

    fig, axes = plt.subplots(2, 2, figsize=(7.9, 5.6))
    rows = []
    width = 0.13
    offs = {("pooled", 0): -2.5, ("pooled", 1): -1.5, ("pooled", 2): -0.5,
            ("reweighted", 0): 0.5, ("reweighted", 1): 1.5, ("reweighted", 2): 2.5}

    for ax, (comp, unit, ylab) in zip(axes.ravel(), PANELS):
        abs_key = "%s_med_abs_%s" % (comp, unit)
        sgn_key = "%s_signed_med_%s" % (comp, unit)
        lo, hi, lo_bar = np.inf, 0.0, np.inf
        for bi, band in enumerate(bands):
            for li, loss in enumerate(("pooled", "reweighted")):
                for ni, (N, q) in enumerate(C.PAIRS):
                    e = pairs[C.run_key(N, q)][loss]["endpoint"]["by_band"][band]
                    v, s = e[abs_key], e[sgn_key]
                    x = bi + offs[(loss, ni)] * width
                    col = (POOLED_SHADES if loss == "pooled" else WEIGHTED_SHADES)[ni]
                    ax.bar(x, v, width=width * 0.92, color=col, zorder=3,
                           edgecolor=C.TEXT2, linewidth=0.35,
                           hatch=("///" if band == C.LOSS_WINDOW_LABEL else None))
                    ax.plot([x], [abs(s)], marker=("^" if s >= 0 else "v"), ms=3.6,
                            color=C.TEXT1, mfc=(C.SURFACE if s >= 0 else C.TEXT1),
                            mew=0.7, ls="none", zorder=5)
                    lo = min(lo, v, abs(s))
                    lo_bar = min(lo_bar, v)
                    hi = max(hi, v, abs(s))
                    if comp == "x":                     # one CSV row per bar, all four
                        pass
                    rows.append(dict(
                        component=comp, unit=("um" if unit == "um" else "dimensionless"),
                        band_GeV=band, n=int(e["n"]),
                        loss=("pooled" if loss == "pooled" else "cost-weighted"),
                        network=net_label(N, q), N=N, q=q,
                        median_abs=v, signed_median=s))
        ax.set_yscale("log")
        # the floor keeps the bars readable: it never drops more than ~1.5
        # decades below the smallest bar, so a handful of very small signed
        # medians (all in the thin >50 GeV band) are clipped; every value is in
        # the companion CSV.
        floor = max(lo * 0.6, lo_bar / 40.0)
        ax.set_ylim(10.0 ** np.floor(np.log10(floor)), hi * 4.0)
        ax.set_ylabel(ylab)
        ax.set_xticks(range(len(bands)))
        ax.set_xticklabels(band_ticks, fontsize=7.4)
        ax.set_xlim(-0.55, len(bands) - 0.35)
        ax.axvline(len(C.BAND_LABELS) - 0.5, color=C.TEXT2, lw=0.7, ls="-", alpha=0.8)
        ax.grid(True, which="major", axis="y", alpha=0.45, zorder=0)
        ax.set_axisbelow(True)
        for bi, band in enumerate(bands):
            n = pairs[C.run_key(64, 2)]["pooled"]["endpoint"]["by_band"][band]["n"]
            ax.text(bi, hi * 3.2, "n = %d" % n, ha="center", va="top",
                    fontsize=6.6, color=C.TEXT2)
        ax.set_xlabel("momentum band [GeV]", fontsize=8)

    handles = ([Patch(facecolor=POOLED_SHADES[i], edgecolor=C.TEXT2, lw=0.35,
                      label="pooled, " + net_label(N, q)) for i, (N, q) in enumerate(C.PAIRS)]
               + [Patch(facecolor=WEIGHTED_SHADES[i], edgecolor=C.TEXT2, lw=0.35,
                        label="cost-weighted, " + net_label(N, q))
                  for i, (N, q) in enumerate(C.PAIRS)]
               + [Line2D([], [], marker="^", ls="none", color=C.TEXT1, mfc=C.SURFACE,
                         ms=4, label="signed median > 0 (plotted at |value|)"),
                  Line2D([], [], marker="v", ls="none", color=C.TEXT1, mfc=C.TEXT1,
                         ms=4, label="signed median < 0 (plotted at |value|)")])
    leg = fig.legend(handles=handles, loc="upper center", ncol=4, frameon=True,
                     framealpha=0.95, bbox_to_anchor=(0.5, 1.10), handlelength=1.5,
                     columnspacing=1.1, borderpad=0.4)
    leg.get_frame().set_edgecolor(C.NEUTRAL)
    fig.tight_layout(h_pad=1.6, w_pad=2.0)

    out_png = os.path.join(C.PAPER, "figures", "fig_components_by_band.png")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, bbox_extra_artists=(leg,))
    plt.close(fig)

    out_csv = os.path.join(C.RESULTS, "fig11_components_by_band.csv")
    with open(out_csv, "w", newline="") as f:
        f.write("# figure 11, fig_components_by_band.png: median |deviation| at z1 per band,\n")
        f.write("# with the signed median beside it.  x and y in micrometres; t_x and t_y\n")
        f.write("# dimensionless (never scaled by 1e3).  All values are taken verbatim from\n")
        f.write("# results/paper_numbers.json group `headline`.\n")
        f.write("# %s.\n" % COLOUR_NOTE)
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote %s" % out_png)
    print("wrote %s" % out_csv)


if __name__ == "__main__":
    main()
