#!/usr/bin/env python
"""Figure 10 - the endpoint error against momentum, one column per network.

WHAT IT SHOWS
  Top row, one panel per network (N = 64, q = 2; N = 128, q = 8; N = 256,
  q = 16): the radial endpoint deviation of every test track from the RK6
  reference at z1 = 7826.0 mm [micrometres, log scale] against the track's
  momentum [GeV, log scale].  The cloud is the cost-weighted-loss run, drawn as
  a hexbin density.  Over it, the per-band median of BOTH losses as a step line
  with a marker at the band centre.  The five band edges (3, 8, 20, 50 GeV) are
  drawn as thin vertical lines, the number of test tracks in each band is
  printed along the top, and the 10-50 GeV loss window is a light shaded span.
  Bottom row: the ratio of the two per-band medians, pooled over cost-weighted,
  on a log axis with 1 marked.

  Slopes do not appear here; positions are micrometres throughout.

COLOUR CONVENTION (identical in figures 10-15)
  pooled loss        grey, hollow markers, dashed line
  cost-weighted loss blue (#2a78d6), filled markers, solid line

INPUTS READ (all read-only)
  scripts/common.py                                  paths, palette, bands,
                                                     deltas(), statistics
  <S>/Block_E_single_network_chain/E1_Network_grid/results/N{064_q02,128_q08,
      256_q16}/chain_states.npz                      test_states (n, N+1, 5),
                                                     pooled loss, extended runs
  <S>/Block_F_reweighted_loss/F1_Training/results/full/N{...}/chain_states.npz
                                                     the cost-weighted runs
  <S>/Block_E_single_network_chain/E0_Track_dataset/results/tracks.npz
                                                     test_truth (n, 257, 5),
                                                     test_P (n,) [GeV]
  results/paper_numbers.json                         group `headline`: every
                                                     per-band median plotted is
                                                     checked against it

OUTPUTS
  figures/fig_headline_vs_p.png
  results/fig10_headline_vs_p.csv     the plotted per-band medians and ratios

  <S> = LHCb_Extrapolation_Project/single_network_chain_discrete_approach

Run:
  PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python fig10_headline_vs_p.py
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"

import sys                                   # noqa: E402
sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
# numbers.py in this folder shadows the standard-library `numbers` module that
# numpy imports; drop the script folder from sys.path before numpy is loaded and
# load common.py explicitly by file location (the guard numbers.py uses).
for _p in ("", ".", HERE):
    while _p in sys.path:
        sys.path.remove(_p)

import importlib.util                        # noqa: E402
import json                                  # noqa: E402

import numpy as np                           # noqa: E402
import matplotlib                            # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402

_spec = importlib.util.spec_from_file_location("sc_paper_common",
                                               os.path.join(HERE, "common.py"))
C = importlib.util.module_from_spec(_spec)
sys.modules["sc_paper_common"] = C
_spec.loader.exec_module(C)

# ------------------------------------------------------------ house style --
POOLED_LINE = C.TEXT2                  # grey
POOLED_FILL = "none"
WEIGHTED_LINE = C.BLUE
WEIGHTED_FILL = C.BLUE
COLOUR_NOTE = ("colour convention: pooled loss = grey, hollow markers, dashed; "
               "cost-weighted loss = blue #2a78d6, filled markers, solid")

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150,
    "figure.facecolor": C.SURFACE, "savefig.facecolor": C.SURFACE,
    "axes.facecolor": C.SURFACE, "axes.edgecolor": C.TEXT2,
    "axes.labelcolor": C.TEXT1, "text.color": C.TEXT1,
    "xtick.color": C.TEXT2, "ytick.color": C.TEXT2,
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.8, "grid.color": C.NEUTRAL, "grid.linewidth": 0.5,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})


def net_label(N, q):
    return "N = %d, q = %d, Δz = %.1f mm" % (N, q, C.dz_mm(N))


def net_title(N, q):
    return "N = %d, q = %d\n\u0394z = %.1f mm" % (N, q, C.dz_mm(N))


def band_centres():
    """Geometric band centres for markers, and the drawing edges [GeV]."""
    lo = [0.0, 3.0, 8.0, 20.0, 50.0]
    hi = [3.0, 8.0, 20.0, 50.0, 150.0]
    lo[0] = 1.6
    return [float(np.sqrt(a * b)) for a, b in zip(lo, hi)], lo, hi


# ------------------------------------------------------------------ data --
def load_all():
    a = C.split_arrays("test")
    P, truth = a["P"], a["truth"]
    out = {}
    for N, q in C.PAIRS:
        for loss, run in (("pooled", C.pooled_run(N, q)),
                          ("weighted", C.reweighted_run(N, q))):
            st = C.load_chain_states(run)
            out[(N, q, loss)] = C.deltas(st[:, -1], truth[:, C.N_MAX])["radial_um"]
    return P, out


def main():
    P, R = load_all()
    J = C.read_json(os.path.join(C.RESULTS, "paper_numbers.json"))
    masks = C.band_masks(P)
    partition = masks[:5]                       # the five bands, no loss window
    window = masks[5]
    centres, lo, hi = band_centres()

    fig = plt.figure(figsize=(7.9, 4.7))
    gs = fig.add_gridspec(2, 3, height_ratios=[3.0, 1.0], hspace=0.10, wspace=0.10)

    rows = []
    xlim = (1.6, 150.0)
    ylim = (2.0, 3.0e4)
    for j, (N, q) in enumerate(C.PAIRS):
        ax = fig.add_subplot(gs[0, j])
        axr = fig.add_subplot(gs[1, j], sharex=ax)
        ref = J["headline"]["pairs"][C.run_key(N, q)]

        ax.axvspan(C.LOSS_WINDOW[0], C.LOSS_WINDOW[1], color=C.NEUTRAL,
                   alpha=0.45, lw=0, zorder=0)
        axr.axvspan(C.LOSS_WINDOW[0], C.LOSS_WINDOW[1], color=C.NEUTRAL,
                    alpha=0.45, lw=0, zorder=0)

        rw = R[(N, q, "weighted")]
        hb = ax.hexbin(P, np.clip(rw, ylim[0], ylim[1]), xscale="log", yscale="log",
                       gridsize=(34, 26), extent=(np.log10(xlim[0]), np.log10(xlim[1]),
                                                  np.log10(ylim[0]), np.log10(ylim[1])),
                       mincnt=1, cmap="Blues", linewidths=0.0, zorder=1)
        hb.set_alpha(0.85)

        for e in (3.0, 8.0, 20.0, 50.0):
            ax.axvline(e, color=C.TEXT2, lw=0.5, ls=":", alpha=0.8, zorder=2)
            axr.axvline(e, color=C.TEXT2, lw=0.5, ls=":", alpha=0.8, zorder=2)

        med = {}
        for loss, key in (("pooled", "pooled"), ("weighted", "reweighted")):
            v = R[(N, q, loss)]
            m = [C.median(v[mk]) for _, mk in partition]
            # the JSON is the authority: every plotted median must reproduce it
            for (lab, _), got in zip(partition, m):
                want = ref[key]["endpoint"]["by_band"][lab]["med_um"]
                assert abs(got - want) <= 1e-9 * max(1.0, abs(want)), (N, q, loss, lab, got, want)
            med[loss] = m
            col = POOLED_LINE if loss == "pooled" else WEIGHTED_LINE
            ls = "--" if loss == "pooled" else "-"
            xs = np.array(lo + [hi[-1]])
            ax.step(xs, np.array(m + [m[-1]]), where="post", color=col, ls=ls,
                    lw=1.5, zorder=4, solid_capstyle="butt")
            ax.plot(centres, m, "o", ms=4.5, color=col, zorder=5,
                    mfc=(POOLED_FILL if loss == "pooled" else WEIGHTED_FILL),
                    mew=1.3, label=("pooled loss" if loss == "pooled" else "cost-weighted loss"))
        # the loss window median, drawn as a short bar over the shaded span
        for loss, key in (("pooled", "pooled"), ("weighted", "reweighted")):
            v = R[(N, q, loss)]
            wmed = C.median(v[window[1]])
            want = ref[key]["endpoint"]["by_band"][C.LOSS_WINDOW_LABEL]["med_um"]
            assert abs(wmed - want) <= 1e-9 * max(1.0, abs(want))
            col = POOLED_LINE if loss == "pooled" else WEIGHTED_LINE
            ax.plot([10.0, 50.0], [wmed, wmed], color=col, lw=2.6, alpha=0.9,
                    ls=(0, (1, 1)) if loss == "pooled" else "-", zorder=6)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(net_title(N, q), pad=16, linespacing=1.15)
        ax.tick_params(labelbottom=False)
        ax.grid(True, which="major", axis="y", alpha=0.45)
        for (lab, mk), cx in zip(partition, centres):
            ax.text(cx, ylim[1] * 0.80, "n = %d" % int(mk.sum()), ha="center", va="top",
                    fontsize=6.4, color=C.TEXT2, rotation=90, zorder=7)
        if j == 0:
            ax.set_ylabel("endpoint radial error at $z_1$ [µm]")
            leg = ax.legend(loc="lower left", frameon=True, framealpha=0.92,
                            handlelength=1.6, borderpad=0.35)
            leg.get_frame().set_edgecolor(C.NEUTRAL)
        else:
            ax.tick_params(labelleft=False)

        ratio = [p / w for p, w in zip(med["pooled"], med["weighted"])]
        axr.axhline(1.0, color=C.TEXT1, lw=0.8, zorder=3)
        axr.plot(centres, ratio, "s-", ms=4.0, lw=1.3, color=C.MAGENTA,
                 mfc=C.MAGENTA, zorder=5)
        axr.set_yscale("log")
        axr.set_ylim(0.2, 16.0)
        axr.set_yticks([0.25, 1.0, 4.0, 10.0])
        axr.set_yticklabels(["0.25", "1", "4", "10"])
        axr.set_xlabel("momentum [GeV]")
        axr.grid(True, which="major", axis="y", alpha=0.45)
        if j == 0:
            axr.set_ylabel("pooled /\ncost-weighted", fontsize=7.6)
        else:
            axr.tick_params(labelleft=False)
        axr.set_xticks([2, 5, 10, 20, 50, 100])
        axr.set_xticklabels(["2", "5", "10", "20", "50", "100"])
        axr.minorticks_off()

        for (lab, mk), p, w, r in zip(partition, med["pooled"], med["weighted"], ratio):
            rows.append(dict(network=net_label(N, q), N=N, q=q, band_GeV=lab,
                             n=int(mk.sum()),
                             pooled_radial_med_um=p, cost_weighted_radial_med_um=w,
                             ratio_pooled_over_cost_weighted=r))
        pw = C.median(R[(N, q, "pooled")][window[1]])
        ww = C.median(R[(N, q, "weighted")][window[1]])
        rows.append(dict(network=net_label(N, q), N=N, q=q, band_GeV=C.LOSS_WINDOW_LABEL,
                         n=int(window[1].sum()), pooled_radial_med_um=pw,
                         cost_weighted_radial_med_um=ww,
                         ratio_pooled_over_cost_weighted=pw / ww))

    cb = fig.colorbar(hb, ax=fig.axes[0::2], fraction=0.016, pad=0.012)
    cb.set_label("test tracks per cell (cost-weighted loss)", fontsize=7.4)
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_edgecolor(C.NEUTRAL)

    out_png = os.path.join(C.PAPER, "figures", "fig_headline_vs_p.png")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)

    out_csv = os.path.join(C.RESULTS, "fig10_headline_vs_p.csv")
    with open(out_csv, "w") as f:
        f.write("# figure 10, fig_headline_vs_p.png: endpoint radial error against momentum.\n")
        f.write("# %s.\n" % COLOUR_NOTE)
        f.write("# positions in micrometres; bands from the per-split P array of tracks.npz;\n")
        f.write("# every median below equals results/paper_numbers.json group `headline`.\n")
    with open(out_csv, "a") as f:
        import csv as _csv
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote %s" % out_png)
    print("wrote %s" % out_csv)


if __name__ == "__main__":
    main()
