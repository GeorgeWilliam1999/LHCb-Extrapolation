#!/usr/bin/env python
"""Figure 13 - the error along the crossing: one step against the whole chain.

WHAT IT SHOWS
  Left: the median radial deviation from the RK6 reference at every plane of
  the chain, for the six runs, against z [mm] from z0 = 2648.2 mm to
  z1 = 7826.0 mm (log y, micrometres).  Each network has its own colour; the
  pooled-loss run is dashed and the cost-weighted-loss run solid.  Plane 0 is
  the track's own starting state and is exactly zero, so the curves start at
  the first output plane.

  Right: the same six networks applied ONCE, from the RK6 truth state on every
  start plane, and compared with the RK6 truth one plane later - the
  single-step error, plotted against the start plane's z on the same log scale.
  The two panels share their y range on purpose: a single application of any of
  these networks is well under 1 micrometre, while the chain of N of them ends
  around 100 micrometres.  The ratio of the two, per run, is printed in the
  right panel.

  No slopes appear in this figure; positions are micrometres throughout.

COLOUR CONVENTION
  Loss: pooled = dashed, cost-weighted = solid (the paper's grey/blue loss
  colours are kept for figures 10-12, 14, 15; here the colour carries the
  network, as three networks must be told apart in both panels).

INPUTS READ (all read-only)
  scripts/common.py             paths, palette, dz_mm(), Z0_MM / Z1_MM
  results/paper_numbers.json    group `along_z`, one entry per run:
                                chain_radial_med_um_per_plane (N+1 values),
                                single_step_radial_med_um_per_plane (N values),
                                chain_over_single_step, dz_mm, quarter/half/end.
                                Those were computed by scripts/numbers.py from
                                the runs' chain_states.npz and from the cached
                                single-step arrays results/cache/single_step_*.npz
                                (the network applied once on every plane through
                                chain_network.load_network), against tracks.npz.

OUTPUTS
  figures/fig_along_z.png
  results/fig13_along_z.csv

Run:
  PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python fig13_along_z.py
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

_spec = importlib.util.spec_from_file_location("sc_paper_common",
                                               os.path.join(HERE, "common.py"))
C = importlib.util.module_from_spec(_spec)
sys.modules["sc_paper_common"] = C
_spec.loader.exec_module(C)

NET_COLOUR = {(64, 2): C.BLUE, (128, 8): C.GREEN, (256, 16): "#c0397a"}
COLOUR_NOTE = ("colour convention: one colour per network (N = 64 blue, N = 128 green, "
               "N = 256 magenta); pooled loss dashed, cost-weighted loss solid.  "
               "In figures 10-12, 14 and 15 the loss itself carries the colour: "
               "pooled = grey, cost-weighted = blue #2a78d6")

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150,
    "figure.facecolor": C.SURFACE, "savefig.facecolor": C.SURFACE,
    "axes.facecolor": C.SURFACE, "axes.edgecolor": C.TEXT2,
    "axes.labelcolor": C.TEXT1, "text.color": C.TEXT1,
    "xtick.color": C.TEXT2, "ytick.color": C.TEXT2,
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.4,
    "axes.linewidth": 0.8, "grid.color": C.NEUTRAL, "grid.linewidth": 0.5,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})


def net_label(N, q):
    return "N = %d, q = %d, Δz = %.1f mm" % (N, q, C.dz_mm(N))


def main():
    J = C.read_json(os.path.join(C.RESULTS, "paper_numbers.json"))
    per_run = J["along_z"]["per_run"]

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(8.0, 3.7), sharey=True)
    rows = []
    for N, q in C.PAIRS:
        col = NET_COLOUR[(N, q)]
        dz = C.dz_mm(N)
        for loss, ls in (("pooled", "--"), ("reweighted", "-")):
            e = per_run["%s %s" % (loss, C.run_key(N, q))]
            assert abs(e["dz_mm"] - dz) < 1e-9
            chain = np.asarray(e["chain_radial_med_um_per_plane"], dtype=float)
            step = np.asarray(e["single_step_radial_med_um_per_plane"], dtype=float)
            z_chain = C.Z0_MM + np.arange(len(chain)) * dz
            z_step = C.Z0_MM + np.arange(len(step)) * dz
            tag = "pooled" if loss == "pooled" else "cost-weighted"
            axl.plot(z_chain[1:], chain[1:], ls=ls, lw=1.4, color=col, zorder=4)
            axr.plot(z_step, step, ls=ls, lw=1.4, color=col, zorder=4)
            for k, (z, v) in enumerate(zip(z_chain, chain)):
                rows.append(dict(panel="left: chain", network=net_label(N, q), N=N, q=q,
                                 loss=tag, plane=k, z_mm=z, radial_med_um=v))
            for k, (z, v) in enumerate(zip(z_step, step)):
                rows.append(dict(panel="right: single step from the RK6 state",
                                 network=net_label(N, q), N=N, q=q, loss=tag,
                                 plane=k, z_mm=z, radial_med_um=v))

    for ax in (axl, axr):
        ax.set_yscale("log")
        ax.set_xlim(C.Z0_MM - 60, C.Z1_MM + 60)
        ax.set_xlabel("z [mm]")
        ax.grid(True, which="major", alpha=0.45)
        ax.set_axisbelow(True)
        ax.set_ylim(2e-2, 4e2)
    axl.set_ylabel("median radial deviation from RK6 [µm]")
    axl.set_title("chained: the network applied to its own output", fontsize=8.8)
    axr.set_title("a single application from the RK6 state", fontsize=8.8)
    axr.axhline(1.0, color=C.TEXT1, lw=0.7, ls=":", zorder=3)
    axr.text(C.Z1_MM - 60, 1.15, "1 µm", ha="right", va="bottom", fontsize=7,
             color=C.TEXT1)

    ratio_txt = "chain end / single step:\n" + "\n".join(
        "  %s  %s  %d×" % (net_label(N, q).split(",")[0],
                                ("pooled" if loss == "pooled" else "cost-weighted"),
                                round(per_run["%s %s" % (loss, C.run_key(N, q))]
                                      ["chain_over_single_step"]))
        for N, q in C.PAIRS for loss in ("pooled", "reweighted"))
    axr.text(0.03, 0.97, ratio_txt, transform=axr.transAxes, va="top", ha="left",
             fontsize=6.4, color=C.TEXT2, linespacing=1.25,
             bbox=dict(facecolor=C.SURFACE, edgecolor=C.NEUTRAL, linewidth=0.5,
                       boxstyle="round,pad=0.30", alpha=0.95))

    handles = [Line2D([], [], color=NET_COLOUR[(N, q)], lw=1.6, label=net_label(N, q))
               for N, q in C.PAIRS]
    handles += [Line2D([], [], color=C.TEXT2, lw=1.4, ls="--", label="pooled loss"),
                Line2D([], [], color=C.TEXT2, lw=1.4, ls="-", label="cost-weighted loss")]
    leg = fig.legend(handles=handles, loc="upper center", ncol=5, frameon=True,
                     framealpha=0.95, bbox_to_anchor=(0.5, 1.08), handlelength=2.0,
                     columnspacing=1.2, borderpad=0.35)
    leg.get_frame().set_edgecolor(C.NEUTRAL)
    fig.tight_layout(w_pad=1.2)

    out_png = os.path.join(C.PAPER, "figures", "fig_along_z.png")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, bbox_extra_artists=(leg,))
    plt.close(fig)

    out_csv = os.path.join(C.RESULTS, "fig13_along_z.csv")
    with open(out_csv, "w", newline="") as f:
        f.write("# figure 13, fig_along_z.png: the median radial deviation from the RK6\n")
        f.write("# reference plane by plane [micrometres], chained (left) and for a single\n")
        f.write("# application from the RK6 state on that plane (right).  Every value is\n")
        f.write("# taken verbatim from results/paper_numbers.json group `along_z`.\n")
        f.write("# %s.\n" % COLOUR_NOTE)
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote %s" % out_png)
    print("wrote %s" % out_csv)


if __name__ == "__main__":
    main()
