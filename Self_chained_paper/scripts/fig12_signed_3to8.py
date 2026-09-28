#!/usr/bin/env python
"""Figure 12 - the signed endpoint deviation of the N = 64, q = 2 network.

WHAT IT SHOWS
  The distribution of the SIGNED deviation of the network's endpoint state at
  z1 = 7826.0 mm from the RK6 reference, for the N = 64, q = 2, dz = 80.9 mm
  network under both losses (step histograms, same binning).  Four columns:
  Delta x and Delta y in micrometres, Delta t_x and Delta t_y dimensionless
  (t = dx/dz; nothing is scaled by 1e3 and no axis says mrad, so the slope
  axes read in scientific notation).

  Top row: the 3-8 GeV band (n = 655).  Bottom row: the 4-6 GeV band
  (n = 286), the region the supervisors asked about.  Each panel annotates,
  for both losses, the signed median, the 68 % half-width (q84 - q16)/2 and
  the RMS; the two position columns also print the fraction of tracks whose
  |deviation| in that component exceeds 1 mm.  The bin range is finite, so the
  count of tracks falling outside the drawn axis is printed too.

COLOUR CONVENTION (identical in figures 10-15)
  pooled loss        grey, hollow markers, dashed line
  cost-weighted loss blue (#2a78d6), filled markers, solid line

INPUTS READ (all read-only)
  scripts/common.py                        paths, deltas(), statistics
  <S>/Block_E_single_network_chain/E1_Network_grid/results/N064_q02/
      chain_states.npz                     test_states (1452, 65, 5), pooled
  <S>/Block_F_reweighted_loss/F1_Training/results/full/N064_q02/
      chain_states.npz                     the cost-weighted run
  <S>/Block_E_single_network_chain/E0_Track_dataset/results/tracks.npz
                                           test_truth (1452, 257, 5), test_P
  results/paper_numbers.json               group `near_5gev`: every median,
                                           half-width and RMS annotated here is
                                           asserted equal to it

OUTPUTS
  figures/fig_signed_3to8.png
  results/fig12_signed_3to8.csv

  <S> = LHCb_Extrapolation_Project/single_network_chain_discrete_approach

Run:
  PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python fig12_signed_3to8.py
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
from matplotlib.ticker import FuncFormatter  # noqa: E402

_spec = importlib.util.spec_from_file_location("sc_paper_common",
                                               os.path.join(HERE, "common.py"))
C = importlib.util.module_from_spec(_spec)
sys.modules["sc_paper_common"] = C
_spec.loader.exec_module(C)

POOLED_LINE = C.TEXT2
WEIGHTED_LINE = C.BLUE
COLOUR_NOTE = ("colour convention: pooled loss = grey (dashed step histogram); "
               "cost-weighted loss = blue #2a78d6 (solid step histogram)")

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150,
    "figure.facecolor": C.SURFACE, "savefig.facecolor": C.SURFACE,
    "axes.facecolor": C.SURFACE, "axes.edgecolor": C.TEXT2,
    "axes.labelcolor": C.TEXT1, "text.color": C.TEXT1,
    "xtick.color": C.TEXT2, "ytick.color": C.TEXT2,
    "font.size": 9, "axes.labelsize": 8.5, "axes.titlesize": 9,
    "xtick.labelsize": 7.4, "ytick.labelsize": 7.4, "legend.fontsize": 7.5,
    "axes.linewidth": 0.8, "grid.color": C.NEUTRAL, "grid.linewidth": 0.5,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})

COMPS = (("x", "um", "$\\Delta x$ at $z_1$ [µm]"),
         ("y", "um", "$\\Delta y$ at $z_1$ [µm]"),
         ("tx", "slope", "$\\Delta t_x$ at $z_1$ (dimensionless)"),
         ("ty", "slope", "$\\Delta t_y$ at $z_1$ (dimensionless)"))
N, Q = 64, 2


def fmt(v, unit):
    return ("%+.1f" % v) if unit == "um" else ("%+.2e" % v)


def fmt_pos(v, unit):
    return ("%.1f" % v) if unit == "um" else ("%.2e" % v)


def main():
    a = C.split_arrays("test")
    P, truth = a["P"], a["truth"]
    J = C.read_json(os.path.join(C.RESULTS, "paper_numbers.json"))
    near = J["near_5gev"]["bands"]

    d = {}
    for loss, run in (("pooled", C.pooled_run(N, Q)), ("reweighted", C.reweighted_run(N, Q))):
        st = C.load_chain_states(run)
        d[loss] = C.deltas(st[:, -1], truth[:, C.N_MAX])

    bands = (("3-8", (P >= 3.0) & (P < 8.0)), ("4-6", (P >= 4.0) & (P < 6.0)))

    fig, axes = plt.subplots(2, 4, figsize=(8.4, 5.0))
    rows = []
    for ri, (band, mask) in enumerate(bands):
        n = int(mask.sum())
        for ci, (comp, unit, xlab) in enumerate(COMPS):
            ax = axes[ri, ci]
            vals = {k: d[k][comp][mask] for k in ("pooled", "reweighted")}
            hw = max(C.halfwidth68(v) for v in vals.values())
            lim = 4.0 * hw
            bins = np.linspace(-lim, lim, 49)
            lines = []
            for loss, col, ls in (("pooled", POOLED_LINE, "--"),
                                  ("reweighted", WEIGHTED_LINE, "-")):
                v = vals[loss]
                ax.hist(v, bins=bins, histtype="step", color=col, lw=1.3, ls=ls,
                        zorder=(3 if loss == "pooled" else 4),
                        label=("pooled loss" if loss == "pooled" else "cost-weighted loss"))
                s_med, s_hw, s_rms = C.signed_median(v), C.halfwidth68(v), C.rms(v)
                # the JSON is the authority for all three
                ref = near[band][loss]["components"][comp]
                for got, want in ((s_med, ref["signed_med_%s" % unit]),
                                  (s_hw, ref["hw68_%s" % unit]),
                                  (s_rms, ref["rms_%s" % unit])):
                    assert abs(got - want) <= 1e-9 * max(1.0, abs(want)), (band, loss, comp, got, want)
                assert int(ref["n"]) == n, (band, ref["n"], n)
                frac1mm = float((np.abs(v) > 1000.0).mean()) if unit == "um" else float("nan")
                outside = int((np.abs(v) > lim).sum())
                tag = "pooled" if loss == "pooled" else "cost-weighted"
                if unit == "um":
                    txt = ("%s\nmed %s\nhw68 %s\nRMS %s\n>1 mm %.1f %%\noff axis %d"
                           % (tag, fmt(s_med, unit), fmt_pos(s_hw, unit),
                              fmt_pos(s_rms, unit), 100.0 * frac1mm, outside))
                else:
                    txt = ("%s\nmed %s\nhw68 %s\nRMS %s\noff axis %d"
                           % (tag, fmt(s_med, unit), fmt_pos(s_hw, unit),
                              fmt_pos(s_rms, unit), outside))
                lines.append((txt, col))
                rows.append(dict(
                    network="N = %d, q = %d, Δz = %.1f mm" % (N, Q, C.dz_mm(N)),
                    band_GeV=band, n=n, component=comp,
                    unit=("um" if unit == "um" else "dimensionless"),
                    loss=tag, signed_median=s_med, halfwidth68=s_hw, rms=s_rms,
                    med_abs=C.med_abs(v),
                    frac_abs_beyond_1mm=frac1mm,
                    axis_limit=lim, n_outside_axis=outside))
            ax.axvline(0.0, color=C.TEXT1, lw=0.6, alpha=0.7, zorder=2)
            ax.set_xlim(-lim, lim)
            e = int(np.floor(np.log10(lim)))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos, _e=e: "%g" % (v / 10.0 ** _e)))
            ax.set_xlabel("%s\n$\\times 10^{%d}$" % (xlab, e), fontsize=7.6, linespacing=1.1)
            ax.grid(True, axis="y", alpha=0.4)
            ax.set_axisbelow(True)
            top = ax.get_ylim()[1]
            ax.set_ylim(0, top * 2.15)
            ax.text(0.02, 0.985, lines[0][0], transform=ax.transAxes, va="top", ha="left",
                    fontsize=5.6, color=lines[0][1], linespacing=1.10)
            ax.text(0.98, 0.985, lines[1][0], transform=ax.transAxes, va="top", ha="right",
                    fontsize=5.6, color=lines[1][1], linespacing=1.10)
            if ci == 0:
                ax.set_ylabel("%s GeV, n = %d\ntest tracks per bin" % (band, n), fontsize=8)

    fig.suptitle("N = %d, q = %d, Δz = %.1f mm: signed endpoint deviation"
                 % (N, Q, C.dz_mm(N)), fontsize=9.5, y=1.045)
    handles = [Line2D([], [], color=POOLED_LINE, ls="--", lw=1.3, label="pooled loss"),
               Line2D([], [], color=WEIGHTED_LINE, ls="-", lw=1.3, label="cost-weighted loss")]
    leg = fig.legend(handles=handles, loc="upper center", ncol=2, frameon=True,
                     framealpha=0.95, bbox_to_anchor=(0.5, 1.012), handlelength=2.0,
                     borderpad=0.35)
    leg.get_frame().set_edgecolor(C.NEUTRAL)
    fig.tight_layout(h_pad=1.4, w_pad=1.1)

    out_png = os.path.join(C.PAPER, "figures", "fig_signed_3to8.png")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)

    out_csv = os.path.join(C.RESULTS, "fig12_signed_3to8.csv")
    with open(out_csv, "w", newline="") as f:
        f.write("# figure 12, fig_signed_3to8.png: signed endpoint deviation of the\n")
        f.write("# N = 64, q = 2 network in 3-8 GeV and 4-6 GeV.  Positions in micrometres,\n")
        f.write("# slopes dimensionless.  signed_median / halfwidth68 / rms are asserted\n")
        f.write("# equal to results/paper_numbers.json group `near_5gev`; frac_abs_beyond_1mm\n")
        f.write("# is the per-component |deviation| > 1 mm fraction, computed here (the JSON\n")
        f.write("# carries only the RADIAL beyond-1-mm fraction).\n")
        f.write("# %s.\n" % COLOUR_NOTE)
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote %s" % out_png)
    print("wrote %s" % out_csv)


if __name__ == "__main__":
    main()
