#!/usr/bin/env python
"""Figure 14 - the anatomy of the chained error, step by step.

WHAT IT SHOWS
  The two N = 64, q = 2, dz = 80.9 mm networks, one per ROW (top: pooled loss,
  bottom: cost-weighted loss).  Each step's output is compared with an RK6
  integration started from the SAME state the network was given, over the same
  dz, at a 1 mm local step; that local deviation is what every panel uses.

  (a) the median |Delta t_x| and |Delta t_y| of a single step against the z of
      that step's output plane.  Dimensionless, log y - no quantity is scaled
      by 1e3 and no axis says mrad.
  (b) what each step costs at the endpoint: the median over test tracks of
      hypot(dx_k + dt_x,k * lever_k, dy_k + dt_y,k * lever_k) [micrometres],
      where lever_k = z1 - z_k is the distance left to run after step k.  An
      early step is expensive because its slope error is carried a long way.
  (c) the running sum of the SIGNED per-step slope error along the track,
      sum_(j<=k) Delta t_x,j, for 50 test tracks (thin lines, one seed fixed in
      this script).  The lines drift in one direction instead of wandering:
      that is what the coherence number annotated in the panel measures -
      each row of (c) carries its own y scale, as the two losses differ by
      about a decade in the size of the drift -
      median over tracks of |sum_k dt| / sum_k |dt|, against 1/sqrt(N) = 0.125
      for steps whose errors were independent.
  (d) how much of the endpoint error the lever-arm model accounts for, for all
      six runs: the median of the model over the median of the measured radial
      endpoint error, with both slopes and with the x slope alone.

COLOUR CONVENTION (identical in figures 10-15)
  pooled loss        grey, hollow markers, dashed line
  cost-weighted loss blue (#2a78d6), filled markers, solid line

INPUTS READ (all read-only)
  scripts/common.py                          paths, palette, statistics
  results/cache/anatomy_{pooled,reweighted}_N064_q02.npz
                                             lx, ly [micrometres] and ltx, lty
                                             [dimensionless], each (64, 1452),
                                             plus final (1452,) - written by
                                             scripts/numbers.py from the runs'
                                             network.pt through
                                             chain_network.step_outputs against
                                             _shared.reference.rk6_rows
  results/paper_numbers.json                 group `anatomy`, per run: the
                                             per-step medians (asserted equal to
                                             the arrays above), the coherences
                                             and the explained fractions

OUTPUTS
  figures/fig_anatomy.png
  results/fig14_anatomy.csv

Run:
  PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python fig14_anatomy.py
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
from matplotlib.patches import Patch         # noqa: E402

_spec = importlib.util.spec_from_file_location("sc_paper_common",
                                               os.path.join(HERE, "common.py"))
C = importlib.util.module_from_spec(_spec)
sys.modules["sc_paper_common"] = C
_spec.loader.exec_module(C)

POOLED_LINE = C.TEXT2
WEIGHTED_LINE = C.BLUE
COLOUR_NOTE = ("colour convention: pooled loss = grey, hollow markers, dashed; "
               "cost-weighted loss = blue #2a78d6, filled markers, solid")

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150,
    "figure.facecolor": C.SURFACE, "savefig.facecolor": C.SURFACE,
    "axes.facecolor": C.SURFACE, "axes.edgecolor": C.TEXT2,
    "axes.labelcolor": C.TEXT1, "text.color": C.TEXT1,
    "xtick.color": C.TEXT2, "ytick.color": C.TEXT2,
    "font.size": 9, "axes.labelsize": 8.2, "axes.titlesize": 8.6,
    "xtick.labelsize": 7.2, "ytick.labelsize": 7.2, "legend.fontsize": 7.0,
    "axes.linewidth": 0.8, "grid.color": C.NEUTRAL, "grid.linewidth": 0.5,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})

N, Q = 64, 2
N_TRACKS_SHOWN = 50
SEED = 20260922


def net_label(Nn, qq):
    return "N = %d, q = %d, Δz = %.1f mm" % (Nn, qq, C.dz_mm(Nn))


def main():
    J = C.read_json(os.path.join(C.RESULTS, "paper_numbers.json"))
    anat = J["anatomy"]["per_run"]
    dz = C.dz_mm(N)
    z_out = C.Z0_MM + (np.arange(N) + 1) * dz          # each step's output plane
    lever = C.Z1_MM - z_out                            # mm left to run after step k

    # ---- the two rows' arrays first, so both rows can share their axes ----
    data = {}
    for loss in ("pooled", "reweighted"):
        with np.load(os.path.join(C.CACHE, "anatomy_%s_%s.npz"
                                  % (loss, C.run_tag(N, Q)))) as z:
            lx, ly, ltx, lty = z["lx"], z["ly"], z["ltx"], z["lty"]
        ref = anat["%s %s" % (loss, C.run_key(N, Q))]
        assert lx.shape == (N, 1452), lx.shape
        # the cached arrays must reproduce the JSON's per-step medians
        for got, want in ((C.med_abs(ltx), ref["per_step_tx_med_abs_slope"]),
                          (C.med_abs(lty), ref["per_step_ty_med_abs_slope"]),
                          (C.med_abs(lx), ref["per_step_x_med_abs_um"]),
                          (C.med_abs(ly), ref["per_step_y_med_abs_um"])):
            assert abs(got - want) <= 1e-9 * max(1.0, abs(want)), (loss, got, want)
        rng = np.random.default_rng(SEED)
        pick = rng.choice(ltx.shape[1], size=N_TRACKS_SHOWN, replace=False)
        data[loss] = dict(
            ref=ref,
            mtx=np.median(np.abs(ltx), axis=1),
            mty=np.median(np.abs(lty), axis=1),
            cost=np.median(np.hypot(lx + ltx * 1e3 * lever[:, None],
                                    ly + lty * 1e3 * lever[:, None]), axis=1),
            cum=np.cumsum(ltx[:, pick], axis=0))

    a_lim = (min(min(d["mtx"].min(), d["mty"].min()) for d in data.values()) * 0.7,
             max(max(d["mtx"].max(), d["mty"].max()) for d in data.values()) * 2.6)
    b_lim = (min(d["cost"].min() for d in data.values()) * 0.7,
             max(d["cost"].max() for d in data.values()) * 1.5)
    # panel (c) is scaled row by row: its point is the DIRECTION of the drift,
    # and the two losses differ in magnitude by about a decade.  Each panel
    # carries its own scientific-notation exponent.
    c_lim = {k: float(np.abs(d["cum"]).max()) * 1.08 for k, d in data.items()}

    fig = plt.figure(figsize=(9.0, 5.1), layout="constrained")
    gs = fig.add_gridspec(2, 4, width_ratios=[1.0, 1.0, 1.0, 1.05])
    ax_d = fig.add_subplot(gs[:, 3])
    rows = []

    for ri, (loss, col, ls) in enumerate((("pooled", POOLED_LINE, "--"),
                                          ("reweighted", WEIGHTED_LINE, "-"))):
        tag = "pooled" if loss == "pooled" else "cost-weighted"
        d = data[loss]
        ref = d["ref"]
        mtx, mty, cost, cum = d["mtx"], d["mty"], d["cost"], d["cum"]

        # (a) per-step slope error
        ax_a = fig.add_subplot(gs[ri, 0])
        ax_a.plot(z_out, mtx, ls="-", lw=1.2, color=col, label="$|\\Delta t_x|$")
        ax_a.plot(z_out, mty, ls=":", lw=1.2, color=col, label="$|\\Delta t_y|$")
        ax_a.set_yscale("log")
        ax_a.set_ylim(*a_lim)
        ax_a.set_ylabel("median |slope error|\nof one step", fontsize=7.4)
        ax_a.set_title("(a) %s loss" % tag, fontsize=8.4)
        leg = ax_a.legend(loc="upper left", frameon=True, framealpha=0.95,
                          handlelength=1.6, borderpad=0.25, labelspacing=0.2, fontsize=6.6)
        leg.get_frame().set_edgecolor(C.NEUTRAL)

        # (b) each step's cost at the endpoint
        ax_b = fig.add_subplot(gs[ri, 1])
        ax_b.plot(z_out, cost, ls="-", lw=1.3, color=col)
        ax_b.set_yscale("log")
        ax_b.set_ylim(*b_lim)
        ax_b.set_ylabel("median cost of one\nstep at $z_1$ [\u00b5m]", fontsize=7.4)
        ax_b.set_title("(b) %s loss" % tag, fontsize=8.4)

        # (c) the running sum of the signed slope error
        ax_c = fig.add_subplot(gs[ri, 2])
        for j in range(N_TRACKS_SHOWN):
            ax_c.plot(z_out, cum[:, j], lw=0.6, color=col, alpha=0.5)
        ax_c.axhline(0.0, color=C.TEXT1, lw=0.7, zorder=4)
        ax_c.set_ylim(-c_lim[loss], c_lim[loss])
        ax_c.set_ylabel("running sum of\nsigned $\\Delta t_x$", fontsize=7.4)
        ax_c.set_title("(c) %s loss, %d tracks" % (tag, N_TRACKS_SHOWN), fontsize=8.4)
        ax_c.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3), useMathText=True)
        ax_c.yaxis.get_offset_text().set_fontsize(6.4)
        ax_c.text(0.03, 0.03,
                  "coherence  $t_x$ %.2f   $t_y$ %.2f\nindependent steps %.3f"
                  % (ref["coherence_tx"], ref["coherence_ty"], ref["coherence_independent"]),
                  transform=ax_c.transAxes, va="bottom", ha="left", fontsize=6.2,
                  color=C.TEXT1, linespacing=1.25,
                  bbox=dict(facecolor=C.SURFACE, edgecolor=C.NEUTRAL, linewidth=0.5,
                            boxstyle="round,pad=0.26", alpha=0.95))

        for ax in (ax_a, ax_b, ax_c):
            ax.set_xlim(C.Z0_MM - 60, C.Z1_MM + 60)
            ax.grid(True, alpha=0.4)
            ax.set_axisbelow(True)
            if ri == 1:
                ax.set_xlabel("z of the step's output plane [mm]", fontsize=7.2)
            else:
                ax.tick_params(labelbottom=False)
            ax.set_xticks([3000, 5000, 7000])

        for k in range(N):
            rows.append(dict(panel="a,b: per step", network=net_label(N, Q),
                             loss=tag, step=k, z_out_mm=z_out[k], lever_mm=lever[k],
                             per_step_tx_med_abs=mtx[k], per_step_ty_med_abs=mty[k],
                             step_cost_at_z1_med_um=cost[k],
                             coherence_tx=ref["coherence_tx"],
                             coherence_ty=ref["coherence_ty"],
                             coherence_independent=ref["coherence_independent"]))

    # (d) the explained fractions of all six runs
    labels, both, xonly = [], [], []
    for Nn, qq in C.PAIRS:
        for loss in ("pooled", "reweighted"):
            e = anat["%s %s" % (loss, C.run_key(Nn, qq))]
            tag = "pooled" if loss == "pooled" else "cost-weighted"
            labels.append("N = %d, q = %d\n%s" % (Nn, qq, tag))
            both.append(e["explained_both_slopes"])
            xonly.append(e["explained_x_slope_only"])
            rows.append(dict(panel="d: explained fraction",
                             network=net_label(Nn, qq), loss=tag, step="",
                             z_out_mm="", lever_mm="", per_step_tx_med_abs="",
                             per_step_ty_med_abs="", step_cost_at_z1_med_um="",
                             coherence_tx=e["coherence_tx"], coherence_ty=e["coherence_ty"],
                             coherence_independent=e["coherence_independent"],
                             explained_both_slopes=e["explained_both_slopes"],
                             explained_x_slope_only=e["explained_x_slope_only"]))
    y = np.arange(len(labels))
    cols = [POOLED_LINE if "pooled" in s2 else WEIGHTED_LINE for s2 in labels]
    ax_d.barh(y - 0.19, both, height=0.34, color=cols, edgecolor=C.TEXT2, lw=0.35, zorder=3)
    ax_d.barh(y + 0.19, xonly, height=0.34, color=cols, edgecolor=C.TEXT2, lw=0.35,
              hatch="///", zorder=3)
    for yy, v in zip(y - 0.19, both):
        ax_d.text(v + 0.02, yy, "%.2f" % v, va="center", ha="left", fontsize=6.2)
    for yy, v in zip(y + 0.19, xonly):
        ax_d.text(v + 0.02, yy, "%.2f" % v, va="center", ha="left", fontsize=6.2)
    ax_d.axvline(1.0, color=C.TEXT1, lw=0.8, ls=":", zorder=4)
    ax_d.set_yticks(y)
    ax_d.set_yticklabels(labels, fontsize=6.4)
    ax_d.invert_yaxis()
    ax_d.set_xlim(0, 1.45)
    ax_d.set_xticks([0, 0.5, 1.0])
    ax_d.set_xlabel("explained fraction of the\nendpoint error", fontsize=7.2)
    ax_d.set_title("(d) all six runs", fontsize=8.4)
    ax_d.grid(True, axis="x", alpha=0.4)
    ax_d.set_axisbelow(True)
    handles = [Patch(facecolor=C.NEUTRAL, edgecolor=C.TEXT2, lw=0.35, label="both slopes"),
               Patch(facecolor=C.NEUTRAL, edgecolor=C.TEXT2, lw=0.35, hatch="///",
                     label="x slope only")]
    leg = ax_d.legend(handles=handles, loc="lower right", frameon=True, framealpha=0.95,
                      handlelength=1.2, borderpad=0.3, labelspacing=0.25, fontsize=6.2)
    leg.get_frame().set_edgecolor(C.NEUTRAL)

    fig.suptitle("the chained error, step by step: %s" % net_label(N, Q), fontsize=9.2)

    out_png = os.path.join(C.PAPER, "figures", "fig_anatomy.png")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)

    fields = ["panel", "network", "loss", "step", "z_out_mm", "lever_mm",
              "per_step_tx_med_abs", "per_step_ty_med_abs", "step_cost_at_z1_med_um",
              "coherence_tx", "coherence_ty", "coherence_independent",
              "explained_both_slopes", "explained_x_slope_only"]
    out_csv = os.path.join(C.RESULTS, "fig14_anatomy.csv")
    with open(out_csv, "w", newline="") as f:
        f.write("# figure 14, fig_anatomy.png: the per-step anatomy of the chained error\n")
        f.write("# for N = 64, q = 2 (panels a-c) and the explained fractions of all six\n")
        f.write("# runs (panel d).  Slopes dimensionless, positions micrometres.\n")
        f.write("# per_step_* and step_cost_at_z1_med_um are medians over the 1,452 test\n")
        f.write("# tracks of the cached arrays results/cache/anatomy_*_N064_q02.npz; their\n")
        f.write("# overall medians are asserted equal to results/paper_numbers.json group\n")
        f.write("# `anatomy`, from which the coherences and explained fractions are taken.\n")
        f.write("# step_cost_at_z1_med_um = median hypot(dx + dtx*lever, dy + dty*lever),\n")
        f.write("# lever = z1 - z_out; it is the per-step breakdown of the lever-arm model\n")
        f.write("# whose summed prediction the JSON carries, and is computed in this script.\n")
        f.write("# %s.\n" % COLOUR_NOTE)
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("wrote %s" % out_png)
    print("wrote %s" % out_csv)


if __name__ == "__main__":
    main()
