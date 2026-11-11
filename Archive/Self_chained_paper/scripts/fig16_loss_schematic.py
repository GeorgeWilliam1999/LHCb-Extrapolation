#!/usr/bin/env python
"""Figure - how one training step produces its residuals (figures/fig_loss_schematic.png).

The Raissi discrete-time construction drawn once, on the real geometry of the
N = 64, q = 2, dz = 80.903 mm network and on one real test track.  Both losses
of this paper share every part of this picture: they differ only in what they
divide the residuals by, which is panel (c)'s last two lines.

WHAT THE FIGURE SHOWS, panel by panel.

(a) THE FORWARD PICTURE.  The first step of the crossing, z_n = z0 to
    z_n + dz.  The input state S_n is the dot on the start plane; the dashed
    line is the straight line it would follow with no field; the three filled
    markers are the network's outputs at z_n + c_j dz for c_1 = 0.21132,
    c_2 = 0.78868 and c_end = 1, drawn as

        output = straight line + correction: sigma(S_n) (x) NN_theta(S_n, z_n).

    The network learns the correction to the straight line, not the stage
    states directly; this departs from Raissi, Perdikaris and Karniadakis,
    whose network emits the states themselves (paper, Section 2.4).  The
    correction to the straight line is real but tiny - 42 um over an
    80.9 mm step for this track - so it is drawn multiplied by EXAGGERATION
    (100x).  The straight line itself and the plane positions are to scale.
    The annotation names the parameterisation, and the line beneath it says
    what the network learns.  (The companion CSV keeps its original series
    labels, "deviation from the straight line", so that the file is unchanged;
    that series is the correction to the straight line.)

(b) THE BACKWARD PICTURE.  From each of the three outputs, the implicit
    Runge-Kutta reconstruction

        R_{n,j} = S_hat_{n,j} - dz SUM_k A_jk f(S_hat_{n,k}, z_n + c_k dz)

    runs the step backwards to the start plane.  Each landing point misses the
    known input S_n by the reconstruction residual r_{n,j} = R_{n,j} - S_n.
    On the scale of the panel the three landings are indistinguishable from
    S_n, so the three gaps are drawn at their true size in the inset, in
    micrometres, one component (x) only; the same construction produces a
    residual in y, t_x and t_y as well.  No reference trajectory is involved
    anywhere: only the input, the tableau, and the field at the network's own
    proposed positions.

(c) THE ALGEBRA.  The q = 2 Gauss-Legendre tableau with its actual numbers,
    including the endpoint row A_{end,k} = b_k that folds Eq. (eq:recon-end)
    into the same matrix, and the one-line form of the loss with the two
    choices of the divisor s that separate the two losses of this paper.

INPUTS READ (all read-only).

  <S>/Block_E_single_network_chain/E0_Track_dataset/results/tracks.npz
                                 the test split; the drawn track is the
                                 median-momentum track of the 3-8 GeV band
                                 (index 275, p = 4.936 GeV).
  <S>/Block_E_single_network_chain/E1_Network_grid/results/N064_q02/
                                 the extended pooled-loss N = 64, q = 2 run:
                                 scale.json + network.pt, loaded read-only
                                 through chain_network.load_network, for the
                                 three real outputs and the real residuals.
  <S>/_shared/irk.py             gauss_legendre(2) for c, A and b.
  <S>/_shared/model.py           reconstruction_residuals, so the residuals
                                 drawn are the ones the trainer forms.
  <S>/_shared/reference.py       make_field, the v8r1 magnet-up map.
  results/paper_numbers.json     group `constants` (the per-run dz).
  scripts/common.py              paths, palette, geometry, loaders.

OUTPUTS.

  figures/fig_loss_schematic.png
  results/fig16_loss_schematic.csv      every plotted value.

Run:
  PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python fig16_loss_schematic.py
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"

import sys                                   # noqa: E402
sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
# numbers.py's guard, for the same reason: this folder holds modules whose names
# shadow the standard library, so drop it from the import path BEFORE numpy is
# imported and load common.py explicitly by file location.
for _p in ("", ".", HERE):
    while _p in sys.path:
        sys.path.remove(_p)

import importlib.util                        # noqa: E402

import numpy as np                           # noqa: E402
import matplotlib                            # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402
from matplotlib.patches import FancyArrowPatch   # noqa: E402

_spec = importlib.util.spec_from_file_location("sc_paper_common",
                                               os.path.join(HERE, "common.py"))
C = importlib.util.module_from_spec(_spec)
sys.modules["sc_paper_common"] = C
_spec.loader.exec_module(C)

FIGURES = os.path.join(C.PAPER, "figures")
PNG = os.path.join(FIGURES, "fig_loss_schematic.png")
CSV = os.path.join(C.RESULTS, "fig16_loss_schematic.csv")

EXAGGERATION = 100.0        # panel (a): the correction to the straight line only

# ------------------------------------------------------------ house style --
P = C.PALETTE
FAINT = "#e5e4df"
BAND_FILL = "#dfe7f2"
INSET_BG = "#f4f4f1"

plt.rcParams.update({
    "figure.facecolor": P["SURFACE"], "axes.facecolor": P["SURFACE"],
    "savefig.facecolor": P["SURFACE"], "axes.edgecolor": P["NEUTRAL"],
    "axes.labelcolor": P["TEXT2"], "text.color": P["TEXT1"],
    "xtick.color": P["TEXT2"], "ytick.color": P["TEXT2"], "font.size": 9,
    "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.grid": True,
    "grid.color": FAINT, "grid.linewidth": 0.6, "legend.frameon": False,
    "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 150,
})

ROWS = []


def rec(panel, series, x, y, xlabel, ylabel, note=""):
    ROWS.append(dict(panel=panel, series=series, x=x, y=y,
                     x_quantity=xlabel, y_quantity=ylabel, note=note))


# ------------------------------------------------------- the real numbers --
def gather():
    """Everything the figure draws, from the trained network and the real track."""
    C.add_experiment_paths()
    import torch
    from _shared.model import LHCbRates, reconstruction_residuals
    from _shared.reference import make_field, gauss_legendre
    from chain_network import load_network, znodes_for

    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)

    N, q = 64, 2
    dz = C.dz_mm(N)
    c, A, b = gauss_legendre(q)

    D = C.load_tracks()
    T = C.split_arrays("test")
    band = (T["P"] >= 3.0) & (T["P"] < 8.0)
    idx = np.where(band)[0]
    # the median-momentum track of the 3-8 GeV band: a deterministic choice
    j = int(idx[np.argsort(T["P"][idx])[len(idx) // 2]])

    fld = make_field(str(D["field"]))
    rates = LHCbRates(make_field(str(D["field"])))
    model = load_network(C.pooled_run(N, q), fld)

    S = T["S0"][j:j + 1]
    z_start = np.array([C.Z0_MM])
    St, zt = torch.as_tensor(S), torch.as_tensor(z_start)
    zn = znodes_for(model, zt)
    At, bt = torch.tensor(A), torch.tensor(b)
    with torch.no_grad():
        out = model(St, zt).numpy()[0]                      # (q+1, 4)
        straight = model.straight(St).numpy()[0]            # (q+1, 4)
        sigma = model.residual_scale(St, zt).numpy()[0]     # (q+1, 4)
        r = (reconstruction_residuals(model, rates, St, dz, zn, At, bt, zt)
             * model.in_scale[:4]).numpy()[0]               # (q+1, 4), physical
    return dict(N=N, q=q, dz=dz, c=c, A=A, b=b, j=j, p=float(T["P"][j]),
                S=S[0], z_start=float(z_start[0]), out=out, straight=straight,
                sigma=sigma, r=r, in_scale=model.in_scale.numpy(),
                znodes=zn.numpy()[0])


# ----------------------------------------------------------------- panels --
def panel_a(ax, G):
    dz, c = G["dz"], G["c"]
    z0 = G["z_start"]
    x0, tx = G["S"][0], G["S"][2]
    cout = np.append(c, 1.0)
    zout = z0 + cout * dz

    zz = np.linspace(z0, z0 + dz, 400)
    ax.plot(zz, tx * (zz - z0), color=P["NEUTRAL"], lw=1.4,
            ls=(0, (5, 3)), zorder=2)

    # the bent path: a quadratic through the exaggerated corrections, purely for
    # the eye; the three markers themselves are the network's real outputs.
    dev = (G["out"][:, 0] - G["straight"][:, 0]) * EXAGGERATION
    coef = np.polyfit(np.append(0.0, cout) * dz, np.append(0.0, dev), 2)
    ax.plot(zz, tx * (zz - z0) + np.polyval(coef, zz - z0),
            color=P["BLUE"], lw=1.9, zorder=3)
    ax.text(z0 + 0.74 * dz, 7.25, "straight line from $S_n$",
            fontsize=7.8, color=P["TEXT2"], ha="center", va="center")
    ax.text(z0 + 0.60 * dz, 1.45,
            "the network's step\n(bend drawn $\\times$%d)" % EXAGGERATION,
            fontsize=7.8, color=P["BLUE"], ha="center", va="center",
            linespacing=1.35)

    ax.plot([z0], [0.0], "o", ms=8.0, color=P["TEXT1"], zorder=6)
    ax.text(z0 + 0.02 * dz, -0.80, "$S_n$", fontsize=10.5, color=P["TEXT1"],
            ha="left", va="center", fontweight="bold")

    names = ["$\\hat{S}_{n,1}$", "$\\hat{S}_{n,2}$",
             "$\\hat{S}_{n,\\mathrm{end}}$"]
    ys = tx * (zout - z0) + dev
    for zi, yi, nm in zip(zout, ys, names):
        ax.plot([zi, zi], [-1.25, yi], color=FAINT, lw=0.8, zorder=1)
        ax.plot([zi], [yi], "o", ms=7.5, color=P["BLUE"],
                markeredgecolor=P["TEXT1"], markeredgewidth=0.6, zorder=6)
        ax.text(zi - 0.025 * dz, yi + 0.62, nm, fontsize=9.5, color=P["BLUE"],
                ha="center", va="bottom")

    ax.annotate("output $=$ straight line + correction:\n$\\qquad \\sigma(S_n)\\odot"
                "\\mathrm{NN}_\\theta(S_n, z_n)$",
                xy=(z0 + 0.45 * dz, tx * 0.45 * dz + np.polyval(coef, 0.45 * dz)),
                xytext=(z0 + 0.06 * dz, 8.1),
                fontsize=8.2, color=P["TEXT1"], linespacing=1.4,
                bbox=dict(boxstyle="round,pad=0.30", fc=INSET_BG, ec=P["NEUTRAL"],
                          lw=0.6),
                arrowprops=dict(arrowstyle="->", lw=0.8, color=P["TEXT2"],
                                connectionstyle="arc3,rad=-0.25"))
    ax.text(z0 + 0.06 * dz, 9.45,
            "the network learns the correction,\nnot the stage states directly",
            fontsize=7.2, color=P["TEXT2"], style="italic", ha="left",
            va="bottom", linespacing=1.25)

    ax.axvline(z0, color=P["TEXT2"], lw=1.0, zorder=1)
    ax.text(z0 + 0.015 * dz, 11.0, "start plane $z_n$", fontsize=7.6,
            color=P["TEXT2"], ha="left", va="center")
    ax.text(z0 + 0.99 * dz, 11.0, "$z_n + \\Delta z$", fontsize=7.6,
            color=P["TEXT2"], ha="right", va="center")
    for zi, nm in zip(zout, ("$c_1$", "$c_2$", "$c_{\\mathrm{end}}$")):
        ax.text(zi, -1.62, nm, fontsize=8.0, color=P["TEXT2"], ha="center",
                va="center")
    ax.set_xlim(z0 - 0.06 * dz, z0 + 1.08 * dz)
    ax.set_ylim(-2.1, 11.8)
    ax.set_xlabel("$z$ [mm]")
    ax.set_ylabel("$x - x_n$ [mm]")
    ax.set_title("(a)  the forward picture", loc="left", pad=13.0)
    ax.text(0.0, 1.012,
            "$N$ = %d, $q$ = %d, $\\Delta z$ = %.3f mm; one test track, $p$ = %.2f GeV"
            % (G["N"], G["q"], G["dz"], G["p"]),
            transform=ax.transAxes, fontsize=7.3, color=P["TEXT2"],
            ha="left", va="bottom")

    for k, nm in enumerate(("1", "2", "end")):
        rec("a", "output " + nm, float(zout[k]), float(tx * (zout[k] - z0)),
            "z_mm", "x_minus_x_n_mm",
            "the real ordinate; the panel draws it with the deviation from the "
            "straight line multiplied by %g" % EXAGGERATION)
        rec("a", "deviation from the straight line, output " + nm,
            float(zout[k]), float(G["out"][k, 0] - G["straight"][k, 0]),
            "z_mm", "x_deviation_mm", "the real, unexaggerated deviation")
        rec("a", "sigma_x, output " + nm, float(zout[k]), float(G["sigma"][k, 0]),
            "z_mm", "sigma_x_mm", "the per-track output scale on x")


def panel_b(ax, G):
    dz, c = G["dz"], G["c"]
    z0 = G["z_start"]
    tx = G["S"][2]
    cout = np.append(c, 1.0)
    zout = z0 + cout * dz
    dev = (G["out"][:, 0] - G["straight"][:, 0]) * EXAGGERATION
    ys = tx * (zout - z0) + dev

    ax.axvline(z0, color=P["TEXT2"], lw=1.0, zorder=1)
    ax.text(z0 + 0.015 * dz, 11.0, "start plane $z_n$", fontsize=7.6,
            color=P["TEXT2"], ha="left", va="center")

    names = ["$\\hat{S}_{n,1}$", "$\\hat{S}_{n,2}$",
             "$\\hat{S}_{n,\\mathrm{end}}$"]
    rads = (0.10, 0.07, 0.05)
    for zi, yi, nm, rad in zip(zout, ys, names, rads):
        ax.add_patch(FancyArrowPatch((zi, yi), (z0, 0.06),
                                     connectionstyle="arc3,rad=%.2f" % rad,
                                     arrowstyle="-|>", mutation_scale=11,
                                     lw=1.3, color=P["GREEN"], zorder=4))
        ax.plot([zi], [yi], "o", ms=7.5, color=P["BLUE"],
                markeredgecolor=P["TEXT1"], markeredgewidth=0.6, zorder=6)
        ax.text(zi - 0.025 * dz, yi + 0.62, nm, fontsize=9.5, color=P["BLUE"],
                ha="center", va="bottom")
    ax.plot([z0], [0.0], "o", ms=8.0, color=P["TEXT1"], zorder=7)
    ax.text(z0 + 0.02 * dz, -0.80, "$S_n$", fontsize=10.5, color=P["TEXT1"],
            ha="left", va="center", fontweight="bold")

    ax.text(z0 + 0.58 * dz, 10.4,
            "$R_{n,j} = \\hat{S}_{n,j} - \\Delta z \\sum_k A_{jk}\\,"
            "f(\\hat{S}_{n,k})$",
            fontsize=8.6, color=P["GREEN"], ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.30", fc=INSET_BG, ec=P["NEUTRAL"], lw=0.6))
    ax.text(z0 + 0.52 * dz, 8.2,
            "each arrow lands at $R_{n,j}$ and misses the\n"
            "known $S_n$ by $r_{n,j} = R_{n,j} - S_n$",
            fontsize=7.9, color=P["TEXT2"], ha="center", va="center",
            linespacing=1.4)
    ax.text(z0 + 0.52 * dz, 6.5,
            "the same happens in $y$, $t_x$ and $t_y$",
            fontsize=7.9, color=P["TEXT2"], ha="center", va="center")

    ax.set_xlim(z0 - 0.06 * dz, z0 + 1.08 * dz)
    ax.set_ylim(-2.1, 11.8)
    ax.set_xlabel("$z$ [mm]")
    ax.set_ylabel("$x - x_n$ [mm]")
    ax.set_title("(b)  the backward picture", loc="left", pad=13.0)
    ax.text(0.0, 1.012, "the same track and the same step as (a)",
            transform=ax.transAxes, fontsize=7.3, color=P["TEXT2"],
            ha="left", va="bottom")

    # ---- the inset: the three gaps at their true size, in micrometres -------
    rx_um = G["r"][:, 0] * 1e3
    span = float(np.max(np.abs(rx_um)))
    ins = ax.inset_axes([0.505, 0.075, 0.470, 0.245])
    labs = ["$r_{n,1}$", "$r_{n,2}$", "$r_{n,\\mathrm{end}}$"]
    yy = np.arange(3)[::-1]
    ins.barh(yy, rx_um, height=0.46, color=[P["MAGENTA"], P["MAGENTA"], P["BLUE"]],
             edgecolor=P["TEXT1"], linewidth=0.5, zorder=3)
    ins.axvline(0.0, color=P["TEXT1"], lw=1.0, zorder=4)
    for y_, v in zip(yy, rx_um):
        ins.text(v + 0.05 * span * (1 if v >= 0 else -1), y_, "%+.3f" % v,
                 fontsize=6.6, color=P["TEXT1"], va="center",
                 ha="left" if v >= 0 else "right")
    ins.set_yticks(yy)
    ins.set_yticklabels(labs, fontsize=7.2)
    ins.set_xlim(-0.46 * span, 1.52 * span)
    ins.set_xlabel("residual in $x$ [$\\mu$m]", fontsize=6.8, labelpad=1.0)
    ins.set_title("the three gaps, at true size", fontsize=6.8, loc="left",
                  fontweight="normal", pad=2.0)
    ins.tick_params(labelsize=6.4, length=2.0)
    ins.set_facecolor(INSET_BG)
    ins.grid(axis="y", visible=False)

    for k, nm in enumerate(("1", "2", "end")):
        for d, (name, unit) in enumerate((("x", "mm"), ("y", "mm"),
                                          ("tx", "dimensionless"),
                                          ("ty", "dimensionless"))):
            rec("b", "residual r_%s in %s" % (nm, name), float(zout[k]),
                float(G["r"][k, d]), "z_mm", "residual_" + unit,
                "reconstruction residual of output %s, physical units" % nm)
        rec("b", "reconstruction R_%s in x" % nm, float(z0),
            float(G["S"][0] + G["r"][k, 0]), "z_mm", "x_mm",
            "R = S_n + r; the arrow's landing point")
    rec("b", "input S_n in x", float(z0), float(G["S"][0]), "z_mm", "x_mm",
        "the known input, test-split track index %d" % G["j"])


def panel_c(ax, G):
    ax.set_axis_off()
    A, b, c = G["A"], G["b"], G["c"]

    ax.text(0.010, 0.97, "the $q = 2$ Gauss-Legendre tableau", fontsize=8.8,
            color=P["TEXT1"], fontweight="bold", ha="left", va="top")

    # the (q+1) x q matrix the code actually uses: A with b appended as the
    # endpoint row, so Eqs. (recon-stage) and (recon-end) are one expression
    x_a0, x_a1 = 0.290, 0.425
    y_rows = (0.655, 0.500, 0.345)
    rows = [("$c_1 = %.5f$" % c[0], A[0, 0], A[0, 1], y_rows[0], P["TEXT1"]),
            ("$c_2 = %.5f$" % c[1], A[1, 0], A[1, 1], y_rows[1], P["TEXT1"]),
            ("$c_{\\mathrm{end}} = 1$", b[0], b[1], y_rows[2], P["GREEN"])]
    ax.text(x_a0, 0.795, "$A_{j1}$", fontsize=8.2, color=P["TEXT2"], ha="center")
    ax.text(x_a1, 0.795, "$A_{j2}$", fontsize=8.2, color=P["TEXT2"], ha="center")
    for lab, a0, a1, y, col in rows:
        ax.text(0.015, y, lab, fontsize=8.6, color=col, ha="left", va="center")
        ax.text(x_a0, y, "$%+.6f$" % a0, fontsize=8.6, color=col,
                ha="center", va="center")
        ax.text(x_a1, y, "$%+.6f$" % a1, fontsize=8.6, color=col,
                ha="center", va="center")
    y_lo, y_hi = 0.275, 0.725
    for xb, tick in ((0.215, +0.016), (0.500, -0.016)):
        ax.plot([xb, xb], [y_lo, y_hi], color=P["NEUTRAL"], lw=1.1)
        ax.plot([xb, xb + tick], [y_hi, y_hi], color=P["NEUTRAL"], lw=1.1)
        ax.plot([xb, xb + tick], [y_lo, y_lo], color=P["NEUTRAL"], lw=1.1)
    ax.text(0.015, 0.175,
            "the third row is $b$: the endpoint is folded in by "
            "$A_{\\mathrm{end},k}\\equiv b_k$.",
            fontsize=7.7, color=P["GREEN"], ha="left", va="center")
    ax.text(0.015, 0.055,
            "$\\sum_k A_{1k} = c_1$,  $\\sum_k A_{2k} = c_2$,  "
            "$\\sum_k b_k = 1$: the row sums are the nodes.",
            fontsize=7.7, color=P["TEXT2"], ha="left", va="center")

    ax.plot([0.550, 0.550], [0.02, 0.99], color=FAINT, lw=1.3)

    ax.text(0.585, 0.97, "the loss, both times", fontsize=8.8, color=P["TEXT1"],
            fontweight="bold", ha="left", va="top")
    ax.text(0.585, 0.790,
            "$\\mathcal{L}(\\theta) = \\mathrm{mean}_{n,j,d}\\ "
            "\\left(\\, r_{n,j,d} \\,/\\, s \\,\\right)^{2}$",
            fontsize=11.5, color=P["TEXT1"], ha="left", va="center")
    ax.text(0.585, 0.545,
            "pooled: $s = s_d$, the round-1 spread of\n"
            "component $d$: four constants, the same for\n"
            "every track and every output plane.",
            fontsize=7.7, color=P["BLUE"], ha="left", va="center",
            linespacing=1.55)
    ax.text(0.585, 0.215,
            "cost-weighted: $s = D_{\\mathrm{ref}}/(a_n g_{n,j,d})$, with\n"
            "$g = (1, 1, \\ell_{n,j}, \\ell_{n,j})$: one value per track,\n"
            "per output plane and per component.",
            fontsize=7.7, color=P["MAGENTA"], ha="left", va="center",
            linespacing=1.55)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("(c)  the algebra behind (a) and (b)", loc="left")

    for j in range(2):
        for k in range(2):
            rec("c", "tableau A[%d,%d]" % (j + 1, k + 1), float(j + 1),
                float(A[j, k]), "row_j", "A_jk",
                "gauss_legendre(2) of _shared/irk.py")
    for k in range(2):
        rec("c", "tableau b[%d]" % (k + 1), 3.0, float(b[k]), "row_j", "b_k",
            "the endpoint row, A_end,k = b_k")
        rec("c", "node c[%d]" % (k + 1), float(k + 1), float(c[k]),
            "node_index", "c_j", "")
    rec("c", "node c_end", 3.0, 1.0, "node_index", "c_j", "the endpoint node")
    for d, nm in enumerate(("x", "y", "tx", "ty")):
        rec("c", "pooled divisor s_%s" % nm, float(d), float(G["in_scale"][d]),
            "component_index", "s_d",
            "the round-1 spread, in_scale of scale.json (mm for x and y)")


def main():
    G = gather()
    fig = plt.figure(figsize=(7.6, 7.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.50], hspace=0.34,
                          wspace=0.24, left=0.082, right=0.985,
                          top=0.925, bottom=0.060)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, :])
    panel_a(axA, G)
    panel_b(axB, G)
    panel_c(axC, G)

    os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(PNG)
    plt.close(fig)

    C.write_csv(CSV, [C.jsonable(r) for r in ROWS],
                ["panel", "series", "x", "y", "x_quantity", "y_quantity", "note"])
    print("wrote %s" % PNG)
    print("wrote %s  (%d rows)" % (CSV, len(ROWS)))
    print("track: test index %d, p = %.4f GeV, S0 = %s"
          % (G["j"], G["p"], np.array2string(G["S"], precision=6)))
    print("residuals in x [um]: %s" % np.array2string(G["r"][:, 0] * 1e3, precision=4))
    print("correction to the straight line in x [mm]: %s"
          % np.array2string(G["out"][:, 0] - G["straight"][:, 0], precision=6))


if __name__ == "__main__":
    main()
