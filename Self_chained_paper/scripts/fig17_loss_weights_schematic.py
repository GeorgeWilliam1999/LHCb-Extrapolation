#!/usr/bin/env python
"""Figure - what each loss does with the residuals (figures/fig_loss_weights_schematic.png).

Figure 16 shows where a residual comes from; this one shows what each of the
two losses then multiplies it by.  Everything is drawn for the N = 64, q = 2,
dz = 80.903 mm geometry, with the real constants of that run.

WHAT THE FIGURE SHOWS, panel by panel.

(a) THE LEVER ARM, GEOMETRICALLY.  The crossing z0 -> z1 drawn as a horizontal
    band.  A slope error dt_x made at an early output plane fans out into a
    displacement (z1 - z) dt_x by the time the track reaches z1; the same
    dt_x made one step before z1 fans into almost nothing.  The wedges are
    drawn with the real measured per-step slope error of the pooled-loss
    N = 64, q = 2 network, 1.3015e-06: from the first Gauss node the fan
    reaches 6.72 um at z1, and the inset draws the same dt_x one step before
    z1, at true scale, reaching 0.105 um.  The lever arm the loss uses is
    (z1 - z) + dz, which is the geometric distance plus one step: the additive
    dz is what keeps the last plane from being weighted to zero, and it makes
    the lever arm 5,241.6 mm at the first Gauss node of the first step against
    80.9 mm at the last endpoint output, a ratio of 65.

(b) THE POOLED LOSS AS A WEIGHT MAP.  The weight a slope residual carries,
    over the plane z along the crossing (vertical) and the track's momentum p
    (horizontal, logarithmic).  Under Eq. (eq:pooled) it is the single
    constant 1 / s_{t_x} = 6.0425 everywhere: one flat colour.

(c) THE COST-WEIGHTED LOSS ON THE SAME AXES.  The weight is
    a_n l(z) / D_ref, with l(z) = (z1 - z) + dz, a_n = sqrt(W(p)) D_ref / D_n
    clamped to [1/5, 5] of the median a_n over the 11,567 training tracks, and
    D_ref = 861.56 mm.  The 10-50 GeV loss window is marked, the two momenta
    where the clamp binds are hatched, and the four worked-example points are
    annotated with their unclamped weights (0.596, 0.0093, 17.8, 0.278).

(d) THE SAME FOR A POSITION RESIDUAL.  The weight is a_n / D_ref: the track
    factor without the lever arm, so the map has no vertical structure at all.

Panels (b), (c) and (d) share one logarithmic colour scale, so the flat panel
and the 1,625-fold range of the drawn (clamped) cost-weighted map can be read
against each other directly.

INPUTS READ (all read-only).

  results/paper_numbers.json     groups `constants` (KAPPA, P_LO, P_HI,
                                 ROLLOFF, W_FLOOR, CLAMP and the per-run dz,
                                 D_ref, i_bar and lever range), `worked_example`
                                 (the four annotated points) and `anatomy`
                                 (the measured per-step slope error of panel a).
  <S>/Block_F_reweighted_loss/F0_Weighting/weighted_loss.py
                                 imported read-only through
                                 common.load_weighted_loss, for band_window,
                                 track_bend, per_track_factor and lever_arms,
                                 so every curve is the module's own function.
  <S>/Block_E_single_network_chain/E0_Track_dataset/results/tracks.npz
                                 the 11,567 training tracks' q/p, for the
                                 median a_n that sets the clamp thresholds
                                 (the one quantity the JSON does not carry),
                                 and the test split's momentum spectrum.
  <S>/.../E1_Network_grid/results/N064_q02/scale.json
                                 in_scale, whose third entry is s_{t_x}.
  scripts/common.py              paths, palette, geometry, loaders.

OUTPUTS.

  figures/fig_loss_weights_schematic.png
  results/fig17_loss_weights_schematic.csv      every plotted value.

Run:
  PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python fig17_loss_weights_schematic.py
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"

import sys                                   # noqa: E402
sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in ("", ".", HERE):
    while _p in sys.path:
        sys.path.remove(_p)

import importlib.util                        # noqa: E402

import numpy as np                           # noqa: E402
import matplotlib                            # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm   # noqa: E402

_spec = importlib.util.spec_from_file_location("sc_paper_common",
                                               os.path.join(HERE, "common.py"))
C = importlib.util.module_from_spec(_spec)
sys.modules["sc_paper_common"] = C
_spec.loader.exec_module(C)

FIGURES = os.path.join(C.PAPER, "figures")
PNG = os.path.join(FIGURES, "fig_loss_weights_schematic.png")
CSV = os.path.join(C.RESULTS, "fig17_loss_weights_schematic.csv")

# ------------------------------------------------------------ house style --
P = C.PALETTE
FAINT = "#e5e4df"
BAND_FILL = "#dfe7f2"
INSET_BG = "#f4f4f1"

# a sequential map in the house blue, light surface to deep blue
CMAP = LinearSegmentedColormap.from_list(
    "house_blue", ["#f7f7f4", "#d7e4f4", "#9dc0e8", "#4e8ddc", "#2a78d6",
                   "#1a4f92", "#10305a"])

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

VMIN, VMAX = 1.0e-4, 30.0        # the shared colour scale of (b), (c) and (d)

ROWS = []


def rec(panel, series, x, y, xlabel, ylabel, note=""):
    ROWS.append(dict(panel=panel, series=series, x=x, y=y,
                     x_quantity=xlabel, y_quantity=ylabel, note=note))


# ------------------------------------------------------- the real numbers --
def gather():
    J = C.read_json(os.path.join(C.RESULTS, "paper_numbers.json"))
    const = J["constants"]
    run = const["per_run"][C.run_key(64, 2)]
    WL = C.load_weighted_loss()

    dz = run["dz_mm"]
    D_ref = run["D_ref_mm"]
    ibar = run["i_bar_T_mm"]
    clamp = const["CLAMP"]

    # the clamp thresholds: the median a_n over the round's batch, which is
    # drawn from the 11,567 training tracks (fig07_weights.py does the same)
    qop_train = np.abs(C.split_arrays("train")["S0"][:, 4])
    a_train = WL.per_track_factor(
        qop_train, dict(D_ref=D_ref, i_bar=ibar, L=C.L_MM, clamp=0.0),
        WL.MODES["full"])
    a_med = float(np.median(a_train))

    in_scale = C.load_scale(C.pooled_run(64, 2))["in_scale"]
    dtx = J["anatomy"]["per_run"]["pooled N=64,q=2"]["per_step_tx_med_abs_slope"]

    # the first Gauss node of the first step: the largest lever arm the loss
    # actually weighs (lever_max_mm in the JSON is the value at c = 0, where no
    # output sits; README.md, discrepancy (a))
    c1 = 0.5 - np.sqrt(3.0) / 6.0
    lev_first_node = C.L_MM + dz * (1.0 - c1)

    return dict(WL=WL, J=J, const=const, run=run, dz=dz, D_ref=D_ref,
                ibar=ibar, clamp=clamp, a_med=a_med, in_scale=in_scale,
                dtx=float(dtx), c1=float(c1), lev_first_node=float(lev_first_node),
                worked=J["worked_example"])


# ----------------------------------------------------------------- panels --
def panel_a(ax, G):
    dz, dtx = G["dz"], G["dtx"]
    z0, z1 = C.Z0_MM, C.Z1_MM
    z_early = z0 + G["c1"] * dz                 # the first Gauss node
    z_late = z1 - dz                            # one step before the far side

    ax.axhline(0.0, color=P["TEXT2"], lw=1.0, zorder=2)
    ax.axvline(z1, color=P["TEXT1"], lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.text(z1 - 40.0, -10.4, "$z_1$", fontsize=9.0, color=P["TEXT1"],
            ha="right", va="center")

    # the early wedge: a slope error dtx opens into (z1 - z) dtx by z1
    zz = np.linspace(z_early, z1, 200)
    fan = (zz - z_early) * dtx * 1e3            # um
    ax.fill_between(zz, -fan, fan, color=P["MAGENTA"], alpha=0.28, zorder=3,
                    linewidth=0)
    ax.plot(zz, fan, color=P["MAGENTA"], lw=1.5, zorder=4)
    ax.plot(zz, -fan, color=P["MAGENTA"], lw=1.5, zorder=4)
    ax.plot([z_early], [0.0], "o", ms=6.0, color=P["MAGENTA"], zorder=6)

    d_early = (z1 - z_early) * dtx * 1e3
    d_late = dz * dtx * 1e3
    ax.annotate("", xy=(z1, d_early), xytext=(z1, -d_early),
                arrowprops=dict(arrowstyle="<->", lw=1.1, color=P["MAGENTA"]))
    ax.text(z1 - 120.0, d_early + 1.15, "$\\pm %.2f\\ \\mu$m at $z_1$" % d_early,
            fontsize=8.2, color=P["MAGENTA"], ha="right", va="center")
    ax.annotate("a slope error $\\Delta t_x$ at the first Gauss node",
                xy=(z_early, 0.0), xytext=(z_early + 240.0, -3.0),
                fontsize=7.9, color=P["MAGENTA"], ha="left", va="center",
                zorder=7,
                bbox=dict(boxstyle="square,pad=0.22", fc=P["SURFACE"], ec="none",
                          alpha=0.88),
                arrowprops=dict(arrowstyle="->", lw=0.7, color=P["MAGENTA"]))

    ax.text(z0 + 120.0, 10.2,
            "the lever arm is $\\ell(z) = (z_1 - z) + \\Delta z$.  At $N$ = 64 it runs from\n"
            "$%s\\,$mm at the first Gauss node to $\\Delta z = %.1f\\,$mm at the last endpoint\n"
            "output: a ratio of $65$.  The additive $\\Delta z$ is why the last output\n"
            "plane carries a small weight rather than none."
            % ("{:,.1f}".format(G["lev_first_node"]), dz),
            fontsize=7.7, color=P["TEXT1"], ha="left", va="top", linespacing=1.6,
            zorder=7,
            bbox=dict(boxstyle="square,pad=0.30", fc=P["SURFACE"], ec="none",
                      alpha=0.88))

    ax.set_xlim(z0 - 120.0, z1 + 340.0)
    ax.set_ylim(-11.5, 11.4)
    ax.set_xlabel("$z$ [mm]")
    ax.set_ylabel("transverse displacement [$\\mu$m]")
    ax.set_title("(a)  the lever arm", loc="left", pad=13.0)
    ax.text(0.0, 1.012,
            "$\\Delta t_x = %.4g$, the measured per-step slope error at $N$ = 64, $q$ = 2"
            % dtx,
            transform=ax.transAxes, fontsize=7.3, color=P["TEXT2"],
            ha="left", va="bottom")
    ax.grid(False)

    # ---- the inset: the same slope error one step before the far side -------
    ins = ax.inset_axes([0.055, 0.045, 0.355, 0.215])
    zl = np.linspace(z_late, z1, 120)
    fl = (zl - z_late) * dtx * 1e3
    ins.fill_between(zl, -fl, fl, color=P["GREEN"], alpha=0.28, linewidth=0)
    ins.plot(zl, fl, color=P["GREEN"], lw=1.4)
    ins.plot(zl, -fl, color=P["GREEN"], lw=1.4)
    ins.plot([z_late], [0.0], "o", ms=4.5, color=P["GREEN"])
    ins.axhline(0.0, color=P["TEXT2"], lw=0.8)
    ins.axvline(z1, color=P["TEXT1"], lw=0.9, ls=(0, (3, 2)))
    ins.annotate("", xy=(z1, d_late), xytext=(z1, -d_late),
                 arrowprops=dict(arrowstyle="<->", lw=0.9, color=P["GREEN"]))
    ins.text(z1 - 0.12 * dz, d_late * 1.28, "$\\pm %.3f\\ \\mu$m" % d_late,
             fontsize=6.8, color=P["GREEN"], ha="right", va="center")
    ins.set_xlim(z_late - 0.30 * dz, z1 + 0.28 * dz)
    ins.set_ylim(-d_late * 2.2, d_late * 2.2)
    ins.set_title("the same $\\Delta t_x$ one step before $z_1$, at true scale "
                  "($z$ in mm)", fontsize=6.8, loc="left", fontweight="normal",
                  pad=2.0)
    ins.tick_params(labelsize=6.2, length=2.0)
    ins.set_facecolor(INSET_BG)
    ins.grid(False)

    rec("a", "slope error used", float("nan"), dtx, "-", "dimensionless",
        "pooled N=64,q=2 per-step median |dtx|")
    rec("a", "lever at the first Gauss node", float(z_early),
        G["lev_first_node"], "z_mm", "lever_mm", "L + dz (1 - c1)")
    rec("a", "lever at the last endpoint", float(z1), float(dz), "z_mm",
        "lever_mm", "the additive dz")
    rec("a", "geometric displacement at z1 from the first Gauss node",
        float(z_early), float(d_early), "z_mm", "displacement_um", "(z1 - z) dtx")
    rec("a", "geometric displacement at z1 from one step before z1",
        float(z_late), float(d_late), "z_mm", "displacement_um", "dz dtx")
    rec("a", "ratio of the two lever arms", float("nan"),
        float(G["lev_first_node"] / dz), "-", "ratio", "")


def _maps(G, nz=220, npix=260):
    WL = G["WL"]
    p = np.logspace(0.0, np.log10(200.0), npix)
    z = np.linspace(C.Z0_MM, C.Z1_MM, nz)
    lev = (C.Z1_MM - z) + G["dz"]
    qop = WL.QOP_TO_GEV / p
    a_un = np.sqrt(WL.band_window(p)) * G["D_ref"] / WL.track_bend(qop, G["ibar"], C.L_MM)
    a_cl = np.clip(a_un, G["a_med"] / G["clamp"], G["a_med"] * G["clamp"])
    slope_map = a_cl[None, :] * lev[:, None] / G["D_ref"]
    pos_map = np.broadcast_to(a_cl[None, :] / G["D_ref"], slope_map.shape).copy()
    pooled = np.full_like(slope_map, 1.0 / G["in_scale"][2])
    return p, z, a_un, a_cl, pooled, slope_map, pos_map


def _decorate_map(ax, p, z, title, subtitle):
    ax.set_xscale("log")
    ax.set_xlim(1.0, 200.0)
    ax.set_ylim(C.Z0_MM, C.Z1_MM)
    ax.set_xlabel("momentum $p$ [GeV]")
    ax.set_ylabel("plane $z$ [mm]")
    ax.set_title(title, loc="left", pad=13.0)
    ax.text(0.0, 1.012, subtitle, transform=ax.transAxes, fontsize=7.3,
            color=P["TEXT2"], ha="left", va="bottom")
    ax.grid(False)
    for pe in C.LOSS_WINDOW:
        ax.axvline(pe, color=P["TEXT1"], lw=0.9, ls=(0, (4, 3)), zorder=5)


def main():
    G = gather()
    p, z, a_un, a_cl, pooled, slope_map, pos_map = _maps(G)
    ext = [1.0, 200.0, C.Z0_MM, C.Z1_MM]
    norm = LogNorm(vmin=VMIN, vmax=VMAX)

    fig = plt.figure(figsize=(7.6, 8.6))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.98, 1.0, 1.0], hspace=0.50,
                          wspace=0.30, left=0.088, right=0.855,
                          top=0.945, bottom=0.052)
    axA = fig.add_subplot(gs[0, :])
    axB = fig.add_subplot(gs[1, 0])
    axCc = fig.add_subplot(gs[1, 1])
    axD = fig.add_subplot(gs[2, 0])

    panel_a(axA, G)

    for ax, M, title, sub in (
            (axB, pooled, "(b)  pooled loss, slope residual",
             "constant: $1/s_{t_x}$ = %.4f everywhere" % (1.0 / G["in_scale"][2])),
            (axCc, slope_map, "(c)  cost-weighted loss, slope residual",
             "$a_n\\,\\ell(z)/D_{\\mathrm{ref}}$, clamped"),
            (axD, pos_map, "(d)  cost-weighted loss, position residual",
             "$a_n/D_{\\mathrm{ref}}$: no lever arm")):
        im = ax.imshow(M, origin="lower", aspect="auto", extent=ext,
                       cmap=CMAP, norm=norm, zorder=1)
        _decorate_map(ax, p, z, title, sub)

    axB.text(14.0, C.Z0_MM + 0.5 * C.L_MM,
             "flat: every track, every plane,\nevery slope residual alike",
             fontsize=7.8, color=P["TEXT2"], ha="center", va="center",
             linespacing=1.45,
             bbox=dict(boxstyle="round,pad=0.30", fc=INSET_BG, ec=P["NEUTRAL"],
                       lw=0.6))

    # where the clamp binds, on (c) and (d)
    p_lo_bind = float(p[a_un < G["a_med"] / G["clamp"]].max()) \
        if (a_un < G["a_med"] / G["clamp"]).any() else None
    p_hi_bind = float(p[a_un > G["a_med"] * G["clamp"]].min()) \
        if (a_un > G["a_med"] * G["clamp"]).any() else None
    for ax in (axCc, axD):
        if p_lo_bind:
            ax.axvspan(1.0, p_lo_bind, facecolor="none", edgecolor=P["TEXT2"],
                       hatch="///", lw=0.0, alpha=0.45, zorder=3)
        if p_hi_bind:
            ax.axvspan(p_hi_bind, 200.0, facecolor="none", edgecolor=P["TEXT2"],
                       hatch="///", lw=0.0, alpha=0.45, zorder=3)
    axD.text(1.35, C.Z0_MM + 0.14 * C.L_MM, "clamped", fontsize=7.0,
             color=P["TEXT1"], ha="left", va="center", rotation=90)
    axD.text(150.0, C.Z0_MM + 0.14 * C.L_MM, "clamped", fontsize=7.0,
             color=P["TEXT1"], ha="left", va="center", rotation=90)

    # the loss window and the worked-example points, on (c)
    for ax in (axCc, axD):
        ax.text(22.0, C.Z0_MM + 0.50 * C.L_MM, "10-50 GeV\nloss window",
                fontsize=7.0, color=P["TEXT1"], ha="center", va="center",
                zorder=6, linespacing=1.3,
                bbox=dict(boxstyle="square,pad=0.18", fc=P["SURFACE"], ec="none",
                          alpha=0.88))
    wk = G["worked"]["cases"]
    z_first = C.Z0_MM + G["c1"] * G["dz"]
    pts = [("20 GeV, first output plane", 20.0, z_first, "left", 6.0, 150.0),
           ("3 GeV, first output plane", 3.0, z_first, "left", 6.0, 150.0),
           ("20 GeV, last output plane", 20.0, C.Z1_MM, "left", 6.0, -260.0),
           ("3 GeV, last output plane", 3.0, C.Z1_MM, "left", 6.0, -260.0)]
    for key, pp, zz, ha, dx, dy in pts:
        v = wk[key]["weight_on_a_slope_residual_per_unit_slope"]
        axCc.plot([pp], [zz], "o", ms=5.0, color=P["MAGENTA"],
                  markeredgecolor=P["TEXT1"], markeredgewidth=0.6, zorder=7)
        axCc.text(pp * 1.14, zz + dy, "%.4g" % v, fontsize=7.2,
                  color=P["TEXT1"], ha=ha, va="center", zorder=7,
                  bbox=dict(boxstyle="square,pad=0.14", fc=P["SURFACE"],
                            ec="none", alpha=0.88))
        rec("c", "worked example: " + key, float(pp), float(v), "p_GeV",
            "a_n * lever / D_ref  [per unit slope]",
            "unclamped, from paper_numbers.json worked_example; the drawn map "
            "is clamped and uses lever = (z1 - z) + dz")
        rec("d", "worked example: " + key, float(pp),
            float(wk[key]["weight_on_a_position_residual_per_mm"]), "p_GeV",
            "a_n / D_ref  [per mm]", "unclamped")

    # ---------------------------------------------------- the shared colour bar --
    pB, pD = axCc.get_position(), axD.get_position()
    cax = fig.add_axes([0.878, pD.y0, 0.020, pB.y1 - pD.y0])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("weight multiplying the residual", fontsize=7.6,
                 color=P["TEXT2"], labelpad=6.0)
    cb.ax.tick_params(labelsize=7.2)
    cb.outline.set_edgecolor(P["NEUTRAL"])

    # the fourth cell of the lower block: the legend of the three maps
    axT = fig.add_subplot(gs[2, 1])
    axT.set_axis_off()
    axT.text(0.0, 0.97, "reading (b), (c) and (d)", fontsize=8.8,
             color=P["TEXT1"], fontweight="bold", ha="left", va="top")
    axT.text(0.0, 0.825,
             "One logarithmic colour scale across the three\n"
             "maps; per unit slope in (b) and (c), per mm in (d).",
             fontsize=7.9, color=P["TEXT1"], ha="left", va="top",
             linespacing=1.5)
    axT.text(0.0, 0.615,
             "(b) is a single flat colour: the pooled loss\n"
             "weighs every slope residual by $1/s_{t_x} = %.4f$,\n"
             "whatever the track and wherever the plane."
             % (1.0 / G["in_scale"][2]),
             fontsize=7.7, color=P["BLUE"], ha="left", va="top", linespacing=1.55)
    axT.text(0.0, 0.375,
             "(c) runs from %.4f to %.3g, a factor %s, and\n"
             "carries both structures: the lever arm vertically\n"
             "and the track factor with its window horizontally."
             % (slope_map.min(), slope_map.max(),
                "{:,.0f}".format(slope_map.max() / slope_map.min())),
             fontsize=7.7, color=P["MAGENTA"], ha="left", va="top",
             linespacing=1.55)
    axT.text(0.0, 0.135,
             "(d) has the horizontal structure only: a position\n"
             "residual gets the track factor but no lever arm,\n"
             "so it runs from %.3g to %.3g per mm."
             % (pos_map.min(), pos_map.max()),
             fontsize=7.7, color=P["GREEN"], ha="left", va="top", linespacing=1.55)
    axT.set_xlim(0, 1)
    axT.set_ylim(0, 1)

    os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(PNG)
    plt.close(fig)

    # ------------------------------------------------------------- the CSV --
    rec("b", "pooled weight on a slope residual", float("nan"),
        float(1.0 / G["in_scale"][2]), "-", "per unit slope",
        "1 / s_tx, constant over the whole map")
    rec("b", "pooled weight on a position residual", float("nan"),
        float(1.0 / G["in_scale"][0]), "-", "per mm", "1 / s_x, for comparison")
    for k, zi in enumerate(z):
        rec("a", "lever arm", float(zi), float((C.Z1_MM - zi) + G["dz"]),
            "z_mm", "lever_mm", "")
    for k, pi in enumerate(p):
        rec("c", "a_n unclamped", float(pi), float(a_un[k]), "p_GeV", "a_n", "")
        rec("c", "a_n clamped", float(pi), float(a_cl[k]), "p_GeV", "a_n",
            "clamped to [1/5, 5] of the median a_n")
        rec("d", "position weight a_n / D_ref", float(pi),
            float(a_cl[k] / G["D_ref"]), "p_GeV", "per mm", "clamped")
    for nm, v, note in (
            ("median a_n over the 11,567 training tracks", G["a_med"], ""),
            ("clamp lower limit", G["a_med"] / G["clamp"], "median / 5"),
            ("clamp upper limit", G["a_med"] * G["clamp"], "median x 5"),
            ("D_ref", G["D_ref"], "mm"),
            ("dz", G["dz"], "mm"),
            ("s_tx", G["in_scale"][2], "the pooled divisor on a slope residual"),
            ("colour scale vmin", VMIN, ""),
            ("colour scale vmax", VMAX, ""),
            ("drawn slope-weight minimum", float(slope_map.min()), "panel c"),
            ("drawn slope-weight maximum", float(slope_map.max()), "panel c"),
            ("drawn slope-weight ratio", float(slope_map.max() / slope_map.min()),
             "panel c: 25 (the clamp) x 65 (the lever arm)"),
            ("worked-example extreme ratio",
             G["worked"]["ratios"]["slope_20GeV_first_over_3GeV_last"],
             "unclamped, paper_numbers.json")):
        rec("constants", nm, float("nan"), float(v), "-", "value", note)

    C.write_csv(CSV, [C.jsonable(r) for r in ROWS],
                ["panel", "series", "x", "y", "x_quantity", "y_quantity", "note"])
    print("wrote %s" % PNG)
    print("wrote %s  (%d rows)" % (CSV, len(ROWS)))
    print("median a_n = %.6f -> clamp [%.6f, %.6f]"
          % (G["a_med"], G["a_med"] / G["clamp"], G["a_med"] * G["clamp"]))
    print("panel (c) drawn range %.6g .. %.6g, ratio %.1f"
          % (slope_map.min(), slope_map.max(), slope_map.max() / slope_map.min()))
    print("panel (d) drawn range %.6g .. %.6g" % (pos_map.min(), pos_map.max()))
    print("panel (b) constant %.6f" % (1.0 / G["in_scale"][2]))
    print("lever at the first Gauss node %.4f mm, at the last endpoint %.4f mm, ratio %.3f"
          % (G["lev_first_node"], G["dz"], G["lev_first_node"] / G["dz"]))


if __name__ == "__main__":
    main()
