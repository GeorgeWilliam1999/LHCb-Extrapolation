#!/usr/bin/env python
"""Figure 7 - the analytic shape of the cost weight (figures/fig_weights.png).

WHAT THE FIGURE SHOWS, panel by panel.

(a) The momentum window W(p) that the cost weight carries, drawn on a
    logarithmic momentum axis from 1 to 200 GeV.  W is one inside the 10-50 GeV
    loss window, falls as a Gaussian in log p outside it, and is floored at
    0.05 so that no track is ever ignored.  The 1/e points, at 5 and at
    100 GeV, and the floor are marked; the loss window is the shaded span.

(b) The per-track factor a_n(p) = sqrt(W(p)) * D_ref / D_n of the
    N = 64, q = 2, dz = 80.9 mm network, on log-log axes.  D_n is the track's
    own total transverse bend across the crossing and D_ref is that run's fixed
    reference bend, so a_n rises with momentum through D_ref/D_n and is pulled
    back down outside the window by sqrt(W).  The unclamped shape is the plain
    line; the clamped shape, limited to [1/5, 5] of the median a_n of the
    batch, is the second line, with the allowed band shaded.

(c) The lever arm l(z) = (z1 - z) + dz of the N = 64 chain, from z0 to z1 on
    linear axes: the distance an output plane's slope error still has to travel
    before it reaches the endpoint.  The additive dz - not a floor - is what
    keeps the last plane from being weighted to zero, and is annotated there.

(d) The combined weight a_n * l / D_ref on a slope residual, for the 3 GeV and
    the 20 GeV track of the worked example, at the first and at the last output
    plane of the chain: four bars on a logarithmic axis, spanning a factor of
    1,908 between the extremes.

INPUTS READ (all read-only).

  results/paper_numbers.json     groups `constants` (KAPPA, QOP_TO_GEV, P_LO,
                                 P_HI, ROLLOFF, W_FLOOR, CLAMP, and the
                                 per-run D_ref, lever range and i_bar) and
                                 `worked_example` (the four bars of panel d and
                                 the 1,908 ratio).
  <S>/Block_F_reweighted_loss/F0_Weighting/weighted_loss.py
                                 imported read-only through common.load_weighted_loss
                                 for band_window, track_bend and lever_arms, so
                                 the curves are the module's own functions.
  <S>/Block_E_single_network_chain/E0_Track_dataset/results/tracks.npz
                                 the 11,567 training tracks' q/p, used for the
                                 one quantity the JSON does not carry: the
                                 median a_n that sets the clamp thresholds.
  scripts/common.py              paths, palette, geometry, loaders.

OUTPUTS.

  figures/fig_weights.png
  results/fig07_weights.csv      every plotted value.

Run:
  PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python fig07_weights.py
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

_spec = importlib.util.spec_from_file_location("sc_paper_common",
                                               os.path.join(HERE, "common.py"))
C = importlib.util.module_from_spec(_spec)
sys.modules["sc_paper_common"] = C
_spec.loader.exec_module(C)

FIGURES = os.path.join(C.PAPER, "figures")
PNG = os.path.join(FIGURES, "fig_weights.png")
CSV = os.path.join(C.RESULTS, "fig07_weights.csv")

# ------------------------------------------------------------ house style --
P = C.PALETTE
FAINT = "#e5e4df"
BAND_FILL = "#dfe7f2"

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


def main():
    J = C.read_json(os.path.join(C.RESULTS, "paper_numbers.json"))
    const = J["constants"]
    run = const["per_run"][C.run_key(64, 2)]
    worked = J["worked_example"]

    WL = C.load_weighted_loss()              # the loss module itself, read-only

    N, q = 64, 2
    dz = run["dz_mm"]
    D_ref = run["D_ref_mm"]
    i_bar = run["i_bar_T_mm"]
    clamp = const["CLAMP"]
    NETNAME = "$N$ = %d, $q$ = %d, $\\Delta z$ = %.1f mm" % (N, q, dz)

    # ---- the one quantity the JSON does not carry: the median a_n ----------
    # a_n is clamped to [1/CLAMP, CLAMP] of the median over the round's own
    # batch of states.  The batch is drawn evenly over the planes from the
    # 11,567 training tracks, so the median over those tracks' q/p is the
    # median the clamp uses.
    qop_train = np.abs(C.split_arrays("train")["S0"][:, 4])
    a_train = WL.per_track_factor(
        qop_train, dict(D_ref=D_ref, i_bar=i_bar, L=C.L_MM, clamp=0.0),
        WL.MODES["full"])
    a_med = float(np.median(a_train))
    a_lo, a_hi = a_med / clamp, a_med * clamp

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.9))
    (axA, axB), (axCc, axD) = axes

    p = np.logspace(np.log10(1.0), np.log10(200.0), 800)

    # ------------------------------------------------------- (a) the window --
    W = WL.band_window(p)
    axA.axvspan(C.LOSS_WINDOW[0], C.LOSS_WINDOW[1], color=BAND_FILL, zorder=0)
    axA.text(22.0, 0.83, "10-50 GeV\nloss window", ha="center", fontsize=7.5,
             color=P["TEXT2"])
    axA.plot(p, W, color=P["BLUE"], lw=1.8, zorder=3)
    axA.axhline(np.exp(-1.0), color=P["TEXT2"], lw=0.8, ls=(0, (4, 3)), zorder=2)
    axA.axhline(const["W_FLOOR"], color=P["MAGENTA"], lw=0.9, ls=(0, (2, 2)), zorder=2)
    for pe in (5.0, 100.0):
        axA.plot([pe], [np.exp(-1.0)], "o", ms=4.5, color=P["GREEN"], zorder=4)
    axA.annotate("1/$e$ at 5 and 100 GeV", xy=(5.0, np.exp(-1.0)),
                 xytext=(1.12, 0.62), fontsize=7.5, color=P["TEXT2"],
                 arrowprops=dict(arrowstyle="-", lw=0.7, color=P["TEXT2"]))
    axA.text(1.12, const["W_FLOOR"] + 0.045, "floor 0.05", fontsize=7.5,
             color=P["MAGENTA"], ha="left")
    axA.set_xscale("log")
    axA.set_xlim(1.0, 200.0)
    axA.set_ylim(0.0, 1.08)
    axA.set_xlabel("momentum $p$ [GeV]")
    axA.set_ylabel("window $W(p)$")
    axA.set_title("(a)  the momentum window", loc="left")
    for xi, yi in zip(p, W):
        rec("a", "W(p)", float(xi), float(yi), "p_GeV", "W")
    for pe in (5.0, 100.0):
        rec("a", "1/e marker", pe, float(np.exp(-1.0)), "p_GeV", "W")
    rec("a", "floor", float("nan"), const["W_FLOOR"], "p_GeV", "W")

    # ------------------------------------------- (b) the per-track factor a_n --
    qop = WL.QOP_TO_GEV / p
    a_un = np.sqrt(WL.band_window(p)) * D_ref / WL.track_bend(qop, i_bar, C.L_MM)
    a_cl = np.clip(a_un, a_lo, a_hi)
    axB.axvspan(C.LOSS_WINDOW[0], C.LOSS_WINDOW[1], color=BAND_FILL, zorder=0)
    axB.axhspan(a_lo, a_hi, color=FAINT, zorder=0)
    axB.plot(p, a_un, color=P["BLUE"], lw=1.6, zorder=3, label="unclamped")
    axB.plot(p, a_cl, color=P["MAGENTA"], lw=1.9, ls=(0, (5, 2)), zorder=4,
             label="clamped to [1/5, 5] of the median")
    axB.axhline(a_med, color=P["TEXT2"], lw=0.7, ls=(0, (1, 2)), zorder=2)
    axB.text(1.15, a_med * 1.14, "median $a_n$ = %.3f" % a_med, fontsize=7.5,
             color=P["TEXT2"])
    axB.set_xscale("log")
    axB.set_yscale("log")
    axB.set_xlim(1.0, 200.0)
    axB.set_xlabel("momentum $p$ [GeV]")
    axB.set_ylabel("per-track factor $a_n$")
    axB.set_title("(b)  the per-track factor", loc="left", pad=15.0)
    axB.text(0.015, 1.01, NETNAME, transform=axB.transAxes, fontsize=7.5,
             color=P["TEXT2"], ha="left", va="bottom")
    axB.legend(loc="lower right", fontsize=7.5)
    for xi, yu, yc in zip(p, a_un, a_cl):
        rec("b", "a_n unclamped", float(xi), float(yu), "p_GeV", "a_n")
        rec("b", "a_n clamped", float(xi), float(yc), "p_GeV", "a_n")
    rec("b", "clamp lower limit", float("nan"), a_lo, "p_GeV", "a_n",
        "median a_n / %g, median over the 11,567 training tracks" % clamp)
    rec("b", "clamp upper limit", float("nan"), a_hi, "p_GeV", "a_n",
        "median a_n x %g" % clamp)
    rec("b", "median a_n", float("nan"), a_med, "p_GeV", "a_n",
        "computed here from tracks.npz train S0[:, 4]")

    # ------------------------------------------------------ (c) the lever arm --
    z = np.linspace(C.Z0_MM, C.Z1_MM, 600)
    lev = (C.Z1_MM - z) + dz
    axCc.plot(z, C.Z1_MM - z, color=P["NEUTRAL"], lw=1.2, ls=(0, (4, 3)), zorder=2,
              label="distance left, $z_1 - z$")
    axCc.plot(z, lev, color=P["GREEN"], lw=1.8, zorder=3,
              label="lever arm $\\ell(z) = (z_1 - z) + \\Delta z$")
    zk = C.Z0_MM + dz * np.arange(1, N + 1)
    axCc.plot(zk, (C.Z1_MM - zk) + dz, ".", ms=2.6, color=P["TEXT2"], zorder=4,
              label="the %d step-end planes" % N)
    axCc.set_xlim(C.Z0_MM - 60.0, C.Z1_MM + 120.0)
    axCc.set_ylim(-150.0, 5600.0)
    axCc.set_xlabel("$z$ [mm]")
    axCc.set_ylabel("lever arm $\\ell$ [mm]")
    axCc.set_title("(c)  the lever arm, $N$ = %d" % N, loc="left")
    axCc.legend(loc="upper right", fontsize=7.5)

    # the additive dz is 1.6 per cent of L, so it needs its own scale to be seen
    ins = axCc.inset_axes([0.10, 0.11, 0.42, 0.33])
    zz = np.linspace(C.Z1_MM - 3.2 * dz, C.Z1_MM, 200)
    ins.plot(zz, C.Z1_MM - zz, color=P["NEUTRAL"], lw=1.2, ls=(0, (4, 3)))
    ins.plot(zz, (C.Z1_MM - zz) + dz, color=P["GREEN"], lw=1.8)
    ins.plot([C.Z1_MM], [dz], "o", ms=4.0, color=P["MAGENTA"], zorder=5)
    ins.annotate("", xy=(C.Z1_MM, 0.0), xytext=(C.Z1_MM, dz),
                 arrowprops=dict(arrowstyle="<->", lw=0.9, color=P["MAGENTA"]))
    ins.text(C.Z1_MM - 0.30 * dz, dz * 0.50,
             "$\\Delta z$ = %.1f mm" % dz, fontsize=7.0, color=P["MAGENTA"],
             ha="right", va="center",
             bbox=dict(boxstyle="square,pad=0.15", fc="#f4f4f1", ec="none"))
    ins.set_xlim(C.Z1_MM - 3.2 * dz, C.Z1_MM + 0.5 * dz)
    ins.set_ylim(-14.0, 3.6 * dz)
    ins.set_title("the last three steps: $\\ell(z_1) = \\Delta z$, not zero",
                  fontsize=7.0, loc="left", fontweight="normal", pad=2.0)
    ins.tick_params(labelsize=6.5, length=2.0)
    ins.set_facecolor("#f4f4f1")
    for zi, li in zip(z, lev):
        rec("c", "lever arm", float(zi), float(li), "z_mm", "lever_mm")
    rec("c", "lever at z0", C.Z0_MM, float(C.L_MM + dz), "z_mm", "lever_mm")
    rec("c", "lever at z1", C.Z1_MM, float(dz), "z_mm", "lever_mm")

    # -------------------------------------------------- (d) the combined weight --
    order = ["20 GeV, first output plane", "3 GeV, first output plane",
             "20 GeV, last output plane", "3 GeV, last output plane"]
    vals, labs, cols = [], [], []
    for k in order:
        c_ = worked["cases"][k]
        vals.append(c_["weight_on_a_slope_residual_per_unit_slope"])
        labs.append(k.replace(", ", ",\n").replace(" output plane", "\noutput plane"))
        cols.append(P["BLUE"] if c_["p_GeV"] == 20.0 else P["MAGENTA"])
        rec("d", k, c_["p_GeV"], c_["weight_on_a_slope_residual_per_unit_slope"],
            "p_GeV", "a_n * lever / D_ref  [per unit slope]",
            "lever = %.3f mm, a_n = %.4f (unclamped)" % (c_["lever_mm"], c_["a_unclamped"]))
    xs = np.arange(4)
    axD.bar(xs, vals, width=0.62, color=cols, edgecolor=P["TEXT1"], linewidth=0.5,
            zorder=3)
    for xi, v in zip(xs, vals):
        axD.text(xi, v * 1.25, ("%.3g" % v), ha="center", fontsize=7.5,
                 color=P["TEXT1"])
    ratio = worked["ratios"]["slope_20GeV_first_over_3GeV_last"]
    axD.annotate("", xy=(0.0, vals[0] * 1.9), xytext=(3.0, vals[0] * 1.9),
                 arrowprops=dict(arrowstyle="<->", lw=0.9, color=P["GREEN"]))
    axD.text(1.5, vals[0] * 2.3, "a factor of %s between the extremes" % ("{:,.0f}".format(ratio)),
             ha="center", fontsize=7.8, color=P["GREEN"])
    axD.set_yscale("log")
    axD.set_xticks(xs)
    axD.set_xticklabels(labs, fontsize=7.3)
    axD.set_ylim(min(vals) / 4.0, max(vals) * 9.0)
    axD.set_ylabel("weight on a slope residual")
    axD.set_title("(d)  the combined weight", loc="left", pad=15.0)
    axD.text(0.015, 1.01, NETNAME, transform=axD.transAxes, fontsize=7.5,
             color=P["TEXT2"], ha="left", va="bottom")
    axD.grid(axis="x", visible=False)
    rec("d", "ratio between the extremes", float("nan"), float(ratio), "-",
        "ratio", "paper_numbers.json worked_example.ratios")

    fig.tight_layout(pad=0.9, w_pad=1.6, h_pad=1.5)
    os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(PNG)
    plt.close(fig)

    C.write_csv(CSV, [C.jsonable(r) for r in ROWS],
                ["panel", "series", "x", "y", "x_quantity", "y_quantity", "note"])
    print("wrote %s" % PNG)
    print("wrote %s  (%d rows)" % (CSV, len(ROWS)))
    print("median a_n over the 11,567 training tracks = %.6f -> clamp [%.6f, %.6f]"
          % (a_med, a_lo, a_hi))


if __name__ == "__main__":
    main()
