#!/usr/bin/env python
"""Data supplement: seven figures that show, to scale, what happens to a track
between the last Upstream Tracker plane and the first fibre-tracker plane, which
physical effect enters where, and how large each one is.

Everything is drawn from the measured data, never from a sketch:
  * the trajectories are the stored RK6 reference tracks of the crossing set;
  * the field profile is read from the v8r1 magnet-up map;
  * the detector positions are the z ranges of the simulated hits themselves;
  * every annotated number is read from a results CSV and copied into
    results/figure_facts.json so the figure and the table cannot drift apart.

Colours are the four validated categorical slots of the house palette, used in
their fixed order (blue, orange, aqua, violet) with the reference track in ink
and the null comparator in muted grey.  Identity is never carried by colour
alone: every series is also named on the figure or directly labelled.

Run:  PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python make_figures.py
Out:  figures/*.png, results/figure_facts.json
"""
import csv
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from _shared.reference import make_field  # noqa: E402

TRACKS = os.path.join(ROOT, "Block_E_single_network_chain", "E0_Track_dataset", "results", "tracks.npz")
CHAIN = os.path.join(ROOT, "Block_F_reweighted_loss", "F1_Training", "results", "full", "N064_q02", "chain_states.npz")
DECOMP = os.path.join(ROOT, "reference_vs_truth", "results", "decomposition_by_band.csv")
HIGHLAND = os.path.join(ROOT, "reference_vs_truth", "results", "highland_air.csv")
THREEREF = os.path.join(ROOT, "Block_F_reweighted_loss", "F2_Analysis", "results", "against_true_state.csv")
COMPAR = os.path.join(ROOT, "Block_F_reweighted_loss", "F3_Analysis", "results", "comparators.csv")
QDZ = os.path.join(ROOT, "Block_F_reweighted_loss", "F3_Analysis", "results", "error_qdz_chain.csv")
FIGS = os.path.join(HERE, "figures")
RES = os.path.join(HERE, "results")

# --- palette: the validated categorical slots, fixed order -------------------
C_NET = "#2a78d6"     # slot 1, blue    - the network
C_TRUTH = "#eb6834"   # slot 2, orange  - the simulated truth
C_EXACT = "#1baf7a"   # slot 3, aqua    - the exact collocation scheme
C_ALT = "#4a3aa7"     # slot 7, violet  - the integrator's own floor
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#9a9a94"
FAINT = "#e5e4df"
SURF = "#fcfcfb"
FIELD_FILL = "#dfe7f2"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": "#c9c8c2", "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.grid": True,
    "grid.color": FAINT, "grid.linewidth": 0.6, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})

FACTS = {}


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def band_col(table, band, col, charge=None, key="band"):
    for r in table:
        if r[key] == band and (charge is None or r.get("charge") == charge):
            v = r[col]
            return float(v) if v not in ("", None) else float("nan")
    return float("nan")


# ---------------------------------------------------------------- load data
D = np.load(TRACKS)
Z0, Z1 = float(D["z0"]), float(D["z1"])
L = Z1 - Z0
PLANES = D["planes"]
S0, TRUTH = D["test_S0"], D["test_truth"]
SPOST, ZPOST, TZPOST = D["test_S_post"], D["test_z_post"], D["test_truth_zpost"]
P, QOP = D["test_P"], D["test_S0"][:, 4]
NN = np.load(CHAIN)["test_states"]          # (n, 65, 5), the chained network
FLD = make_field("up")

DEC = rows(DECOMP)
HIG = rows(HIGHLAND)
TR3 = [r for r in rows(THREEREF) if r["N"] == "64"]
CMP = {r["what"]: r for r in rows(COMPAR)}
QDZR = rows(QDZ)

BANDS = ["2-5 GeV", "5-10 GeV", "10-25 GeV", "25-50 GeV", "50-100 GeV"]
BANDS3 = ["2-5 GeV", "5-10 GeV", "10-25 GeV", "25-200 GeV"]

MED_NN = band_col(TR3, "all", "nn_vs_rk6_pos_med_um")
MED_TRUE = band_col(TR3, "all", "rk6_vs_true_pos_med_um")
MED_LINE = float(CMP["straight line"]["med_um"])
MED_EXACT = float(QDZR[0]["exact_med_um"])
# The reference integrator's own floor: read from the filed table rather than typed here,
# because typing it is how it went wrong once. gates.json G3 records the 5 mm against 1 mm
# step convergence in MILLIMETRES (2.8459e-08 mm); in micrometres that is 2.846e-05, which is
# 28 picometres. An earlier version of this script hard-coded 0.0285 um, a thousand times too
# large, following the same slip in Data_generation_exploration/Data/README.md ("28 nm").
RK6_SELF = os.path.join(ROOT, "reference_vs_truth", "results", "rk6_self_consistency.csv")
RK6_STEP_FLOOR = float([r for r in rows(RK6_SELF)
                        if r["what"].startswith("RK6 step convergence")
                        and r["what"].endswith("median")][0]["value_um"])
DZ64 = L / 64.0
GL2 = np.array([0.21132487, 0.78867513])   # q = 2 Gauss-Legendre nodes

FACTS["scales"] = dict(z0_mm=Z0, z1_mm=Z1, L_mm=L, dz_N64_mm=DZ64,
                       median_network_vs_rk6_um=MED_NN,
                       median_rk6_vs_true_um=MED_TRUE,
                       median_straight_line_um=MED_LINE,
                       median_exact_scheme_um=MED_EXACT,
                       rk6_step_convergence_um=RK6_STEP_FLOOR)


def radial_um(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]) * 1e3)


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    p = os.path.join(FIGS, name)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("  wrote figures/%s" % name)


def scalebar(ax, length, label, colour=INK2, frac=(0.06, 0.08)):
    """A horizontal scale bar in data units, placed low-left."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xs = x0 + frac[0] * (x1 - x0)
    ys = y0 + frac[1] * (y1 - y0)
    ax.plot([xs, xs + length], [ys, ys], "-", color=colour, lw=2.2, solid_capstyle="butt")
    for xx in (xs, xs + length):
        ax.plot([xx, xx], [ys - 0.015 * (y1 - y0), ys + 0.015 * (y1 - y0)], "-", color=colour, lw=1.4)
    ax.text(xs + length / 2, ys + 0.03 * (y1 - y0), label, ha="center", va="bottom",
            fontsize=8, color=colour)


# ============================================================ 1. the crossing
def fig1_crossing():
    """The crossing at true scale, with the field that bends it."""
    pick = []
    for lo, hi in ((2.5, 3.5), (9, 12), (40, 70)):
        m = np.where((P > lo) & (P < hi) & (np.abs(S0[:, 0]) < 90) & (np.abs(S0[:, 2]) < 0.05))[0]
        if len(m):
            pick.append(int(m[np.argmin(np.abs(S0[m, 0]))]))
    zz = PLANES
    fig = plt.figure(figsize=(11.6, 5.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.35, 1.0], hspace=0.30)
    ax = fig.add_subplot(gs[0])
    axb = fig.add_subplot(gs[1])

    zf = np.linspace(Z0, Z1, 400)
    _, By, _ = FLD(np.zeros_like(zf), np.zeros_like(zf), zf)
    By = np.abs(By)

    # field as a background wash on the trajectory panel, so the eye links them
    for k in range(len(zf) - 1):
        ax.axvspan(zf[k], zf[k + 1], color=FIELD_FILL, alpha=float(By[k] / By.max()) * 0.85, lw=0)

    ramp = ["#9dc3ec", "#5b9ae0", "#14508f"]     # one hue, light to dark = rising momentum
    for j, i in enumerate(pick):
        x = TRUTH[i, :, 0]
        ax.plot(zz, x, "-", color=ramp[j], lw=2.0, zorder=4)
        line = S0[i, 0] + S0[i, 2] * (zz - Z0)
        if j == 0:
            ax.plot(zz, line, "--", color=MUTED, lw=1.3, zorder=3, label="no field at all (straight line)")
            ax.annotate("", xy=(Z1, x[-1]), xytext=(Z1, line[-1]),
                        arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.2))
            ax.text(Z1 - 110, 0.5 * (x[-1] + line[-1]), "the bend,\n%.0f mm" % abs(x[-1] - line[-1]),
                    ha="right", va="center", fontsize=8.5, color=INK2)
        ax.text(zz[-1] + 55, x[-1], "%.0f GeV" % P[i], color=ramp[j], fontsize=9,
                va="center", fontweight="bold")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(Z0 - 80, Z1 + 620)
    ax.axvline(Z0, color=INK2, lw=1.0)
    ax.axvline(Z1, color=INK2, lw=1.0)
    ylo, yhi = ax.get_ylim()
    ax.text(Z0 + 60, ylo + 0.06 * (yhi - ylo), "last UT plane\nz = %.1f mm" % Z0, fontsize=8,
            color=INK2, va="bottom", ha="left")
    ax.text(Z1 - 60, ylo + 0.06 * (yhi - ylo), "first SciFi plane\nz = %.1f mm" % Z1, fontsize=8,
            color=INK2, va="bottom", ha="right")
    ax.set_ylabel("x  [mm]")
    ax.set_title("The magnet crossing at true scale: 5,177.8 mm long, and the paths that cross it\n"
                 "(equal scale on both axes, so the curvature is the real curvature)", loc="left")
    ax.legend(loc="lower left", fontsize=8.5, bbox_to_anchor=(0.005, 0.30))
    ax.grid(False)

    axb.fill_between(zf, 0, By, color=C_NET, alpha=0.18, lw=0)
    axb.plot(zf, By, "-", color=C_NET, lw=1.8)
    axb.set_xlim(Z0 - 80, Z1 + 430)
    axb.set_ylim(0, 1.18)
    axb.set_xlabel("z  [mm]   (the beam direction)")
    axb.set_ylabel("|B$_y$| on axis  [T]")
    imax = int(np.argmax(By))
    axb.annotate("peak %.2f T at z = %.0f mm" % (By[imax], zf[imax]),
                 xy=(zf[imax], By[imax]), xytext=(zf[imax] + 380, 0.98),
                 fontsize=8.5, color=INK2,
                 arrowprops=dict(arrowstyle="->", color=INK2, lw=1.0))
    axb.annotate("0.21 T", xy=(Z0, By[0]), xytext=(Z0 + 120, 0.20), fontsize=8, color=INK2,
                 arrowprops=dict(arrowstyle="->", color=INK2, lw=0.9))
    axb.set_title("The field along the same axis: the bend is done in the middle, not at the ends", loc="left",
                  fontsize=9, fontweight="normal")
    FACTS["fig1"] = dict(tracks_GeV=[float(P[i]) for i in pick],
                         bend_mm=float(abs(TRUTH[pick[0], -1, 0] - (S0[pick[0], 0] + S0[pick[0], 2] * L))),
                         peak_By_T=float(By.max()), peak_By_z_mm=float(zf[int(np.argmax(By))]),
                         By_at_z0_T=float(By[0]), By_at_z1_T=float(By[-1]))
    save(fig, "fig1_crossing_to_scale.png")


# ====================================================== 2. the zoom cascade
def fig2_zoom():
    """Four decades of zoom at the SciFi plane: who sits where."""
    d_nn = np.hypot(NN[:, -1, 0] - TRUTH[:, -1, 0], NN[:, -1, 1] - TRUTH[:, -1, 1]) * 1e3
    d_tr = np.hypot(SPOST[:, 0] - TZPOST[:, 0], SPOST[:, 1] - TZPOST[:, 1]) * 1e3
    score = np.abs(np.log(np.maximum(d_nn, 1e-3) / MED_NN)) + np.abs(np.log(np.maximum(d_tr, 1e-3) / MED_TRUE))
    i = int(np.argmin(np.where((P > 4) & (P < 25), score, 1e9)))

    nn = (NN[i, -1, :2] - TRUTH[i, -1, :2]) * 1e3        # um, relative to RK6 at z1
    tr = (SPOST[i, :2] - TZPOST[i, :2]) * 1e3            # um, relative to RK6 at the particle's plane
    sl = (np.array([S0[i, 0] + S0[i, 2] * L, S0[i, 1] + S0[i, 3] * L]) - TRUTH[i, -1, :2]) * 1e3

    spans = [6.0e5, 8.0e3, 4.0e2, 1.2e2]
    units = [("mm", 1e3), ("mm", 1e3), ("um", 1.0), ("um", 1.0)]
    titles = ["a.  the whole deflection", "b.  the truth separates",
              "c.  the network separates", "d.  the network's error"]
    fig, axes = plt.subplots(1, 4, figsize=(14.2, 4.1))
    fig.subplots_adjust(wspace=0.42)
    for k, ax in enumerate(axes):
        s = spans[k]
        un, uf = units[k]
        f = 1.0 / uf
        ax.set_xlim(-s * f, s * f)
        ax.set_ylim(-s * f, s * f)
        ax.set_aspect("equal")
        ax.axhline(0, color=FAINT, lw=0.8, zorder=1)
        ax.axvline(0, color=FAINT, lw=0.8, zorder=1)
        # population medians as rings
        for r, col, lab in ((MED_TRUE, C_TRUTH, "typical truth gap"), (MED_NN, C_NET, "typical network error")):
            if 0.02 * s < r < 2.2 * s:
                th = np.linspace(0, 2 * np.pi, 200)
                ax.plot(r * np.cos(th) * f, r * np.sin(th) * f, ":", color=col, lw=1.1, alpha=0.75, zorder=2)
        if k == 0:
            ax.plot(sl[0] * f, sl[1] * f, "s", ms=8, color=MUTED, zorder=5)
            ax.annotate("straight line\n(no field): %.0f mm" % (np.hypot(*sl) / 1e3),
                        xy=(sl[0] * f, sl[1] * f), xytext=(0.06, 0.16), textcoords="axes fraction",
                        fontsize=8, color=INK2, arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.0))
        for pt, col, nm in ((tr, C_TRUTH, "the truth"), (nn, C_NET, "the network")):
            r = np.hypot(*pt)
            if r < 0.97 * s:
                ax.plot(pt[0] * f, pt[1] * f, "o", ms=9 if col == C_TRUTH else 8, color=col, zorder=6,
                        markeredgecolor=SURF, markeredgewidth=1.6)
            elif r < 60 * s:
                # off this panel: point at it from the middle, and say how far
                u = np.array(pt) / r
                lab = "%s:\n%s that way" % (nm, ("%.1f mm" % (r / 1e3)) if r > 900 else ("%.0f \u00b5m" % r))
                ax.annotate("", xy=(u[0] * 0.92 * s * f, u[1] * 0.92 * s * f),
                            xytext=(u[0] * 0.52 * s * f, u[1] * 0.52 * s * f),
                            arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8))
                ax.text(0.04, 0.05, lab, transform=ax.transAxes, fontsize=7.8, color=col,
                        ha="left", va="bottom")
        ax.plot(0, 0, "P", ms=9, color=INK, zorder=8, markeredgecolor=SURF, markeredgewidth=1.4)
        if k < 3:
            nx = spans[k + 1] * f
            ax.add_patch(Rectangle((-nx, -nx), 2 * nx, 2 * nx, fill=False, ec=INK2, lw=1.1, zorder=9))
            if spans[k + 1] / spans[k] > 0.02:      # only label a square big enough to see
                ax.annotate("the square is\nthe next panel", xy=(-nx, nx), xycoords="data",
                            xytext=(0.035, 0.955), textcoords="axes fraction", fontsize=7.4,
                            color=INK2, va="top", ha="left",
                            arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8, alpha=0.8))
        if k == 0:
            ax.annotate("reference, truth and network\nare all inside this one marker",
                        xy=(0, 0), xytext=(0.50, 0.80), textcoords="axes fraction", fontsize=7.8,
                        color=INK2, ha="center",
                        arrowprops=dict(arrowstyle="->", color=INK2, lw=0.9))
        ax.set_title(titles[k], loc="left", fontsize=9.5)
        ax.set_xlabel("$\\Delta x$  [%s]" % ("mm" if un == "mm" else "µm"))
        if k == 0:
            ax.set_ylabel("$\\Delta y$  [mm]")
        ax.tick_params(labelsize=8)
    # one legend for the set, built from proxies so identity is never colour alone
    hs = [plt.Line2D([], [], marker="P", ls="", color=INK, ms=9, label="the field-only reference (origin)"),
          plt.Line2D([], [], marker="o", ls="", color=C_TRUTH, ms=9, label="where the particle really was (simulated truth)"),
          plt.Line2D([], [], marker="o", ls="", color=C_NET, ms=8, label="where the network put it"),
          plt.Line2D([], [], marker="s", ls="", color=MUTED, ms=8, label="no field at all (straight line)"),
          plt.Line2D([], [], ls=":", color=INK2, label="dotted rings: the median over all 1,452 test tracks")]
    fig.legend(handles=hs, loc="lower center", ncols=3, fontsize=8.5, bbox_to_anchor=(0.5, -0.10))
    fig.suptitle("Zooming in on the far end of the crossing, one track, each panel about ten times closer than the last",
                 x=0.012, ha="left", fontsize=11, fontweight="bold")
    FACTS["fig2"] = dict(track_index=i, p_GeV=float(P[i]),
                         network_minus_reference_um=[float(nn[0]), float(nn[1])],
                         truth_minus_reference_um=[float(tr[0]), float(tr[1])],
                         straight_line_minus_reference_um=[float(sl[0]), float(sl[1])],
                         panel_half_widths_um=spans)
    save(fig, "fig2_zoom_cascade.png")


# ================================================= 3. where each effect enters
def fig3_pipeline():
    """The detector along z, what happens where, and who models it."""
    d_nn = np.hypot(NN[:, -1, 0] - TRUTH[:, -1, 0], NN[:, -1, 1] - TRUTH[:, -1, 1]) * 1e3
    i = int(np.argmin(np.where((P > 4) & (P < 8), np.abs(d_nn - MED_NN), 1e9)))

    fig = plt.figure(figsize=(12.6, 6.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.95, 1.0], hspace=0.60)
    ax = fig.add_subplot(gs[0])
    axt = fig.add_subplot(gs[1])

    zf = np.linspace(Z0, Z1, 300)
    _, By, _ = FLD(np.zeros_like(zf), np.zeros_like(zf), zf)
    By = np.abs(By)
    for k in range(len(zf) - 1):
        ax.axvspan(zf[k], zf[k + 1], color=FIELD_FILL, alpha=float(By[k] / By.max()) * 0.9, lw=0)
    ax.text(0.5 * (Z0 + Z1), 0.90, "the dipole magnet", ha="center", fontsize=9.5,
            color="#2f5c92", transform=ax.get_xaxis_transform(), fontweight="bold")

    # detector envelopes, from the z range of the simulated hits themselves
    for z_lo, z_hi, lab in ((-288.2, 750.7, "VELO"), (2306.7, 2663.3, "UT"), (7817.3, 9412.1, "SciFi")):
        ax.axvspan(z_lo, z_hi, color="#d8d7d0", alpha=0.85, lw=0, zorder=2)
        ax.text(0.5 * (z_lo + z_hi), 0.035, lab, ha="center", fontsize=9, color=INK2,
                transform=ax.get_xaxis_transform(), fontweight="bold", zorder=3)

    zz = np.concatenate([np.linspace(-288.2, Z0, 60), PLANES])
    xx = np.concatenate([S0[i, 0] + S0[i, 2] * (np.linspace(-288.2, Z0, 60) - Z0), TRUTH[i, :, 0]])
    ax.plot(zz, xx, "-", color=INK, lw=2.0, zorder=6)
    ax.plot(Z0, S0[i, 0], "o", ms=8, color=INK, zorder=7, markeredgecolor=SURF, markeredgewidth=1.5)
    ax.plot(ZPOST[i], SPOST[i, 0], "o", ms=9, color=C_TRUTH, zorder=7, markeredgecolor=SURF, markeredgewidth=1.5)

    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0 - 0.95 * (y1 - y0), y1 + 0.55 * (y1 - y0))

    def note(zpos, xpos, text, fx, fy, colour=INK2):
        """Anchor the box at a fixed place on the axes and point it at the data."""
        ax.annotate(text, xy=(zpos, xpos), xycoords="data", xytext=(fx, fy),
                    textcoords="axes fraction", fontsize=8.3, color=colour, ha="center",
                    va="center", zorder=12,
                    arrowprops=dict(arrowstyle="->", color=colour, lw=1.0,
                                    connectionstyle="arc3,rad=0.12"),
                    bbox=dict(boxstyle="round,pad=0.32", fc=SURF, ec=colour, lw=0.9, alpha=0.97))

    note(200, np.interp(200, zz, xx),
         "1.  the particle is made here.\nthe momentum it has at this point\nis the one the reference is given",
         0.115, 0.88, C_TRUTH)
    note(1600, np.interp(1600, zz, xx),
         "2.  it loses 16 to 18 MeV in the VELO\nand the UT before the magnet, and the\nreference is never told",
         0.245, 0.15, C_TRUTH)
    note(Z0, S0[i, 0],
         "3.  the reference and the network both\nstart from the real state on this plane",
         0.455, 0.90, INK)
    note(0.5 * (Z0 + Z1), np.interp(0.5 * (Z0 + Z1), zz, xx),
         "4.  five metres of air and structure:\nthe particle is kicked at random,\nand neither the reference nor the\nnetwork can follow a random kick",
         0.630, 0.14, C_TRUTH)
    note(ZPOST[i], SPOST[i, 0],
         "5.  the simulated hit.\nthis is the truth\nwe compare with",
         0.905, 0.80, C_TRUTH)

    ax.set_xlim(-500, 9600)
    ax.set_xlabel("z  [mm]   (detector envelopes are the z range of the simulated hits)")
    ax.set_ylabel("x  [mm]")
    ax.set_title("Where each effect enters, along one real %.0f GeV track" % P[i], loc="left")
    ax.grid(False)

    # the ledger: which description contains which effect
    axt.axis("off")
    cols = ["bending in the field", "energy loss", "multiple scattering",
            "detector resolution"]
    rws = [("the simulated truth (the hit)", ["yes", "yes", "yes", "not applied"]),
           ("the reference track (RK6)", ["yes", "no", "no", "n/a"]),
           ("what the network is trained on", ["yes", "no", "no", "n/a"])]
    nc, nr = len(cols) + 1, len(rws) + 1
    cw, ch = 1.0 / nc, 1.0 / (nr + 0.6)
    for j, c in enumerate(cols):
        axt.text((j + 1.5) * cw, 1 - 0.5 * ch, c, ha="center", va="center", fontsize=8.6,
                 color=INK, fontweight="bold")
    for r, (name, vals) in enumerate(rws):
        yy = 1 - (r + 1.5) * ch
        axt.text(0.5 * cw, yy, name, ha="center", va="center", fontsize=8.6, color=INK)
        for j, v in enumerate(vals):
            col = {"yes": "#0f7a4f", "no": "#b4402a"}.get(v, INK2)
            mark = {"yes": "✓  in", "no": "✗  absent"}.get(v, v)
            axt.add_patch(Rectangle(((j + 1) * cw + 0.006, yy - 0.42 * ch), cw - 0.012, 0.84 * ch,
                                    fc="#eef6f1" if v == "yes" else ("#fbeeea" if v == "no" else "#f2f2ef"),
                                    ec="none"))
            axt.text((j + 1.5) * cw, yy, mark, ha="center", va="center", fontsize=8.6, color=col)
    axt.plot([0, 1], [1 - ch, 1 - ch], "-", color="#c9c8c2", lw=1.0)
    axt.text(0, 1.10, "What each description of the crossing contains", fontsize=9.5,
             fontweight="bold", color=INK, va="bottom")
    axt.text(0, -0.12, "The reference solves the equation of motion in the measured field and nothing else. "
                       "The network is trained on the residual of that same equation, so it inherits exactly the same scope. "
                       "No digitisation or pattern recognition enters anywhere in this pipeline.",
             fontsize=8.3, color=INK2, va="top")
    FACTS["fig3"] = dict(track_index=i, p_GeV=float(P[i]),
                         velo_z=[-288.2, 750.7], ut_z=[2306.7, 2663.3], scifi_z=[7817.3, 9412.1])
    save(fig, "fig3_where_effects_enter.png")


# =============================================== 4. the budget, on a log axis
def fig4_budget():
    x = np.arange(len(BANDS3))
    nn = [band_col(TR3, b, "nn_vs_rk6_pos_med_um") for b in BANDS3]
    gap = [band_col(TR3, b, "rk6_vs_true_pos_med_um") for b in BANDS3]
    sysm = []
    for b in BANDS3:
        bb = "25-50 GeV" if b == "25-200 GeV" else b
        sysm.append(abs(band_col(DEC, bb, "median_dx_um", "q+")))
    fig, ax = plt.subplots(figsize=(9.6, 6.8))
    ax.set_yscale("log")
    ax.axhline(MED_LINE, color=MUTED, lw=2.0, ls="-")
    ax.text(len(BANDS3) - 0.45, MED_LINE, " no field at all\n %.0f mm" % (MED_LINE / 1e3),
            color=MUTED, fontsize=8.6, va="center")
    ax.plot(x, gap, "-o", color=C_TRUTH, lw=2.2, ms=8, mec=SURF, mew=1.5, label="reference to simulated truth")
    ax.plot(x, sysm, "--^", color=C_TRUTH, lw=1.7, ms=7, alpha=0.75, label="of which is the stale start momentum")
    ax.plot(x, nn, "-o", color=C_NET, lw=2.2, ms=8, mec=SURF, mew=1.5, label="network to reference")
    ax.axhline(MED_EXACT, color=C_EXACT, lw=1.8, ls="-")
    ax.text(len(BANDS3) - 0.45, MED_EXACT, " the exact scheme\n %.2f µm" % MED_EXACT,
            color=C_EXACT, fontsize=8.6, va="center")
    ax.axhline(RK6_STEP_FLOOR, color=C_ALT, lw=1.8, ls="--")
    ax.text(len(BANDS3) - 0.45, RK6_STEP_FLOOR,
            " the integrator's own floor\n %.1f × 10$^{-5}$ µm (28 picometres)" % (RK6_STEP_FLOOR * 1e5),
            color=C_ALT, fontsize=8.6, va="center")
    for xx, v in zip(x, gap):
        ax.annotate("%.2f mm" % (v / 1e3), (xx, v), textcoords="offset points", xytext=(0, 11),
                    ha="center", fontsize=8.2, color=C_TRUTH)
    for xx, v in zip(x, nn):
        ax.annotate("%.0f µm" % v, (xx, v), textcoords="offset points", xytext=(0, -16),
                    ha="center", fontsize=8.2, color=C_NET)
    ax.set_xticks(x)
    ax.set_xticklabels(BANDS3)
    ax.set_xlim(-0.35, len(BANDS3) + 0.62)
    ax.set_ylim(3e-6, 4e6)
    ax.set_yticks([1e-5, 1e-3, 1e-1, 1e1, 1e3, 1e5])
    ax.set_xlabel("momentum of the track")
    ax.set_ylabel("distance at the first SciFi plane  [µm, logarithmic]")
    ax.set_title("Every error in the problem on one axis, and they span ten decades", loc="left")
    ax.legend(loc="lower left", fontsize=8.8, bbox_to_anchor=(0.005, 0.30))
    ax.grid(True, which="major", axis="y")
    FACTS["fig4"] = dict(bands=BANDS3, network_vs_reference_um=nn, reference_vs_truth_um=gap,
                         stale_momentum_um=sysm)
    save(fig, "fig4_error_budget.png")


# ================================================ 5. the charge tells them apart
def fig5_charge():
    d = SPOST - TZPOST
    bend = TZPOST[:, 0] - (S0[:, 0] + S0[:, 2] * (ZPOST - Z0))
    q = np.sign(QOP)
    m = (P > 5) & (P < 10)
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6))
    ax = axes[0]
    drawn = {}
    for sgn, col, lab in ((+1, C_NET, "positive tracks"), (-1, C_TRUTH, "negative tracks")):
        s = m & (q == sgn)
        ax.plot(bend[s], d[s, 0] * 1e3, "o", ms=3.0, color=col, alpha=0.45, mec="none", label=lab)
        med = np.median(d[s, 0]) * 1e3
        drawn["median_dx_%s_test_split_um" % ("qplus" if sgn > 0 else "qminus")] = float(med)
        drawn["n_%s_test_split" % ("qplus" if sgn > 0 else "qminus")] = int(s.sum())
        ax.axhline(med, color=col, lw=1.8, ls="--")
        ax.text(0.985, med, "median %+0.0f µm " % med, color=col, fontsize=8.8, va="center",
                ha="right", transform=ax.get_yaxis_transform(),
                bbox=dict(boxstyle="round,pad=0.22", fc=SURF, ec="none", alpha=0.9))
    ax.axhline(0, color=INK2, lw=1.0)
    ax.set_ylim(-6000, 6000)
    ax.set_xlabel("how far the field bent the track  [mm]")
    ax.set_ylabel("truth minus reference, in x  [µm]")
    ax.set_title("a.  the centre flips with the charge: that is a momentum error", loc="left", fontsize=9.5)
    ax.legend(loc="upper left", fontsize=8.6, markerscale=2.4)

    ax = axes[1]
    pc = np.array([band_col(DEC, b, "median_P_GeV", "both") for b in BANDS])
    for sgn, col, lab, mk in ((+1, C_NET, "positive tracks", "o"), (-1, C_TRUTH, "negative tracks", "s")):
        key = "q+" if sgn > 0 else "q-"
        rel = np.array([band_col(DEC, b, "median_dx_over_bend", key) for b in BANDS])
        ax.plot(pc, rel * 1e3, mk + "-", color=col, lw=1.9, ms=7, mec=SURF, mew=1.3, label=lab)
    ax.axhline(0, color=INK2, lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("momentum  [GeV]")
    ax.set_ylabel("over-bend, as parts per thousand of the bend")
    ax.set_title("b.  as a fraction of the bend the two charges agree: one momentum deficit", loc="left", fontsize=9.5)
    ax.legend(loc="upper right", fontsize=8.6)
    ax.text(0.03, 0.06, "about 16 to 18 MeV of momentum,\nlost before the starting plane",
            transform=ax.transAxes, fontsize=8.4, color=INK2,
            bbox=dict(boxstyle="round,pad=0.35", fc=SURF, ec=FAINT))
    fig.suptitle("Splitting the gap by charge separates a momentum error from a random one",
                 x=0.012, ha="left", fontsize=11, fontweight="bold")
    # Both populations are recorded, because the left panel is drawn on the test crossings
    # while the decomposition table covers all 14,482. The page must say which it quotes.
    FACTS["fig5"] = dict(band="5-10 GeV",
                         median_dx_qplus_all_crossings_um=band_col(DEC, "5-10 GeV", "median_dx_um", "q+"),
                         median_dx_qminus_all_crossings_um=band_col(DEC, "5-10 GeV", "median_dx_um", "q-"),
                         n_qplus_all_crossings=int(band_col(DEC, "5-10 GeV", "n", "q+")),
                         n_qminus_all_crossings=int(band_col(DEC, "5-10 GeV", "n", "q-")),
                         panel_a_is_drawn_on="the 1,452 test crossings only", **drawn)
    save(fig, "fig5_charge_separation.png")


# ====================================================== 6. the scattering floor
def fig6_scattering():
    p = np.array([float(r["median_P_GeV"]) for r in HIG])
    meas = np.array([float(r["measured_halfwidth68_dx_um"]) for r in HIG])
    air = np.array([float(r["highland_air_width_um"]) for r in HIG])
    eff = np.array([float(r["effective_x_over_X0"]) for r in HIG])
    keep = p > 2.0
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(p[keep], meas[keep], "o", ms=9, color=C_TRUTH, mec=SURF, mew=1.5,
            label="measured width of the gap", zorder=5)
    ax.plot(p[keep], air[keep], "-", color=INK2, lw=1.9, label="scattering in the air alone (Highland)")
    ax.fill_between(p[keep], air[keep], meas[keep], color=C_TRUTH, alpha=0.13, lw=0)
    ax.plot(p[keep], np.full(keep.sum(), MED_NN), "-", color=C_NET, lw=2.0,
            label="the network's own error, for scale")
    for xx, yy, e in zip(p[keep], meas[keep], eff[keep]):
        ax.annotate("%.1f%% X$_0$" % (e * 100), (xx, yy), textcoords="offset points",
                    xytext=(6, 9), fontsize=8.0, color=C_TRUTH)
    ax.set_xlim(p[keep].min() * 0.72, p[keep].max() * 1.65)
    ax.set_xlabel("momentum  [GeV]")
    ax.set_ylabel("68 % half-width of the gap in x  [µm]")
    ax.set_title("The random part of the gap is multiple scattering, and it falls as 1/p", loc="left")
    ax.legend(loc="upper right", fontsize=8.8)
    ax.text(0.03, 0.06,
            "The measured width sits a little above the air-only curve\n"
            "because the gap is not only air: the numbers beside the points\n"
            "are the material this implies, in radiation lengths.",
            transform=ax.transAxes, fontsize=8.4, color=INK2,
            bbox=dict(boxstyle="round,pad=0.35", fc=SURF, ec=FAINT))
    FACTS["fig6"] = dict(p_GeV=p[keep].tolist(), measured_um=meas[keep].tolist(),
                         air_only_um=air[keep].tolist(), effective_x_over_X0=eff[keep].tolist())
    save(fig, "fig6_scattering_floor.png")


# ============================================== 7. what one step actually is
def fig7_one_step():
    d_nn = np.hypot(NN[:, -1, 0] - TRUTH[:, -1, 0], NN[:, -1, 1] - TRUTH[:, -1, 1]) * 1e3
    i = int(np.argmin(np.where((P > 4) & (P < 8), np.abs(d_nn - MED_NN), 1e9)))
    k0 = 4 * 24                      # a step in the middle of the crossing, stride 4 = N of 64
    k1 = k0 + 4
    zA, zB = PLANES[k0], PLANES[k1]
    sub = np.linspace(zA, zB, 80)
    xs = np.interp(sub, PLANES[k0:k1 + 1], TRUTH[i, k0:k1 + 1, 0])
    line = TRUTH[i, k0, 0] + TRUTH[i, k0, 2] * (sub - zA)
    dev = (xs - line) * 1e3          # um

    err = 0.43
    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    ax.axhline(0, color=MUTED, lw=1.6, ls="--")
    ax.text(zA + 0.015 * (zB - zA), 0, " a straight line from the start state", fontsize=8.3,
            color=MUTED, va="bottom", ha="left")
    ax.plot(sub, dev, "-", color=INK, lw=2.4, zorder=5)
    ax.fill_between(sub, 0, dev, color=INK, alpha=0.05, lw=0)
    span = abs(dev).max()
    for j, c in enumerate(GL2):
        zc = zA + c * (zB - zA)
        dc = float(np.interp(zc, sub, dev))
        ax.plot([zc, zc], [dc - 0.055 * span, dc + 0.055 * span], "-", color=C_EXACT, lw=1.2, zorder=4)
        ax.plot(zc, dc, "o", ms=11, color=C_EXACT, mec=SURF, mew=1.7, zorder=6)
        ax.annotate("stage state %d of 2:\nthe network predicts this" % (j + 1), xy=(zc, dc),
                    xytext=(zc + (0.13 if j == 0 else 0.05) * (zB - zA),
                            dc + (-0.24 if j == 0 else 0.33) * span),
                    fontsize=8.3, color="#0e6647", ha="left",
                    arrowprops=dict(arrowstyle="->", color="#0e6647", lw=1.0))
    ax.plot(zA, 0, "P", ms=12, color=INK, mec=SURF, mew=1.5, zorder=7)
    ax.plot(zB, dev[-1], "P", ms=12, color=INK, mec=SURF, mew=1.5, zorder=7)
    ax.annotate("the track state goes in here", xy=(zA, 0), xytext=(zA + 0.10 * (zB - zA), -0.17 * span),
                fontsize=8.6, color=INK2, ha="left",
                arrowprops=dict(arrowstyle="->", color=INK2, lw=1.0))
    ax.annotate("and the step ends here", xy=(zB, dev[-1]),
                xytext=(zB - 0.30 * (zB - zA), dev[-1] + 0.16 * span),
                fontsize=8.6, color=INK2, ha="right",
                arrowprops=dict(arrowstyle="->", color=INK2, lw=1.0))

    # the honest vertical exaggeration, computed from the rendered axes box
    fig.canvas.draw()
    bb = ax.get_window_extent()
    xr = (zB - zA) / bb.width
    yr = (2.0 * span * 1e-3) / bb.height          # um -> mm
    stretch = xr / yr
    ax.set_xlabel("z  [mm]    one step of %.1f mm, taken from the middle of the crossing\n"
                  "horizontal in millimetres, vertical in micrometres: the vertical axis is "
                  "stretched about %d times" % (DZ64, int(round(stretch / 10.0) * 10)))
    ax.set_ylabel("departure from a straight line  [µm]")
    ax.set_title("One step of the chain: what the network is asked for, and how small it is", loc="left")
    ax.text(0.985, 0.94,
            "No measured position enters this picture. The network is scored on\n"
            "whether its own stage states satisfy the equation of motion, which\n"
            "is why it needs no labels to train on.",
            transform=ax.transAxes, fontsize=8.4, color=INK2, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.35", fc=SURF, ec=FAINT))

    # inset: the endpoint at the scale of the network's own single-step error
    axi = ax.inset_axes([0.065, 0.10, 0.33, 0.29])
    axi.set_facecolor("#f4f4f1")
    zin = zB - 1.2
    msk = sub >= zin
    axi.plot(sub[msk], dev[msk], "-", color=INK, lw=2.2)
    axi.errorbar([zB], [dev[-1]], yerr=[err], color=C_NET, lw=3.0, capsize=7, capthick=2.2, zorder=6)
    axi.plot(zB, dev[-1], "P", ms=9, color=INK, mec=SURF, mew=1.2, zorder=7)
    axi.set_ylim(dev[-1] - 2.2, dev[-1] + 2.2)
    axi.set_xlim(zin, zB + 0.30)
    axi.tick_params(labelsize=7)
    axi.set_yticks([dev[-1] - 2, dev[-1], dev[-1] + 2])
    axi.set_yticklabels(["−2", "0", "+2"])
    axi.set_xticks([])
    axi.set_ylabel("µm", fontsize=7.5)
    axi.set_title("the last 1.2 mm of the step, magnified 150 times:\n"
                  "the network's own error is 0.43 µm, the blue bar",
                  fontsize=7.8, loc="left", fontweight="normal", color=C_NET)
    for s_ in axi.spines.values():
        s_.set_edgecolor("#c9c8c2")
    FACTS["fig7"] = dict(track_index=i, p_GeV=float(P[i]), dz_mm=float(zB - zA),
                         departure_from_line_um=float(dev[-1]),
                         gauss_legendre_nodes=GL2.tolist(), single_step_error_um=err)
    save(fig, "fig7_one_step.png")


def main():
    os.makedirs(RES, exist_ok=True)
    print("crossing %.1f -> %.1f mm (L = %.1f), step at N=64 is %.2f mm" % (Z0, Z1, L, DZ64))
    print("medians [um]: network-vs-reference %.1f, reference-vs-truth %.0f, straight line %.0f, exact %.3f"
          % (MED_NN, MED_TRUE, MED_LINE, MED_EXACT))
    fig1_crossing()
    fig2_zoom()
    fig3_pipeline()
    fig4_budget()
    fig5_charge()
    fig6_scattering()
    fig7_one_step()
    with open(os.path.join(RES, "figure_facts.json"), "w") as f:
        json.dump(FACTS, f, indent=1)
    print("  wrote results/figure_facts.json")


if __name__ == "__main__":
    sys.exit(main())
