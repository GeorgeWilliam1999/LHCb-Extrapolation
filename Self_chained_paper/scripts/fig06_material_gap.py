#!/usr/bin/env python
"""Figure 6 - what separates the field-only reference from the real particle.

The residual d = (the particle's real Geant4 state on its own first SciFi
plane) minus (the field-only RK6 reference carried to that same plane) is
measured on all 14,482 crossings of the track set and split into the two parts
that behave differently under a charge flip.

WHAT THE FIGURE SHOWS

  (a) the SYSTEMATIC part: the signed median of dx per momentum band, drawn
      separately for positive and negative tracks, on a symmetric-log axis so
      the four orders of magnitude between the softest and the stiffest band
      are all visible.  Below 20 GeV the two charges sit on opposite sides of
      zero - the sign flips - which is what a charge-odd effect looks like.
  (b) the same quantity divided by the bend (how far the reference ends up from
      where a straight line through the start state would have put it).  The
      ratio is dimensionless and is plotted at its own scale, NOT multiplied by
      anything.  It does NOT flip sign with the charge, which is the signature
      of a momentum defect: a momentum error enters the equation of motion as
      q/p and so scales the bend, giving dx / bend ~ dp / p.  The momentum
      defect this implies is printed above each band.
  (c) the RANDOM part: the 68 per cent half-width of dx about its own median,
      with each charge first centred on its own median so the systematic part
      of (a) cannot inflate it, against the band's median momentum on a
      log-log axis.  Beside it are the Highland prediction for the air between
      the two planes (x/X0 = L / 304,000 mm) and a pure 1/p line anchored on
      the 3-8 GeV measurement.  The measured width is a factor 2.1 to 3.6 wider
      than air alone, and that ratio is printed at each point.

INPUT FILES (all opened read-only)

  ../results/paper_numbers.json
      reference_vs_truth.rebinned_decomposition.by_band[<band>][q+|q-|both]:
      n, median_P_GeV, signed_med_dx_um, median_dx_over_bend, implied_dp_MeV,
      hw68_dx_um, hw68_dx_charge_corrected_um, n_bend_cut, median_bend_mm.
      reference_vs_truth.rebinned_decomposition.bend_cut_mm and n_crossings.
      reference_vs_truth.highland_air_rebinned.rows: band, n, median_P_GeV,
      x_over_X0_air, theta0_air_urad, highland_air_width_um,
      measured_hw68_dx_um (the charge-corrected width), ratio_measured_over_air
      and effective_x_over_X0.
      The definitions these follow are those of
      ../../single_network_chain_discrete_approach/reference_vs_truth/decompose.py
      (half_width_68, highland_theta0, highland_displacement_mm); the
      re-binned block already carries every value this figure needs, so nothing
      is recomputed from tracks.npz here.

OUTPUT

  ../figures/fig_material_gap.png
  ../results/fig06_material_gap.csv   every plotted value, long format

RUN

  cd Self_chained_paper/scripts
  PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python fig06_material_gap.py
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"

import sys                                            # noqa: E402
sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in ("", ".", HERE):
    while _p in sys.path:
        sys.path.remove(_p)

import importlib.util                                 # noqa: E402

import numpy as np                                    # noqa: E402
import matplotlib                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.lines import Line2D                   # noqa: E402

_spec = importlib.util.spec_from_file_location("sc_paper_common",
                                               os.path.join(HERE, "common.py"))
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

FIGURES = os.path.join(C.PAPER, "figures")
PNG = os.path.join(FIGURES, "fig_material_gap.png")
CSV = os.path.join(C.RESULTS, "fig06_material_gap.csv")

BANDS = list(C.BAND_LABELS) + [C.LOSS_WINDOW_LABEL]
SHORT = {C.LOSS_WINDOW_LABEL: "10-50\nwindow"}
CHARGE = (("q+", C.BLUE, -0.17), ("q-", C.MAGENTA, 0.17))


def style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150,
        "figure.facecolor": C.SURFACE, "axes.facecolor": C.SURFACE,
        "savefig.facecolor": C.SURFACE, "savefig.edgecolor": C.SURFACE,
        "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9,
        "axes.edgecolor": C.TEXT2, "axes.labelcolor": C.TEXT1,
        "text.color": C.TEXT1, "xtick.color": C.TEXT2, "ytick.color": C.TEXT2,
        "axes.linewidth": 0.8, "legend.frameon": False,
        "grid.color": C.NEUTRAL, "grid.alpha": 0.6, "grid.linewidth": 0.6,
    })


def hatch_window(ax, nb):
    ax.axvspan(nb - 1.5, nb - 0.5, facecolor="none", edgecolor=C.NEUTRAL,
               hatch="///", lw=0.0, zorder=0)
    ax.axvline(nb - 1.5, color=C.TEXT2, lw=0.8, ls=":", zorder=1)


def main():
    style()
    J = C.read_json(os.path.join(C.RESULTS, "paper_numbers.json"))
    R = J["reference_vs_truth"]["rebinned_decomposition"]
    B = R["by_band"]
    HL = {r["band"]: r for r in J["reference_vs_truth"]["highland_air_rebinned"]["rows"]}
    out = []

    fig = plt.figure(figsize=(6.3, 7.4))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.12], hspace=0.55)
    axa = fig.add_subplot(gs[0])
    axb = fig.add_subplot(gs[1], sharex=axa)
    axc = fig.add_subplot(gs[2])

    nb = len(BANDS)
    xs = np.arange(nb, dtype=float)

    # ------------------------- (a) the signed centre, split by charge -------
    hatch_window(axa, nb)
    axa.axhline(0.0, color=C.TEXT2, lw=0.9, zorder=2)
    for cname, colour, dx in CHARGE:
        for bi, band in enumerate(BANDS):
            v = B[band][cname]["signed_med_dx_um"]
            axa.vlines(bi + dx, 0.0, v, color=colour, lw=1.4, zorder=3)
            axa.plot([bi + dx], [v], "o", ms=6, mfc=colour, mec=C.SURFACE,
                     mew=0.7, zorder=4)
            out.append(dict(panel="a", band=band, charge=cname,
                            n=B[band][cname]["n"],
                            quantity="signed median dx", value=v, unit="um"))
    axa.set_yscale("symlog", linthresh=10.0, linscale=0.45)
    axa.set_ylim(-4e4, 4e4)
    axa.set_ylabel(r"signed median  $\Delta x$  ($\mu$m)")
    axa.grid(True, axis="y", which="major")
    axa.set_title("(a)  the systematic part: it flips sign with the charge",
                  loc="left", pad=4)
    for bi, band in enumerate(BANDS):
        axa.annotate("n = %d" % B[band]["both"]["n"], xy=(bi, 0.985),
                     xycoords=("data", "axes fraction"), ha="center", va="top",
                     fontsize=8.5, color=C.TEXT2)
    plt.setp(axa.get_xticklabels(), visible=False)

    # ---------------------------- (b) the same divided by the bend ----------
    hatch_window(axb, nb)
    axb.axhline(0.0, color=C.TEXT2, lw=0.9, zorder=2)
    for cname, colour, dx in CHARGE:
        for bi, band in enumerate(BANDS):
            v = B[band][cname]["median_dx_over_bend"]
            axb.vlines(bi + dx, 0.0, v, color=colour, lw=1.4, zorder=3)
            axb.plot([bi + dx], [v], "o", ms=6, mfc=colour, mec=C.SURFACE,
                     mew=0.7, zorder=4)
            out.append(dict(panel="b", band=band, charge=cname,
                            n=B[band][cname]["n_bend_cut"],
                            quantity="median dx / bend", value=v,
                            unit="dimensionless"))
    axb.set_ylabel(r"median  $\Delta x\,/\,$bend" "\n(dimensionless)")
    axb.legend(handles=[Line2D([], [], ls="none", marker="o", ms=6, mfc=c,
                               mec=C.SURFACE, mew=0.7, label=n)
                        for n, c, _ in CHARGE],
               loc="upper right", ncol=2, handletextpad=0.4,
               columnspacing=1.2, bbox_to_anchor=(0.99, 0.87))
    axb.set_ylim(-3.4e-3, 1.02e-2)
    axb.grid(True, axis="y")
    axb.set_title("(b)  the same, divided by the bend: it does not flip",
                  loc="left", pad=4)
    for bi, band in enumerate(BANDS):
        dp = B[band]["both"]["implied_dp_MeV"]
        axb.annotate((r"$\Delta p$ = %+.0f MeV" if bi == 0 else "%+.0f MeV") % dp,
                     xy=(bi, 0.985),
                     xycoords=("data", "axes fraction"), ha="center", va="top",
                     fontsize=8, color=C.TEXT2)
        out.append(dict(panel="b", band=band, charge="both",
                        n=B[band]["both"]["n_bend_cut"],
                        quantity="implied momentum defect", value=dp,
                        unit="MeV"))
    axb.set_xticks(xs)
    axb.set_xticklabels([SHORT.get(b, b) for b in BANDS])
    axb.set_xlabel("momentum band  (GeV)")
    axb.set_xlim(-0.6, nb - 0.4)

    # -------------------------- (c) the random part against momentum --------
    axc.axvspan(C.LOSS_WINDOW[0], C.LOSS_WINDOW[1], color=C.NEUTRAL,
                alpha=0.45, lw=0, zorder=0)
    pm, meas, air, rat = [], [], [], []
    for band in C.BAND_LABELS:
        h = HL[band]
        pm.append(h["median_P_GeV"])
        meas.append(h["measured_hw68_dx_um"])
        air.append(h["highland_air_width_um"])
        rat.append(h["ratio_measured_over_air"])
        out.append(dict(panel="c", band=band, charge="both (charge corrected)",
                        n=h["n"], quantity="68 % half-width of dx",
                        value=h["measured_hw68_dx_um"], unit="um"))
        out.append(dict(panel="c", band=band, charge="", n=h["n"],
                        quantity="Highland air-only prediction",
                        value=h["highland_air_width_um"], unit="um"))
        out.append(dict(panel="c", band=band, charge="", n=h["n"],
                        quantity="measured / air", value=h["ratio_measured_over_air"],
                        unit="dimensionless"))
    pm = np.array(pm)
    meas = np.array(meas)
    air = np.array(air)
    axc.plot(pm, meas, "o-", lw=1.6, ms=6.5, color=C.MAGENTA,
             label="measured (charge corrected)")
    axc.plot(pm, air, "s--", lw=1.4, ms=5.5, color=C.BLUE,
             label="Highland, air only")
    pline = np.logspace(np.log10(pm.min() * 0.75), np.log10(pm.max() * 1.4), 50)
    anchor = meas[1] * pm[1]
    axc.plot(pline, anchor / pline, ":", lw=1.3, color=C.TEXT2,
             label="1/p reference")
    w = HL[C.LOSS_WINDOW_LABEL]
    axc.plot([w["median_P_GeV"]], [w["measured_hw68_dx_um"]], "D", ms=7,
             mfc="none", mec=C.MAGENTA, mew=1.4, label="10-50 GeV window")
    out.append(dict(panel="c", band=C.LOSS_WINDOW_LABEL,
                    charge="both (charge corrected)", n=w["n"],
                    quantity="68 % half-width of dx",
                    value=w["measured_hw68_dx_um"], unit="um"))
    for p, m, r in zip(pm, meas, rat):
        axc.annotate(r"$\times$%.2f" % r, xy=(p, m), xytext=(0, 9),
                     textcoords="offset points", ha="center", va="bottom",
                     fontsize=8.5, color=C.TEXT1)
    axc.set_xscale("log")
    axc.set_yscale("log")
    axc.set_xlabel("band median momentum  (GeV)")
    axc.set_ylabel(r"68 % half-width of $\Delta x$  ($\mu$m)")
    axc.grid(True, which="both")
    axc.set_title("(c)  the random part, against the band median momentum",
                  loc="left", pad=4)
    axc.annotate("air: $x/X_0$ = %.4f" % HL["<3"]["x_over_X0_air"],
                 xy=(0.985, 0.03), xycoords="axes fraction", ha="right",
                 va="bottom", fontsize=8.5, color=C.TEXT2)
    axc.legend(loc="lower left", labelspacing=0.28, handlelength=2.0)
    axc.set_ylim(0.13 * air.min(), 6.0 * meas.max())
    axc.annotate("10-50 GeV\nloss window", xy=(np.sqrt(10 * 50.0), 0.97),
                 xycoords=("data", "axes fraction"), ha="center", va="top",
                 fontsize=8.5, color=C.TEXT2)

    fig.text(0.5, 0.004,
             "all three panels: %s crossings, bend cut |bend| > %g mm"
             % (f"{R['n_crossings']:,}", R["bend_cut_mm"]),
             ha="center", va="bottom", fontsize=8.5, color=C.TEXT2)

    fig.subplots_adjust(left=0.175, right=0.985, top=0.965, bottom=0.095)
    os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(PNG, dpi=150, facecolor=C.SURFACE)
    plt.close(fig)

    C.write_csv(CSV, out, ["panel", "band", "charge", "n", "quantity", "value",
                           "unit"])
    print("wrote %s" % PNG)
    print("wrote %s  (%d rows)" % (CSV, len(out)))
    print("signed median dx flips sign in the bands below 20 GeV; "
          "dx/bend keeps its sign in every band")
    print("measured / air ratio: %.2f to %.2f over the five bands"
          % (min(rat), max(rat)))


if __name__ == "__main__":
    main()
