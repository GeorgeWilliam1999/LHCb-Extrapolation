#!/usr/bin/env python
"""Figure 1 - the crossing the network has to cover, in three registers.

WHAT THE FIGURE SHOWS

  (top)     |B| on the beam axis (x = y = 0) between the last UT plane
            z0 = 2648.2 mm and the first SciFi plane z1 = 7826.0 mm, read from
            the LHCb v8r1 magnet-UP field map.  This is the same curve
            `E0_Track_dataset/build_tracks.py` draws in its overview figure.
  (middle)  the plane grids of the three chains used in the paper, N = 64, 128
            and 256, drawn as ticks at z0 + k dz with dz labelled.  The N = 256
            grid is the 257-plane grid every truth track is stored on.
  (bottom)  x(z) for six real test tracks - the nearest track to 3, 12 and
            45 GeV in each charge - taken from the stored RK6 truth paths on
            the 257-plane grid, to scale, in mm.  The 1/p bend is the whole
            reason a single step across the magnet cannot work.

INPUT FILES (all opened read-only)

  ../results/paper_numbers.json
      dataset.geometry (z0, z1, L, the 257 planes, dz per N) and
      dataset.field_map (which polarity, the map path and its md5).
  /cvmfs/lhcb.cern.ch/lib/lhcb/DBASE/FieldMap/v8r1/cdf/field.v8r1.up.bin
      the field map itself, loaded through
      ../../single_network_chain_discrete_approach/_shared/field_v8r1.py
      (imported by file location; nothing in that folder is written).
  ../../single_network_chain_discrete_approach/Block_E_single_network_chain/
      E0_Track_dataset/results/tracks.npz
      test_P, test_S0 (for the charge, carried in q/p) and test_truth
      (1452, 257, 5) for the paths.

OUTPUT

  ../figures/fig_crossing.png
  ../results/fig01_crossing.csv   z, |B| and the six x(z) paths on the 257 planes

RUN

  cd Self_chained_paper/scripts
  PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python fig01_crossing.py
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
# `numbers.py` lives beside this file and shadows the standard-library module
# numpy imports while it initialises, so the script directory is dropped from
# the import path before numpy is loaded and common.py is loaded by location.
for _p in ("", ".", HERE):
    while _p in sys.path:
        sys.path.remove(_p)

import csv                                            # noqa: E402
import importlib.util                                 # noqa: E402
import json                                           # noqa: E402

import numpy as np                                    # noqa: E402
import matplotlib                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402

_spec = importlib.util.spec_from_file_location("sc_paper_common",
                                               os.path.join(HERE, "common.py"))
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

FIGURES = os.path.join(C.PAPER, "figures")
PNG = os.path.join(FIGURES, "fig_crossing.png")
CSV = os.path.join(C.RESULTS, "fig01_crossing.csv")

N_SHOWN = (64, 128, 256)
TARGET_P = (3.0, 12.0, 45.0)


def style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150,
        "figure.facecolor": C.SURFACE, "axes.facecolor": C.SURFACE,
        "savefig.facecolor": C.SURFACE, "savefig.edgecolor": C.SURFACE,
        "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9.5,
        "axes.edgecolor": C.TEXT2, "axes.labelcolor": C.TEXT1,
        "text.color": C.TEXT1, "xtick.color": C.TEXT2, "ytick.color": C.TEXT2,
        "axes.linewidth": 0.8, "legend.frameon": False,
        "grid.color": C.NEUTRAL, "grid.alpha": 0.6, "grid.linewidth": 0.6,
    })


def load_field(which):
    """The vendored v8r1 loader, imported by file location (read-only)."""
    spec = importlib.util.spec_from_file_location(
        "sc_field_v8r1", os.path.join(C.SHARED, "field_v8r1.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    path = ("/cvmfs/lhcb.cern.ch/lib/lhcb/DBASE/FieldMap/v8r1/cdf/field.v8r1.%s.bin"
            % which)
    return mod.FieldV8R1(path), path


def pick_tracks(P, S0):
    """(target_p, charge) -> row index of the nearest test track in |p|."""
    q = np.sign(S0[:, 4])
    picks = []
    for p in TARGET_P:
        for sign, name in ((+1.0, "q+"), (-1.0, "q-")):
            m = np.flatnonzero(q == sign)
            i = m[int(np.argmin(np.abs(P[m] - p)))]
            picks.append((p, name, int(i), float(P[i])))
    return picks


def main():
    style()
    J = C.read_json(os.path.join(C.RESULTS, "paper_numbers.json"))
    geom = J["dataset"]["geometry"]
    z0, z1, L = geom["z0_mm"], geom["z1_mm"], geom["L_mm"]
    which = J["dataset"]["field_map"]["which"]

    fld, field_path = load_field(which)
    D = C.split_arrays("test")
    planes = np.asarray(C.load_tracks()["planes"])
    assert planes.size == 257 and abs(planes[0] - z0) < 1e-6
    assert abs(planes[-1] - z1) < 1e-6

    zz = np.linspace(z0, z1, 2001)
    Bx, By, Bz = fld(np.zeros_like(zz), np.zeros_like(zz), zz)
    Bmag = np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2)

    picks = pick_tracks(D["P"], D["S0"])

    fig = plt.figure(figsize=(6.3, 6.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.05, 0.62, 1.35], hspace=0.16)
    axB = fig.add_subplot(gs[0])
    axG = fig.add_subplot(gs[1], sharex=axB)
    axX = fig.add_subplot(gs[2], sharex=axB)

    # ---------------------------------------------------------------- |B| --
    axB.plot(zz, Bmag, "-", lw=1.6, color=C.TEXT1)
    axB.fill_between(zz, 0, Bmag, color=C.BLUE, alpha=0.10, lw=0)
    axB.set_ylabel("|B| on the axis  (T)")
    axB.set_ylim(0, 1.30 * Bmag.max())
    axB.grid(True, axis="y")
    kpeak = int(np.argmax(Bmag))
    axB.annotate("peak %.3f T at z = %.0f mm" % (Bmag[kpeak], zz[kpeak]),
                 xy=(zz[kpeak], Bmag[kpeak]), xytext=(0, 5),
                 textcoords="offset points", ha="center", va="bottom",
                 fontsize=9, color=C.TEXT2)
    axB.set_title("v8r1 magnet-%s field map, %d test tracks stored on 257 planes"
                  % (which, D["P"].size), color=C.TEXT2, pad=6)

    # ------------------------------------------------------- the plane grids --
    for i, N in enumerate(N_SHOWN):
        zs = z0 + np.arange(N + 1) * (L / N)
        axG.plot(zs, np.full_like(zs, float(i)), "|", ms=9 if N < 128 else 6,
                 mew=0.8 if N < 128 else 0.5, color=C.BLUE)
    axG.set_yticks(range(len(N_SHOWN)))
    axG.set_yticklabels([r"$N$ = %d" % N for N in N_SHOWN])
    axG.set_ylim(-0.6, len(N_SHOWN) - 0.4)
    for i, N in enumerate(N_SHOWN):
        axG.annotate(r"$\Delta z$ = %.1f mm" % (L / N), xy=(z1 - 140, i),
                     ha="right", va="center", fontsize=9, color=C.TEXT2,
                     bbox=dict(fc=C.SURFACE, ec="none", pad=1.2))
    axG.grid(True, axis="x")
    axG.tick_params(axis="y", length=0)

    # ------------------------------------------------------------- the tracks --
    colours = {3.0: C.MAGENTA, 12.0: C.BLUE, 45.0: C.GREEN}
    rows_x = {}
    for p, cname, i, ptrue in picks:
        xz = D["truth"][i, :, 0]
        rows_x["%.0fGeV_%s" % (p, cname)] = xz
        axX.plot(planes, xz, "-" if cname == "q+" else "--", lw=1.5,
                 color=colours[p],
                 label="%.1f GeV, %s" % (ptrue, cname))
    axX.axhline(0.0, color=C.NEUTRAL, lw=0.8, zorder=0)
    axX.set_ylabel("x  (mm)")
    axX.set_xlabel("z  (mm)")
    axX.grid(True)
    axX.legend(ncol=3, loc="upper left", handlelength=2.2, columnspacing=1.2,
               borderaxespad=0.3)

    # ---------------------------------------------- z0 / z1 on every panel --
    for ax in (axB, axG, axX):
        for z, txt in ((z0, "z0"), (z1, "z1")):
            ax.axvline(z, color=C.TEXT2, lw=0.9, ls=":", zorder=0)
        ax.set_xlim(z0 - 90, z1 + 90)
    axB.annotate("$z_0$ = %.1f mm\nlast UT plane" % z0, xy=(z0, 0),
                 xytext=(6, 4), textcoords="offset points", ha="left",
                 va="bottom", fontsize=9, color=C.TEXT2)
    axB.annotate("$z_1$ = %.1f mm\nfirst SciFi plane" % z1, xy=(z1, 0),
                 xytext=(-6, 4), textcoords="offset points", ha="right",
                 va="bottom", fontsize=9, color=C.TEXT2)
    axG.annotate(r"$L$ = %.1f mm" % L, xy=(0.5 * (z0 + z1), -0.58),
                 ha="center", va="bottom", fontsize=9, color=C.TEXT2,
                 bbox=dict(fc=C.SURFACE, ec="none", pad=1.2))
    for ax in (axB, axG):
        plt.setp(ax.get_xticklabels(), visible=False)

    fig.subplots_adjust(left=0.135, right=0.985, top=0.935, bottom=0.075)
    os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(PNG, dpi=150, facecolor=C.SURFACE)
    plt.close(fig)

    # ------------------------------------------------------------ the CSV --
    Bp = np.sqrt(np.sum(np.asarray(fld(np.zeros_like(planes),
                                       np.zeros_like(planes), planes)) ** 2,
                        axis=0))
    names = list(rows_x)
    rows = []
    for k in range(planes.size):
        r = {"plane": k, "z_mm": float(planes[k]), "B_T_on_axis": float(Bp[k])}
        for nm in names:
            r["x_mm_" + nm] = float(rows_x[nm][k])
        rows.append(r)
    C.write_csv(CSV, rows)

    print("wrote %s" % PNG)
    print("wrote %s" % CSV)
    print("field: %s" % field_path)
    print("tracks picked (target GeV, charge, row, true p):")
    for p, cname, i, ptrue in picks:
        print("   %5.1f  %s  row %5d  p = %7.3f GeV" % (p, cname, i, ptrue))
    print("dz: " + ", ".join("N=%d -> %.4f mm" % (N, L / N) for N in N_SHOWN))
    print("|B| peak %.4f T at z = %.1f mm" % (Bmag.max(), zz[int(np.argmax(Bmag))]))


if __name__ == "__main__":
    main()
