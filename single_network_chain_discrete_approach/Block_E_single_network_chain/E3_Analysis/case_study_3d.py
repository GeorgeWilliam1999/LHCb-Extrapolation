#!/usr/bin/env python
"""E3 - the case-study figure in three dimensions: the starting x as a third axis.

`case_study.py` draws each component's error against momentum (top row) and the
signed error in the 10-20 GeV band (bottom row). This script redraws all eight
panels with the track's STARTING x - its x on the last UT plane, z0 - as the
third dimension:

  top row     a surface over the (momentum, starting x) plane whose height is
              the MEDIAN |error| of the tracks in that cell (log height); cells
              with fewer than MIN_TRACKS tracks are left empty;
  bottom row  the 10-20 GeV band as a two-dimensional histogram over (signed
              error, starting x), drawn as bars whose height is the number of
              tracks.

A flat version of the same numbers is written beside it, because a surface seen
from one angle hides cells behind ridges; the flat maps are the ones to read
values off, and the table behind both is `results/case_study_error_vs_p_x0.csv`.

Run:     PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python case_study_3d.py [--N 64 --q 2]
Outputs: figures/case_study_components_3d.png, figures/case_study_components_x0_maps.png,
         results/case_study_error_vs_p_x0.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
E1 = os.path.join(HERE, "..", "E1_Network_grid", "results")
TRACKS = os.path.join(HERE, "..", "E0_Track_dataset", "results", "tracks.npz")
SPLIT = "test"
COMPONENTS = (("x", 0, 1e3, "µm"), ("y", 1, 1e3, "µm"), ("tx", 2, 1e3, "mrad"), ("ty", 3, 1e3, "mrad"))
P_EDGES = np.array([1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100, 200.0])
X0_EDGES = np.array([-700, -400, -250, -150, -75, 0, 75, 150, 250, 400, 700.0])
BAND = (10.0, 20.0)
MIN_TRACKS = 5


def pick_best():
    rows = list(csv.DictReader(open(os.path.join(HERE, "results", "convergence.csv"))))
    b = min(rows, key=lambda r: float(r["val_last8_mean"]))
    return int(b["N"]), int(b["q"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--N", type=int, default=None)
    ap.add_argument("--q", type=int, default=None)
    a = ap.parse_args(argv)
    bN, bq = pick_best()
    N, q = (a.N or bN), (a.q or bq)
    run = os.path.join(E1, "N%03d_q%02d" % (N, q))
    rec = json.load(open(os.path.join(run, "record.json")))

    D = np.load(TRACKS)
    n_max = int(D["n_max"])
    truth_end = np.asarray(D["%s_truth" % SPLIT])[:, n_max]
    S0 = np.asarray(D["%s_S0" % SPLIT])
    x0 = S0[:, 0]
    P = np.asarray(D["%s_P" % SPLIT])
    end = np.load(os.path.join(run, "chain_states.npz"))["%s_states" % SPLIT][:, -1]
    d = end[:, :4] - truth_end[:, :4]
    band = (P >= BAND[0]) & (P <= BAND[1])

    # -- the table behind both figures ------------------------------------------
    rows = []
    med = {}
    for name, i, scale, unit in COMPONENTS:
        v = np.abs(d[:, i]) * scale
        grid = np.full((len(P_EDGES) - 1, len(X0_EDGES) - 1), np.nan)
        cnt = np.zeros_like(grid)
        for ip, (plo, phi) in enumerate(zip(P_EDGES[:-1], P_EDGES[1:])):
            for ix, (xlo, xhi) in enumerate(zip(X0_EDGES[:-1], X0_EDGES[1:])):
                m = (P >= plo) & (P < phi) & (x0 >= xlo) & (x0 < xhi)
                cnt[ip, ix] = m.sum()
                if m.sum() >= MIN_TRACKS:
                    grid[ip, ix] = np.median(v[m])
                rows.append(dict(component=name, unit=unit, p_lo=plo, p_hi=phi, x0_lo=xlo, x0_hi=xhi,
                                 n=int(m.sum()),
                                 med=float(np.median(v[m])) if m.sum() else float("nan"),
                                 signed_median=float(np.median(d[m, i]) * scale) if m.sum() else float("nan")))
        med[name] = (grid, cnt)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "case_study_error_vs_p_x0.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize
    title = ("Block E case study in 3D: N = %d, q = %d (dz = %.1f mm), %d test tracks, "
             "starting x as the third axis" % (N, q, rec["dz_mm"], len(P)))
    pc = np.sqrt(P_EDGES[:-1] * P_EDGES[1:])
    xc = 0.5 * (X0_EDGES[:-1] + X0_EDGES[1:])
    PP, XX = np.meshgrid(np.log10(pc), xc, indexing="ij")

    fig = plt.figure(figsize=(21, 10))
    for j, (name, i, scale, unit) in enumerate(COMPONENTS):
        grid, cnt = med[name]
        ax = fig.add_subplot(2, 4, j + 1, projection="3d")
        Z = np.log10(grid)
        norm = Normalize(np.nanmin(Z), np.nanmax(Z))
        ax.plot_surface(PP, XX, np.ma.masked_invalid(Z), facecolors=cm.viridis_r(norm(Z)),
                        rstride=1, cstride=1, linewidth=0.2, shade=False, antialiased=True)
        ax.set_xlabel("momentum [GeV]", fontsize=8, labelpad=2)
        ax.set_ylabel("starting x at z0 [mm]", fontsize=8, labelpad=2)
        ax.set_zlabel("median |%s error| [%s]" % (name, unit), fontsize=8, labelpad=2)
        ticks = np.array([1, 3, 10, 30, 100])
        ax.set_xticks(np.log10(ticks), [str(t) for t in ticks], fontsize=7)
        ax.tick_params(labelsize=7)
        zt = np.arange(np.floor(np.nanmin(Z)), np.ceil(np.nanmax(Z)) + 0.1)
        ax.set_zticks(zt, ["%g" % 10 ** t for t in zt], fontsize=7)
        ax.view_init(elev=26, azim=-131)
        ax.set_title("%s: median |error| over (momentum, starting x)" % name, fontsize=9)

    for j, (name, i, scale, unit) in enumerate(COMPONENTS):
        ax = fig.add_subplot(2, 4, j + 5, projection="3d")
        sgn = d[band, i] * scale
        lim = np.quantile(np.abs(sgn), 0.98)
        e_edges = np.linspace(-lim, lim, 13)
        xb = np.array([-700, -250, -75, 75, 250, 700.0])
        H, _, _ = np.histogram2d(np.clip(sgn, -lim, lim), x0[band], bins=[e_edges, xb])
        ec = 0.5 * (e_edges[:-1] + e_edges[1:])
        xcb = 0.5 * (xb[:-1] + xb[1:])
        EE, XB = np.meshgrid(ec, xcb, indexing="ij")
        dx = (e_edges[1] - e_edges[0]) * 0.85
        dy = np.diff(xb).mean() * 0.5
        norm = Normalize(0, H.max())
        ax.bar3d(EE.ravel() - dx / 2, XB.ravel() - dy / 2, np.zeros(H.size), dx, dy, H.ravel(),
                 color=cm.viridis(norm(H.ravel())), shade=True)
        ax.set_xlabel("signed %s error [%s]" % (name, unit), fontsize=8, labelpad=2)
        ax.set_ylabel("starting x at z0 [mm]", fontsize=8, labelpad=2)
        ax.set_zlabel("tracks", fontsize=8, labelpad=2)
        ax.tick_params(labelsize=7)
        ax.view_init(elev=24, azim=-128)
        ax.set_title("%s, %g-%g GeV (%d tracks)" % (name, BAND[0], BAND[1], band.sum()), fontsize=9)
    fig.suptitle(title + " — top: median |error|; bottom: the 10-20 GeV band, signed", fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "figures", "case_study_components_3d.png"), dpi=110)
    plt.close(fig)

    # -- the flat version of the same numbers -------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(20, 8.5))
    for j, (name, i, scale, unit) in enumerate(COMPONENTS):
        grid, cnt = med[name]
        ax = axes[0, j]
        pm = ax.pcolormesh(P_EDGES, X0_EDGES, np.ma.masked_invalid(grid).T, cmap="viridis_r",
                           norm=matplotlib.colors.LogNorm())
        fig.colorbar(pm, ax=ax, pad=0.02).set_label("median |%s error| [%s]" % (name, unit), fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("momentum [GeV]", fontsize=9)
        ax.set_ylabel("starting x at z0 [mm]", fontsize=9)
        ax.set_title("%s: median |error| (cells with < %d tracks blank)" % (name, MIN_TRACKS), fontsize=9)
        ax = axes[1, j]
        sgn = d[band, i] * scale
        lim = np.quantile(np.abs(sgn), 0.98)
        e_edges = np.linspace(-lim, lim, 13)
        xb = np.array([-700, -250, -75, 75, 250, 700.0])
        H, _, _ = np.histogram2d(np.clip(sgn, -lim, lim), x0[band], bins=[e_edges, xb])
        pm = ax.pcolormesh(e_edges, xb, H.T, cmap="viridis")
        fig.colorbar(pm, ax=ax, pad=0.02).set_label("tracks", fontsize=8)
        ax.axvline(0, color="w", lw=1)
        ax.set_xlabel("signed %s error, %g-%g GeV [%s]" % (name, BAND[0], BAND[1], unit), fontsize=9)
        ax.set_ylabel("starting x at z0 [mm]", fontsize=9)
        ax.set_title("%s in the band (%d tracks)" % (name, band.sum()), fontsize=9)
    fig.suptitle(title + " — the same numbers, flat", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "case_study_components_x0_maps.png"), dpi=120)
    print("wrote figures/case_study_components_3d.png, case_study_components_x0_maps.png, "
          "results/case_study_error_vs_p_x0.csv")

    # what the extra axis actually shows
    for name, i, scale, unit in COMPONENTS:
        grid, cnt = med[name]
        prof = np.array([np.nanmedian(grid[:, ix]) for ix in range(grid.shape[1])])
        print("  %-3s median |error| against starting x: " % name
              + "  ".join("%+5.0f mm: %.3g" % (xc[ix], prof[ix]) for ix in range(len(xc)) if np.isfinite(prof[ix])))


if __name__ == "__main__":
    main()
