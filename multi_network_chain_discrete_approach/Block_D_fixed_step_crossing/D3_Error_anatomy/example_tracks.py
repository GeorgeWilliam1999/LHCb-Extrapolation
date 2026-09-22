#!/usr/bin/env python
"""D3 - individual tracks: 10 to 20 GeV, in the middle of the acceptance.

The selection is the test particles with 10 <= p <= 20 GeV and 3 < eta < 4,
the central band of the acceptance (away from the beam line and the outer
edges). Four of them are shown: the tracks nearest the 25th, 50th, 75th and
95th percentile of the best single step's position error within the
selection, so that the examples run from a good track to a bad one rather than
being hand-picked.

For each track: who it is, its start state on z0, the RK6 state on z1, how
far the field deflects it in x and in y, and every best chain's error on z1
in x, y, tx and ty, beside the exact collocation scheme at the same N and q.
Along the crossing: every chain's signed x and y error on each of its planes.

Run:
    PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python example_tracks.py

Outputs:
    results/example_tracks.csv            one row per (track, chain) at z1
    results/example_tracks_along_z.csv    one row per (track, chain, plane)
    figures/example_tracks_along_z.png    signed x and y error against z
"""
import csv
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BD = os.path.abspath(os.path.join(HERE, ".."))
OUT, FIG = os.path.join(HERE, "results"), os.path.join(HERE, "figures")
CHAINS = [(1, 16), (4, 5), (16, 15), (64, 7), (128, 7)]
QUANTILES = (0.25, 0.50, 0.75, 0.95)
NAMES = {211: "pion", 321: "kaon", 2212: "proton", 13: "muon"}


def main():
    d = np.load(os.path.join(BD, "D0_Crossing_dataset", "results", "crossing_particles.npz"))
    Z0, Z1, planes = float(d["z0"]), float(d["z1"]), d["planes"]
    S0, T = d["test_S0"], d["test_truth"]
    P, ETA, PID, EVT, MCK = d["test_P"], d["test_ETA"], d["test_PID"], d["test_EVT"], d["test_MCKEY"]
    sel = np.flatnonzero((P >= 10) & (P <= 20) & (ETA > 3) & (ETA < 4))
    net = {c: np.load(os.path.join(BD, "D1_Chain_grid", "results", "N%03d_q%02d" % c,
                                   "states.npz"))["test_states"] for c in CHAINS}
    exact = {c: np.load(os.path.join(BD, "D2_Comparators", "results",
                                     "exact_N%03d_q%02d_states.npz" % c))["states"] for c in CHAINS}
    e1 = net[CHAINS[0]][:, -1] - T[:, -1]
    pos1 = np.maximum(np.abs(e1[:, 0]), np.abs(e1[:, 1])) * 1e3
    picks = []
    for qq in QUANTILES:
        target = np.quantile(pos1[sel], qq)
        i = sel[np.argmin(np.abs(pos1[sel] - target))]
        picks.append((qq, int(i)))
    print("selection: %d test tracks; |x1| <= %.0f mm, |y1| <= %.0f mm"
          % (len(sel), np.abs(T[sel, -1, 0]).max(), np.abs(T[sel, -1, 1]).max()))

    L = Z1 - Z0
    rows, along = [], []
    for qq, i in picks:
        base = dict(
            example="p%02d" % int(qq * 100), evt=int(EVT[i]), mckey=int(MCK[i]),
            species=NAMES.get(abs(int(PID[i])), str(int(PID[i]))), charge=int(np.sign(S0[i, 4])),
            p_gev=float(P[i]), eta=float(ETA[i]),
            x0_mm=S0[i, 0], y0_mm=S0[i, 1], tx0_mrad=S0[i, 2] * 1e3, ty0_mrad=S0[i, 3] * 1e3,
            x1_mm=T[i, -1, 0], y1_mm=T[i, -1, 1], tx1_mrad=T[i, -1, 2] * 1e3, ty1_mrad=T[i, -1, 3] * 1e3,
            field_bend_x_mm=T[i, -1, 0] - (S0[i, 0] + S0[i, 2] * L),
            field_bend_y_mm=T[i, -1, 1] - (S0[i, 1] + S0[i, 3] * L),
            field_dtx_mrad=(T[i, -1, 2] - S0[i, 2]) * 1e3, field_dty_mrad=(T[i, -1, 3] - S0[i, 3]) * 1e3,
        )
        for (N, q) in CHAINS:
            stride = (len(planes) - 1) // N
            e = net[(N, q)][i, -1] - T[i, -1]
            ex = exact[(N, q)][i, -1] - T[i, -1]
            r = dict(base, N=N, q=q, dx_um=e[0] * 1e3, dy_um=e[1] * 1e3,
                     dtx_mrad=e[2] * 1e3, dty_mrad=e[3] * 1e3,
                     exact_pos_um=max(abs(ex[0]), abs(ex[1])) * 1e3)
            rows.append(r)
            for k in range(N + 1):
                ek = net[(N, q)][i, k] - T[i, k * stride]
                along.append(dict(example=base["example"], evt=base["evt"], mckey=base["mckey"],
                                  N=N, q=q, plane=k, z_mm=float(planes[k * stride]),
                                  dx_um=ek[0] * 1e3, dy_um=ek[1] * 1e3,
                                  dtx_mrad=ek[2] * 1e3, dty_mrad=ek[3] * 1e3))
    os.makedirs(OUT, exist_ok=True)
    for path, rr in (("example_tracks.csv", rows), ("example_tracks_along_z.csv", along)):
        with open(os.path.join(OUT, path), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rr[0].keys()))
            w.writeheader()
            w.writerows(rr)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, len(picks), figsize=(4.2 * len(picks), 6.4), sharex=True)
    colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for j, (qq, i) in enumerate(picks):
        for (N, q), col in zip(CHAINS, colours):
            a = [r for r in along if r["evt"] == int(EVT[i]) and r["mckey"] == int(MCK[i]) and r["N"] == N]
            z = [r["z_mm"] for r in a]
            ax[0, j].plot(z, [r["dx_um"] for r in a], "-o", color=col, ms=3 if N <= 16 else 0, lw=1.2,
                          label="N = %d, q = %d" % (N, q))
            ax[1, j].plot(z, [r["dy_um"] for r in a], "-o", color=col, ms=3 if N <= 16 else 0, lw=1.2)
        b = [r for r in rows if r["evt"] == int(EVT[i]) and r["mckey"] == int(MCK[i])][0]
        ax[0, j].set_title("%s percentile: %s %+d, %.1f GeV, eta %.2f\nfield bend x %.0f mm, y %.2f mm"
                           % (b["example"][1:] + "th", b["species"], b["charge"], b["p_gev"], b["eta"],
                              b["field_bend_x_mm"], b["field_bend_y_mm"]), fontsize=9)
        for r_ in (0, 1):
            ax[r_, j].axhline(0, color="0.6", lw=0.8)
            ax[r_, j].grid(alpha=0.3)
        ax[1, j].set_xlabel("z [mm]")
    ax[0, 0].set_ylabel("x error against RK6 [um]")
    ax[1, 0].set_ylabel("y error against RK6 [um]")
    ax[0, 0].legend(fontsize=8)
    fig.suptitle("Test tracks, 10-20 GeV, 3 < eta < 4: signed error on every plane of the best chain at each N",
                 fontsize=11)
    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, "example_tracks_along_z.png"), dpi=130)
    print("picks:", picks)


if __name__ == "__main__":
    main()
