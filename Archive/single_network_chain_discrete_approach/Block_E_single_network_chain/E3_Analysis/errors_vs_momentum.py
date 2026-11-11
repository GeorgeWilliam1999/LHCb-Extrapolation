#!/usr/bin/env python
"""E3 - every Block E network's endpoint error against momentum, per component.

At the end plane z1 the error of a track is four numbers: dx, dy in µm and
dtx, dty in mrad, each against the RK6 track. This script draws them against
the track's momentum for all 16 networks - one figure per component, a 4 x 4
panel grid (step count down, stage count across), each panel a two-dimensional
histogram with the running median and the 95th percentile drawn on top - and
writes the medians per momentum bin as a table.

Run:     PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python errors_vs_momentum.py
Outputs: figures/error_vs_p_{x,y,tx,ty}.png, results/error_vs_p.csv
"""
from __future__ import annotations

import csv
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
E1 = os.path.join(HERE, "..", "E1_Network_grid", "results")
TRACKS = os.path.join(HERE, "..", "E0_Track_dataset", "results", "tracks.npz")
SPLIT = "test"
COMPONENTS = (("x", 0, 1e3, "µm"), ("y", 1, 1e3, "µm"), ("tx", 2, 1e3, "mrad"), ("ty", 3, 1e3, "mrad"))
P_EDGES = np.array([1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100, 200.0])


def main():
    D = np.load(TRACKS)
    n_max = int(D["n_max"])
    truth_end = np.asarray(D["%s_truth" % SPLIT])[:, n_max]
    P = np.asarray(D["%s_P" % SPLIT])
    runs = sorted(glob.glob(os.path.join(E1, "N[0-9][0-9][0-9]_q[0-9][0-9]")))
    err, meta = {}, {}
    for run in runs:
        rec = json.load(open(os.path.join(run, "record.json")))
        end = np.load(os.path.join(run, "chain_states.npz"))["%s_states" % SPLIT][:, -1]
        err[(rec["N"], rec["q"])] = end[:, :4] - truth_end[:, :4]
        meta[(rec["N"], rec["q"])] = rec
    Ns = sorted({k[0] for k in err})
    qs = sorted({k[1] for k in err})

    rows = []
    for (N, q), d in sorted(err.items()):
        for name, i, scale, unit in COMPONENTS:
            a = np.abs(d[:, i]) * scale
            for lo, hi in zip(P_EDGES[:-1], P_EDGES[1:]):
                m = (P >= lo) & (P < hi)
                if m.sum() < 5:
                    continue
                rows.append(dict(N=N, q=q, component=name, unit=unit, p_lo=lo, p_hi=hi,
                                 n=int(m.sum()), med=float(np.median(a[m])),
                                 p95=float(np.quantile(a[m], 0.95)),
                                 signed_mean=float(np.mean(d[m, i]) * scale)))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "error_vs_p.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    centres = np.sqrt(P_EDGES[:-1] * P_EDGES[1:])
    for name, i, scale, unit in COMPONENTS:
        fig, ax = plt.subplots(len(Ns), len(qs), figsize=(4.0 * len(qs), 3.1 * len(Ns)),
                               sharex=True, sharey=True)
        vals = np.concatenate([np.abs(d[:, i]) * scale for d in err.values()])
        hi = np.quantile(vals, 0.995)
        for r, N in enumerate(Ns):
            for c, q in enumerate(qs):
                a = ax[r, c]
                d = err[(N, q)]
                v = np.abs(d[:, i]) * scale
                hb = a.hexbin(P, np.clip(v, 1e-4 * hi, hi), xscale="log", yscale="log", gridsize=34,
                              mincnt=1, norm=LogNorm(), cmap="Blues")
                cb = fig.colorbar(hb, ax=a, pad=0.02)
                cb.set_label("test tracks per cell", fontsize=7)
                cb.ax.tick_params(labelsize=6)
                med = [np.median(v[(P >= lo) & (P < h2)]) if ((P >= lo) & (P < h2)).sum() > 4 else np.nan
                       for lo, h2 in zip(P_EDGES[:-1], P_EDGES[1:])]
                p95 = [np.quantile(v[(P >= lo) & (P < h2)], 0.95) if ((P >= lo) & (P < h2)).sum() > 4 else np.nan
                       for lo, h2 in zip(P_EDGES[:-1], P_EDGES[1:])]
                a.plot(centres, med, "r-o", ms=3, lw=1.4, label="median per momentum bin")
                a.plot(centres, p95, "r--", lw=1, label="95th percentile")
                a.set_title("N = %d, q = %d   (median %.3g %s)"
                            % (N, q, np.median(v), unit), fontsize=9)
                a.grid(alpha=0.25, which="both")
                if r == len(Ns) - 1:
                    a.set_xlabel("momentum [GeV]")
                if c == 0:
                    a.set_ylabel("|%s error| [%s]" % (name, unit))
        ax[0, 0].legend(fontsize=7)
        fig.suptitle("Block E: |%s| error at the SciFi plane against momentum. Each hexagon is coloured by how many of the 1,452 test tracks fall in it (log scale)." % name,
                     fontsize=12)
        fig.tight_layout()
        os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
        fig.savefig(os.path.join(HERE, "figures", "error_vs_p_%s.png" % name), dpi=120)
        plt.close(fig)
        print("wrote figures/error_vs_p_%s.png" % name)
    print("wrote results/error_vs_p.csv")


if __name__ == "__main__":
    main()
