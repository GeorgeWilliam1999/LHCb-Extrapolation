#!/usr/bin/env python
"""E3 - where along the crossing the chain error is built up.

Every chain's states are saved on all of its planes, so the radial error
against the RK6 track can be read off plane by plane: the growth curve. Read
beside `single_step_vs_z.csv` (what one step costs where) it separates the two
things that set the endpoint error - how good a step is, and how far the errors
it leaves have still to be carried.

Run:     PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python along_z.py
Outputs: results/error_vs_z.csv, figures/error_vs_z.png
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


def main():
    D = np.load(TRACKS)
    Z0, L, n_max = float(D["z0"]), float(D["L"]), int(D["n_max"])
    truth = np.asarray(D["%s_truth" % SPLIT])
    rows = []
    for run in sorted(glob.glob(os.path.join(E1, "N[0-9][0-9][0-9]_q[0-9][0-9]"))):
        rec = json.load(open(os.path.join(run, "record.json")))
        N, q = rec["N"], rec["q"]
        stride = n_max // N
        st = np.load(os.path.join(run, "chain_states.npz"))["%s_states" % SPLIT]
        for k in range(N + 1):
            t = truth[:, k * stride]
            r = np.hypot(st[:, k, 0] - t[:, 0], st[:, k, 1] - t[:, 1]) * 1e3
            dtx = np.abs(st[:, k, 2] - t[:, 2]) * 1e3
            rows.append(dict(N=N, q=q, plane=k, z_mm=float(Z0 + k * L / N),
                             med_um=float(np.median(r)), p95_um=float(np.quantile(r, 0.95)),
                             tx_med_mrad=float(np.median(dtx))))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "error_vs_z.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Ns = sorted({x["N"] for x in rows})
    qs = sorted({x["q"] for x in rows})
    fig, ax = plt.subplots(1, len(Ns), figsize=(4.3 * len(Ns), 4.2), sharey=True)
    for i, N in enumerate(Ns):
        for q in qs:
            s = [x for x in rows if x["N"] == N and x["q"] == q]
            ax[i].plot([x["z_mm"] / 1000 for x in s], [x["med_um"] for x in s], "-", lw=1.3,
                       label="q = %d" % q)
        ax[i].set_title("N = %d (dz = %.0f mm)" % (N, 5177.8 / N), fontsize=10)
        ax[i].set_xlabel("z [m]")
        ax[i].set_yscale("log")
        ax[i].grid(alpha=0.3, which="both")
    ax[0].set_ylabel("median radial error against the RK6 track [µm]")
    ax[0].legend(fontsize=8)
    fig.suptitle("Block E: how the error grows along the crossing, test tracks", fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "figures", "error_vs_z.png"), dpi=130)
    print("wrote results/error_vs_z.csv, figures/error_vs_z.png")
    for N in Ns:
        s = [x for x in rows if x["N"] == N and x["q"] == qs[0]]
        q1 = s[len(s) // 4]["med_um"]; h = s[len(s) // 2]["med_um"]; e = s[-1]["med_um"]
        print("  N=%3d q=%d: quarter way %6.1f um, half way %6.1f, end %6.1f  (end/half %.2f)"
              % (N, qs[0], q1, h, e, e / h))


if __name__ == "__main__":
    main()
