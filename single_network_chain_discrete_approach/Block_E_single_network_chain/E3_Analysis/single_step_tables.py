#!/usr/bin/env python
"""E3 - error(q, dz) for a SINGLE step, and how it varies along the magnet.

The chain table mixes what a network does in one step with how that error
accumulates over N of them. This one isolates the step: every network is
applied ONCE, from the RK6 state on each of its start planes, and its output is
compared with the RK6 state one step later. The error is again radial,
r = sqrt(dx^2 + dy^2), so the two tables are directly comparable; the ratio
between them says how much of the chain error is accumulation.

Every test track contributes on every start plane (1,452 x N pairs per model),
so the same numbers also give the error against z, which shows where along the
magnet a network is weak - the fringe field at the ends bends far less than the
middle, and the step is the same length everywhere.

Run:     PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python single_step_tables.py
Outputs: results/error_qdz_single_step.csv, results/single_step_vs_z.csv,
         figures/single_step.png
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"

import csv     # noqa: E402
import glob    # noqa: E402
import json    # noqa: E402
import sys     # noqa: E402
import time    # noqa: E402

import numpy as np   # noqa: E402
import torch         # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
E1 = os.path.join(HERE, "..", "E1_Network_grid")
sys.path.insert(0, E1)

import use_shared    # noqa: E402,F401
from _shared.reference import make_field           # noqa: E402
from chain_network import load_network, step_outputs, straight_numpy   # noqa: E402

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

TRACKS = os.path.join(HERE, "..", "E0_Track_dataset", "results", "tracks.npz")
SPLIT = "test"


def main():
    D = np.load(TRACKS)
    Z0, L, n_max = float(D["z0"]), float(D["L"]), int(D["n_max"])
    truth = np.asarray(D["%s_truth" % SPLIT])
    P = np.asarray(D["%s_P" % SPLIT])
    fld = make_field(str(D["field"]))
    rows, by_z = [], []
    for run in sorted(glob.glob(os.path.join(E1, "results", "N[0-9][0-9][0-9]_q[0-9][0-9]"))):
        rec = json.load(open(os.path.join(run, "record.json")))
        N, q = rec["N"], rec["q"]
        stride, dz = n_max // N, L / N
        model = load_network(run, fld)
        t0 = time.time()
        n_tr = len(truth)
        S = truth[:, 0:n_max:stride].reshape(-1, 5)                       # (n_tr*N, 5)
        nxt = truth[:, stride:n_max + 1:stride].reshape(-1, 5)
        plane = np.tile(np.arange(N), n_tr) if False else np.repeat(np.arange(N)[None, :], n_tr, 0).reshape(-1)
        z_start = Z0 + plane * dz
        out = step_outputs(model, S, z_start)[:, -1, :]
        r = np.hypot(out[:, 0] - nxt[:, 0], out[:, 1] - nxt[:, 1]) * 1e3
        line = straight_numpy(S, dz)
        r_line = np.hypot(line[:, 0] - nxt[:, 0], line[:, 1] - nxt[:, 1]) * 1e3
        p = np.repeat(P[:, None], N, axis=1).reshape(-1)
        rows.append(dict(N=N, q=q, dz_mm=round(dz, 2), n_pairs=int(len(r)),
                         step_med_um=float(np.median(r)), step_p95_um=float(np.quantile(r, 0.95)),
                         step_mean_um=float(r.mean()),
                         straight_line_step_med_um=float(np.median(r_line)),
                         step_over_straight=float(np.median(r) / np.median(r_line)),
                         step_med_um_p_above_10GeV=float(np.median(r[p >= 10])),
                         step_med_um_p_below_5GeV=float(np.median(r[p < 5])),
                         chain_med_um=rec["test"]["vs_rk6_endpoint"]["pos_med_um"],
                         wall_s=round(time.time() - t0, 1)))
        for k in range(N):
            m = plane == k
            by_z.append(dict(N=N, q=q, plane=k, z_mm=float(Z0 + k * dz),
                             step_med_um=float(np.median(r[m])),
                             step_p95_um=float(np.quantile(r[m], 0.95)),
                             straight_line_med_um=float(np.median(r_line[m]))))
        print("N=%3d q=%2d: single step median %7.2f um (p95 %8.1f), straight line %9.1f um, "
              "ratio %.4f | %.0f s" % (N, q, rows[-1]["step_med_um"], rows[-1]["step_p95_um"],
                                       rows[-1]["straight_line_step_med_um"],
                                       rows[-1]["step_over_straight"], rows[-1]["wall_s"]), flush=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    for name, data in (("error_qdz_single_step.csv", rows), ("single_step_vs_z.csv", by_z)):
        with open(os.path.join(HERE, "results", name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Ns = sorted({x["N"] for x in rows})
    qs = sorted({x["q"] for x in rows})
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    for N in Ns:
        v = [x["step_med_um"] for x in sorted([r for r in rows if r["N"] == N], key=lambda d: d["q"])]
        ax[0].plot(qs, v, "-o", label="N = %d (dz = %.0f mm)" % (N, 5177.8 / N))
        c = ax[0].lines[-1].get_color()
        ch = [x["chain_med_um"] for x in sorted([r for r in rows if r["N"] == N], key=lambda d: d["q"])]
        ax[1].plot(qs, np.array(ch) / np.array(v), "-o", color=c, label="N = %d" % N)
        z = [x for x in by_z if x["N"] == N and x["q"] == qs[1]]
        ax[2].plot([x["z_mm"] / 1000 for x in z], [x["step_med_um"] for x in z], "-", color=c,
                   label="N = %d, q = %d" % (N, qs[1]))
    for a, t, yl in ((ax[0], "One step, median radial error", "median radial error of one step [µm]"),
                     (ax[1], "Chain error / single-step error", "ratio (N steps would give N if errors added)"),
                     (ax[2], "Where along the magnet the step error sits", "median radial error of one step [µm]")):
        a.set_title(t, fontsize=10)
        a.set_ylabel(yl, fontsize=9)
        a.grid(alpha=0.3, which="both")
        a.legend(fontsize=8)
    for a in ax[:2]:
        a.set_xscale("log", base=2)
        a.set_xticks(qs, [str(q) for q in qs])
        a.set_xlabel("stage count q")
    ax[0].set_yscale("log")
    ax[1].set_yscale("log")
    ax[2].set_xlabel("z of the step's start plane [m]")
    ax[2].set_yscale("log")
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "figures", "single_step.png"), dpi=130)
    print("wrote results/error_qdz_single_step.csv, single_step_vs_z.csv, figures/single_step.png")


if __name__ == "__main__":
    main()
