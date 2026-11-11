#!/usr/bin/env python
"""E3 - what the endpoint error is made of, step by step.

The chain error is 150-250 µm while one step is under a micron, so the error is
built by accumulation. This script takes it apart on the test tracks of one
network:

  local error      at every step, the network's output against RK6 taken from
                   the SAME state the network was given (so it is what this
                   step added, not what it inherited): radial, and the slope
                   parts dtx, dty;
  coherence        along one track, do those slope increments point the same
                   way? |sum| / sum|..| is 1 when every step pushes the same
                   way and about 1/sqrt(N) when they are independent;
  lever arm        each step's slope error multiplied by the distance left to
                   z1, summed. If the endpoint error is mostly this, then what
                   matters is not the size of a step's error but how early it
                   is made;
  what is needed   inverting that sum gives the per-step accuracy a target
                   endpoint error would need.

Run:     PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python error_anatomy.py [--N 64 --q 2]
Outputs: results/error_anatomy.csv (per step), results/error_anatomy_summary.json,
         figures/error_anatomy.png
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"

import argparse   # noqa: E402
import csv        # noqa: E402
import json       # noqa: E402
import sys        # noqa: E402
import time       # noqa: E402

import numpy as np   # noqa: E402
import torch         # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
E1 = os.path.join(HERE, "..", "E1_Network_grid")
TRACKS = os.path.join(HERE, "..", "E0_Track_dataset", "results", "tracks.npz")
sys.path.insert(0, E1)

import use_shared   # noqa: E402,F401
from _shared.reference import make_field, rk6_rows   # noqa: E402
from chain_network import load_network, step_outputs  # noqa: E402

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

SPLIT = "test"
RK6_LOCAL_STEP = 1.0     # mm; the local reference only needs ~1e-3 µm
TARGETS_UM = (10.0, 1.0)


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
    run = os.path.join(E1, "results", "N%03d_q%02d" % (N, q))
    rec = json.load(open(os.path.join(run, "record.json")))
    D = np.load(TRACKS)
    Z0, Z1, L, n_max = float(D["z0"]), float(D["z1"]), float(D["L"]), int(D["n_max"])
    dz, stride = L / N, n_max // N
    fld = make_field(str(D["field"]))
    model = load_network(run, fld)
    S0 = np.asarray(D["%s_S0" % SPLIT])
    truth = np.asarray(D["%s_truth" % SPLIT])
    n = len(S0)

    t0 = time.time()
    S = S0.copy()
    local_pos, local_tx, local_ty = np.empty((N, n)), np.empty((N, n)), np.empty((N, n))
    for k in range(N):
        z = Z0 + k * dz
        out = step_outputs(model, S, np.full(n, z))[:, -1, :]
        ref = rk6_rows(S, z, z + dz, step=RK6_LOCAL_STEP, field=fld)
        local_pos[k] = np.hypot(out[:, 0] - ref[:, 0], out[:, 1] - ref[:, 1]) * 1e3
        local_tx[k] = (out[:, 2] - ref[:, 2]) * 1e3
        local_ty[k] = (out[:, 3] - ref[:, 3]) * 1e3
        S = np.concatenate([out, S[:, 4:5]], axis=1)
    end = S
    truth_end = truth[:, n_max]
    final_r = np.hypot(end[:, 0] - truth_end[:, 0], end[:, 1] - truth_end[:, 1]) * 1e3
    print("carried %d tracks through %d steps in %.0f s" % (n, N, time.time() - t0))

    # -- the lever-arm model --------------------------------------------------------
    lever = (Z1 - (Z0 + (np.arange(N) + 1) * dz))[:, None]        # mm left after step k
    dx_from_slope = (local_tx * lever * 1e-3).sum(axis=0) * 1e3     # µm: mrad -> rad, mm -> µm
    dx_local = 0.0                                                   # position part, below
    lever_only = np.abs(dx_from_slope)
    coherence = np.abs(local_tx.sum(axis=0)) / np.abs(local_tx).sum(axis=0)
    independent = 1.0 / np.sqrt(N)

    rows = []
    for k in range(N):
        rows.append(dict(step=k, z_mm=float(Z0 + k * dz), lever_mm=float(lever[k, 0]),
                         local_pos_med_um=float(np.median(local_pos[k])),
                         local_tx_med_mrad=float(np.median(np.abs(local_tx[k]))),
                         local_tx_bias_mrad=float(np.median(local_tx[k])),
                         local_ty_med_mrad=float(np.median(np.abs(local_ty[k]))),
                         carried_to_end_um=float(np.median(np.abs(local_tx[k] * lever[k, 0]))) * 1e-3 * 1e3))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "error_anatomy.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    med_local = float(np.median(local_pos))
    med_tx = float(np.median(np.abs(local_tx)))
    need = {("%g um target" % t): {
        "per_step_slope_mrad": float(med_tx * t / np.median(lever_only)),
        "factor_better": float(np.median(lever_only) / t)} for t in TARGETS_UM}
    summary = dict(
        N=N, q=q, dz_mm=dz, n_tracks=int(n),
        final_med_um=float(np.median(final_r)),
        local_step_med_um=med_local,
        local_step_slope_med_mrad=med_tx,
        local_step_slope_bias_mrad=float(np.median(local_tx)),
        sum_of_local_steps_um=float(np.median(local_pos.sum(axis=0))),
        lever_arm_prediction_med_um=float(np.median(lever_only)),
        lever_arm_over_final=float(np.median(lever_only) / np.median(final_r)),
        coherence_med=float(np.median(coherence)),
        coherence_if_independent=float(independent),
        needed_for=need,
        rk6_local_step_mm=RK6_LOCAL_STEP, wall_s=round(time.time() - t0, 1))
    with open(os.path.join(HERE, "results", "error_anatomy_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    zz = np.array([r["z_mm"] for r in rows]) / 1000
    ax[0].plot(zz, [r["local_pos_med_um"] for r in rows], "-o", ms=3, label="radial")
    ax[0].plot(zz, [r["local_tx_med_mrad"] for r in rows], "-s", ms=3, label="|dtx| [mrad]")
    ax[0].plot(zz, np.abs([r["local_tx_bias_mrad"] for r in rows]), "-^", ms=3, label="|median dtx| (bias)")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("z of the step [m]")
    ax[0].set_ylabel("local error added by one step")
    ax[0].set_title("What each step adds (µm and mrad)", fontsize=10)
    ax[0].legend(fontsize=8)
    ax[1].plot(zz, [r["carried_to_end_um"] for r in rows], "-o", ms=3, color="#d62728")
    ax[1].set_xlabel("z of the step [m]")
    ax[1].set_ylabel("its slope error × the distance left [µm]")
    ax[1].set_title("What each step costs at the SciFi plane", fontsize=10)
    ax[2].hist(np.clip(coherence, 0, 1), bins=40, color="#4c72b0")
    ax[2].axvline(independent, color="k", ls="--", label="independent steps: %.2f" % independent)
    ax[2].axvline(np.median(coherence), color="r", label="median %.2f" % np.median(coherence))
    ax[2].set_xlabel("coherence of a track's slope increments")
    ax[2].set_ylabel("tracks")
    ax[2].set_title("Do the steps push the same way?", fontsize=10)
    ax[2].legend(fontsize=8)
    for x in ax:
        x.grid(alpha=0.3, which="both")
    fig.suptitle("Block E: anatomy of the endpoint error, N = %d, q = %d (%d test tracks)"
                 % (N, q, n), fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "figures", "error_anatomy.png"), dpi=130)
    print("wrote results/error_anatomy.csv, error_anatomy_summary.json, figures/error_anatomy.png")


if __name__ == "__main__":
    main()
