#!/usr/bin/env python
"""E3 - every trained network on all three splits: is it over-trained?

A network is trained on states drawn from the TRAINING tracks only. The
validation and test tracks are never seen. Over-training would show up as a
network that is better on the tracks it trained on than on the ones it did not,
so this script measures both, the same way, for every finished network:

  * the chain: each split's tracks carried from their real state at z0 through
    the N steps, scored at z1 against the RK6 track - median and 95th
    percentile of the position error, and x, y, tx, ty separately;
  * the loss itself: the label-free RK-PINN loss on 32,000 (track, plane)
    states drawn from the split's RK6 tracks, exactly as training draws them.
    This is the quantity that was optimised, so it is the sharper test.

Over-training is judged on the ratios test/train of both measures. Equal within
a few percent means none.

Run:     PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python evaluate_splits.py
Output:  results/split_comparison.csv   one row per (N, q, split)
         results/overtraining.csv       one row per (N, q): the ratios
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"

import csv      # noqa: E402
import glob     # noqa: E402
import json     # noqa: E402
import sys      # noqa: E402
import time     # noqa: E402

import numpy as np   # noqa: E402
import torch         # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
E1 = os.path.join(HERE, "..", "E1_Network_grid")
sys.path.insert(0, E1)

import use_shared    # noqa: E402,F401
from _shared.model import LHCbRates, physics_loss          # noqa: E402
from _shared.reference import gauss_legendre, make_field    # noqa: E402
from chain_network import carry, load_network, znodes_for   # noqa: E402
from train_network import draw_states                       # noqa: E402

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

TRACKS = os.path.join(HERE, "..", "E0_Track_dataset", "results", "tracks.npz")
SPLITS = ("train", "val", "test")
ROWS = ["N", "q", "split", "n_tracks", "pos_med_um", "pos_p95_um", "x_med_um", "y_med_um",
        "tx_med_mrad", "ty_med_mrad", "loss_on_rk6_states", "n_states"]


def main():
    D = np.load(TRACKS)
    Z0, L, n_max = float(D["z0"]), float(D["L"]), int(D["n_max"])
    field = str(D["field"])
    fld = make_field(field)
    rates = LHCbRates(make_field(field))
    rows, ratios = [], []
    for rec_path in sorted(glob.glob(os.path.join(E1, "results", "N*_q*", "record.json"))):
        run = os.path.dirname(rec_path)
        rec = json.load(open(rec_path))
        N, q = rec["N"], rec["q"]
        dz, stride = L / N, n_max // N
        z_starts = Z0 + np.arange(N) * dz
        model = load_network(run, fld)
        c, A_np, b_np = gauss_legendre(q)
        A, b = torch.tensor(A_np), torch.tensor(b_np)
        per = {}
        t0 = time.time()
        for s in SPLITS:
            S0 = np.asarray(D["%s_S0" % s])
            end = carry(model, S0, Z0, N)[:, -1]
            truth = np.asarray(D["%s_truth" % s])[:, n_max]
            d = end[:, :4] - truth[:, :4]
            pos = np.abs(d[:, :2]).max(axis=1) * 1e3
            # the trained objective, on this split's own RK6 track states
            states = np.asarray(D["%s_truth" % s])[:, 0:n_max:stride]
            Sd, zd, _ = draw_states(states, z_starts, rec["states_per_round"],
                                    np.random.default_rng([20260918, N, q]))
            St, zt = torch.as_tensor(Sd), torch.as_tensor(zd)
            with torch.no_grad():
                loss = float(physics_loss(model, rates, St, dz, znodes_for(model, zt), A, b, zt).item())
            row = dict(N=N, q=q, split=s, n_tracks=len(S0),
                       pos_med_um=float(np.median(pos)), pos_p95_um=float(np.quantile(pos, 0.95)),
                       x_med_um=float(np.median(np.abs(d[:, 0])) * 1e3),
                       y_med_um=float(np.median(np.abs(d[:, 1])) * 1e3),
                       tx_med_mrad=float(np.median(np.abs(d[:, 2])) * 1e3),
                       ty_med_mrad=float(np.median(np.abs(d[:, 3])) * 1e3),
                       loss_on_rk6_states=loss, n_states=int(len(Sd)))
            rows.append(row)
            per[s] = row
        ratios.append(dict(
            N=N, q=q, converged=rec["converged"], confirmed=rec["confirmed"], restarts=rec["restarts"],
            train_pos_med_um=per["train"]["pos_med_um"], val_pos_med_um=per["val"]["pos_med_um"],
            test_pos_med_um=per["test"]["pos_med_um"],
            test_over_train_pos=per["test"]["pos_med_um"] / per["train"]["pos_med_um"],
            val_over_train_pos=per["val"]["pos_med_um"] / per["train"]["pos_med_um"],
            train_loss=per["train"]["loss_on_rk6_states"], test_loss=per["test"]["loss_on_rk6_states"],
            test_over_train_loss=per["test"]["loss_on_rk6_states"] / per["train"]["loss_on_rk6_states"]))
        print("N=%3d q=%2d  chain median um: train %7.1f  val %7.1f  test %7.1f  (test/train %.3f) | "
              "loss train %.3e test %.3e (%.3f) | %.0f s"
              % (N, q, per["train"]["pos_med_um"], per["val"]["pos_med_um"], per["test"]["pos_med_um"],
                 ratios[-1]["test_over_train_pos"], per["train"]["loss_on_rk6_states"],
                 per["test"]["loss_on_rk6_states"], ratios[-1]["test_over_train_loss"],
                 time.time() - t0), flush=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "split_comparison.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ROWS)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(HERE, "results", "overtraining.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ratios[0].keys()))
        w.writeheader()
        w.writerows(ratios)
    r = np.array([x["test_over_train_pos"] for x in ratios])
    lr = np.array([x["test_over_train_loss"] for x in ratios])
    print("\n%d networks: test/train chain median %.3f to %.3f (median %.3f); "
          "test/train loss %.3f to %.3f (median %.3f)"
          % (len(ratios), r.min(), r.max(), np.median(r), lr.min(), lr.max(), np.median(lr)))


if __name__ == "__main__":
    main()
