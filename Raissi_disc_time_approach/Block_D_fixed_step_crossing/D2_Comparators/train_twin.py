#!/usr/bin/env python
"""D2 - the one supervised twin: start state at z0 -> end state at z1, no
stages, trained on the RK6 endpoint (George 2026-09-14: one twin, taking the
starting state and predicting the final cross-magnet state only; the label is
the RK6 endpoint, material effects ignored).

The network is the Block D class with q = 0 (`chain_model.FrozenResidualNetwork`:
two hidden layers of 128, straight-line-residual output, one output block = the
endpoint), trained with the shared trainer's protocol in `data` mode and the
residual-normalised MSE (`chain_model.residual_data_loss`), so the twin sees
the same O(1) target as the physics networks. Everything about the optimiser is
the shared trainer's.

Usage:
    python train_twin.py [--seed 0]

Outputs in results/twin/:
    twin.pt, twin_history.csv, twin.json    the trainer's, plus
    twin_scores.json                        endpoint vs RK6 and vs the real
                                            SciFi state, per momentum band
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"

import argparse
import json
import sys
import time

import numpy as np
import torch

import use_shared                                        # noqa: F401
from _shared import train as shared_train
from _shared.evaluate import predict
from _shared.reference import RK6_STEP, make_field, rk6_rows

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "D1_Chain_grid"))
from chain_model import (DEPTH, WIDTH, FrozenResidualNetwork,   # noqa: E402
                         residual_data_loss, straight_line_states)
from metrics import component_stats, pos_err_um, slope_err_mrad, stats   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(HERE, "..", "D0_Crossing_dataset", "results",
                            "crossing_particles.npz")
SPLITS = ("train", "val", "test")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "twin"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=WIDTH)
    ap.add_argument("--depth", type=int, default=DEPTH)
    ap.add_argument("--outer-cap", type=int, default=400)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--no-confirm", action="store_true")
    ap.add_argument("--score-only", action="store_true",
                    help="do not train: rebuild the twin from its checkpoint and rescore")
    a = ap.parse_args(argv)
    t0 = time.time()
    D = {k: v for k, v in np.load(os.path.abspath(a.data), allow_pickle=False).items()}
    Z0, Z1, L = float(D["z0"]), float(D["z1"]), float(D["L"])
    n_max = int(D["n_max"])
    field = str(D["field"])
    fld = make_field(field)
    os.makedirs(a.out, exist_ok=True)

    # the dataset: q = 0, ref = the RK6 endpoint only
    c = np.zeros(0)
    arrays = dict(kind="frozen", znodes=c, zout=np.array([Z1]), z0=Z0, z1=Z1, q=0,
                  c=c, field=field)
    for s in SPLITS:
        S = D["%s_S0" % s]
        if s == "train" and a.n_train:
            S = S[:a.n_train]
        arrays["%s_S" % s] = S
        arrays["%s_P" % s] = D["%s_P" % s][:len(S)]
        arrays["%s_ref" % s] = D["%s_truth" % s][:len(S), n_max][:, None, :]
        arrays["%s_z0" % s] = np.full(len(S), Z0)
        arrays["%s_dz" % s] = np.full(len(S), L)
    arrays["in_scale"] = arrays["train_S"].std(axis=0)
    st = straight_line_states(arrays["train_S"], L, c)
    arrays["out_scale"] = np.append(st.reshape(-1, 4).std(axis=0), arrays["train_S"][:, 4].std())
    data_path = os.path.join(a.out, "twin_data.npz")
    np.savez_compressed(data_path, **arrays)

    def factory(q, in_scale, out_scale, width=a.width, depth=a.depth, n_extra=0):
        return FrozenResidualNetwork(0, in_scale, out_scale, width=width, depth=depth,
                                     c=c, z0=Z0, dz=L, field=fld)

    shared_train.OneStepNetwork = factory
    shared_train.data_loss = residual_data_loss
    shared_train.gauss_legendre = lambda q: (np.zeros(0), np.zeros((0, 0)), np.zeros(0))

    def score_split_q0(model, data, split="test"):
        """The shared scorer without the stage term, which is empty at q = 0."""
        S = np.asarray(data["%s_S" % split])
        ref = np.asarray(data["%s_ref" % split])
        out = predict(model, S)
        end_err = pos_err_um(out[:, -1, :], ref[:, -1, :])
        straight = np.concatenate([S[:, :2] + S[:, 2:4] * L, S[:, 2:4]], axis=1)
        return {"endpoint_med_um": float(np.median(end_err)),
                "endpoint_p95_um": float(np.quantile(end_err, 0.95)),
                "stage_med_um": float("nan"),
                "slope_med_mrad": float(np.median(slope_err_mrad(out[:, -1, :], ref[:, -1, :]))),
                "straight_med_um": float(np.median(pos_err_um(straight, ref[:, -1, :]))),
                "n": int(len(S))}, out

    shared_train.score_split = score_split_q0

    rec_path = os.path.join(a.out, "twin.json")
    if a.score_only and os.path.exists(rec_path):
        with open(rec_path) as f:
            record = json.load(f)
    else:
        record = shared_train.main([
        "--data", data_path, "--mode", "data", "--seed", str(a.seed), "--q", "0",
        "--width", str(a.width), "--depth", str(a.depth), "--out", a.out,
        "--tag", "twin", "--field", field, "--outer-cap", str(a.outer_cap)]
            + (["--no-confirm"] if a.no_confirm else []))

    torch.manual_seed(a.seed)
    model = factory(0, arrays["in_scale"], arrays["out_scale"])
    model.load_state_dict(torch.load(os.path.join(a.out, "twin.pt"), weights_only=True))
    model.eval()
    scores = {"seed": a.seed, "width": a.width, "depth": a.depth,
              "n_parameters": int(sum(p.numel() for p in model.parameters())),
              "converged": bool(record["converged"]), "restarts": int(record["restarts"]),
              "wall_s": float(record["wall_s"])}
    for s in ("val", "test"):
        S = arrays["%s_S" % s]
        out = predict(model, S)[:, -1, :]
        pred = np.concatenate([out, S[:, 4:5]], axis=1)
        truth = D["%s_truth" % s][:, n_max]
        carried = rk6_rows(pred, Z1, D["%s_z_post" % s], step=RK6_STEP, field=fld)
        pb = D["%s_PBAND" % s]
        scores[s] = {
            "vs_rk6_endpoint": dict(stats(pos_err_um(pred, truth), slope_err_mrad(pred, truth), pb),
                                    components=component_stats(pred, truth, pb)),
            "vs_real_scifi_state": dict(stats(pos_err_um(carried, D["%s_S_post" % s]),
                                              slope_err_mrad(carried, D["%s_S_post" % s]), pb),
                                        components=component_stats(carried, D["%s_S_post" % s], pb))}
    with open(os.path.join(a.out, "twin_scores.json"), "w") as f:
        json.dump(scores, f, indent=1)
    os.remove(data_path)
    print("TWIN: test endpoint vs RK6 median %.4g um (p95 %.4g), vs real SciFi %.4g um; "
          "converged %s; %.0f s" % (scores["test"]["vs_rk6_endpoint"]["pos_med_um"],
                                    scores["test"]["vs_rk6_endpoint"]["pos_p95_um"],
                                    scores["test"]["vs_real_scifi_state"]["pos_med_um"],
                                    scores["converged"], time.time() - t0))
    return scores


if __name__ == "__main__":
    main()
