#!/usr/bin/env python
"""[v2 — official-sample data] The six runs: {physics, data} x seeds {0,1,2}, VDP optimiser protocol.

Full-batch L-BFGS (max_iter 200, strong Wolfe, history 120), up to 6 restarts,
early stop after two consecutive restarts improving < 1%. Saves after every
run and resumes. Scoring on the held-out test split: endpoint + per-stage
error vs the fp64 RK4 reference, in um and the agreed rho.

Run:  /data/bfys/gscriven/conda/envs/TE/bin/python training.py
Outputs: results/summary.csv, results/histories.csv, results/predictions.npz,
         results/one_step_{mode}_seed{s}.pt
"""
import csv
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "S0_Baseline_data_exploration"))
sys.path.insert(0, os.path.join(HERE, "..", "S1_Simple_first_pass"))
sys.path.insert(0, os.path.join(HERE, "..", "S2_One_step_network"))  # model.py, field_torch.py
from reference_card import rho  # noqa: E402
from irk import tableau  # noqa: E402
from model import (LHCbRates, OneStepNetwork, data_loss, physics_loss)  # noqa: E402

torch.set_default_dtype(torch.float64)
RES = os.path.join(HERE, "results")

SEEDS = (0, 1, 2)
MODES = ("physics", "data")
OUTER, MAX_ITER = 150, 200   # v2: straight to stall (v1's 6-restart cap was a budget artefact)

d = np.load(os.path.join(RES, "frozen_leg_data.npz"))
Q, Z0, Z1 = int(d["q"]), float(d["z0"]), float(d["z1"])
DZ = Z1 - Z0
c, A_np, b_np = tableau(Q)
A, b = torch.tensor(A_np), torch.tensor(b_np)
ZN = torch.tensor(d["znodes"])
S_tr = torch.tensor(d["train_S"])
REF_tr = torch.tensor(d["train_ref"])
in_scale, out_scale = d["in_scale"], d["out_scale"]

rates = LHCbRates()


def train_one(mode, seed):
    torch.manual_seed(seed)
    model = OneStepNetwork(Q, in_scale, out_scale)
    opt = torch.optim.LBFGS(model.parameters(), max_iter=MAX_ITER,
                            history_size=120, tolerance_grad=1e-13,
                            tolerance_change=1e-16, line_search_fn="strong_wolfe")

    def loss_now():
        if mode == "physics":
            return physics_loss(model, rates, S_tr, DZ, ZN, A, b)
        return data_loss(model, S_tr, REF_tr)

    def closure():
        opt.zero_grad()
        loss = loss_now()
        loss.backward()
        return loss

    history, previous, stalled = [], float("inf"), 0
    for outer in range(OUTER):
        t0 = time.time()
        opt.step(closure)
        loss = loss_now().item()
        history.append(dict(mode=mode, seed=seed, outer=outer, loss=loss,
                            wall_s=time.time() - t0))
        print("  %s seed %d outer %d: loss %.6e (%.0f s)"
              % (mode, seed, outer, loss, history[-1]["wall_s"]), flush=True)
        if previous - loss < 1e-2 * max(loss, 1e-30):
            stalled += 1
        else:
            stalled = 0
        previous = loss
        if stalled >= 2 and outer >= 2:
            break
    return model, history


def score(model, which="test"):
    S = torch.tensor(d[which + "_S"])
    ref = d[which + "_ref"]                     # (N, q+1, 5) numpy
    with torch.no_grad():
        out = model(S).numpy()                  # (N, q+1, 4)
    end_err = np.abs(out[:, -1, :2] - ref[:, -1, :2]).max(axis=1) * 1e3  # um
    stage_err = np.abs(out[:, :-1, :2] - ref[:, :-1, :2]).max(axis=(1, 2)) * 1e3
    r = rho(ref[:, -1, :4], out[:, -1, :])
    straight = d[which + "_S"][:, :2] + d[which + "_S"][:, 2:4] * DZ
    line_err = np.abs(straight - ref[:, -1, :2]).max(axis=1) * 1e3
    return {
        "endpoint_med_um": float(np.median(end_err)),
        "endpoint_p95_um": float(np.quantile(end_err, 0.95)),
        "stage_med_um": float(np.median(stage_err)),
        "rho_mean": float(r.mean()), "rho_median": float(np.median(r)),
        "straight_med_um": float(np.median(line_err)),
        "n": int(len(S)),
    }, out, end_err


def main():
    all_hist, summary, preds = [], [], {}
    for mode in MODES:
        for seed in SEEDS:
            ckpt = os.path.join(RES, "one_step_%s_seed%d.pt" % (mode, seed))
            if os.path.exists(ckpt):
                print("resume: %s exists, skipping train" % ckpt, flush=True)
                model = OneStepNetwork(Q, in_scale, out_scale)
                model.load_state_dict(torch.load(ckpt, weights_only=True))
            else:
                print("training %s seed %d" % (mode, seed), flush=True)
                model, hist = train_one(mode, seed)
                all_hist += hist
                torch.save(model.state_dict(), ckpt)
            sc, out, end_err = score(model)
            sc.update(mode=mode, seed=seed)
            summary.append(sc)
            preds["%s_seed%d_out" % (mode, seed)] = out
            preds["%s_seed%d_end_err_um" % (mode, seed)] = end_err
            print("  scored:", json.dumps(sc), flush=True)

    with open(os.path.join(RES, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary[0].keys())
        w.writeheader(); w.writerows(summary)
    if all_hist:
        with open(os.path.join(RES, "histories.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_hist[0].keys())
            w.writeheader(); w.writerows(all_hist)
    np.savez_compressed(os.path.join(RES, "predictions.npz"),
                        test_P=d["test_P"], **preds)
    print("done.")


if __name__ == "__main__":
    main()
