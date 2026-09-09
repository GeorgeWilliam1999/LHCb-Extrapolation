#!/usr/bin/env python
"""Continue the six step-2 runs until the stall criterion actually fires.

The first attempt capped every run at 6 L-BFGS restarts, and histories.csv
shows all six runs were still improving 20-35% per restart at the cap: the
early stop (two consecutive restarts improving < 1%) never fired, so the
reported errors were budget-limited, not converged. This script resumes each
run from its saved checkpoint and keeps restarting until the stall criterion
fires, with a safety cap of 60 total restarts per run.

Protocol is otherwise identical to training.py. One honest caveat: the
L-BFGS curvature history is not checkpointed, so it restarts empty at each
resume boundary; the strong-Wolfe line search rebuilds it within a restart.

Idempotent and crash-safe: every restart appends its row to histories.csv
and overwrites the checkpoint before the next one starts, so rerunning this
script continues wherever it stopped. Stall detection carries across resume
boundaries via the last recorded loss in histories.csv.

The pre-continuation (budget-6) summary/histories/predictions are snapshot
in results/budget6/ (also in git at 557a22f).

Run:  /data/bfys/gscriven/conda/envs/TE/bin/python continue_training.py
Outputs: updated results/one_step_{mode}_seed{s}.pt, results/histories.csv,
         results/summary.csv (now also train/val medians + endpoint slope),
         results/predictions.npz
"""
import csv
import json
import os
import time

import numpy as np
import torch

from training import (MAX_ITER, MODES, RES, SEEDS, Q, DZ, ZN, A, b, d,
                      S_tr, REF_tr, in_scale, out_scale, rates, score)
from model import OneStepNetwork, data_loss, physics_loss

torch.set_default_dtype(torch.float64)

OUTER_CAP = 150         # total restarts incl. the original 6 - safety only
# (first continuation pass capped at 60: every run was still improving
#  ~2%/restart there; the improvement rate decays ~1/n, predicting the <1%
#  stall fires around restart ~135, so 150 gives it room to fire for real)
HIST = os.path.join(RES, "histories.csv")


def read_history():
    with open(HIST) as f:
        return list(csv.DictReader(f))


def append_history(row):
    with open(HIST, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mode", "seed", "outer", "loss", "wall_s"])
        w.writerow(row)


def continue_one(mode, seed):
    rows = [r for r in read_history()
            if r["mode"] == mode and int(r["seed"]) == seed]
    done = 1 + max(int(r["outer"]) for r in rows)
    previous = float(rows[-1]["loss"])

    torch.manual_seed(seed)                     # irrelevant post-init, kept for parity
    model = OneStepNetwork(Q, in_scale, out_scale)
    ckpt = os.path.join(RES, "one_step_%s_seed%d.pt" % (mode, seed))
    model.load_state_dict(torch.load(ckpt, weights_only=True))
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

    stalled = 0
    for outer in range(done, OUTER_CAP):
        t0 = time.time()
        opt.step(closure)
        loss = loss_now().item()
        torch.save(model.state_dict(), ckpt)
        append_history(dict(mode=mode, seed=seed, outer=outer, loss=loss,
                            wall_s=round(time.time() - t0, 1)))
        print("  %s seed %d outer %d: loss %.6e (%.0f s)"
              % (mode, seed, outer, loss, time.time() - t0), flush=True)
        if previous - loss < 1e-2 * max(loss, 1e-30):
            stalled += 1
        else:
            stalled = 0
        previous = loss
        if stalled >= 2:
            print("  %s seed %d: stalled at outer %d - converged" % (mode, seed, outer),
                  flush=True)
            return model, True
    print("  %s seed %d: hit the %d-restart safety cap without stalling"
          % (mode, seed, OUTER_CAP), flush=True)
    return model, False


def main():
    summary, preds = [], {}
    for mode in MODES:
        for seed in SEEDS:
            print("continuing %s seed %d" % (mode, seed), flush=True)
            model, converged = continue_one(mode, seed)
            sc, out, end_err = score(model)
            slope = np.abs(out[:, -1, 2:4]
                           - d["test_ref"][:, -1, 2:4]).max(axis=1) * 1e3
            for which in ("train", "val"):
                sc["%s_med_um" % which] = score(model, which)[0]["endpoint_med_um"]
            sc.update(mode=mode, seed=seed, converged=converged,
                      slope_med_mrad=float(np.median(slope)))
            summary.append(sc)
            preds["%s_seed%d_out" % (mode, seed)] = out
            preds["%s_seed%d_end_err_um" % (mode, seed)] = end_err
            print("  scored:", json.dumps(sc), flush=True)

    with open(os.path.join(RES, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary[0].keys())
        w.writeheader(); w.writerows(summary)
    np.savez_compressed(os.path.join(RES, "predictions.npz"),
                        test_P=d["test_P"], **preds)
    print("done.")


if __name__ == "__main__":
    main()
