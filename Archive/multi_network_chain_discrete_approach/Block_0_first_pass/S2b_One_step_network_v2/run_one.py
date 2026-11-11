#!/usr/bin/env python
"""Train ONE v2 run (mode, seed) to genuine stall. Crash-safe and resumable.

Why this exists: training.py runs the six configurations serially and only
saves a checkpoint when a whole run finishes. On a contended node that is both
slow (wall-clock = sum of six) and fragile (a kill loses the run in progress).
This trains a single (mode, seed) with a bounded thread count, checkpoints and
appends its history after EVERY restart, and resumes from wherever it stopped —
so the six can run concurrently and survive interruption.

Protocol identical to training.py / the v1 experiment: full-batch L-BFGS
(max_iter 200, strong Wolfe, history 120), restart until two consecutive
restarts each improve < 1% (the stall criterion), safety cap 150 restarts.

Run:  OMP_NUM_THREADS=4 python run_one.py <mode> <seed>
Outputs: results/one_step_{mode}_seed{s}.pt, results/hist_{mode}_seed{s}.csv
"""
import csv
import os
import sys
import time

import torch

MODE, SEED = sys.argv[1], int(sys.argv[2])

from training import (MAX_ITER, RES, Q, DZ, ZN, A, b, S_tr, REF_tr,  # noqa: E402
                      in_scale, out_scale, rates)
from model import OneStepNetwork, data_loss, physics_loss  # noqa: E402

torch.set_default_dtype(torch.float64)

OUTER_CAP = 400
CKPT = os.path.join(RES, "one_step_%s_seed%d.pt" % (MODE, SEED))
HIST = os.path.join(RES, "hist_%s_seed%d.csv" % (MODE, SEED))


def history_rows():
    if not os.path.exists(HIST):
        return []
    with open(HIST) as f:
        return list(csv.DictReader(f))


def append_history(row):
    new = not os.path.exists(HIST)
    with open(HIST, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mode", "seed", "outer", "loss", "wall_s"])
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    torch.manual_seed(SEED)
    model = OneStepNetwork(Q, in_scale, out_scale)
    rows = history_rows()
    if rows and os.path.exists(CKPT):
        model.load_state_dict(torch.load(CKPT, weights_only=True))
        start = 1 + max(int(r["outer"]) for r in rows)
        previous = float(rows[-1]["loss"])
        print("resuming %s seed %d at restart %d (loss %.6e)"
              % (MODE, SEED, start, previous), flush=True)
    else:
        start, previous = 0, float("inf")

    opt = torch.optim.LBFGS(model.parameters(), max_iter=MAX_ITER,
                            history_size=120, tolerance_grad=1e-13,
                            tolerance_change=1e-16, line_search_fn="strong_wolfe")

    def loss_now():
        if MODE == "physics":
            return physics_loss(model, rates, S_tr, DZ, ZN, A, b)
        return data_loss(model, S_tr, REF_tr)

    def closure():
        opt.zero_grad()
        loss = loss_now()
        loss.backward()
        return loss

    stalled = 0
    for outer in range(start, OUTER_CAP):
        t0 = time.time()
        opt.step(closure)
        loss = loss_now().item()
        torch.save(model.state_dict(), CKPT)
        append_history(dict(mode=MODE, seed=SEED, outer=outer, loss=loss,
                            wall_s=round(time.time() - t0, 1)))
        print("  %s seed %d outer %d: loss %.6e (%.0f s)"
              % (MODE, SEED, outer, loss, time.time() - t0), flush=True)
        if previous - loss < 1e-2 * max(loss, 1e-30):
            stalled += 1
        else:
            stalled = 0
        previous = loss
        if stalled >= 2 and outer >= 2:
            print("  %s seed %d: STALLED at outer %d - converged" % (MODE, SEED, outer),
                  flush=True)
            return
    print("  %s seed %d: hit the %d-restart safety cap without stalling"
          % (MODE, SEED, OUTER_CAP), flush=True)


if __name__ == "__main__":
    main()
