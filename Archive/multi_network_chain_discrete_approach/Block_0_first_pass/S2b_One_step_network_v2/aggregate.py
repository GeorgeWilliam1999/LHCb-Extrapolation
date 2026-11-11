#!/usr/bin/env python
"""Score the six finished v2 runs and write the combined result files.

Reads the per-run checkpoints and history files written by run_one.py and
produces the same artefacts the v1 experiment produced, so the two are
directly comparable: summary.csv (with train/val medians, endpoint slope and
the converged flag), histories.csv, predictions.npz.

Run:  /data/bfys/gscriven/conda/envs/TE/bin/python aggregate.py
"""
import csv
import json
import os

import numpy as np
import torch

from training import RES, SEEDS, MODES, Q, in_scale, out_scale, d, score
from model import OneStepNetwork

torch.set_default_dtype(torch.float64)


def run_history(mode, seed):
    path = os.path.join(RES, "hist_%s_seed%d.csv" % (mode, seed))
    with open(path) as f:
        return list(csv.DictReader(f))


summary, preds, all_hist = [], {}, []
for mode in MODES:
    for seed in SEEDS:
        ckpt = os.path.join(RES, "one_step_%s_seed%d.pt" % (mode, seed))
        model = OneStepNetwork(Q, in_scale, out_scale)
        model.load_state_dict(torch.load(ckpt, weights_only=True))
        rows = run_history(mode, seed)
        all_hist += rows
        losses = [float(r["loss"]) for r in rows]
        # the stall criterion, recomputed from the recorded history
        converged = (len(losses) >= 3
                     and all(losses[-i - 2] - losses[-i - 1]
                             < 1e-2 * max(losses[-i - 1], 1e-30) for i in range(2)))
        sc, out, end_err = score(model)
        slope = np.abs(out[:, -1, 2:4] - d["test_ref"][:, -1, 2:4]).max(axis=1) * 1e3
        for which in ("train", "val"):
            sc["%s_med_um" % which] = score(model, which)[0]["endpoint_med_um"]
        sc.update(mode=mode, seed=seed, converged=bool(converged),
                  slope_med_mrad=float(np.median(slope)),
                  restarts=len(losses), final_loss=losses[-1])
        summary.append(sc)
        preds["%s_seed%d_out" % (mode, seed)] = out
        preds["%s_seed%d_end_err_um" % (mode, seed)] = end_err
        print(json.dumps(sc), flush=True)

with open(os.path.join(RES, "summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=summary[0].keys())
    w.writeheader(); w.writerows(summary)
with open(os.path.join(RES, "histories.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["mode", "seed", "outer", "loss", "wall_s"])
    w.writeheader(); w.writerows(all_hist)
np.savez_compressed(os.path.join(RES, "predictions.npz"), test_P=d["test_P"], **preds)
print("wrote summary.csv, histories.csv, predictions.npz")
