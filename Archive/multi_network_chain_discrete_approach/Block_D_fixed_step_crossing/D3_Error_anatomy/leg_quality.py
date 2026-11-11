#!/usr/bin/env python
"""D3 - how well did every one of the 4,260 legs actually train?

Every leg of Block D was declared converged by the shared trainer's rule: two
consecutive L-BFGS restarts each improving the loss by less than 1 percent,
followed by a confirmation pass with a fresh optimiser that must also fail to
move the loss. This script reads every leg's record and loss history and asks
two further questions the rule does not.

1. Did the optimiser actually run to the end? A restart is 200 L-BFGS
   iterations and takes about as long as the leg's first restarts. A restart
   that finishes in a small fraction of that time and leaves the loss exactly
   unchanged was stopped inside the optimiser (the gradient fell below its
   fixed tolerance), not by the loss. If the leg's final confirmation block is
   made of such restarts, the confirmation tested nothing; if the stall itself
   was declared on such restarts, the stall is an artefact too.
2. Did the network beat the straight line on its own leg? The record holds the
   network's own-step endpoint error (against the fine propagation of the
   states it was given) and the straight line's error on the same states.

Run:
    PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python leg_quality.py

Outputs:
    results/leg_quality.csv          one row per leg
    results/leg_quality_summary.csv  one row per step count N
"""
import csv
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CHAINS = os.path.join(HERE, "..", "D1_Chain_grid", "results")
OUT = os.path.join(HERE, "results")
QUIT_WALL = 0.2       # a restart shorter than this fraction of the leg's first restarts
QUIT_GAIN = 1e-6      # ... that also leaves the loss unchanged to this relative precision


def read_history(path):
    with open(path) as f:
        return [dict(phase=r["phase"], loss=float(r["loss"]), wall=float(r["wall_s"]))
                for r in csv.DictReader(f)]


def rel_gain(prev, cur):
    return (prev - cur) / prev if prev > 0 else np.nan


def main():
    rows = []
    for cdir in sorted(glob.glob(os.path.join(CHAINS, "N[0-9][0-9][0-9]_q[0-9][0-9]"))):
        name = os.path.basename(cdir)
        N, q = int(name[1:4]), int(name[6:8])
        for rec_path in sorted(glob.glob(os.path.join(cdir, "leg[0-9][0-9][0-9].json"))):
            tag = os.path.basename(rec_path)[:-5]
            k = int(tag[3:])
            rec = json.load(open(rec_path))
            sc = json.load(open(os.path.join(cdir, tag + "_scale.json")))
            hist = read_history(os.path.join(cdir, tag + "_history.csv"))
            stall = [h for h in hist if h["phase"] == "stall"]
            conf = [h for h in hist if h["phase"] == "confirm"]
            # the last confirmation block and the stall restarts just before it
            last_conf, i = [], len(hist) - 1
            while i >= 0 and hist[i]["phase"] == "confirm":
                last_conf.insert(0, hist[i])
                i -= 1
            before = [h for h in hist[:i + 1] if h["phase"] == "stall"][-3:]
            first_wall = np.median([h["wall"] for h in stall[:5]])
            stall_wall = np.median([h["wall"] for h in before]) if before else np.nan
            conf_wall = np.median([h["wall"] for h in last_conf]) if last_conf else np.nan
            quit_flags = [h["wall"] < QUIT_WALL * first_wall and
                          (j > 0 and abs(rel_gain(hist[j - 1]["loss"], h["loss"])) < QUIT_GAIN)
                          for j, h in enumerate(hist)]
            n_quit = int(sum(quit_flags))
            confirm_quit = bool(last_conf) and all(quit_flags[len(hist) - len(last_conf):])
            stall_quit = bool(before) and all(quit_flags[j] for j, h in enumerate(hist[:i + 1])
                                              if h["phase"] == "stall" and h in before[-2:])
            gains = [rel_gain(a["loss"], b["loss"]) for a, b in zip(before[:-1], before[1:])]
            conf_gain = (rel_gain(before[-1]["loss"], last_conf[-1]["loss"])
                         if before and last_conf else np.nan)
            te, tr = rec["test"], rec["train"]
            rows.append(dict(
                N=N, q=q, leg=k, z0=sc["z0"], z1=sc["z1"],
                converged=rec.get("converged"), confirmed=rec.get("confirmed"),
                confirm_attempts=rec.get("confirm_attempts"), restarts=rec.get("restarts"),
                n_stall=len(stall), n_confirm=len(conf), final_loss=rec.get("final_loss"),
                last_stall_gain=gains[-1] if gains else np.nan,
                stall_wall_s=stall_wall, confirm_wall_s=conf_wall,
                confirm_wall_ratio=conf_wall / stall_wall if stall_wall else np.nan,
                confirm_gain=conf_gain,
                first_wall_s=first_wall, n_quit_restarts=n_quit,
                confirm_quit=confirm_quit, stall_quit=stall_quit,
                own_test_um=te["endpoint_med_um"], straight_test_um=te["straight_med_um"],
                own_over_straight_test=te["endpoint_med_um"] / te["straight_med_um"],
                own_train_um=tr["endpoint_med_um"],
                own_over_straight_train=tr["endpoint_med_um"] / tr["straight_med_um"],
                rho_median_test=te.get("rho_median"),
            ))
    os.makedirs(OUT, exist_ok=True)
    keys = list(rows[0].keys())
    with open(os.path.join(OUT, "leg_quality.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    summ = []
    for N in sorted({r["N"] for r in rows}):
        R = [r for r in rows if r["N"] == N]
        ratio = np.array([r["own_over_straight_test"] for r in R])
        cw = np.array([r["confirm_wall_ratio"] for r in R])
        summ.append(dict(
            N=N, legs=len(R),
            converged=sum(bool(r["converged"]) for r in R),
            at_restart_cap=sum(int(r["restarts"]) >= 400 for r in R),
            confirm_is_optimiser_quit=sum(r["confirm_quit"] for r in R),
            stall_is_optimiser_quit=sum(r["stall_quit"] for r in R),
            legs_with_any_quit_restart=sum(r["n_quit_restarts"] > 0 for r in R),
            final_loss_median=float(np.median([r["final_loss"] for r in R])),
            last_stall_gain_median=float(np.nanmedian([r["last_stall_gain"] for r in R])),
            confirm_wall_ratio_median=float(np.nanmedian(cw)),
            own_over_straight_median=float(np.median(ratio)),
            own_over_straight_p90=float(np.quantile(ratio, 0.9)),
            legs_ratio_ge_0p5=int((ratio >= 0.5).sum()),
            legs_ratio_ge_0p8=int((ratio >= 0.8).sum()),
            legs_ratio_ge_1=int((ratio >= 1.0).sum()),
        ))
    with open(os.path.join(OUT, "leg_quality_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys()))
        w.writeheader()
        w.writerows(summ)
    for s in summ:
        print(s)


if __name__ == "__main__":
    main()
