#!/usr/bin/env python
"""E1 - re-score every network from its current weights and rewrite its record.

Training was stopped by hand on 2026-09-18 (the chain error had been flat for
ten rounds while the loss kept falling within rounds). A run that was stopped
mid-flight still carries the record of its last completed pass, so the record
and the weights disagree. This script rebuilds each record from the checkpoint
that is actually on disk: it carries the validation and test tracks through the
chain, scores them exactly as the trainer does, and rewrites `record.json`,
`chain_states.npz` and the progress file.

`stopped_by_hand` marks the runs whose training was cut short this way, and
`restarts` is what the history actually contains.

Run:  PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python finalise.py
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"

import csv     # noqa: E402
import glob    # noqa: E402
import json    # noqa: E402
import time    # noqa: E402

import numpy as np   # noqa: E402
import torch         # noqa: E402

import use_shared    # noqa: E402,F401
from _shared.reference import make_field                    # noqa: E402
from chain_network import carry, load_network               # noqa: E402
from metrics import chain_scores                            # noqa: E402

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

HERE = os.path.dirname(os.path.abspath(__file__))
TRACKS = os.path.join(HERE, "..", "E0_Track_dataset", "results", "tracks.npz")


def main():
    D = {k: v for k, v in np.load(TRACKS, allow_pickle=False).items()}
    Z0 = float(D["z0"])
    fld = make_field(str(D["field"]))
    for run in sorted(glob.glob(os.path.join(HERE, "results", "N[0-9][0-9][0-9]_q[0-9][0-9]"))):
        sc = json.load(open(os.path.join(run, "scale.json")))
        prog = json.load(open(os.path.join(run, "progress.json")))
        hist = list(csv.DictReader(open(os.path.join(run, "history.csv"))))
        old = json.load(open(os.path.join(run, "record.json"))) if os.path.exists(
            os.path.join(run, "record.json")) else {}
        N, q = sc["N"], sc["q"]
        model = load_network(run, fld)
        t0 = time.time()
        states, cost = {}, {}
        for s in ("val", "test"):
            t1 = time.time()
            states[s] = carry(model, D["%s_S0" % s], Z0, N)
            cost[s] = (time.time() - t1) / len(states[s]) * 1e6
        scores = chain_scores(states, D, N, fld)
        np.savez_compressed(os.path.join(run, "chain_states.npz"),
                            **{"%s_states" % s: v for s, v in states.items()})
        rec = dict(
            N=N, q=q, dz_mm=sc["dz"], seed=sc["seed"], width=sc["width"], depth=sc["depth"],
            field=sc["field"], n_parameters=int(sum(p.numel() for p in model.parameters())),
            n_train_tracks=sc["n_train_tracks"], states_per_round=sc["states_per_round"],
            round_restarts=old.get("round_restarts"), outer_cap=old.get("outer_cap"),
            round_cap=old.get("round_cap"),
            rounds=int(prog["round"]), restarts=len(hist),
            converged=bool(prog.get("converged")), confirmed=prog.get("confirmed"),
            hit_cap=prog.get("hit_cap"),
            stopped_by_hand=bool(prog.get("phase") != "done"),
            early_stop_restarts=int(sum(int(r["early_stop"]) for r in hist)),
            final_loss=float(hist[-1]["loss_after"]),
            train_wall_s=round(sum(float(r["wall_s"]) for r in hist), 1),
            carry_us_per_track=cost, rescored=time.strftime("%Y-%m-%d %H:%M"), **scores)
        with open(os.path.join(run, "record.json"), "w") as f:
            json.dump(rec, f, indent=1)
        prog["phase"] = "done"
        with open(os.path.join(run, "progress.json"), "w") as f:
            json.dump(prog, f, indent=1)
        t = rec["test"]["vs_rk6_endpoint"]
        print("N=%3d q=%2d: %4d restarts, %2d rounds%s -> test median %.1f um (p95 %.1f), %.0f s"
              % (N, q, rec["restarts"], rec["rounds"], " (stopped by hand)" if rec["stopped_by_hand"] else "",
                 t["pos_med_um"], t["pos_p95_um"], time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
