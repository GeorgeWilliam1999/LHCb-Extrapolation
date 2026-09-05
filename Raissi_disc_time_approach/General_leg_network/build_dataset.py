#!/usr/bin/env python
"""Build the one-network-for-all-legs training set, and size it by a timing smoke.

The dataset is the shared general-leg builder (`_shared/prepare.py`), which gives
every sample its own start plane z0 and step length dz and hands the network the
normalised (z0, dz) as two extra inputs.

How the training size is chosen. The training runs must finish overnight: 150
L-BFGS restarts is the safety cap, so one restart has to stay well under two
minutes. The rule stated before looking at any number: take the largest N in
{2000, 4000, 8000, 16000} for which ONE L-BFGS restart (200 iterations, physics
loss, 4x100 network, one thread) takes under 90 s, so that 150 restarts is about
four hours.

The timing is measured on truncated copies of one large pool, so that only the
number of training states differs between the four measurements.

Outputs
    results/general_legs.npz        the dataset the runs train on
    results/general_legs_meta.json  the builder's own meta (counts, scales)
    results/dataset_meta.json       the choice of N, the timing table, the
                                    per-leg / per-split / per-momentum mix and
                                    the sign distribution of dz per leg type
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"

import csv
import json
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

import use_shared                                        # noqa: F401
from _shared.prepare import general_leg_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
SHARED = os.path.join(os.path.dirname(HERE), "_shared")
PY = "/data/bfys/gscriven/conda/envs/TE/bin/python"
CANDIDATES = (2000, 4000, 8000, 16000)
BUDGET_S = 90.0
LEGS = ("A", "B", "C")
BANDS = ((1.0, 5.0), (5.0, 20.0), (20.0, 200.0))
SPLITS = ("train", "val", "test")


def band_of(p):
    for i, (lo, hi) in enumerate(BANDS):
        if lo <= p < hi:
            return i
    return len(BANDS) - 1 if p >= BANDS[-1][0] else 0


def time_one_restart(npz_path, n, workdir):
    """Wall time of a single physics L-BFGS restart (200 iterations) at 4x100."""
    tag = "timing_n%d" % n
    cmd = [PY, os.path.join(SHARED, "train.py"), "--data", npz_path,
           "--mode", "physics", "--seed", "0", "--width", "100", "--depth", "4",
           "--out", workdir, "--tag", tag, "--outer-cap", "1", "--no-confirm",
           "--max-iter", "200"]
    env = dict(os.environ)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    total = time.time() - t0
    if r.returncode != 0:
        raise SystemExit("timing run failed for N=%d:\n%s" % (n, r.stderr[-3000:]))
    with open(os.path.join(workdir, tag + "_history.csv")) as f:
        rows = list(csv.DictReader(f))
    return float(rows[0]["wall_s"]), total


def truncate_train(arrays, n, out_path):
    """A copy of the pool with the training split cut to its first n rows."""
    out = {}
    for k, v in arrays.items():
        v = np.asarray(v)
        if k.startswith("train_"):
            out[k] = v[:n]
        else:
            out[k] = v
    np.savez_compressed(out_path, **out)


def describe(arrays, npz_meta):
    """Counts, momentum mix and dz sign distribution, per split and leg type."""
    d = {"counts": {}, "per_leg": {}, "momentum_mix": {}, "dz_sign": {}}
    for split in SPLITS:
        L = np.asarray(arrays["%s_LEG" % split])
        P = np.asarray(arrays["%s_P" % split])
        dz = np.asarray(arrays["%s_dz" % split])
        d["counts"][split] = int(len(L))
        d["per_leg"][split] = {t: int((L == i).sum()) for i, t in enumerate("ABCD")}
        mix, sign = {}, {}
        for i, t in enumerate("ABCD"):
            m = L == i
            if not m.any():
                continue
            b = np.array([band_of(p) for p in P[m]])
            mix[t] = {"1-5GeV": int((b == 0).sum()), "5-20GeV": int((b == 1).sum()),
                      "20-200GeV": int((b == 2).sum()),
                      "p_median_GeV": float(np.median(P[m]))}
            sign[t] = {"forward_dz_gt_0": int((dz[m] > 0).sum()),
                       "backward_dz_lt_0": int((dz[m] < 0).sum()),
                       "dz_abs_median_mm": float(np.median(np.abs(dz[m])))}
        d["momentum_mix"][split] = mix
        d["dz_sign"][split] = sign
    d["builder_meta"] = npz_meta
    return d


def main():
    results = os.path.join(HERE, "results")
    os.makedirs(results, exist_ok=True)
    pool_npz = os.path.join(results, "timing_pool.npz")

    if os.path.exists(pool_npz):
        print("== reusing the timing pool already on disk ==", flush=True)
        _d = np.load(pool_npz)
        pool = {k: _d[k] for k in _d.files}
    else:
        print("== building the timing pool (n_train %d) =="
              % max(CANDIDATES), flush=True)
        t0 = time.time()
        pool = general_leg_dataset(legs=LEGS, q=8, n_train=max(CANDIDATES),
                                   field="down", out_npz=pool_npz,
                                   seed=20260718, n_eval=1000, fiducial=True,
                                   verbose=True)
        print("pool built in %.0f s" % (time.time() - t0), flush=True)

    workdir = tempfile.mkdtemp(prefix="timing_", dir=results)
    table = []
    for n in CANDIDATES:
        # the candidate is the builder's cap; the fiducial requirement removes
        # about 1% of the states, so the truncation takes as many rows as the
        # pool actually holds after the cut.
        n_rows = min(n, len(pool["train_S"]))
        p = os.path.join(workdir, "pool_n%d.npz" % n)
        truncate_train(pool, n_rows, p)
        restart_s, total_s = time_one_restart(p, n, workdir)
        table.append({"n_train_cap": n, "n_rows_timed": n_rows,
                      "restart_s": restart_s,
                      "process_s": round(total_s, 1),
                      "projected_150_restarts_h": round(restart_s * 150 / 3600.0, 2),
                      "under_budget": bool(restart_s < BUDGET_S)})
        print("  N=%5d  one restart %.1f s  -> 150 restarts %.2f h"
              % (n, restart_s, restart_s * 150 / 3600.0), flush=True)

    ok = [r["n_train_cap"] for r in table if r["under_budget"]]
    chosen = max(ok) if ok else min(r["n_train_cap"] for r in table)
    print("== chosen n_train = %d ==" % chosen, flush=True)

    out_npz = os.path.join(results, "general_legs.npz")
    print("== building the training set (n_train %d, n_eval 4000) ==" % chosen,
          flush=True)
    arrays = general_leg_dataset(legs=LEGS, q=8, n_train=chosen, field="down",
                                 out_npz=out_npz, seed=20260718, n_eval=4000,
                                 fiducial=True, verbose=True)
    with open(os.path.splitext(out_npz)[0] + "_meta.json") as f:
        npz_meta = json.load(f)

    meta = {
        "experiment": "A3a - one network for legs A, B, C",
        "dataset": os.path.relpath(out_npz, HERE),
        "legs": list(LEGS), "q": 8, "field": "down", "fiducial": True,
        "n_train_chosen": chosen, "n_eval": 4000, "seed": 20260718,
        "sizing_rule": ("largest N in %s whose single physics L-BFGS restart "
                        "(200 iterations, 4x100, one thread) runs under %.0f s"
                        % (list(CANDIDATES), BUDGET_S)),
        "timing_table": table,
        "stratification": ("the shared builder caps the pool with a plain random "
                           "permutation - it does NOT stratify by leg type or by "
                           "momentum, so the mix below is the natural mix of the "
                           "training set and is recorded rather than imposed"),
    }
    meta.update(describe(arrays, npz_meta))
    with open(os.path.join(results, "dataset_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)

    shutil.rmtree(workdir, ignore_errors=True)
    os.remove(pool_npz)
    pm = os.path.splitext(pool_npz)[0] + "_meta.json"
    if os.path.exists(pm):
        os.remove(pm)
    print(json.dumps({k: meta[k] for k in
                      ("n_train_chosen", "counts", "per_leg", "timing_table")},
                     indent=1))


if __name__ == "__main__":
    main()
