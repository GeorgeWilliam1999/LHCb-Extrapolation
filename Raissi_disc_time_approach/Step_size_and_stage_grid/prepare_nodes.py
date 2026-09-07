#!/usr/bin/env python
"""Build the per-q training set: the RK6 reference at the Gauss nodes.

`../Magnet_tracks_dataset/results/magnet_tracks_v3.npz` stores each row's start
state and its **end** state. A discrete-time network is scored - and, in the
data arm, trained - against the reference at the q Gauss-Legendre nodes as well,
and those node planes z_j = z0 + c_j*dz move with q. So one file per q is
needed, and this script builds it.

    results/grid_q<qq>.npz

in the layout `../_shared/train.py` and `../_shared/evaluate.py` already
understand (`kind = "general"`, per-sample `z0`, `dz`, `znodes` and the
normalised `extra`), plus the Block C columns the per-stratum scoring needs
(`STRATUM`, `DIRECTION`, `P`, `ETA`).

## How the node states are computed

Each row is marched with `rk6_rows` at 0.1 mm from its own z0 to node 1, then
from node 1 to node 2, and so on to the endpoint, in fp64 on the **MagUp** map -
the same integrator, step and polarity the dataset's own `Y` column was built
with. Marching in segments rather than restarting from z0 for every node costs
one crossing per row instead of q/2 of them, and the price is that a node is
reached through q shortened last steps rather than one. That price is measured,
not assumed: the composed endpoint is compared with the dataset's own `Y`
(which `../Magnet_tracks_dataset` produced in a single unsegmented march) and
the difference is recorded in the meta json under
`endpoint_composition_vs_dataset_um`. `Y` itself is then used as the endpoint
row of the reference, so the endpoint label is bit-identical to the published
dataset and only the interior nodes come from the segmented march.

## The rows

The evaluation splits are taken whole (2,000 rows per stratum per split). The
training split is subsampled **evenly across the six strata** to `--n-train`
rows in total, with a fixed permutation seeded once per stratum, so that every
q uses the *same* training rows and two points of the grid differ only in q.
`--n-train 36000` takes the whole train split.

    PY=/data/bfys/gscriven/conda/envs/TE/bin/python
    PYTHONNOUSERSITE=1 $PY prepare_nodes.py --q 20 --n-train 36000 --workers 12
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"

import argparse
import json
import time
from multiprocessing import Pool

import numpy as np

import use_shared                                        # noqa: F401
from _shared.prepare import STRATUM_NAMES
from _shared.reference import (RK6_STEP, field_md5, field_path,
                               gauss_legendre, make_field, rk6_rows)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
SOURCE = os.path.join(HERE, "results", "magnet_tracks_v3_residual.npz")
SPLITS = ("train", "val", "test")
ROW_SEED = 20260907
_FIELD_WHICH = "up"


# ------------------------------------------------------ the worker's march --
def _node_reference(args):
    """(index, ref (n, q+1, 5)) for one chunk of rows, marched node by node."""
    idx, S, z0, zout, which, step = args
    fld = make_field(which)
    ref = np.empty((len(S), zout.shape[1], 5))
    cur, zprev = np.asarray(S, dtype=np.float64).copy(), np.asarray(z0).copy()
    for j in range(zout.shape[1]):
        cur = rk6_rows(cur, zprev, zout[:, j], step=step, field=fld)
        ref[:, j] = cur
        zprev = zout[:, j]
    return idx, ref


def node_reference(S, z0, zout, which=_FIELD_WHICH, step=RK6_STEP, workers=1):
    """(N, q+1, 5) the RK6 reference at every node plane and the endpoint.

    Rows are dealt round-robin into `workers` chunks after sorting by |dz|, so
    every chunk carries the same mix of long and short steps and the wall time
    is the mean rather than the max.
    """
    n = len(S)
    if n == 0:
        return np.empty((0, zout.shape[1], 5))
    if workers <= 1:
        return _node_reference((np.arange(n), S, z0, zout, which, step))[1]
    order = np.argsort(np.abs(zout[:, -1] - z0), kind="stable")[::-1]
    chunks = [order[k::workers] for k in range(workers)]
    chunks = [c for c in chunks if len(c)]
    tasks = [(c, S[c], z0[c], zout[c], which, step) for c in chunks]
    ref = np.empty((n, zout.shape[1], 5))
    with Pool(len(tasks)) as pool:
        for idx, part in pool.imap_unordered(_node_reference, tasks):
            ref[idx] = part
    return ref


# ------------------------------------------------------------ the row pick --
def pick_rows(STRAT, SPLIT, split_index, n_total):
    """Row indices for one split: `n_total` spread evenly over the strata."""
    per = int(np.ceil(n_total / len(STRATUM_NAMES)))
    out = []
    for i in range(len(STRATUM_NAMES)):
        pool = np.flatnonzero((SPLIT == split_index) & (STRAT == i))
        rng = np.random.default_rng(ROW_SEED + 1000 * split_index + i)
        take = pool if len(pool) <= per else pool[rng.permutation(len(pool))[:per]]
        out.append(np.sort(take))
    return np.concatenate(out)


def _straight_out_scale(S, dzs):
    """The straight-line output scale of `_shared.prepare`, same formula."""
    n, k = len(S), dzs.shape[-1]
    straight = np.repeat(S[:, None, :], k, axis=1)
    d = np.broadcast_to(dzs, (n, k))
    straight[:, :, 0] += S[:, None, 2] * d
    straight[:, :, 1] += S[:, None, 3] * d
    return straight.reshape(-1, 5).std(axis=0)


def build(q, n_train, n_eval, source=SOURCE, which=_FIELD_WHICH,
          step=RK6_STEP, workers=1, out_npz=None, verbose=True):
    t0 = time.time()
    d = np.load(os.path.abspath(source))
    X = np.asarray(d["X"], dtype=np.float64)
    Y = np.asarray(d["Y"], dtype=np.float64)
    STRAT, DIR, SPLIT = np.asarray(d["STRATUM"]), np.asarray(d["DIRECTION"]), \
        np.asarray(d["SPLIT"])
    P, ETA = np.asarray(d["P"]), np.asarray(d["ETA"])

    c, _, _ = gauss_legendre(q)
    arrays = {"kind": "general", "q": q, "c": c, "field": which,
              "stratum_names": np.array(list(STRATUM_NAMES)),
              "rk6_step_mm": np.array(step),
              "row_seed": np.array(ROW_SEED)}
    picked, comp = {}, {}
    for si, split in enumerate(SPLITS):
        rows = pick_rows(STRAT, SPLIT, si, n_train if split == "train" else n_eval)
        picked[split] = rows
        S, z0, dz = X[rows, :5], X[rows, 5], X[rows, 6]
        znodes = z0[:, None] + c[None, :] * dz[:, None]              # (N, q)
        zout = np.concatenate([znodes, (z0 + dz)[:, None]], axis=1)  # (N, q+1)
        ref = node_reference(S, z0, zout, which=which, step=step, workers=workers)
        err_um = np.abs(ref[:, -1, :2] - Y[rows, :2]).max(axis=1) * 1e3
        comp[split] = {"median_um": float(np.median(err_um)),
                       "p95_um": float(np.quantile(err_um, 0.95)),
                       "max_um": float(err_um.max())}
        ref[:, -1] = Y[rows]        # the dataset's own unsegmented endpoint
        if not np.isfinite(ref).all():
            raise SystemExit("non-finite reference in split %s" % split)
        arrays["%s_S" % split] = S
        arrays["%s_ref" % split] = ref
        arrays["%s_z0" % split] = z0
        arrays["%s_dz" % split] = dz
        arrays["%s_znodes" % split] = znodes
        arrays["%s_STRATUM" % split] = STRAT[rows]
        arrays["%s_DIRECTION" % split] = DIR[rows]
        arrays["%s_P" % split] = P[rows]
        arrays["%s_ETA" % split] = ETA[rows]
        if verbose:
            print("  q=%d %-5s %6d rows  (%.0f s so far)"
                  % (q, split, len(rows), time.time() - t0), flush=True)

    tr_S = arrays["train_S"]
    tr_z0, tr_dz = arrays["train_z0"], arrays["train_dz"]
    in_scale = tr_S.std(axis=0)
    out_scale = _straight_out_scale(
        tr_S, np.concatenate([c[None, :] * tr_dz[:, None], tr_dz[:, None]],
                             axis=1))
    extra_mean = np.array([tr_z0.mean(), tr_dz.mean()])
    extra_scale = np.array([tr_z0.std(), tr_dz.std()])
    extra_scale[extra_scale == 0] = 1.0
    arrays.update(in_scale=in_scale, out_scale=out_scale,
                  extra_mean=extra_mean, extra_scale=extra_scale)
    for split in SPLITS:
        z0, dz = arrays["%s_z0" % split], arrays["%s_dz" % split]
        arrays["%s_extra" % split] = np.stack(
            [(z0 - extra_mean[0]) / extra_scale[0],
             (dz - extra_mean[1]) / extra_scale[1]], axis=1)

    meta = {
        "kind": "general", "built_by": "Step_size_and_stage_grid/prepare_nodes.py",
        "source_dataset": os.path.relpath(os.path.abspath(source), HERE),
        "q": q, "n_train_requested": n_train, "n_eval_requested": n_eval,
        "counts": {s: int(len(picked[s])) for s in SPLITS},
        "counts_per_stratum": {
            s: {STRATUM_NAMES[i]: int((STRAT[picked[s]] == i).sum())
                for i in range(len(STRATUM_NAMES))} for s in SPLITS},
        "reference": {
            "integrator": "fp64 fixed-step RK6 (Butcher, 7 stages, order 6)",
            "step_mm": step,
            "marched": "segment by segment through the q Gauss nodes; the "
                       "endpoint row is the dataset's own Y",
            "field": {"which": which, "file": field_path(which),
                      "md5": field_md5(which)},
        },
        "endpoint_composition_vs_dataset_um": comp,
        "row_seed": ROW_SEED,
        "in_scale": in_scale.tolist(), "out_scale": out_scale.tolist(),
        "extra_mean": extra_mean.tolist(), "extra_scale": extra_scale.tolist(),
        "wall_s": round(time.time() - t0, 1),
    }
    if out_npz:
        os.makedirs(os.path.dirname(os.path.abspath(out_npz)) or ".",
                    exist_ok=True)
        np.savez_compressed(os.path.abspath(out_npz), **arrays)
        with open(os.path.splitext(os.path.abspath(out_npz))[0] + "_meta.json",
                  "w") as f:
            json.dump(meta, f, indent=1)
    if verbose:
        print("q=%d done in %.0f s; endpoint composition vs the dataset: "
              "median %.2e um, max %.2e um"
              % (q, meta["wall_s"], comp["test"]["median_um"],
                 max(comp[s]["max_um"] for s in SPLITS)), flush=True)
    return arrays, meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--q", type=int, required=True)
    ap.add_argument("--n-train", type=int, default=36000)
    ap.add_argument("--n-eval", type=int, default=12000)
    ap.add_argument("--source", default=SOURCE)
    ap.add_argument("--field", choices=("down", "up"), default=_FIELD_WHICH)
    ap.add_argument("--step", type=float, default=RK6_STEP)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    out = a.out or os.path.join(RESULTS, "grid_q%02d.npz" % a.q)
    build(a.q, a.n_train, a.n_eval, source=a.source, which=a.field,
          step=a.step, workers=a.workers, out_npz=out)


if __name__ == "__main__":
    main()
