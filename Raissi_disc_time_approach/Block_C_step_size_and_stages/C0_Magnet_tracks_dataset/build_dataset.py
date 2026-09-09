#!/usr/bin/env python
"""C0 - build the magnet-to-magnet track dataset.

A thin driver. Everything it does lives in
`../../_shared/prepare.magnet_tracks_dataset`, so that the dataset can be rebuilt
from the shared package by anyone, and so that the selection it uses
(`magnet_leg_rows`) is literally the same code the fine-reference study
(`../C1_Fine_reference`) drew its 200 legs with.

    PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python build_dataset.py

Options worth knowing:

    --particles N     how many particles to integrate (a compute budget: the
                      RK6 path costs about a tenth of a second per leg)
    --field up|down   the polarity. Default 'up': the sample is a MagUp sample,
                      see check_polarity.py. 'down' builds the twin.
    --quick           a 40-particle smoke build, for checking the plumbing

Outputs:
    results/magnet_tracks_v3.npz        X, Y and the labels
    results/magnet_tracks_v3_meta.json  the cut cascade and every count
    results/dense_states.npz            the RK6 paths, every 10 mm
    results/dataset_meta.json           a copy of the meta, at the name C0 asks
                                        for, with the cut cascade at the top
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"

import argparse
import json

import use_shared                                            # noqa: F401
from _shared.prepare import magnet_tracks_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--particles", type=int, default=6000)
    ap.add_argument("--field", choices=("up", "down"), default="up")
    ap.add_argument("--step", type=float, default=0.1,
                    help="the RK6 reference step in mm (C1.4 says 0.1)")
    ap.add_argument("--sample-mm", type=float, default=10.0)
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-eval", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260718)
    ap.add_argument("--tag", default="magnet_tracks_v3")
    ap.add_argument("--quick", action="store_true",
                    help="40 particles, for checking the plumbing")
    a = ap.parse_args()
    if a.quick:
        a.particles, a.n_train, a.n_eval = 40, 200, 60
        a.tag += "_quick"

    os.makedirs(RESULTS, exist_ok=True)
    out_npz = os.path.join(RESULTS, a.tag + ".npz")
    dense_npz = os.path.join(RESULTS, ("dense_states_quick.npz" if a.quick
                                       else "dense_states.npz"))
    arrays = magnet_tracks_dataset(
        out_npz=out_npz, dense_npz=dense_npz, n_particles=a.particles,
        n_train=a.n_train, n_eval=a.n_eval, sample_mm=a.sample_mm,
        step=a.step, field=a.field, seed=a.seed, verbose=True)

    meta = arrays["_meta"]
    meta["built_by"] = "C0_Magnet_tracks_dataset/build_dataset.py"
    meta["dense_states_file"] = os.path.relpath(dense_npz, HERE)
    meta["dataset_file"] = os.path.relpath(out_npz, HERE)
    name = "dataset_meta_quick.json" if a.quick else "dataset_meta.json"
    with open(os.path.join(RESULTS, name), "w") as f:
        json.dump(meta, f, indent=1)

    print("\ncut cascade")
    for c in meta["cut_cascade"]:
        print("  %-58s %8d -> %8d   (%d particles)"
              % (c["cut"], c["rows_in"], c["rows_out"], c["particles_out"]))
    print("\nrows: %d" % len(arrays["X"]))


if __name__ == "__main__":
    main()
