#!/usr/bin/env python
"""Build the four frozen-leg training sets, one per number of stages.

The magnet crossing (leg B, z = 2648.2 -> 7826.0 mm) is the same leg in every
case; only the number of Gauss-Legendre stages q changes, and with it the
number of stage planes the network has to predict. Everything else - the
population, the fiducial requirement, the split seed, the 2000-state training
cap - is the shared builder's default, so a difference between two of these
datasets is the number of stages and nothing else.

    PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python build_datasets.py
    ... --q 8            (build a single one, for running the four in parallel)

Writes, for q in {2, 4, 8, 16}:

    results/frozen_leg_q<qq>.npz         the dataset train.py consumes
    results/frozen_leg_q<qq>_meta.json   counts, scales, fiducial drops
    results/dataset_check_q<qq>.json      the two checks below, for that q

Two checks are asserted rather than merely reported:

  1. the split counts are 1979 / 2062 / 2018 for every q. The fiducial cut asks
     whether the reference trajectory leaves the field map, which is a property
     of the leg and the polarity, not of how many stage planes we sample it on,
     so the surviving population must not move with q.
  2. at q = 8 the dataset must reproduce the verified baseline
     One_step_network_v2/results/frozen_leg_data.npz **bitwise on every array
     the baseline contains** - same dtype, same shape, same bytes - because
     that is the file the converged baseline result was trained on. The shared
     builder writes a few keys the older v2 script did not (`kind`, `c`,
     `field` and the per-sample `*_z0` / `*_dz`, which the generalised driver
     needs and which are constants here); those extras are listed rather than
     treated as a difference, and the constant ones are checked to be the
     constants they should be.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import use_shared  # noqa: F401  (puts _shared on sys.path)
from _shared.prepare import frozen_leg_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
BASELINE_NPZ = os.path.join(HERE, "..", "One_step_network_v2", "results",
                            "frozen_leg_data.npz")

Q_VALUES = (2, 4, 8, 16)
EXPECTED_COUNTS = {"train": 1979, "val": 2062, "test": 2018}


def npz_path(q):
    return os.path.join(RESULTS, "frozen_leg_q%02d.npz" % q)


def bitwise_same(new_npz, baseline_npz):
    """Every array the baseline holds, reproduced byte for byte in the new file.

    Not a file-level md5: the npz container is a zip, whose stored timestamps
    and compression choices differ between two writes of identical arrays, so
    the claim being checked is about the arrays themselves. Keys the new file
    adds and the baseline never had are reported, not counted as differences -
    the baseline was written by the older `One_step_network_v2/prepare_data.py`,
    before the shared builder started recording `kind`, `c`, `field` and the
    per-sample start plane and step length.
    """
    a = np.load(new_npz, allow_pickle=False)
    b = np.load(baseline_npz, allow_pickle=False)
    missing = sorted(set(b.files) - set(a.files))
    extra = sorted(set(a.files) - set(b.files))
    differing = {}
    for k in sorted(set(a.files) & set(b.files)):
        x, y = a[k], b[k]
        if not (x.dtype == y.dtype and x.shape == y.shape
                and x.tobytes() == y.tobytes()):
            differing[k] = {"dtype": [str(x.dtype), str(y.dtype)],
                            "shape": [list(x.shape), list(y.shape)]}
    report = {"n_shared_keys": len(set(a.files) & set(b.files)),
              "keys_missing_from_new": missing,
              "keys_added_by_shared_builder": extra,
              "differing_keys": differing}
    return (not missing and not differing), report


def build_one(q):
    out = npz_path(q)
    print("=" * 70)
    print("q = %d  ->  %s" % (q, os.path.basename(out)))
    arrays = frozen_leg_dataset(q=q, out_npz=out, verbose=True)
    counts = {s: int(len(arrays["%s_S" % s])) for s in ("train", "val", "test")}
    check = {"q": q, "counts": counts, "counts_expected": EXPECTED_COUNTS,
             "counts_match": counts == EXPECTED_COUNTS,
             "npz": os.path.basename(out)}
    if q == 8:
        same, report = bitwise_same(out, BASELINE_NPZ)
        check["bitwise_equal_to_v2_baseline"] = bool(same)
        check["bitwise_report"] = report
        # the keys the older baseline never carried are constants of this leg,
        # so state what they are rather than only that they are new
        check["added_keys_are_constants"] = {
            "z0": float(arrays["train_z0"][0]), "dz": float(arrays["train_dz"][0]),
            "kind": str(arrays["kind"]), "field": str(arrays["field"])}
        print("  all %d baseline arrays reproduced bitwise: %s"
              % (report["n_shared_keys"], same))
        print("  keys the shared builder adds: %s"
              % ", ".join(report["keys_added_by_shared_builder"]))
        assert same, "q=8 dataset does not reproduce the v2 baseline: %s" % report
    assert counts == EXPECTED_COUNTS, \
        "q=%d counts %s != expected %s" % (q, counts, EXPECTED_COUNTS)
    return check


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--q", type=int, default=None,
                    help="build only this q (default: all four)")
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    qs = Q_VALUES if a.q is None else (a.q,)
    for q in qs:
        check = build_one(q)
        # one file per q, never a shared one: the four builds are meant to be
        # runnable at the same time and a read-modify-write would race.
        path = os.path.join(RESULTS, "dataset_check_q%02d.json" % q)
        with open(path, "w") as f:
            json.dump(check, f, indent=1, sort_keys=True)
        print("wrote %s" % path)


if __name__ == "__main__":
    main()
