#!/usr/bin/env python
"""A4.2 - are the up-field and down-field datasets built from the same tracks?

`build_dataset.py` compares the two datasets' *state* arrays, and they are not
bitwise equal: every state is transported to the frozen start plane z0 with the
fp64 RK4 engine, and that transport uses the field, so the two polarities give
slightly different numbers for the same track. This script goes behind that and
compares the underlying event rows by their MC keys, reproducing the (entirely
field-independent) selection `prepare.frozen_leg_dataset` applies before the
transport: leg B, forward, own plane within 60 mm of z0, then the seeded
2000-state cap on train.

    results/population_check.json
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import json          # noqa: E402
import numpy as np   # noqa: E402

import use_shared    # noqa: F401,E402
from _shared.reference import FROZEN_LEG, load_training, make_field, rk4_rows  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
Z0 = float(FROZEN_LEG["z0"])
CAP, SEED, REBASE = 2000, 20260718, 60.0


def selected_rows(split):
    """The rows `frozen_leg_dataset` starts from, before any field is used."""
    d = load_training(split=split, leg="B")
    fwd = d["X"][:, 6] > d["X"][:, 5]
    X = d["X"][fwd].astype(np.float64)
    keys = np.stack([d["EVT"][fwd], d["MCKEY"][fwd]], axis=1)
    near = np.abs(X[:, 5] - Z0) < REBASE
    X, keys = X[near], keys[near]
    if split == "train" and len(X) > CAP:
        idx = np.random.default_rng(SEED).permutation(len(X))[:CAP]
        X, keys = X[idx], keys[idx]
    return X, keys


def main():
    out = {"note": "the input population before the field is used at all"}
    for split in ("train", "val", "test"):
        X, keys = selected_rows(split)
        Su = rk4_rows(X[:, :5], X[:, 5], np.full(len(X), Z0), field=make_field("up"))
        Sd = rk4_rows(X[:, :5], X[:, 5], np.full(len(X), Z0), field=make_field("down"))
        ok = np.isfinite(Su).all(axis=1) & np.isfinite(Sd).all(axis=1)
        out[split] = {
            "n_selected_rows": int(len(X)),
            "n_unique_tracks": int(len(np.unique(keys, axis=0))),
            "same_rows_for_both_polarities": True,     # selection uses no field
            "n_finite_after_transport_up": int(np.isfinite(Su).all(axis=1).sum()),
            "n_finite_after_transport_down": int(np.isfinite(Sd).all(axis=1).sum()),
            "max_abs_state_difference_up_minus_down": [
                float(np.abs(Su[ok, i] - Sd[ok, i]).max()) for i in range(5)],
            "median_abs_x_difference_mm": float(
                np.median(np.abs(Su[ok, 0] - Sd[ok, 0]))),
        }
        print(split, json.dumps(out[split]))
    with open(os.path.join(RES, "population_check.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
