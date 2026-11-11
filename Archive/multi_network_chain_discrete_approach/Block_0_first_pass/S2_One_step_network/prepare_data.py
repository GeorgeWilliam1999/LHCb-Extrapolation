#!/usr/bin/env python
"""Freeze the leg: re-base the real cross-magnet states to the exact modal
planes and package everything the six runs need.

- Take the forward B legs from the event-derived training set (per split).
- Transport each state from its own UT plane to exactly z0 = 2648.2 mm with
  the fp64 RK4 engine (|dz| <= ~25 mm, exact within the label convention),
  set the target to z1 = 7826.0 mm.
- Compute, for the data twin and for scoring only: the RK4 reference state at
  every Gauss node z0 + c_j*dz (q = 8) and at z1.
- Record the fixed, label-free scales: input scale = std of the input states
  (train), output scale = std of the STRAIGHT-LINE-propagated states over the
  node span (no labels involved), residual scale = input scale.

Output: results/frozen_leg_data.npz
Run:  /data/bfys/gscriven/conda/envs/TE/bin/python prepare_data.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "S0_Baseline_data_exploration"))
sys.path.insert(0, os.path.join(HERE, "..", "S1_Simple_first_pass"))
from reference_card import FROZEN_LEG, load_training, rk4_rows  # noqa: E402
from irk import tableau, verify_tableau  # noqa: E402

RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

Q = 8
Z0, Z1 = FROZEN_LEG["z0"], FROZEN_LEG["z1"]
N_TRAIN = 2000
SEED = 20260718

assert verify_tableau(Q)["passes"]
c, A, b = tableau(Q)
ZNODES = Z0 + c * (Z1 - Z0)          # the q fixed stage planes
ZOUT = np.append(ZNODES, Z1)         # nodes + endpoint


def rebase(split, cap=None):
    d = load_training(split=split, leg=1)
    fwd = d["X"][:, 6] > d["X"][:, 5]
    X = d["X"][fwd].astype(np.float64)
    P = d["P"][fwd].astype(np.float64)
    near = np.abs(X[:, 5] - Z0) < 60.0          # own UT plane within 60 mm
    X, P = X[near], P[near]
    S = rk4_rows(X[:, :5], X[:, 5], np.full(len(X), Z0))   # exact transport
    ok = np.isfinite(S).all(axis=1)
    S, P = S[ok], P[ok]
    if cap is not None and len(S) > cap:
        idx = np.random.default_rng(SEED).permutation(len(S))[:cap]
        S, P = S[idx], P[idx]
    # reference at all nodes + endpoint (for the twin / scoring only)
    ref = np.empty((len(S), Q + 1, 5))
    cur, zprev = S.copy(), Z0
    for j, zt in enumerate(ZOUT):
        cur = rk4_rows(cur, np.full(len(S), zprev), np.full(len(S), zt))
        ref[:, j] = cur
        zprev = zt
    return S, P, ref


train_S, train_P, train_ref = rebase("train", cap=N_TRAIN)
val_S, val_P, val_ref = rebase("val")
test_S, test_P, test_ref = rebase("test")
print("states: train %d  val %d  test %d" % (len(train_S), len(val_S), len(test_S)))

# fixed label-free scales
in_scale = train_S.std(axis=0)                       # (5,)
# straight-line propagation of the training states to every node (no labels)
dzs = ZOUT - Z0
straight = np.repeat(train_S[:, None, :], Q + 1, axis=1)
straight[:, :, 0] += train_S[:, None, 2] * dzs[None, :, None][:, :, 0]
straight[:, :, 1] += train_S[:, None, 3] * dzs[None, :, None][:, :, 0]
out_scale = straight.reshape(-1, 5).std(axis=0)      # (5,)

np.savez_compressed(
    os.path.join(RES, "frozen_leg_data.npz"),
    train_S=train_S, train_P=train_P, train_ref=train_ref,
    val_S=val_S, val_P=val_P, val_ref=val_ref,
    test_S=test_S, test_P=test_P, test_ref=test_ref,
    znodes=ZNODES, zout=ZOUT, z0=Z0, z1=Z1, q=Q,
    in_scale=in_scale, out_scale=out_scale,
)
meta = {
    "leg": {"z0": Z0, "z1": Z1, "q": Q},
    "counts": {"train": len(train_S), "val": len(val_S), "test": len(test_S)},
    "in_scale": in_scale.tolist(), "out_scale": out_scale.tolist(),
    "rebase": "own UT plane within 60 mm, fp64 RK4 transport to z0 (exact)",
    "seed": SEED,
}
with open(os.path.join(RES, "frozen_leg_meta.json"), "w") as f:
    json.dump(meta, f, indent=1)
print(json.dumps(meta, indent=1))
