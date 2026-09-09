#!/usr/bin/env python
"""Assemble the real particle paths the network will be chained along.

A chain is one particle's own ordered sequence of planes. For every particle in
a split we take its FORWARD training rows (z1 > z0), collect the planes they
name - every z0 and every z1 - and sort them. Consecutive planes in that list
are the legs of the chain, and the chain starts from the state stored at the
first plane.

Why the planes and not the rows. The obvious rule, "row i is followed by row j
when z1(i) equals z0(j)", almost never fires four times in a row: only 15 of the
7,636 test particles have four rows that butt together exactly, because the
training set stores at most three plane-to-plane legs per particle and they need
not be adjacent. 42% of consecutive row pairs do butt together exactly; the rest
leave a gap, and the gap is simply a piece of the particle's path that no stored
row covers. Treating the sorted planes as the waypoints keeps every stored plane
in the chain and fills those gaps with one further leg of the same kind, which
is what a real extrapolator would have to do anyway. It gives 4,563 test and
4,551 val particles with four or more legs.

The reference is not a stored label: it is the fp64 RK4 path from the same start
state through the same planes (`chain_reference`), so the network and the truth
begin at the same place and the comparison is of the propagation alone.

Leg D. The downstream leg (first T state back to the primary vertex, one giant
backward step) is kept separately, as the composite test: the same network is
asked to walk it in two steps, first back to the particle's UT plane and then
back to the vertex plane, and is scored against the stored D-leg label.

Outputs
    results/chains_test.npz, results/chains_val.npz
    results/leg_d_test.npz
    results/chains_meta.json      how many particles and legs qualified
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"

import json

import numpy as np

import use_shared                                        # noqa: F401
from _shared.reference import FROZEN_LEG, load_training

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
MIN_LEGS = 4
ROUND_MM = 3          # the planes are stored fp32; round before uniquing
LEG_NAMES = "ABCD"


def particle_key(d):
    return d["EVT"].astype(np.int64) * 1000000 + d["MCKEY"].astype(np.int64)


def build_split(split):
    d = load_training(split=split)
    X = d["X"].astype(np.float64)
    L = d["LEG"].astype(np.int8)
    P = d["P"].astype(np.float64)
    key = particle_key(d)
    order = np.argsort(key, kind="stable")
    key, X, L, P = key[order], X[order], L[order], P[order]
    uk, start = np.unique(key, return_index=True)
    end = np.append(start[1:], len(key))

    S0, planes, nlegs, pmom, kept_key, legmix = [], [], [], [], [], []
    n_seen = 0
    for a, b in zip(start, end):
        n_seen += 1
        sl = slice(a, b)
        fwd = X[sl, 6] > X[sl, 5]
        if fwd.sum() < 2:
            continue
        Xf, Lf = X[sl][fwd], L[sl][fwd]
        wp = np.unique(np.round(np.concatenate([Xf[:, 5], Xf[:, 6]]), ROUND_MM))
        if len(wp) - 1 < MIN_LEGS:
            continue
        # the start state: the row that begins at the first plane
        i0 = int(np.argmin(np.abs(Xf[:, 5] - wp[0])))
        S0.append(Xf[i0, :5])
        planes.append(wp)
        nlegs.append(len(wp) - 1)
        pmom.append(P[sl][fwd][i0])
        kept_key.append(key[a])
        legmix.append("".join(sorted({LEG_NAMES[i] for i in Lf})))

    nlegs = np.asarray(nlegs, dtype=np.int32)
    maxw = int(nlegs.max()) + 1
    W = np.full((len(nlegs), maxw), np.nan)
    for i, wp in enumerate(planes):
        W[i, :len(wp)] = wp
    out = dict(S0=np.asarray(S0), planes=W, n_legs=nlegs,
               P=np.asarray(pmom), key=np.asarray(kept_key, dtype=np.int64),
               leg_mix=np.asarray(legmix))
    meta = {
        "split": split,
        "particles_seen": n_seen,
        "particles_kept": int(len(nlegs)),
        "min_legs": MIN_LEGS,
        "legs_total": int(nlegs.sum()),
        "n_legs_histogram": {str(k): int((nlegs == k).sum())
                             for k in sorted(set(nlegs.tolist()))},
        "leg_length_mm_quantiles": {},
    }
    dz = np.diff(W, axis=1)
    dz = dz[np.isfinite(dz)]
    meta["leg_length_mm_quantiles"] = {
        q: float(np.quantile(dz, float(q)))
        for q in ("0.01", "0.1", "0.5", "0.9", "0.99")}
    return out, meta


def build_leg_d(split="test"):
    """The D leg of every test particle that has one, plus its UT waypoint."""
    d = load_training(split=split)
    X = d["X"].astype(np.float64)
    Y = d["Y"].astype(np.float64)
    L = d["LEG"].astype(np.int8)
    P = d["P"].astype(np.float64)
    key = particle_key(d)
    order = np.argsort(key, kind="stable")
    key, X, Y, L, P = key[order], X[order], Y[order], L[order], P[order]
    uk, start = np.unique(key, return_index=True)
    end = np.append(start[1:], len(key))

    S0, z_t, z_mid, z_pv, lab, pmom, src, kk = [], [], [], [], [], [], [], []
    for a, b in zip(start, end):
        sl = slice(a, b)
        dmask = L[sl] == 3
        if not dmask.any():
            continue
        i = int(np.where(dmask)[0][0])
        Xd, Yd = X[sl][i], Y[sl][i]
        bmask = L[sl] == 1
        if bmask.any():
            Xb = X[sl][bmask][0]
            mid = float(min(Xb[5], Xb[6]))        # the UT side of the B leg
            source = "own_B_leg"
        else:
            mid = float(FROZEN_LEG["z0"])
            source = "frozen_leg_z0"
        if not (Xd[6] < mid < Xd[5]):             # the waypoint must lie inside
            continue
        S0.append(Xd[:5]); z_t.append(Xd[5]); z_mid.append(mid)
        z_pv.append(Xd[6]); lab.append(Yd); pmom.append(P[sl][i])
        src.append(source); kk.append(key[a])

    out = dict(S0=np.asarray(S0), z_t=np.asarray(z_t), z_mid=np.asarray(z_mid),
               z_pv=np.asarray(z_pv), label=np.asarray(lab),
               P=np.asarray(pmom), waypoint_source=np.asarray(src),
               key=np.asarray(kk, dtype=np.int64))
    meta = {
        "split": split,
        "particles_with_D_leg": int(len(kk)),
        "waypoint_from_own_B_leg": int((out["waypoint_source"] == "own_B_leg").sum()),
        "waypoint_from_frozen_leg_z0": int((out["waypoint_source"] == "frozen_leg_z0").sum()),
        "dz_first_step_mm_median": float(np.median(out["z_mid"] - out["z_t"])),
        "dz_second_step_mm_median": float(np.median(out["z_pv"] - out["z_mid"])),
        "dz_whole_leg_mm_median": float(np.median(out["z_pv"] - out["z_t"])),
    }
    return out, meta


def main():
    os.makedirs(RESULTS, exist_ok=True)
    meta = {"rule": ("per particle: forward rows only; the planes they name, "
                     "sorted and uniqued to %d decimals, are the waypoints; "
                     "consecutive waypoints are the legs; keep >= %d legs"
                     % (ROUND_MM, MIN_LEGS))}
    for split in ("test", "val"):
        arrays, m = build_split(split)
        np.savez_compressed(os.path.join(RESULTS, "chains_%s.npz" % split),
                            **arrays)
        meta[split] = m
        print("%s: %d particles, %d legs, lengths %s"
              % (split, m["particles_kept"], m["legs_total"],
                 m["n_legs_histogram"]))
    darr, dmeta = build_leg_d("test")
    np.savez_compressed(os.path.join(RESULTS, "leg_d_test.npz"), **darr)
    meta["leg_d_test"] = dmeta
    print("leg D: %d test particles" % dmeta["particles_with_D_leg"])
    with open(os.path.join(RESULTS, "chains_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)


if __name__ == "__main__":
    main()
