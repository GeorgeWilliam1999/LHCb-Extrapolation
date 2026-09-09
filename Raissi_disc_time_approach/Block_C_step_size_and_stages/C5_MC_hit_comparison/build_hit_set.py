#!/usr/bin/env python
"""C5.1 - the hit-to-hit test set: real steps that end on a real simulated hit.

Every score in this folder is against the state the **simulated particle
actually had** on the next sensor plane, so the set is built from the harvested
MCHit states themselves rather than from any labelled training row.

    ../../../Data_generation_exploration/Official_xdigi/results/states.npz

holds one row per plane crossing: `evt`, `mc_key`, `det`, `x`, `y`, `z`, `tx`,
`ty`, `pid`, `q`, `p_GeV`, `eta`. Sorting a particle's rows by z and taking
**consecutive pairs** gives a step whose start and end states are both measured
- the leg-C geometry of the v2 training set. A pair that jumps the magnet
(a UT plane to a SciFi plane) is the leg-B geometry. Both directions are kept:
the reversed pair starts at the later hit and ends on the earlier one.

**The label caveat that makes this folder necessary.** The v2 training rows'
`Y` column is *not* usable as the end state here: it was computed by
field-only propagation on the **MagDown** map, and the sample is a MagUp sample
(`../C0_Magnet_tracks_dataset/README.md`), so `Y` is the right start state pushed
through the wrong polarity. Nothing in this folder reads `Y`. The v2 file is
opened for one thing only - its per-particle train/val/test label, which is a
property of the split and not of the labels.

## The cuts, and why each is here

| cut | why |
|---|---|
| 2 < eta < 5, 1 < p < 200 GeV | the domain every network in Block C was trained on |
| non-electron | as `../C0_Magnet_tracks_dataset` C0.1: an electron's "material effect" is bremsstrahlung, a different measurement |
| both planes inside 2200 < z < 9500 mm | the network's (z0, dz) inputs were normalised on the magnet-to-magnet population; a VELO start plane is outside that range and the network would be extrapolating in its own input space. VP->VP steps are the majority of all consecutive pairs and are dropped for this reason alone |
| the particle is in the **v2 test** split | no particle whose legs trained anything |
| the particle is **not** in the v3 train or val split | `../C0_Magnet_tracks_dataset`'s split is 60/20/20 by particle over a 6,000-particle subsample of the same population, and its seed differs from v2's, so a v2-test particle can be a v3-train one. Those are removed. |
| \|dz\| >= 0.05 mm | below that the two "hits" are the two faces of one sensor |

The surviving pairs are binned by |dz| and a capped, seeded sample is drawn
from each bin, because the cross-magnet steps cost about 0.13 s each of fine
reference and the whole point is to keep this a one-process job.

    PYTHONNOUSERSITE=1 python build_hit_set.py

writes `results/hit_set.npz` (gitignored) and `results/hit_set_meta.json`.
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import argparse
import json
import time

import numpy as np

import use_shared                                     # noqa: F401
from _shared.reference import C_QP

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
DATA = os.path.abspath(os.path.join(
    HERE, "..", "..", "Data_generation_exploration", "Official_xdigi"))
STATES = os.path.join(DATA, "results", "states.npz")
V2 = os.path.join(DATA, "training_v2", "train_official_v2.npz")
V3 = os.path.abspath(os.path.join(
    HERE, "..", "C0_Magnet_tracks_dataset", "results", "magnet_tracks_v3.npz"))

SEED = 20260908
ETA_RANGE = (2.0, 5.0)
P_RANGE = (1.0, 200.0)
Z_WINDOW = (2200.0, 9500.0)
DZ_MIN = 0.05

# The reporting strata. They are contiguous, unlike the six the grid was
# trained on, because a hit-to-hit step length is set by the detector geometry
# and cannot be drawn: there would be nothing in the gaps between
# "0.05-0.2 mm" and "0.5-2 mm".
STRATA = (
    ("0.05-2 mm", 0.05, 2.0, 4000),
    ("2-20 mm", 2.0, 20.0, 4000),
    ("20-200 mm", 20.0, 200.0, 6000),
    ("200-2000 mm", 200.0, 2000.0, 4000),
    ("cross-magnet", 2000.0, 1e9, 3000),
)
P_BANDS = (("1-5 GeV", 1.0, 5.0), ("5-20 GeV", 5.0, 20.0),
           ("20-200 GeV", 20.0, 200.0))


def particle_key(evt, mckey):
    """The (event, MC key) pair as one int64, the way the v2 builder does it."""
    return np.asarray(evt, dtype=np.int64) * 10_000_000 \
        + np.asarray(mckey, dtype=np.int64)


def split_labels(path, split_field="SPLIT"):
    """particle key -> split label, read off a built training set."""
    d = np.load(path)
    k = particle_key(d["EVT"], d["MCKEY"])
    u, i = np.unique(k, return_index=True)
    return u, np.asarray(d[split_field])[i]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=RESULTS)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    d = np.load(STATES, allow_pickle=True)
    det_names = [str(s) for s in d["det_names"]]
    key = particle_key(d["evt"], d["mc_key"])
    z = np.asarray(d["z"], dtype=np.float64)
    order = np.lexsort((z, key))
    key, z = key[order], z[order]
    x = np.asarray(d["x"], dtype=np.float64)[order]
    y = np.asarray(d["y"], dtype=np.float64)[order]
    tx = np.asarray(d["tx"], dtype=np.float64)[order]
    ty = np.asarray(d["ty"], dtype=np.float64)[order]
    det = np.asarray(d["det"])[order]
    pid = np.asarray(d["pid"])[order]
    charge = np.asarray(d["q"], dtype=np.float64)[order]
    p = np.asarray(d["p_GeV"], dtype=np.float64)[order]
    eta = np.asarray(d["eta"], dtype=np.float64)[order]

    cascade = []

    def note(label, n_pairs, n_particles, why=""):
        cascade.append({"cut": label, "pairs_out": int(n_pairs),
                        "particles_out": int(n_particles), "note": why})

    # ------------------------------------------------ the consecutive pairs --
    same = key[1:] == key[:-1]
    i0 = np.flatnonzero(same)              # index of the earlier hit
    i1 = i0 + 1
    note("consecutive plane crossings of one particle (forward)",
         len(i0), len(np.unique(key[i0])), "sorted by z within the particle")

    keep = z[i1] - z[i0] >= DZ_MIN
    i0, i1 = i0[keep], i1[keep]
    note("|dz| >= %.2g mm" % DZ_MIN, len(i0), len(np.unique(key[i0])),
         "below this the two hits are the two faces of one sensor")

    keep = ((z[i0] > Z_WINDOW[0]) & (z[i0] < Z_WINDOW[1])
            & (z[i1] > Z_WINDOW[0]) & (z[i1] < Z_WINDOW[1]))
    dropped_vp = int((~keep).sum())
    i0, i1 = i0[keep], i1[keep]
    note("both planes inside %g < z < %g mm" % Z_WINDOW, len(i0),
         len(np.unique(key[i0])),
         "the network's (z0, dz) normalisation is the magnet-to-magnet one; "
         "%d pairs removed, almost all VELO-to-VELO" % dropped_vp)

    keep = (eta[i0] > ETA_RANGE[0]) & (eta[i0] < ETA_RANGE[1])
    i0, i1 = i0[keep], i1[keep]
    note("%g < eta < %g" % ETA_RANGE, len(i0), len(np.unique(key[i0])))

    keep = (p[i0] > P_RANGE[0]) & (p[i0] < P_RANGE[1])
    i0, i1 = i0[keep], i1[keep]
    note("%g < p < %g GeV" % P_RANGE, len(i0), len(np.unique(key[i0])))

    keep = np.abs(pid[i0]) != 11
    i0, i1 = i0[keep], i1[keep]
    note("non-electron", len(i0), len(np.unique(key[i0])),
         "an electron's residual is bremsstrahlung, a different measurement")

    u2, s2 = split_labels(V2)
    idx = np.searchsorted(u2, key[i0])
    idx = np.clip(idx, 0, len(u2) - 1)
    in2 = (u2[idx] == key[i0])
    keep = in2 & (s2[idx] == 2)
    i0, i1 = i0[keep], i1[keep]
    note("the particle is in the v2 TEST split", len(i0),
         len(np.unique(key[i0])), "80/10/10 by particle, seed 20260718")

    u3, s3 = split_labels(V3)
    idx = np.searchsorted(u3, key[i0])
    idx = np.clip(idx, 0, len(u3) - 1)
    in3 = (u3[idx] == key[i0]) if len(u3) else np.zeros(len(i0), bool)
    bad = in3 & ((s3[idx] == 0) | (s3[idx] == 1))
    n_leak = int(bad.sum())
    i0, i1 = i0[~bad], i1[~bad]
    note("the particle is NOT in the magnet_tracks_v3 train or val split",
         len(i0), len(np.unique(key[i0])),
         "%d pairs removed: the v3 split is 60/20/20 on its own seed, so a "
         "v2-test particle can be a v3-train one and its dense path is what "
         "the grid trained on" % n_leak)

    # ------------------------------------------------------- both directions --
    # A reversed pair starts at the later hit and ends on the earlier one; the
    # truth is the earlier hit's measured state. Nothing is propagated to build
    # it, so the backward half is exactly as truthful as the forward half.
    A = np.concatenate([i0, i1])
    B = np.concatenate([i1, i0])
    direction = np.concatenate([np.ones(len(i0), np.int8),
                                -np.ones(len(i0), np.int8)])
    note("both directions", len(A), len(np.unique(key[A])),
         "the reversed pair ends on the earlier hit's measured state")

    dz = z[B] - z[A]

    # --------------------------------------------------------- the sampling --
    rng = np.random.default_rng(a.seed)
    stratum = np.full(len(A), -1, np.int8)
    chosen = []
    per_stratum = []
    for si, (name, lo, hi, cap) in enumerate(STRATA):
        pool = np.flatnonzero((np.abs(dz) >= lo) & (np.abs(dz) < hi))
        stratum[pool] = si
        take = pool if len(pool) <= cap else np.sort(
            pool[rng.permutation(len(pool))[:cap]])
        chosen.append(take)
        per_stratum.append({"stratum": name, "index": si,
                            "abs_dz_mm": [lo, hi if hi < 1e8 else None],
                            "available": int(len(pool)), "cap": cap,
                            "drawn": int(len(take))})
    sel = np.sort(np.concatenate(chosen))
    A, B, dz, direction, stratum = A[sel], B[sel], dz[sel], direction[sel], \
        stratum[sel]

    # ---------------------------------------------------------- the arrays --
    qop = C_QP * charge[A] / p[A]
    S0 = np.column_stack([x[A], y[A], tx[A], ty[A], qop])
    HIT = np.column_stack([x[B], y[B], tx[B], ty[B], qop])   # the truth
    out = dict(
        S0=S0, HIT=HIT, z0=z[A], z1=z[B], dz=dz, DIRECTION=direction,
        STRATUM=stratum, P=p[A], ETA=eta[A], PID=pid[A],
        EVT=(key[A] // 10_000_000).astype(np.int32),
        MCKEY=(key[A] % 10_000_000).astype(np.int64),
        DET0=det[A], DET1=det[B],
        stratum_names=np.array([s[0] for s in STRATA]),
        p_band_names=np.array([b[0] for b in P_BANDS]),
        p_band_edges=np.array([[b[1], b[2]] for b in P_BANDS]),
        det_names=np.array(det_names),
    )
    np.savez_compressed(os.path.join(a.out, "hit_set.npz"), **out)

    meta = {
        "kind": "hit_to_hit_v1",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "what": "steps whose start and end states are both measured MCHit "
                "crossings of the same simulated particle",
        "source_states": STATES,
        "split_source": V2,
        "leak_guard_source": V3,
        "label_caveat": "the v2 rows' Y column is field-only propagation on "
                        "the MagDown map and is NOT used anywhere in this "
                        "folder; the end state is the particle's own next hit",
        "cuts": {"eta": list(ETA_RANGE), "p_GeV": list(P_RANGE),
                 "z_window_mm": list(Z_WINDOW), "abs_dz_min_mm": DZ_MIN,
                 "drop_electrons": True},
        "cut_cascade": cascade,
        "seed": a.seed,
        "per_stratum": per_stratum,
        "n_rows": int(len(S0)),
        "n_particles": int(len(np.unique(key[A]))),
        "direction_counts": {"forward": int((direction == 1).sum()),
                             "backward": int((direction == -1).sum())},
        "abs_dz_mm": {"min": float(np.abs(dz).min()),
                      "median": float(np.median(np.abs(dz))),
                      "max": float(np.abs(dz).max())},
        "p_bands": [{"name": b[0], "lo": b[1], "hi": b[2],
                     "n": int(((p[A] >= b[1]) & (p[A] < b[2])).sum())}
                    for b in P_BANDS],
        "plane_pairs": {"%s->%s" % (det_names[i], det_names[j]):
                        int(((det[A] == i) & (det[B] == j)).sum())
                        for i in range(len(det_names))
                        for j in range(len(det_names))
                        if ((det[A] == i) & (det[B] == j)).any()},
        "wall_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(a.out, "hit_set_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)

    for c in cascade:
        print("%-62s %8d pairs  %6d particles"
              % (c["cut"], c["pairs_out"], c["particles_out"]))
    for s in per_stratum:
        print("   %-14s available %8d, drawn %5d"
              % (s["stratum"], s["available"], s["drawn"]))
    print("%d rows, %d particles -> results/hit_set.npz (%.1f s)"
          % (meta["n_rows"], meta["n_particles"], meta["wall_s"]))


if __name__ == "__main__":
    main()
