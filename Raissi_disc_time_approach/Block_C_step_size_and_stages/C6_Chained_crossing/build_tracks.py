#!/usr/bin/env python
"""C6.2a - the 1,000 crossing tracks every network is chained over.

The chain experiment asks one question of all 720 trained networks, so the
track list has to be built **once**, saved, and reused: two networks may only
differ by the network. This script writes that list.

## Where the tracks come from

`../C3_Step_size_and_stage_grid` trains and scores on `results/grid_q<qq>.npz`,
whose evaluation splits are the whole test split of
`../C0_Magnet_tracks_dataset`'s v3 set - 2,000 rows per stratum. The full-crossing
stratum of that split is 2,000 whole magnet crossings, 1,001 of them forward
(the particle's last UT plane to its first SciFi plane) and 999 backward. The
first 500 of each, in row order, are this experiment's tracks.

The row order is recovered exactly, not guessed: `prepare_nodes.pick_rows` is
deterministic given the dataset's `STRATUM` and `SPLIT` columns, and at
`n_eval = 12000` it takes every test row of every stratum. `check` below
asserts that the recovered start states, planes and end states are identical
to `grid_q08.npz`'s own `test_*` arrays, element for element.

## What is stored per track

| array | meaning |
|---|---|
| `S0` (N, 5) | the start state (x, y, tx, ty, qop), fp64 |
| `z0`, `z1`, `dz` | the start plane, the far plane and the signed crossing length |
| `REF` (N, 5) | the **fine reference** state at the far plane - the v3 set's own `Y`, an RK6 march at 0.1 mm on the MagUp map |
| `HIT` (N, 5) | the **particle's real state at the far plane**: its first SciFi crossing for a forward track, its last UT crossing for a backward one, read out of the harvested MCHit states |
| `HIT_Z` (N,) | the z of that hit; it agrees with `z1` to under a micron and is stored so the difference is auditable rather than assumed |
| `DENSE_Z`, `DENSE_S` | the fine reference's own path, one state every 10 mm, flattened; `DENSE_OFF` and `DENSE_N` index into it per track |
| `EVT`, `MCKEY` | the particle |
| `ROW` | the row of `magnet_tracks_v3_residual.npz` the track is |

The hit is obtained the way `../C5_MC_hit_comparison` obtains it: from
`Official_xdigi/results/states.npz`, the harvested plane crossings themselves.
**The v2 training set's `Y` column is never read** - it is field-only
propagation on the MagDown map and the sample is MagUp
(`../C0_Magnet_tracks_dataset/README.md`).

    PYTHONNOUSERSITE=1 python build_tracks.py

writes `results/chain_tracks.npz` (gitignored), `results/chain_tracks.csv`
(the committed track list) and `results/chain_tracks_meta.json`.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"

import argparse
import csv
import json
import sys
import time

import numpy as np

import use_shared                                        # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GRID = os.path.join(ROOT, "C3_Step_size_and_stage_grid")
DENSE = os.path.join(ROOT, "C0_Magnet_tracks_dataset", "results",
                     "dense_states.npz")
V3 = os.path.join(GRID, "results", "magnet_tracks_v3_residual.npz")
STATES = os.path.abspath(os.path.join(
    ROOT, "..", "Data_generation_exploration", "Official_xdigi", "results",
    "states.npz"))
RESULTS = os.path.join(HERE, "results")

N_PER_DIRECTION = 500
FULL_CROSSING_STRATUM = 5
TEST_SPLIT = 2
N_EVAL = 12000


def test_rows(v3):
    """The v3 test rows, in the order `../C3_Step_size_and_stage_grid` uses."""
    sys.path.insert(0, GRID)
    from prepare_nodes import pick_rows                   # noqa: E402
    return pick_rows(np.asarray(v3["STRATUM"]), np.asarray(v3["SPLIT"]),
                     TEST_SPLIT, N_EVAL)


def check_against_grid(v3, rows):
    """Assert the recovered rows are the grid's own test split, exactly."""
    g = np.load(os.path.join(GRID, "results", "grid_q08.npz"))
    X, Y = np.asarray(v3["X"]), np.asarray(v3["Y"])
    out = {
        "start_state_max_abs_diff": float(np.abs(g["test_S"] - X[rows, :5]).max()),
        "z0_max_abs_diff": float(np.abs(g["test_z0"] - X[rows, 5]).max()),
        "dz_max_abs_diff": float(np.abs(g["test_dz"] - X[rows, 6]).max()),
        "end_state_max_abs_diff":
            float(np.abs(np.asarray(g["test_ref"])[:, -1] - Y[rows]).max()),
        "n": int(len(rows)),
    }
    for k, v in out.items():
        if k != "n" and v != 0.0:
            raise SystemExit("recovered test rows differ from the grid's "
                             "own split: %s = %r" % (k, v))
    return out


def far_plane_hits(evt, mckey, direction):
    """(N, 5) the particle's real state at the far plane, and (N,) its z.

    Forward (direction +1): the particle's **first** SciFi (FT) crossing.
    Backward (direction -1): its **last** upstream-tracker (UT) crossing.
    Both are measured states of the simulated particle, not propagations.
    """
    d = np.load(STATES, allow_pickle=True)
    det_names = [str(s) for s in d["det_names"]]
    i_ft, i_ut = det_names.index("FT"), det_names.index("UT")
    key = np.asarray(d["evt"], dtype=np.int64) * 10_000_000 \
        + np.asarray(d["mc_key"], dtype=np.int64)
    z = np.asarray(d["z"], dtype=np.float64)
    order = np.lexsort((z, key))
    key, z = key[order], z[order]
    det = np.asarray(d["det"])[order]
    x = np.asarray(d["x"], dtype=np.float64)[order]
    y = np.asarray(d["y"], dtype=np.float64)[order]
    tx = np.asarray(d["tx"], dtype=np.float64)[order]
    ty = np.asarray(d["ty"], dtype=np.float64)[order]
    charge = np.asarray(d["q"], dtype=np.float64)[order]
    p = np.asarray(d["p_GeV"], dtype=np.float64)[order]
    from _shared.reference import C_QP

    want = np.asarray(evt, dtype=np.int64) * 10_000_000 \
        + np.asarray(mckey, dtype=np.int64)
    lo = np.searchsorted(key, want, "left")
    hi = np.searchsorted(key, want, "right")
    HIT = np.empty((len(want), 5))
    HZ = np.empty(len(want))
    for i in range(len(want)):
        if hi[i] <= lo[i]:
            raise SystemExit("particle %d not in the harvested states" % want[i])
        sl = slice(lo[i], hi[i])
        want_det = i_ft if direction[i] > 0 else i_ut
        idx = np.flatnonzero(det[sl] == want_det)
        if not len(idx):
            raise SystemExit("particle %d has no %s crossing"
                             % (want[i], det_names[want_det]))
        j = lo[i] + (idx[0] if direction[i] > 0 else idx[-1])
        HIT[i] = (x[j], y[j], tx[j], ty[j], C_QP * charge[j] / p[j])
        HZ[i] = z[j]
    return HIT, HZ


def dense_paths(leg_index):
    """The 10 mm reference states of the chosen legs, flattened and indexed."""
    d = np.load(DENSE)
    n_nodes = np.asarray(d["leg_n_nodes"], dtype=np.int64)
    start = np.concatenate([[0], np.cumsum(n_nodes)])
    Z, S = np.asarray(d["Z"]), np.asarray(d["S"])
    LI = np.asarray(d["LEG_INDEX"])
    take = [np.arange(start[i], start[i] + n_nodes[i]) for i in leg_index]
    for k, i in enumerate(leg_index):          # the contiguity assumption
        if not (LI[take[k]] == i).all():
            raise SystemExit("dense states of leg %d are not contiguous" % i)
    off = np.concatenate([[0], np.cumsum([len(t) for t in take])])[:-1]
    flat = np.concatenate(take)
    return (Z[flat].astype(np.float64), S[flat].astype(np.float64),
            off.astype(np.int64),
            np.array([len(t) for t in take], dtype=np.int64),
            float(np.asarray(d["sample_mm"])))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--per-direction", type=int, default=N_PER_DIRECTION)
    ap.add_argument("--out", default=RESULTS)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    v3 = np.load(V3)
    rows = test_rows(v3)
    parity = check_against_grid(v3, rows)
    STRAT = np.asarray(v3["STRATUM"])[rows]
    DIR = np.asarray(v3["DIRECTION"])[rows]
    fc = rows[STRAT == FULL_CROSSING_STRATUM]
    fdir = DIR[STRAT == FULL_CROSSING_STRATUM]
    fwd = fc[fdir == 1][:a.per_direction]
    bwd = fc[fdir == -1][:a.per_direction]
    if len(fwd) < a.per_direction or len(bwd) < a.per_direction:
        raise SystemExit("only %d forward / %d backward crossings available"
                         % (len(fwd), len(bwd)))
    sel = np.concatenate([fwd, bwd])

    X, Y = np.asarray(v3["X"]), np.asarray(v3["Y"])
    S0, z0, dz = X[sel, :5], X[sel, 5], X[sel, 6]
    REF = Y[sel]
    EVT = np.asarray(v3["EVT"])[sel]
    MCKEY = np.asarray(v3["MCKEY"])[sel]
    DIRECTION = np.asarray(v3["DIRECTION"])[sel]
    LEG_INDEX = np.asarray(v3["LEG_INDEX"])[sel]

    HIT, HIT_Z = far_plane_hits(EVT, MCKEY, DIRECTION)
    dz_hit = HIT_Z - (z0 + dz)
    DZ, DS, DOFF, DN, sample_mm = dense_paths(LEG_INDEX)

    # the dense path must start on the start plane and end on the far plane
    ends = np.array([DZ[DOFF[i] + DN[i] - 1] for i in range(len(sel))])
    starts = np.array([DZ[DOFF[i]] for i in range(len(sel))])
    if np.abs(starts - z0).max() > 1e-9 or np.abs(ends - (z0 + dz)).max() > 1e-9:
        raise SystemExit("dense path endpoints do not match the leg")

    hit_err_um = np.abs(HIT[:, :2] - REF[:, :2]).max(axis=1) * 1e3

    out = dict(S0=S0, z0=z0, z1=z0 + dz, dz=dz, REF=REF, HIT=HIT, HIT_Z=HIT_Z,
               DIRECTION=DIRECTION, EVT=EVT, MCKEY=MCKEY,
               LEG_INDEX=LEG_INDEX, ROW=sel,
               P=np.asarray(v3["P"])[sel], ETA=np.asarray(v3["ETA"])[sel],
               PID=np.asarray(v3["PID"])[sel],
               DENSE_Z=DZ, DENSE_S=DS, DENSE_OFF=DOFF, DENSE_N=DN,
               dense_sample_mm=np.array(sample_mm))
    np.savez_compressed(os.path.join(a.out, "chain_tracks.npz"), **out)

    with open(os.path.join(a.out, "chain_tracks.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["track", "row", "evt", "mckey", "direction", "p_GeV",
                    "eta", "pid", "z0_mm", "z1_mm", "dz_mm", "hit_z_mm",
                    "hit_minus_far_plane_z_mm", "hit_vs_reference_um"])
        for i in range(len(sel)):
            w.writerow([i, int(sel[i]), int(EVT[i]), int(MCKEY[i]),
                        int(DIRECTION[i]), "%.6g" % out["P"][i],
                        "%.6g" % out["ETA"][i], int(out["PID"][i]),
                        "%.6f" % z0[i], "%.6f" % (z0[i] + dz[i]),
                        "%.6f" % dz[i], "%.6f" % HIT_Z[i],
                        "%.3e" % dz_hit[i], "%.6g" % hit_err_um[i]])

    meta = {
        "kind": "chain_tracks_v1",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "what": "the 1,000 whole-magnet crossings every network is chained "
                "over in Block C step C6",
        "source": os.path.relpath(V3, HERE),
        "rows": "the full-crossing stratum of the v3 TEST split, in "
                "prepare_nodes.pick_rows order; the first %d forward and the "
                "first %d backward" % (a.per_direction, a.per_direction),
        "split_parity_vs_grid_q08": parity,
        "n_tracks": int(len(sel)),
        "n_forward": int((DIRECTION == 1).sum()),
        "n_backward": int((DIRECTION == -1).sum()),
        "n_particles": int(len(np.unique(EVT.astype(np.int64) * 10_000_000
                                         + MCKEY))),
        "crossing_length_mm": {
            "min": float(np.abs(dz).min()), "median": float(np.median(np.abs(dz))),
            "max": float(np.abs(dz).max())},
        "momentum_GeV": {"min": float(out["P"].min()),
                         "median": float(np.median(out["P"])),
                         "max": float(out["P"].max())},
        "far_plane_hit": {
            "source": STATES,
            "rule": "forward -> the particle's first FT crossing; backward -> "
                    "its last UT crossing",
            "never_uses": "the v2 training set's Y column (MagDown "
                          "field-only propagation of a MagUp sample)",
            "hit_z_minus_far_plane_z_mm": {
                "max_abs": float(np.abs(dz_hit).max()),
                "median_abs": float(np.median(np.abs(dz_hit)))},
            "hit_vs_fine_reference_um": {
                "median": float(np.median(hit_err_um)),
                "p95": float(np.quantile(hit_err_um, 0.95)),
                "median_forward": float(np.median(hit_err_um[DIRECTION == 1])),
                "median_backward": float(np.median(hit_err_um[DIRECTION == -1])),
            },
        },
        "dense_reference": {
            "source": os.path.relpath(DENSE, HERE),
            "sample_mm": sample_mm,
            "matched_by": "LEG_INDEX, cross-checked against EVT, MCKEY and "
                          "DIRECTION",
            "states_per_track": {"min": int(DN.min()), "max": int(DN.max())},
        },
        "wall_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(a.out, "chain_tracks_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)

    print("%d tracks (%d forward, %d backward) from %d particles"
          % (meta["n_tracks"], meta["n_forward"], meta["n_backward"],
             meta["n_particles"]))
    print("crossing length %.1f - %.1f mm, median %.1f"
          % (meta["crossing_length_mm"]["min"],
             meta["crossing_length_mm"]["max"],
             meta["crossing_length_mm"]["median"]))
    print("the material floor on these tracks: median |real hit - fine "
          "reference| = %.1f um, p95 %.1f um"
          % (meta["far_plane_hit"]["hit_vs_fine_reference_um"]["median"],
             meta["far_plane_hit"]["hit_vs_fine_reference_um"]["p95"]))
    print("-> results/chain_tracks.npz, chain_tracks.csv (%.1f s)"
          % meta["wall_s"])
    return meta


if __name__ == "__main__":
    main()
