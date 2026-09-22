#!/usr/bin/env python
"""E0 - the tracks of Block E: the start state on the last UT plane, the
sixth-order track across the crossing on 257 planes, and the particle's real
state on the first SciFi plane.

This is Block D's `D0_Crossing_dataset/build_dataset.py` with two changes and
nothing else:

  * the plane grid is z_k = z0 + k L/256, k = 0 .. 256, so that every plane of
    the Block E step counts N = 2, 64, 128 and 256 lies on it (256 is
    divisible by all four); Block D's grid was L/128;
  * the training split is NOT capped. Block D capped it at 2,000 particles
    because every leg had its own network; a Block E network trains on
    32,000 (track, plane) states per round drawn from all training tracks.
    `--n-train` still applies Block D's seeded cap if given, so the Block D
    subset can be rebuilt exactly.

The selection, the transport of the real last-UT state to z0, the fiducial
cut, the splits and the material comparison are Block D's, line for line; the
gate `check_against_block_d.py` confirms that the validation and test
particles and their states on the shared planes agree with Block D's file.

## What each particle carries

    S0          its real last-UT state, transported to z0 exactly with RK6
                (0.1 mm) - the state every chain starts from
    truth       the RK6 track of S0 through all 257 planes, (n, 257, 5)
    truth_zpost the same track carried on from z1 to the particle's own SciFi
                plane z_post
    S_post      the particle's REAL state on that plane (the data ground
                truth; it differs from truth_zpost by the material crossed)
    S_pre, z_pre, z_post, P, ETA, PID, EVT, MCKEY, PBAND   as in Block D

Run:
    PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python build_tracks.py --workers 16

Outputs:
    results/tracks.npz          the arrays above, per split
    results/tracks_meta.json    the cut cascade, counts, material floor, cost
    figures/tracks_overview.png |B| on the axis along the crossing and the four
                                plane grids
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
from _shared.prepare import (P_BANDS, _inside_map, _particle_key,
                             magnet_leg_rows, p_band_index)
from _shared.reference import (DATA_NPZ, FROZEN_LEG, RK6_STEP, field_md5,
                               make_field, rk6_rows)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")

Z0 = float(FROZEN_LEG["z0"])
Z1 = float(FROZEN_LEG["z1"])
L = Z1 - Z0
N_VALUES = (2, 64, 128, 256)
N_MAX = max(N_VALUES)
PLANES = Z0 + np.arange(N_MAX + 1) * (L / N_MAX)      # 257 planes
FIELD = "up"
WINDOW_MM = 60.0
SEED = 20260718                                       # Block D's cap seed
SPLITS = ("train", "val", "test")
SPLIT_CODE = {"train": 0, "val": 1, "test": 2}


def _march_planes(args):
    """RK6 through the 257 planes for one chunk of start states (worker)."""
    S0, = args
    fld = make_field(FIELD)
    n = len(S0)
    truth = np.empty((n, N_MAX + 1, 5))
    truth[:, 0] = S0
    cur = S0.copy()
    for k in range(1, N_MAX + 1):
        cur = rk6_rows(cur, PLANES[k - 1], PLANES[k], step=RK6_STEP, field=fld)
        truth[:, k] = cur
    return truth


def march_planes(S0, workers):
    chunks = np.array_split(np.arange(len(S0)), max(1, min(workers, len(S0) // 200 or 1)))
    if workers <= 1 or len(chunks) == 1:
        return _march_planes((S0,))
    with Pool(len(chunks)) as pool:
        parts = pool.map(_march_planes, [(S0[c],) for c in chunks])
    return np.concatenate(parts, axis=0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--n-train", type=int, default=None,
                    help="cap the training split with Block D's seeded permutation (default: no cap)")
    ap.add_argument("--window-mm", type=float, default=WINDOW_MM)
    ap.add_argument("--out", default=os.path.join(RESULTS, "tracks.npz"))
    a = ap.parse_args(argv)
    t0 = time.time()
    fld = make_field(FIELD)

    # -- the rows, both directions, and the forward/backward pairing ----------
    r = magnet_leg_rows(verbose=True)
    key = _particle_key(r["EVT"], r["MCKEY"])
    fwd = r["DIRECTION"] == 1
    back_of = {k: i for i, k in zip(np.flatnonzero(~fwd), key[~fwd])}
    fi = np.flatnonzero(fwd)
    bi = np.array([back_of[k] for k in key[fi]])
    z_pre, z_post = r["z0"][fi], r["z1"][fi]
    assert np.allclose(r["z0"][bi], z_post) and np.allclose(r["z1"][bi], z_pre), \
        "the backward row of a particle must run from its SciFi plane to its UT plane"
    cascade = list(r["cascade"])
    cascade.append({"cut": "forward rows only", "rows_in": int(len(key)),
                    "rows_removed": int(len(key) - len(fi)), "rows_out": int(len(fi)),
                    "particles_out": int(len(fi)), "note": "one row per particle"})

    win = (np.abs(z_pre - Z0) < a.window_mm) & (np.abs(z_post - Z1) < a.window_mm)
    cascade.append({"cut": "|z_pre - z0| < %g mm and |z_post - z1| < %g mm"
                           % (a.window_mm, a.window_mm),
                    "rows_in": int(len(fi)), "rows_removed": int((~win).sum()),
                    "rows_out": int(win.sum()), "particles_out": int(win.sum()),
                    "note": "z0 = %.1f, z1 = %.1f mm" % (Z0, Z1)})
    print("  %-38s %7d -> %7d  (-%d)" % ("window at both ends", len(fi),
                                          win.sum(), (~win).sum()))
    fi, bi = fi[win], bi[win]
    z_pre, z_post = z_pre[win], z_post[win]
    S_pre = r["S0"][fi]                    # real state on the last UT plane
    S_post = r["S0"][bi]                   # real state on the first SciFi plane
    n = len(fi)

    # -- transport to z0, the track across the crossing, and on to z_post ------
    t1 = time.time()
    S0 = rk6_rows(S_pre, z_pre, Z0, step=RK6_STEP, field=fld)
    truth = march_planes(S0, a.workers)
    truth_zpost = rk6_rows(truth[:, N_MAX], Z1, z_post, step=RK6_STEP, field=fld)
    t_rk6 = time.time() - t1

    finite = np.isfinite(truth).all(axis=(1, 2)) & np.isfinite(truth_zpost).all(axis=1)
    inside = _inside_map(truth, fld)
    keep = finite & inside
    cascade.append({"cut": "fiducial: RK6 track stays inside the field map",
                    "rows_in": int(n), "rows_removed": int((~keep).sum()),
                    "rows_out": int(keep.sum()), "particles_out": int(keep.sum()),
                    "note": "x, y within the map at all 257 planes, all finite"})
    print("  %-38s %7d -> %7d  (-%d)" % ("fiducial", n, keep.sum(), (~keep).sum()))

    cols = dict(S0=S0, truth=truth, truth_zpost=truth_zpost, S_pre=S_pre,
                S_post=S_post, z_pre=z_pre, z_post=z_post,
                P=r["P"][fi], ETA=r["ETA"][fi], PID=r["PID"][fi],
                EVT=r["EVT"][fi], MCKEY=r["MCKEY"][fi],
                PBAND=p_band_index(r["P"][fi]))
    cols = {k: np.asarray(v)[keep] for k, v in cols.items()}
    split = np.asarray(r["SPLIT_V2"][fi])[keep]

    # -- the material floor: real SciFi state against the field-only truth ----
    dpos = np.abs(cols["S_post"][:, :2] - cols["truth_zpost"][:, :2]).max(axis=1) * 1e3
    dslope = np.abs(cols["S_post"][:, 2:4] - cols["truth_zpost"][:, 2:4]).max(axis=1) * 1e3
    floor = {"all": {"pos_med_um": float(np.median(dpos)),
                     "pos_p95_um": float(np.quantile(dpos, 0.95)),
                     "slope_med_mrad": float(np.median(dslope)), "n": int(len(dpos))}}
    for i, (lo, hi) in enumerate(P_BANDS):
        m = cols["PBAND"] == i
        if m.any():
            floor["%g-%g GeV" % (lo, hi)] = {
                "pos_med_um": float(np.median(dpos[m])),
                "pos_p95_um": float(np.quantile(dpos[m], 0.95)),
                "slope_med_mrad": float(np.median(dslope[m])), "n": int(m.sum())}

    # -- splits (training uncapped unless --n-train) -----------------------------
    arrays = dict(z0=Z0, z1=Z1, L=L, planes=PLANES, n_max=N_MAX,
                  N_values=np.array(N_VALUES), field=FIELD,
                  window_mm=a.window_mm, rk6_step_mm=RK6_STEP)
    counts, uncapped = {}, {}
    for s in SPLITS:
        m = np.flatnonzero(split == SPLIT_CODE[s])
        uncapped[s] = int(len(m))
        if s == "train" and a.n_train is not None and len(m) > a.n_train:
            m = np.sort(m[np.random.default_rng(SEED).permutation(len(m))[:a.n_train]])
        counts[s] = int(len(m))
        for k, v in cols.items():
            arrays["%s_%s" % (s, k)] = v[m]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, **arrays)

    meta = {
        "what": "Block E tracks: start state at z0, RK6 track on 257 planes, "
                "real state on the first SciFi plane",
        "created": time.strftime("%Y-%m-%d %H:%M"),
        "source_training_set": os.path.abspath(DATA_NPZ),
        "field": {"which": FIELD, "md5": field_md5(FIELD)},
        "crossing": {"z0_mm": Z0, "z1_mm": Z1, "L_mm": L,
                     "N_values": list(N_VALUES),
                     "dz_mm": {str(N): L / N for N in N_VALUES},
                     "planes": "z0 + k L/256, k = 0..256; chain N starts its steps on every 256/N-th"},
        "reference": "RK6 (Butcher 7-stage, order 6) at %g mm, marched plane to "
                     "plane; transport z_pre -> z0 and z1 -> z_post with the same" % RK6_STEP,
        "cut_cascade": cascade,
        "counts": counts,
        "particles_before_any_cap": uncapped,
        "train_cap": {"n_train": a.n_train, "seed": SEED if a.n_train is not None else None},
        "material_floor_real_SciFi_state_vs_field_only_truth": floor,
        "pre_plane_z_mm": {"min": float(cols["z_pre"].min()), "max": float(cols["z_pre"].max())},
        "post_plane_z_mm": {"min": float(cols["z_post"].min()), "max": float(cols["z_post"].max())},
        "cost": {"rk6_wall_s": round(t_rk6, 1), "workers": a.workers,
                 "total_wall_s": round(time.time() - t0, 1)},
        "file": os.path.relpath(a.out, HERE),
    }
    with open(os.path.splitext(a.out)[0] + "_meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps({k: meta[k] for k in ("counts", "material_floor_real_SciFi_state_vs_field_only_truth", "cost")}, indent=1))

    # -- the figure -------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        zz = np.linspace(Z0, Z1, 2000)
        Bx, By, Bz = fld(np.zeros_like(zz), np.zeros_like(zz), zz)
        B = np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2)
        fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                               gridspec_kw={"height_ratios": [3, 2]})
        ax[0].plot(zz, B, "k-")
        ax[0].set_ylabel("|B| on the axis  [T]")
        ax[0].set_title("The crossing: z0 = %.1f mm (last UT plane) to z1 = %.1f mm "
                        "(first SciFi plane), L = %.1f mm, field v8r1 %s" % (Z0, Z1, L, FIELD))
        for i, N in enumerate(N_VALUES):
            zs = Z0 + np.arange(N + 1) * L / N
            ax[1].plot(zs, np.full_like(zs, i), "|", ms=12 if N < 64 else 5,
                       label="N = %d, dz = %.1f mm" % (N, L / N))
        ax[1].set_yticks(range(len(N_VALUES)))
        ax[1].set_yticklabels(["N = %d (dz = %.1f mm)" % (N, L / N) for N in N_VALUES])
        ax[1].set_xlabel("z  [mm]")
        ax[1].set_ylim(-0.7, len(N_VALUES) - 0.3)
        ax[1].grid(axis="x", alpha=0.3)
        fig.tight_layout()
        os.makedirs(FIGURES, exist_ok=True)
        fig.savefig(os.path.join(FIGURES, "tracks_overview.png"), dpi=130)
    except Exception as e:                                   # pragma: no cover
        print("figure skipped:", e)
    print("wrote %s  (%.0f s)" % (a.out, time.time() - t0))
    return meta


if __name__ == "__main__":
    main()
