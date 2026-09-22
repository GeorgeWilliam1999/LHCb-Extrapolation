#!/usr/bin/env python
"""D0 - the crossing particles: the start state on the last UT plane, the
sixth-order truth along the crossing, and the particle's real state on the
first SciFi plane.

The crossing is the frozen leg of the whole programme, z0 = 2648.2 mm (the
modal last-UT plane) to z1 = 7826.0 mm (the modal first-SciFi plane),
L = 5177.8 mm. Block D trains one network per fixed step of that crossing,
for N = 1, 4, 16, 64, 128 steps, so this script lays down the plane grid of
the finest case - 129 planes z_k = z0 + k L/128 - and every coarser grid is a
subset of it (128 is divisible by 1, 4, 16 and 64).

## Who is in the set

The cross-magnet rows of the official MagUp sample selected by
`_shared.prepare.magnet_leg_rows` (leg B; the pre-magnet plane is a UT plane;
both directions present for the particle; 2 < eta < 5; 1 < p < 200 GeV;
non-electron), forward direction only, and then two window cuts:

    |z_pre  - z0| < WINDOW_MM     the particle's last UT plane is one of the
                                  four UT layers around z0 (2641.8, 2648.2,
                                  2656.8, 2663.2 mm)
    |z_post - z1| < WINDOW_MM     its first SciFi plane is the first SciFi
                                  layer around z1 (7825.5 - 7826.7 mm)

so that every particle starts and ends within a few centimetres of the frozen
planes and the transport to them is a short field-only step.

## What each particle carries

    S0          its real last-UT state, transported to z0 exactly with the
                fine sixth-order reference (RK6, 0.1 mm) - the state every
                chain of Block D starts from
    truth       the RK6 trajectory of S0 through all 129 planes (n, 129, 5):
                the "fine Runge-Kutta" comparator, plane by plane
    truth_zpost the same trajectory carried on from z1 to the particle's own
                SciFi plane z_post
    S_post      the particle's REAL state on that plane - the start state of
                its backward row in the training set - which is the "data
                ground truth" comparator. It differs from truth_zpost by the
                material the particle crossed, which no field-only method can
                predict; that difference is recorded as the material floor.

The fiducial requirement drops the particles whose RK6 trajectory leaves the
field map in x or y (a handful). Splits are the training set's own by-particle
splits; the training split is capped at N_TRAIN particles by a seeded
permutation (George 2026-09-14: 2,000 to start, more if the legs do not
converge), validation and test are not capped.

Run:
    PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python build_dataset.py [--workers 8]

Outputs:
    results/crossing_particles.npz        the arrays above, per split
    results/crossing_particles_meta.json  the cut cascade, counts, floors, cost
    figures/crossing_geometry.png         |B| on the axis along the crossing
                                          and the five plane grids
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
N_VALUES = (1, 4, 16, 64, 128)
N_MAX = max(N_VALUES)
PLANES = Z0 + np.arange(N_MAX + 1) * (L / N_MAX)      # 129 planes
FIELD = "up"
WINDOW_MM = 60.0
N_TRAIN = 2000
SEED = 20260718
SPLITS = ("train", "val", "test")
SPLIT_CODE = {"train": 0, "val": 1, "test": 2}


def _march_planes(args):
    """RK6 through the 129 planes for one chunk of start states (worker)."""
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
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-train", type=int, default=N_TRAIN)
    ap.add_argument("--window-mm", type=float, default=WINDOW_MM)
    ap.add_argument("--out", default=os.path.join(RESULTS, "crossing_particles.npz"))
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

    # -- transport to z0, the truth along the crossing, and on to z_post -------
    t1 = time.time()
    S0 = rk6_rows(S_pre, z_pre, Z0, step=RK6_STEP, field=fld)
    truth = march_planes(S0, a.workers)
    truth_zpost = rk6_rows(truth[:, N_MAX], Z1, z_post, step=RK6_STEP, field=fld)
    t_rk6 = time.time() - t1

    finite = np.isfinite(truth).all(axis=(1, 2)) & np.isfinite(truth_zpost).all(axis=1)
    inside = _inside_map(truth, fld)
    keep = finite & inside
    cascade.append({"cut": "fiducial: RK6 trajectory stays inside the field map",
                    "rows_in": int(n), "rows_removed": int((~keep).sum()),
                    "rows_out": int(keep.sum()), "particles_out": int(keep.sum()),
                    "note": "x, y within the map at all 129 planes, all finite"})
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

    # -- splits and the training cap -------------------------------------------
    arrays = dict(z0=Z0, z1=Z1, L=L, planes=PLANES, n_max=N_MAX,
                  N_values=np.array(N_VALUES), field=FIELD,
                  window_mm=a.window_mm, rk6_step_mm=RK6_STEP)
    counts, capped = {}, {}
    for s in SPLITS:
        m = np.flatnonzero(split == SPLIT_CODE[s])
        capped[s] = int(len(m))
        if s == "train" and len(m) > a.n_train:
            m = np.sort(m[np.random.default_rng(SEED).permutation(len(m))[:a.n_train]])
        counts[s] = int(len(m))
        for k, v in cols.items():
            arrays["%s_%s" % (s, k)] = v[m]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, **arrays)

    meta = {
        "what": "Block D crossing particles: start state at z0, RK6 truth at "
                "129 planes, real state on the first SciFi plane",
        "created": time.strftime("%Y-%m-%d %H:%M"),
        "source_training_set": os.path.abspath(DATA_NPZ),
        "field": {"which": FIELD, "md5": field_md5(FIELD)},
        "crossing": {"z0_mm": Z0, "z1_mm": Z1, "L_mm": L,
                     "N_values": list(N_VALUES),
                     "dz_mm": {str(N): L / N for N in N_VALUES},
                     "planes": "z0 + k L/128, k = 0..128; chain N uses every 128/N-th"},
        "reference": "RK6 (Butcher 7-stage, order 6) at %g mm, marched plane to "
                     "plane; transport z_pre -> z0 and z1 -> z_post with the same" % RK6_STEP,
        "cut_cascade": cascade,
        "counts": counts,
        "particles_before_train_cap": capped,
        "train_cap": {"n_train": a.n_train, "seed": SEED},
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
            ax[1].plot(zs, np.full_like(zs, i), "|", ms=12 if N < 64 else 6,
                       label="N = %d, dz = %.0f mm" % (N, L / N))
        ax[1].set_yticks(range(len(N_VALUES)))
        ax[1].set_yticklabels(["N = %d" % N for N in N_VALUES])
        ax[1].set_xlabel("z  [mm]")
        ax[1].set_ylim(-0.7, len(N_VALUES) - 0.3)
        ax[1].grid(axis="x", alpha=0.3)
        fig.tight_layout()
        os.makedirs(FIGURES, exist_ok=True)
        fig.savefig(os.path.join(FIGURES, "crossing_geometry.png"), dpi=130)
    except Exception as e:                                   # pragma: no cover
        print("figure skipped:", e)
    print("wrote %s  (%.0f s)" % (a.out, time.time() - t0))
    return meta


if __name__ == "__main__":
    main()
