"""The dataset builders: what a one-step network is trained and scored on.

Two builders, both producing the same kind of `.npz` (plus a `_meta.json` with
the counts and scales) so that `train.py` can consume either:

  frozen_leg_dataset  the baseline geometry - every sample starts on the same
                      plane and ends on the same plane, so the q stage planes
                      are constants of the problem. This is
                      `One_step_network_v2/prepare_data.py` generalised to any
                      number of stages and to either field polarity; with
                      q = 8 and the MagDown map it reproduces that file's
                      output (checked in `smoke_tests.py`).

  general_leg_dataset every sample carries its own start plane z0 and step
                      length dz, so the stage planes are per sample and the
                      network gets (z0, dz) as two extra normalised inputs.

Both apply the fiducial requirement introduced in One_step_network_v2: the
reference trajectory must stay inside the field map, i.e. inside the region
where the ODE being solved is defined at all. A soft track that bends out past
|x| = 4 m is outside both the map and the LHCb acceptance, and its "reference"
is computed against a clamped, meaningless field; such states carried half of
the whole physics loss in v2 while the data twin absorbed them silently.

Scales are label-free by construction:
  in_scale  = std of the training input states (5 components);
  out_scale = std of the STRAIGHT-LINE-propagated states over the node span
              (no reference integration involved);
  extra_*   = mean and std of (z0, dz) over the training set (general only).

Run:  PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python prepare.py --help
"""
from __future__ import annotations

import json
import os

import numpy as np

try:
    from .reference import (FROZEN_LEG, field_bounds, leg_indices, load_training,
                            make_field, rk4_rows, gauss_legendre)
except ImportError:                       # pragma: no cover  (run as a script)
    from reference import (FROZEN_LEG, field_bounds, leg_indices, load_training,
                           make_field, rk4_rows, gauss_legendre)

SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------- helpers ---
def _meta_path(out_npz):
    return os.path.splitext(out_npz)[0] + "_meta.json"


def _inside_map(ref, field):
    """Rows whose whole reference trajectory stays inside the map in x and y."""
    lo, hi = field_bounds(field)
    out = ((ref[:, :, 0] < lo[0]) | (ref[:, :, 0] > hi[0])
           | (ref[:, :, 1] < lo[1]) | (ref[:, :, 1] > hi[1])).any(axis=1)
    return ~out


def _straight_out_scale(S, dzs):
    """Scale of the outputs: the straight-line states over the node span.

    dzs: (q+1,) for a frozen leg, or (N, q+1) when every sample has its own.
    """
    n, k = (len(S), dzs.shape[-1])
    straight = np.repeat(S[:, None, :], k, axis=1)
    d = np.broadcast_to(dzs, (n, k))
    straight[:, :, 0] += S[:, None, 2] * d
    straight[:, :, 1] += S[:, None, 3] * d
    return straight.reshape(-1, 5).std(axis=0)


def _write(out_npz, arrays, meta):
    if out_npz is None:
        return
    os.makedirs(os.path.dirname(os.path.abspath(out_npz)) or ".", exist_ok=True)
    np.savez_compressed(out_npz, **arrays)
    with open(_meta_path(out_npz), "w") as f:
        json.dump(meta, f, indent=1)


# ------------------------------------------------------------ frozen leg ----
def frozen_leg_dataset(q=8, field="down", n_train=2000, seed=20260718,
                       rebase_mm=60.0, fiducial=True, out_npz=None,
                       z0=None, z1=None, training_npz=None, verbose=True):
    """Every sample on the same leg: z0 -> z1, q fixed stage planes.

    q          number of Gauss-Legendre stages
    field      'down' or 'up' - the field polarity the references are built with
    n_train    cap on the training states (val and test are not capped)
    seed       the permutation seed for that cap
    rebase_mm  a state is used if its own plane is within this many mm of z0;
               it is then transported to z0 exactly with the fp64 RK4 engine
    fiducial   drop states whose reference trajectory leaves the field map
    out_npz    where to write the dataset (a `_meta.json` goes beside it)

    Returns the dict of arrays that is written to the npz.
    """
    fld = make_field(field)
    Z0 = float(FROZEN_LEG["z0"] if z0 is None else z0)
    Z1 = float(FROZEN_LEG["z1"] if z1 is None else z1)
    c, A, b = gauss_legendre(q)
    znodes = Z0 + c * (Z1 - Z0)                 # the q fixed stage planes
    zout = np.append(znodes, Z1)                # nodes + endpoint
    dropped = {}

    def build(split, cap=None):
        d = load_training(split=split, leg="B", npz=training_npz)
        fwd = d["X"][:, 6] > d["X"][:, 5]
        X = d["X"][fwd].astype(np.float64)
        P = d["P"][fwd].astype(np.float64)
        near = np.abs(X[:, 5] - Z0) < rebase_mm
        X, P = X[near], P[near]
        S = rk4_rows(X[:, :5], X[:, 5], np.full(len(X), Z0), field=fld)
        ok = np.isfinite(S).all(axis=1)
        S, P = S[ok], P[ok]
        if cap is not None and len(S) > cap:
            idx = np.random.default_rng(seed).permutation(len(S))[:cap]
            S, P = S[idx], P[idx]
        ref = np.empty((len(S), q + 1, 5))
        cur, zprev = S.copy(), Z0
        for j, zt in enumerate(zout):
            cur = rk4_rows(cur, np.full(len(S), zprev), np.full(len(S), zt),
                           field=fld)
            ref[:, j] = cur
            zprev = zt
        keep = np.isfinite(ref).all(axis=(1, 2))
        if fiducial:
            keep &= _inside_map(ref, fld)
        dropped[split] = int((~keep).sum())
        if verbose and dropped[split]:
            print("  fiducial: dropped %d of %d %s states leaving the field map"
                  % (dropped[split], len(S), split))
        return S[keep], P[keep], ref[keep]

    data = {}
    for split in SPLITS:
        data[split] = build(split, cap=n_train if split == "train" else None)
    if verbose:
        print("states: " + "  ".join("%s %d" % (s, len(data[s][0])) for s in SPLITS))

    train_S = data["train"][0]
    in_scale = train_S.std(axis=0)
    out_scale = _straight_out_scale(train_S, zout - Z0)

    arrays = dict(kind="frozen", znodes=znodes, zout=zout, z0=Z0, z1=Z1, q=q,
                  c=c, in_scale=in_scale, out_scale=out_scale, field=field)
    for split in SPLITS:
        S, P, ref = data[split]
        arrays["%s_S" % split] = S
        arrays["%s_P" % split] = P
        arrays["%s_ref" % split] = ref
        arrays["%s_z0" % split] = np.full(len(S), Z0)
        arrays["%s_dz" % split] = np.full(len(S), Z1 - Z0)

    meta = {
        "kind": "frozen",
        "leg": {"z0": Z0, "z1": Z1, "q": q},
        "field": field,
        "counts": {s: int(len(data[s][0])) for s in SPLITS},
        "dropped_by_fiducial": dropped,
        "in_scale": in_scale.tolist(),
        "out_scale": out_scale.tolist(),
        "rebase": "own plane within %g mm, fp64 RK4 transport to z0 (exact)" % rebase_mm,
        "fiducial": bool(fiducial),
        "seed": seed,
        "n_train_cap": n_train,
    }
    _write(out_npz, arrays, meta)
    if verbose:
        print(json.dumps(meta, indent=1))
    return arrays


# ----------------------------------------------------------- general leg ----
def general_leg_dataset(legs=("A", "B", "C"), q=8, n_train=2000, field="down",
                        out_npz=None, seed=20260718, n_eval=2000,
                        fiducial=True, training_npz=None, verbose=True):
    """Every sample its own leg: start plane z0, step dz, per-sample stage planes.

    legs      which leg types to draw from ('A' vertex fetch, 'B' cross-magnet,
              'C' plane-to-plane, 'D' downstream); both directions are kept
    q         number of Gauss-Legendre stages
    n_train   cap on the training states
    n_eval    cap on the val and test states (these pools are large and the
              reference integration is the expensive part)
    field     'down' or 'up'
    fiducial  drop states whose reference trajectory leaves the field map

    Returns the dict of arrays that is written to the npz. In addition to the
    frozen-leg contents it carries, per split, `*_z0`, `*_dz`, `*_znodes`
    (N, q) and `*_extra` (N, 2) - the normalised (z0, dz) the network is given.
    """
    fld = make_field(field)
    c, A, b = gauss_legendre(q)
    idx_legs = leg_indices(legs)
    dropped = {}

    def build(split, cap):
        d = load_training(split=split, leg=idx_legs, npz=training_npz)
        X = d["X"].astype(np.float64)
        P = d["P"].astype(np.float64)
        L = d["LEG"].astype(np.int8)
        good = np.abs(X[:, 6] - X[:, 5]) > 1e-6
        X, P, L = X[good], P[good], L[good]
        if cap is not None and len(X) > cap:
            sel = np.random.default_rng(seed).permutation(len(X))[:cap]
            X, P, L = X[sel], P[sel], L[sel]
        S = X[:, :5]
        z0, z1 = X[:, 5], X[:, 6]
        dz = z1 - z0
        znodes = z0[:, None] + c[None, :] * dz[:, None]        # (N, q)
        zout = np.concatenate([znodes, z1[:, None]], axis=1)   # (N, q+1)
        ref = np.empty((len(S), q + 1, 5))
        cur, zprev = S.copy(), z0.copy()
        for j in range(q + 1):
            cur = rk4_rows(cur, zprev, zout[:, j], field=fld)
            ref[:, j] = cur
            zprev = zout[:, j]
        keep = np.isfinite(ref).all(axis=(1, 2)) & np.isfinite(S).all(axis=1)
        if fiducial:
            keep &= _inside_map(ref, fld)
        dropped[split] = int((~keep).sum())
        if verbose and dropped[split]:
            print("  fiducial: dropped %d of %d %s states leaving the field map"
                  % (dropped[split], len(S), split))
        return (S[keep], P[keep], L[keep], ref[keep], z0[keep], dz[keep],
                znodes[keep])

    data = {}
    for split in SPLITS:
        data[split] = build(split, cap=n_train if split == "train" else n_eval)
    if verbose:
        print("states: " + "  ".join("%s %d" % (s, len(data[s][0])) for s in SPLITS))

    train_S, _, _, _, tr_z0, tr_dz, _ = data["train"]
    in_scale = train_S.std(axis=0)
    out_scale = _straight_out_scale(
        train_S, np.concatenate([c[None, :] * tr_dz[:, None], tr_dz[:, None]], axis=1))
    extra_mean = np.array([tr_z0.mean(), tr_dz.mean()])
    extra_scale = np.array([tr_z0.std(), tr_dz.std()])
    extra_scale[extra_scale == 0] = 1.0

    arrays = dict(kind="general", q=q, c=c, in_scale=in_scale,
                  out_scale=out_scale, extra_mean=extra_mean,
                  extra_scale=extra_scale, field=field,
                  legs=np.array(list(legs)))
    for split in SPLITS:
        S, P, L, ref, z0, dz, znodes = data[split]
        arrays["%s_S" % split] = S
        arrays["%s_P" % split] = P
        arrays["%s_LEG" % split] = L
        arrays["%s_ref" % split] = ref
        arrays["%s_z0" % split] = z0
        arrays["%s_dz" % split] = dz
        arrays["%s_znodes" % split] = znodes
        arrays["%s_extra" % split] = np.stack(
            [(z0 - extra_mean[0]) / extra_scale[0],
             (dz - extra_mean[1]) / extra_scale[1]], axis=1)

    meta = {
        "kind": "general",
        "legs": list(legs),
        "q": q,
        "field": field,
        "counts": {s: int(len(data[s][0])) for s in SPLITS},
        "per_leg_counts": {s: {t: int((data[s][2] == i).sum())
                               for i, t in enumerate("ABCD")} for s in SPLITS},
        "dropped_by_fiducial": dropped,
        "in_scale": in_scale.tolist(),
        "out_scale": out_scale.tolist(),
        "extra_mean": extra_mean.tolist(),
        "extra_scale": extra_scale.tolist(),
        "extra_inputs": "(z0, dz), normalised by extra_mean/extra_scale",
        "fiducial": bool(fiducial),
        "seed": seed,
        "n_train_cap": n_train,
        "n_eval_cap": n_eval,
    }
    _write(out_npz, arrays, meta)
    if verbose:
        print(json.dumps(meta, indent=1))
    return arrays


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--kind", choices=("frozen", "general"), default="frozen")
    ap.add_argument("--q", type=int, default=8)
    ap.add_argument("--field", choices=("down", "up"), default="down")
    ap.add_argument("--legs", default="A,B,C")
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-eval", type=int, default=2000)
    ap.add_argument("--no-fiducial", action="store_true")
    ap.add_argument("--out", required=True, help="output .npz path")
    a = ap.parse_args()
    if a.kind == "frozen":
        frozen_leg_dataset(q=a.q, field=a.field, n_train=a.n_train,
                           fiducial=not a.no_fiducial, out_npz=a.out)
    else:
        general_leg_dataset(legs=tuple(a.legs.split(",")), q=a.q, field=a.field,
                            n_train=a.n_train, n_eval=a.n_eval,
                            fiducial=not a.no_fiducial, out_npz=a.out)
