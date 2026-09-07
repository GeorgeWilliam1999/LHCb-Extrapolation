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
import time

import numpy as np

try:
    from .reference import (DATA_NPZ, FROZEN_LEG, KAPPA, LOAD_FIELDS, RK6_STEP, field_bounds,
                            field_md5, field_path, leg_indices, load_training,
                            make_field, rk4_rows, rk6_dense_rows, rk6_rows,
                            gauss_legendre)
except ImportError:                       # pragma: no cover  (run as a script)
    from reference import (DATA_NPZ, FROZEN_LEG, KAPPA, LOAD_FIELDS, RK6_STEP, field_bounds,
                           field_md5, field_path, leg_indices, load_training,
                           make_field, rk4_rows, rk6_dense_rows, rk6_rows,
                           gauss_legendre)

SPLITS = ("train", "val", "test")

# The pre-magnet plane of a leg-B row is whatever tracker plane the particle
# last crossed before z = 2800 mm. For most particles that is a UT plane, but a
# particle with no UT hit contributes its last VELO plane instead. The two
# populations are cleanly separated in z - the VELO ends near 770 mm and the UT
# starts near 2320 mm, with nothing in between - so this boundary splits them
# and no row sits near it. `magnet_leg_rows` records the observed clusters.
UT_VELO_BOUNDARY_MM = 1500.0


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



# ======================================================================== #
# Block C: the magnet-to-magnet track dataset                              #
# ======================================================================== #
# One builder and one loader, plus the selection they share with the fine
# reference study (`../Fine_reference`). Nothing above this line is touched.

# |dz| strata: five log-uniform bands plus the whole crossing. The bands are
# equal in size by construction, so a network trained on this set sees the
# short steps as often as the long ones - which is the defect wave 1 of
# ../General_leg_network ran into from the other side.
STRATA = (
    ("0.05-0.2 mm", 0.05, 0.2),
    ("0.5-2 mm", 0.5, 2.0),
    ("5-20 mm", 5.0, 20.0),
    ("50-200 mm", 50.0, 200.0),
    ("500-2000 mm", 500.0, 2000.0),
    ("full crossing", None, None),
)
STRATUM_NAMES = tuple(s[0] for s in STRATA)
FULL_CROSSING = len(STRATA) - 1

# The momentum bands the dataset is reported in (GeV).
P_BANDS = ((1.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 25.0), (25.0, 200.0))


def p_band_index(P):
    """Index into P_BANDS for each momentum; -1 outside every band."""
    idx = np.full(len(P), -1, dtype=np.int8)
    for i, (lo, hi) in enumerate(P_BANDS):
        idx[(P >= lo) & (P < hi if i + 1 < len(P_BANDS) else P <= hi)] = i
    return idx


def _particle_key(EVT, MCKEY):
    return EVT.astype(np.int64) * 10_000_000 + MCKEY.astype(np.int64)


def magnet_leg_rows(training_npz=None, eta_range=(2.0, 5.0),
                    p_range=(1.0, 200.0), drop_electrons=True,
                    require_ut_plane=True, require_both_directions=True,
                    verbose=True):
    """The cross-magnet legs of the training set, with the cut cascade recorded.

    Leg B of the training set is "the last tracker plane the particle crossed
    before z = 2800 mm, to the first one it crossed after z = 7000 mm", built
    in both directions as two separate rows. This selects the subset of those
    rows that Block C is about, and returns the cascade - rows in, rows removed
    by each cut, rows out - so that the dataset can say exactly what it is.

    The cuts, in the order applied:

      1. the pre-magnet plane is a UT plane, not a VELO one. A particle with no
         UT hit contributes its last VELO plane instead, which makes a 7 m leg
         through the whole of the UT rather than the magnet-to-magnet step this
         dataset is about. The two populations are separated by a 1.5 m gap in
         z (`UT_VELO_BOUNDARY_MM`); the observed clusters are reported.
      2. both directions survive for the particle, so that forward and backward
         legs are a matched pair.
      3. eta_range on the truth pseudorapidity at the particle's origin.
      4. p_range on the truth momentum.
      5. electrons dropped (bremsstrahlung makes their field-only label
         meaningless well before the material effects that are out of scope
         anyway).

    The fiducial requirement is NOT here: it needs the RK6 trajectory, so it is
    applied by `magnet_tracks_dataset` once the paths have been integrated, and
    appended to this cascade.

    Returns a dict with the surviving rows (S0, z0, z1, DIRECTION, P, ETA, PID,
    EVT, MCKEY, CORE, SPLIT_V2), the cascade, and the plane clusters.
    """
    d = load_training(leg="B", npz=training_npz,
                      fields=LOAD_FIELDS + ("ETA", "ORIGIN_R"))
    X = d["X"].astype(np.float64)
    keep = np.ones(len(X), dtype=bool)
    cascade = []

    def stage(name, mask, note=""):
        nonlocal keep
        before = int(keep.sum())
        keep = keep & mask
        after = int(keep.sum())
        cascade.append({"cut": name, "rows_in": before,
                        "rows_removed": before - after, "rows_out": after,
                        "particles_out": int(len(np.unique(
                            _particle_key(d["EVT"], d["MCKEY"])[keep]))),
                        "note": note})
        if verbose:
            print("  %-38s %7d -> %7d  (-%d)"
                  % (name, before, after, before - after))
        return after

    z0, z1 = X[:, 5], X[:, 6]
    forward = z1 > z0
    z_pre = np.where(forward, z0, z1)
    z_post = np.where(forward, z1, z0)

    cascade.append({"cut": "leg-B rows in the training set (both directions)",
                    "rows_in": len(X), "rows_removed": 0, "rows_out": len(X),
                    "particles_out": int(len(np.unique(
                        _particle_key(d["EVT"], d["MCKEY"])))),
                    "note": "load_training(leg='B')"})
    if verbose:
        print("  %-38s %7d" % ("leg-B rows in", len(X)))

    if require_ut_plane:
        stage("pre-magnet plane is a UT plane",
              z_pre > UT_VELO_BOUNDARY_MM,
              "z_pre > %g mm; below it the particle had no UT hit and the row "
              "starts on a VELO plane" % UT_VELO_BOUNDARY_MM)

    if require_both_directions:
        key = _particle_key(d["EVT"], d["MCKEY"])
        okkey = set(np.unique(key[keep & forward])) & set(
            np.unique(key[keep & ~forward]))
        stage("both directions kept for the particle",
              np.isin(key, np.fromiter(okkey, dtype=np.int64,
                                       count=len(okkey))
                      if okkey else np.empty(0, dtype=np.int64)),
              "forward and backward legs are a matched pair")

    if eta_range is not None:
        stage("%g < eta < %g" % eta_range,
              (d["ETA"] > eta_range[0]) & (d["ETA"] < eta_range[1]),
              "truth pseudorapidity at the particle's origin (ETA column)")
    if p_range is not None:
        stage("%g < p < %g GeV" % p_range,
              (d["P"] > p_range[0]) & (d["P"] < p_range[1]),
              "already the training set's own domain cut, so it removes little")
    if drop_electrons:
        stage("non-electron", np.abs(d["PID"]) != 11,
              "|PID| != 11")
    if require_both_directions:            # the cuts above can break a pair
        key = _particle_key(d["EVT"], d["MCKEY"])
        okkey = set(np.unique(key[keep & forward])) & set(
            np.unique(key[keep & ~forward]))
        stage("both directions still kept after the cuts",
              np.isin(key, np.fromiter(okkey, dtype=np.int64,
                                       count=len(okkey))
                      if okkey else np.empty(0, dtype=np.int64)))

    def clusters(z, gap=200.0):
        u = np.sort(np.unique(np.round(z, 1)))
        if not len(u):
            return []
        out, start, prev = [], u[0], u[0]
        for v in u[1:]:
            if v - prev > gap:
                out.append((start, prev))
                start = v
            prev = v
        out.append((start, prev))
        return [{"z_lo": float(a), "z_hi": float(b),
                 "n_rows": int(((z >= a - 0.05) & (z <= b + 0.05)).sum())}
                for a, b in out]

    out = {
        "S0": X[keep, :5], "z0": z0[keep], "z1": z1[keep],
        "DIRECTION": np.where(forward[keep], 1, -1).astype(np.int8),
        "P": d["P"][keep].astype(np.float64),
        "ETA": d["ETA"][keep].astype(np.float64),
        "PID": d["PID"][keep], "EVT": d["EVT"][keep], "MCKEY": d["MCKEY"][keep],
        "CORE": d["CORE"][keep], "SPLIT_V2": d["SPLIT"][keep],
        "cascade": cascade,
        "plane_clusters": {
            "pre_magnet_all_leg_B_rows": clusters(z_pre),
            "pre_magnet_selected": clusters(z_pre[keep]),
            "post_magnet_selected": clusters(z_post[keep]),
        },
        "cuts": {"eta_range": list(eta_range) if eta_range else None,
                 "p_range": list(p_range) if p_range else None,
                 "drop_electrons": bool(drop_electrons),
                 "require_ut_plane": bool(require_ut_plane),
                 "ut_velo_boundary_mm": UT_VELO_BOUNDARY_MM,
                 "require_both_directions": bool(require_both_directions)},
    }
    if verbose:
        print("  selected %d legs from %d particles"
              % (len(out["P"]), cascade[-1]["particles_out"]))
    return out


def magnet_tracks_dataset(out_npz=None, dense_npz=None, n_particles=6000,
                          n_train=6000, n_eval=2000, sample_mm=10.0,
                          step=RK6_STEP, field="up", seed=20260718,
                          training_npz=None, eta_range=(2.0, 5.0),
                          p_range=(1.0, 200.0), verbose=True):
    """Build the magnet-to-magnet dataset (Block C, C0) and its metadata.

    NOTE ON THE FIELD POLARITY. `field` defaults to 'up', not to the 'down'
    that every experiment in this folder before Block C used, because the
    sample the start states come from is a MagUp sample. Propagating its states
    through the MagDown map bends them the wrong way - by 0.9 m at the median
    across the magnet, and with the sign of the bend wrong for 100% of the
    15,257 particles. `../Magnet_tracks_dataset/check_polarity.py` measures it.
    Building this dataset on 'down' would make the fiducial cut throw away
    almost every soft backward leg for a reason that is an artefact, so the
    default is the polarity the events were simulated with. Pass field='down'
    for a twin.

    The whole construction in one place:

      1. `magnet_leg_rows` selects the cross-magnet legs and records the cuts.
      2. `n_particles` particles are drawn from the survivors with `seed` -
         a compute cap, not a physics cut, because step 3 costs about a tenth
         of a second per leg and the fiducial requirement cannot be applied
         before it.
      3. every leg is integrated with the RK6 reference at `step`, keeping the
         state every `sample_mm` along the way (`rk6_dense_rows`). These dense
         states are the path; they are written to `dense_npz`.
      4. the fiducial requirement: the whole RK6 path must stay inside the
         field map, i.e. inside the region where the ODE is defined at all. A
         particle is dropped if either of its two legs leaves.
      5. the particles are split 60/20/20 BY PARTICLE with `seed`, so no
         particle's states appear in two splits.
      6. for each stratum and split, start states are drawn uniformly from the
         dense states of that split's legs, a direction is drawn, |dz| is drawn
         log-uniform inside the stratum, and the end state is RK6 from the
         start state to z0 + dz. The start state is itself on the RK6 path, so
         nothing is re-based. The full-crossing stratum is not drawn: it is
         every leg, start plane to end plane, and its label is the endpoint
         step 3 already computed.

    Returns the dict of arrays written to `out_npz`; the metadata goes to
    `<out_npz without .npz>_meta.json`.
    """
    t_start = time.time()
    fld = make_field(field)
    lo, hi = field_bounds(fld)
    rng = np.random.default_rng(seed)

    if verbose:
        print("[1] selection")
    sel = magnet_leg_rows(training_npz=training_npz, eta_range=eta_range,
                          p_range=p_range, verbose=verbose)
    cascade = list(sel["cascade"])

    # ---- 2. the compute cap, by particle ---------------------------------
    key = _particle_key(sel["EVT"], sel["MCKEY"])
    parts = np.unique(key)
    n_cap = min(int(n_particles), len(parts))
    chosen = np.sort(rng.permutation(len(parts))[:n_cap])
    capped = np.isin(key, parts[chosen])
    cascade.append({
        "cut": "compute cap: %d particles drawn at random" % n_cap,
        "rows_in": len(key), "rows_removed": int((~capped).sum()),
        "rows_out": int(capped.sum()), "particles_out": n_cap,
        "note": "a budget cap, not a physics cut: the RK6 path costs ~0.1 s "
                "per leg and the fiducial requirement needs it. Seed %d."
                % seed})
    if verbose:
        print("  %-38s %7d -> %7d" % ("compute cap", len(key), capped.sum()))
    L = {k: sel[k][capped] for k in
         ("S0", "z0", "z1", "DIRECTION", "P", "ETA", "PID", "EVT", "MCKEY",
          "CORE", "SPLIT_V2")}

    # ---- 3. the RK6 paths -------------------------------------------------
    if verbose:
        print("[2] RK6 at %g mm on %d legs, sampled every %g mm"
              % (step, len(L["P"]), sample_mm))
    t0 = time.time()
    Zg, Sg, valid = rk6_dense_rows(L["S0"], L["z0"], L["z1"],
                                   sample_mm=sample_mm, step=step, field=fld)
    dense_wall = time.time() - t0
    if verbose:
        print("  %.1f s  (%.3f s per leg)  grid %s"
              % (dense_wall, dense_wall / max(1, len(L["P"])), Sg.shape))

    # ---- 4. the fiducial requirement --------------------------------------
    out_of_map = ((Sg[:, :, 0] < lo[0]) | (Sg[:, :, 0] > hi[0])
                  | (Sg[:, :, 1] < lo[1]) | (Sg[:, :, 1] > hi[1]))
    bad = (out_of_map & valid).any(axis=1) | (~np.isfinite(Sg[:, :, :5]).all(
        axis=2) & valid).any(axis=1)
    keykept = _particle_key(L["EVT"], L["MCKEY"])
    fid = ~bad
    lost_one = int(len(np.unique(keykept[bad])))
    cascade.append({
        "cut": "fiducial: the whole RK6 path stays inside the field map",
        "rows_in": int(len(bad)), "rows_removed": int(bad.sum()),
        "rows_out": int(fid.sum()),
        "particles_out": int(len(np.unique(keykept[fid]))),
        "particles_that_lost_at_least_one_leg": lost_one,
        "note": "map corners x, y in [%g, %g] mm. Applied PER LEG, not per "
                "particle: a particle whose backward leg leaves the map still "
                "contributes its forward one. Dropping the particle would "
                "remove the soft tracks preferentially, which are the ones the "
                "extrapolator finds hardest." % (lo[0], hi[0])})
    if verbose:
        print("  %-38s %7d -> %7d" % ("fiducial (RK6 path in map)",
                                      len(bad), fid.sum()))
    L = {k: v[fid] for k, v in L.items()}
    Zg, Sg, valid = Zg[fid], Sg[fid], valid[fid]
    keykept = keykept[fid]
    n_legs = len(L["P"])

    # ---- 5. the split, by particle ----------------------------------------
    parts = np.unique(keykept)
    perm = np.random.default_rng(seed).permutation(len(parts))
    ntr, nva = int(0.6 * len(parts)), int(0.2 * len(parts))
    lab = np.full(len(parts), 2, dtype=np.int8)
    lab[perm[:ntr]] = 0
    lab[perm[ntr:ntr + nva]] = 1
    LEG_SPLIT = lab[np.searchsorted(parts, keykept)]

    # ---- 6. the samples ----------------------------------------------------
    if verbose:
        print("[3] drawing the strata")
    z_lo = np.minimum(L["z0"], L["z1"])       # the UT plane of this leg
    z_hi = np.maximum(L["z0"], L["z1"])       # the SciFi plane of this leg
    node_z = Zg
    n_nodes = valid.sum(axis=1)

    cols = {k: [] for k in ("X", "Y", "STRATUM", "DIRECTION", "LEG_DIR", "P",
                            "ETA", "PID", "EVT", "MCKEY", "SPLIT", "LEG_INDEX")}
    stratum_report = []
    flips = clips = 0
    for si, (name, dz_lo, dz_hi) in enumerate(STRATA):
        for split, target in ((0, n_train), (1, n_eval), (2, n_eval)):
            pool = np.flatnonzero(LEG_SPLIT == split)
            if si == FULL_CROSSING:
                take = pool if len(pool) <= target else np.sort(
                    rng.permutation(len(pool))[:target])
                legs = pool[take] if len(pool) > target else pool
                last = np.array([np.flatnonzero(valid[i])[-1] for i in legs])
                S0 = Sg[legs, 0].copy()
                zz0 = node_z[legs, 0].copy()
                dz = node_z[legs, last] - zz0
                Y = Sg[legs, last].copy()
                nodes = np.zeros(len(legs), dtype=np.int32)
                available = len(pool)
            else:
                available = int(n_nodes[pool].sum()) if len(pool) else 0
                nsamp = min(target, available)
                # a (leg, node) pair drawn uniformly over all dense states
                w = n_nodes[pool].astype(np.float64)
                pick = rng.choice(len(pool), size=nsamp, replace=True,
                                  p=w / w.sum())
                legs = pool[pick]
                nodes = np.array([rng.integers(0, n_nodes[i]) for i in legs],
                                 dtype=np.int32)
                S0 = Sg[legs, nodes].copy()
                zz0 = node_z[legs, nodes].copy()
                mag = np.exp(rng.uniform(np.log(dz_lo), np.log(dz_hi), nsamp))
                sgn = rng.choice(np.array([-1.0, 1.0]), size=nsamp)
                room_up = z_hi[legs] - zz0
                room_dn = zz0 - z_lo[legs]
                room = np.where(sgn > 0, room_up, room_dn)
                other = np.where(sgn > 0, room_dn, room_up)
                flip = (mag > room) & (mag <= other)
                sgn = np.where(flip, -sgn, sgn)
                flips += int(flip.sum())
                room = np.where(sgn > 0, room_up, room_dn)
                clip = mag > room
                clips += int(clip.sum())
                mag = np.minimum(mag, room)
                dz = sgn * mag
                Y = rk6_rows(S0, zz0, zz0 + dz, step=step, field=fld)
            cols["X"].append(np.column_stack([S0, zz0, dz]))
            cols["Y"].append(Y)
            cols["STRATUM"].append(np.full(len(dz), si, dtype=np.int8))
            cols["DIRECTION"].append(np.sign(dz).astype(np.int8))
            cols["LEG_DIR"].append(L["DIRECTION"][legs])
            for k in ("P", "ETA", "PID", "EVT", "MCKEY"):
                cols[k].append(L[k][legs])
            cols["SPLIT"].append(np.full(len(dz), split, dtype=np.int8))
            cols["LEG_INDEX"].append(legs.astype(np.int32))
            stratum_report.append({
                "stratum": name, "index": si,
                "split": ("train", "val", "test")[split],
                "target": target, "drawn": int(len(dz)),
                "population_available": int(available),
                "population_limited": bool(len(dz) < target)})
            if verbose:
                print("  %-14s %-5s target %5d -> %5d"
                      % (name, ("train", "val", "test")[split], target,
                         len(dz)))

    arrays = {k: np.concatenate(v) for k, v in cols.items()}
    arrays["X"] = arrays["X"].astype(np.float64)
    arrays["Y"] = arrays["Y"].astype(np.float64)
    arrays["P"] = arrays["P"].astype(np.float64)
    arrays["ETA"] = arrays["ETA"].astype(np.float64)
    arrays["stratum_names"] = np.array(STRATUM_NAMES)
    arrays["p_bands"] = np.array(P_BANDS)
    arrays["rk6_step_mm"] = np.array(step)
    arrays["sample_mm"] = np.array(sample_mm)
    arrays["field"] = np.array(field)
    arrays["seed"] = np.array(seed)

    # ---- the dense states, in the flat form a consumer wants ---------------
    if dense_npz is not None:
        os.makedirs(os.path.dirname(os.path.abspath(dense_npz)) or ".",
                    exist_ok=True)
        flat = valid.ravel()
        rows_leg = np.repeat(np.arange(n_legs), valid.shape[1])[flat]
        np.savez(dense_npz,
                 EVT=L["EVT"][rows_leg].astype(np.int32),
                 MCKEY=L["MCKEY"][rows_leg].astype(np.int64),
                 DIRECTION=L["DIRECTION"][rows_leg].astype(np.int8),
                 LEG_INDEX=rows_leg.astype(np.int32),
                 SPLIT=LEG_SPLIT[rows_leg].astype(np.int8),
                 P=L["P"][rows_leg].astype(np.float64),
                 Z=Zg.ravel()[flat].astype(np.float64),
                 S=Sg.reshape(-1, Sg.shape[2])[flat].astype(np.float64),
                 leg_EVT=L["EVT"].astype(np.int32),
                 leg_MCKEY=L["MCKEY"].astype(np.int64),
                 leg_DIRECTION=L["DIRECTION"].astype(np.int8),
                 leg_SPLIT=LEG_SPLIT.astype(np.int8),
                 leg_P=L["P"].astype(np.float64),
                 leg_ETA=L["ETA"].astype(np.float64),
                 leg_z0=L["z0"], leg_z1=L["z1"],
                 leg_n_nodes=n_nodes.astype(np.int32),
                 sample_mm=np.array(sample_mm), rk6_step_mm=np.array(step),
                 field=np.array(field))
        if verbose:
            print("[4] dense states: %d, %s" % (flat.sum(), dense_npz))

    # ---- the metadata ------------------------------------------------------
    pb = p_band_index(arrays["P"])
    counts = {}
    for si, name in enumerate(STRATUM_NAMES):
        m = arrays["STRATUM"] == si
        counts[name] = {
            ("train", "val", "test")[sp]: {
                "forward(dz>0)": int((m & (arrays["SPLIT"] == sp)
                                      & (arrays["DIRECTION"] > 0)).sum()),
                "backward(dz<0)": int((m & (arrays["SPLIT"] == sp)
                                       & (arrays["DIRECTION"] < 0)).sum()),
                "total": int((m & (arrays["SPLIT"] == sp)).sum()),
            } for sp in (0, 1, 2)}
        counts[name]["all"] = int(m.sum())
    meta = {
        "kind": "magnet_tracks_v3",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "what": "cross-magnet steps of every length, from the last UT plane to "
                "the first SciFi plane, labelled by the fine RK6 reference",
        "source_training_set": os.path.abspath(training_npz or DATA_NPZ),
        "reference": {
            "integrator": "fp64 fixed-step RK6 (Butcher, 7 stages, order 6)",
            "step_mm": step,
            "dense_sampling_mm": sample_mm,
            "field": {"which": field, "file": field_path(field),
                      "md5": field_md5(field)},
            "kappa": KAPPA,
            "qop_convention": "qop = 0.299792458 q / p[GeV] (Allen)",
            "verification": "../Fine_reference/results/tableau_checks.json and "
                            "reference_convergence.csv",
        },
        "cut_cascade": cascade,
        "cuts": sel["cuts"],
        "plane_clusters": sel["plane_clusters"],
        "strata": [{"index": i, "name": n,
                    "abs_dz_mm": None if lo_ is None else [lo_, hi_],
                    "draw": ("every leg, start plane to end plane, the track's "
                             "own direction" if lo_ is None else
                             "|dz| log-uniform, direction drawn at random")}
                   for i, (n, lo_, hi_) in enumerate(STRATA)],
        "counts_per_stratum_direction_split": counts,
        "counts_per_stratum_and_split": [s for s in stratum_report],
        "momentum_bands_GeV": [list(b) for b in P_BANDS],
        "counts_per_momentum_band": {
            "%g-%g" % P_BANDS[i]: {
                "all": int((pb == i).sum()),
                **{("train", "val", "test")[sp]:
                   int(((pb == i) & (arrays["SPLIT"] == sp)).sum())
                   for sp in (0, 1, 2)}} for i in range(len(P_BANDS))},
        "legs": {"n_legs": int(n_legs),
                 "n_particles": int(len(parts)),
                 "forward": int((L["DIRECTION"] > 0).sum()),
                 "backward": int((L["DIRECTION"] < 0).sum()),
                 "dense_states": int(valid.sum())},
        "split": {"by": "particle", "seed": seed, "fractions": [0.6, 0.2, 0.2],
                  "particles": {"train": int((lab == 0).sum()),
                                "val": int((lab == 1).sum()),
                                "test": int((lab == 2).sum())},
                  "warning": "this is NOT the v2 training set's 80/10/10 split; "
                             "a particle in this set's train split may be in "
                             "the v2 test split. The v2 label is carried as "
                             "SPLIT_V2 in the dense file for anyone who needs "
                             "to intersect them."},
        "clipping": {"direction_flipped_to_fit": int(flips),
                     "magnitude_clipped_to_the_window": int(clips),
                     "rule": "the drawn direction is used if the step fits "
                             "inside [UT plane, SciFi plane]; otherwise it is "
                             "flipped; |dz| is clipped only if neither fits"},
        "format": {"X": "(x, y, tx, ty, qop, z0, dz) fp64",
                   "Y": "(x, y, tx, ty, qop) at z0 + dz, fp64",
                   "DIRECTION": "sign(dz)",
                   "LEG_DIR": "+1 if the parent leg runs UT -> SciFi, -1 back",
                   "LEG_INDEX": "row in the dense file's leg_* arrays"},
        "wall_s": round(time.time() - t_start, 1),
        "dense_wall_s": round(dense_wall, 1),
    }
    if out_npz is not None:
        os.makedirs(os.path.dirname(os.path.abspath(out_npz)) or ".",
                    exist_ok=True)
        np.savez_compressed(out_npz, **arrays)
        with open(_meta_path(out_npz), "w") as f:
            json.dump(meta, f, indent=1)
    arrays["_meta"] = meta
    if verbose:
        print("[5] %d rows -> %s  (%.0f s)"
              % (len(arrays["X"]), out_npz, meta["wall_s"]))
    return arrays


def load_magnet_tracks(npz, split=None, stratum=None, direction=None,
                       p_range=None):
    """The loader for `magnet_tracks_dataset`'s output.

    split:     None | 'train' | 'val' | 'test'
    stratum:   None | an index | a name from STRATUM_NAMES | a sequence of either
    direction: None | +1 (dz > 0) | -1 (dz < 0)
    p_range:   None | (lo, hi) in GeV
    """
    d = np.load(os.path.abspath(npz))
    per_row = [k for k in d.files if getattr(d[k], "ndim", 0) >= 1
               and len(d[k]) == len(d["X"]) and k not in ("p_bands",)]
    m = np.ones(len(d["X"]), dtype=bool)
    if split is not None:
        m &= d["SPLIT"] == {"train": 0, "val": 1, "test": 2}[split]
    if stratum is not None:
        want = [stratum] if isinstance(stratum, (str, int, np.integer)) else list(stratum)
        idx = [STRATUM_NAMES.index(s) if isinstance(s, str) else int(s)
               for s in want]
        m &= np.isin(d["STRATUM"], idx)
    if direction is not None:
        m &= d["DIRECTION"] == np.sign(direction)
    if p_range is not None:
        m &= (d["P"] >= p_range[0]) & (d["P"] <= p_range[1])
    out = {k: d[k][m] for k in per_row}
    for k in ("stratum_names", "p_bands", "rk6_step_mm", "sample_mm", "field",
              "seed"):
        if k in d.files:
            out[k] = d[k]
    return out

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
