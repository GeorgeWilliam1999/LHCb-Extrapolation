#!/usr/bin/env python
"""C6.1 + C6.2 for one trained network - the unit of work on the farm.

One job, one tag. It does two things and writes two files.

**C6.1, the single step, per component.** The network is run once on the whole
v3 TEST split of its own `../Step_size_and_stage_grid/results/grid_q<qq>.npz`
(12,000 rows, all six strata) through the grid's own scoring path -
`grid_model.build_grid_model` and `_shared.evaluate.predict` - and the four
signed components of the endpoint error against the fine reference are stored
per row:

    dx, dy      mm              dtx, dty    dimensionless

`--check` asserts that max(|dx|, |dy|) medians per stratum reproduce the run's
own `by_stratum` block in `../Step_size_and_stage_grid/results/<tag>.json` to
better than 1e-9 relative, so the new numbers are the old ones split into
components and nothing else.

**C6.2, the chain.** The same network is walked across the whole magnet on the
1,000 tracks of `results/chain_tracks.npz` at every step length of
`chain.COLUMNS`, by the rule in `chain.py`. Recorded per column:

* at the far plane, the four components against the **fine reference** and
  against the **particle's real hit**, per track;
* along the chain, at 50 evenly spaced fractions of the crossing, the four
  components and max(|dx|, |dy|) against the fine reference's own path there,
  as medians and 95th percentiles over tracks, forward, backward and pooled.

    PYTHONNOUSERSITE=1 python chain_one.py --tag w64_d4_q08_physics_s0

writes `results/single_step/<tag>.npz`, `results/chains/<tag>.npz` (both
gitignored) and `results/records/<tag>.json`, which is what `aggregate.py`
reads.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"

import argparse
import json
import sys
import time

import numpy as np
import torch

import use_shared                                        # noqa: F401
import chain as C
from _shared.evaluate import predict
from _shared.prepare import STRATUM_NAMES

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GRID = os.path.join(ROOT, "Step_size_and_stage_grid")
RESULTS = os.path.join(HERE, "results")
DIRECTIONS = (("all", None), ("forward", 1), ("backward", -1))


def load_model(tag, q, width, depth):
    """The trained checkpoint, rebuilt on the grid's own dataset constants."""
    sys.path.insert(0, GRID)
    from grid_model import build_grid_model                  # noqa: E402
    data = np.load(os.path.join(GRID, "results", "grid_q%02d.npz" % q))
    ckpt = os.path.join(GRID, "results", tag + ".pt")
    if not os.path.exists(ckpt):
        raise SystemExit("no checkpoint %s" % ckpt)
    m = build_grid_model(data, width=width, depth=depth)
    m.load_state_dict(torch.load(ckpt, weights_only=True))
    m.eval()
    return m, data


# ------------------------------------------------------ C6.1 the single step --
def single_step(model, data, tag, out_dir, check=True):
    S = np.asarray(data["test_S"])
    ref = np.asarray(data["test_ref"])
    extra = np.asarray(data["test_extra"])
    strat = np.asarray(data["test_STRATUM"])
    direction = np.asarray(data["test_DIRECTION"])
    out = predict(model, S, extra)
    err = C.component_errors(out[:, -1, :], ref[:, -1, :])

    checks = {}
    if check:
        with open(os.path.join(GRID, "results", tag + ".json")) as f:
            rec = json.load(f)
        mx = np.abs(err[:, :2]).max(axis=1) * 1e3
        for row in rec["by_stratum"]:
            if row["split"] != "test" or row["direction"] != "all":
                continue
            m = (np.ones(len(err), bool) if row["stratum"] < 0
                 else strat == row["stratum"])
            mine = float(np.median(mx[m]))
            theirs = float(row["endpoint_med_um"])
            rel = abs(mine - theirs) / max(theirs, 1e-300)
            checks[row["stratum_name"]] = {"grid_json_um": theirs,
                                           "recomputed_um": mine,
                                           "rel_diff": rel, "n": int(m.sum())}
            if rel > 1e-9:
                raise SystemExit(
                    "single-step scoring does not reproduce the grid record: "
                    "%s, %.6e vs %.6e" % (row["stratum_name"], mine, theirs))

    os.makedirs(os.path.join(out_dir, "single_step"), exist_ok=True)
    np.savez_compressed(
        os.path.join(out_dir, "single_step", tag + ".npz"),
        d=err.astype(np.float32), STRATUM=strat.astype(np.int8),
        DIRECTION=direction.astype(np.int8),
        P=np.asarray(data["test_P"]).astype(np.float32),
        ETA=np.asarray(data["test_ETA"]).astype(np.float32),
        ROW=np.arange(len(err), dtype=np.int32),
        components=np.array(["dx_mm", "dy_mm", "dtx", "dty"]))

    rows = []
    names = list(STRATUM_NAMES)
    for si in range(-1, len(names)):
        sm = np.ones(len(err), bool) if si < 0 else (strat == si)
        for dname, dval in DIRECTIONS:
            m = sm if dval is None else (sm & (direction == dval))
            st = C.component_stats(err, m)
            for comp in C.COMPONENTS:
                rows.append(dict(stratum="all" if si < 0 else names[si],
                                 direction=dname, component=comp,
                                 unit=C.UNITS[comp], **st[comp]))
    return rows, checks


# ------------------------------------------------------------ C6.2 the chain --
def chains(model, data, tracks, tag, out_dir, progress=None):
    extra_mean = np.asarray(data["extra_mean"])
    extra_scale = np.asarray(data["extra_scale"])
    S0 = np.asarray(tracks["S0"])
    z0 = np.asarray(tracks["z0"])
    dz = np.asarray(tracks["dz"])
    REF = np.asarray(tracks["REF"])
    HIT = np.asarray(tracks["HIT"])
    DIR = np.asarray(tracks["DIRECTION"])
    DZ, DS = np.asarray(tracks["DENSE_Z"]), np.asarray(tracks["DENSE_S"])
    DOFF, DN = np.asarray(tracks["DENSE_OFF"]), np.asarray(tracks["DENSE_N"])

    saved, summary, growth = {}, [], []
    for name, nominal, n_per in C.COLUMNS:
        key = C.COLUMN_KEYS[name]
        idx = C.column_tracks(DIR, n_per)
        t0 = time.time()
        S_end, CP_S, CP_Z, n_steps, step = C.chain_column(
            model, S0[idx], z0[idx], dz[idx], nominal, extra_mean, extra_scale,
            progress=progress)
        wall = time.time() - t0

        e_fine = C.component_errors(S_end, REF[idx])
        e_hit = C.component_errors(S_end, HIT[idx])
        # the fine reference's own path at the checkpoint planes
        cp_ref = np.empty_like(CP_S)
        for a, t in enumerate(idx):
            cp_ref[a] = C.dense_interp(DZ, DS, DOFF[t], DN[t], CP_Z[a])
        cp_err = CP_S - cp_ref
        frac = CP_Z - z0[idx][:, None]
        frac = frac / dz[idx][:, None]

        saved.update({
            "far_fine_" + key: e_fine.astype(np.float32),
            "far_hit_" + key: e_hit.astype(np.float32),
            "track_" + key: idx.astype(np.int32),
            "n_steps_" + key: n_steps.astype(np.int64),
            "step_mm_" + key: step.astype(np.float64),
            "cp_err_" + key: cp_err.astype(np.float32),
            "cp_frac_" + key: frac.astype(np.float32),
        })

        d_idx = DIR[idx]
        for dname, dval in DIRECTIONS:
            m = np.ones(len(idx), bool) if dval is None else (d_idx == dval)
            for label, e in (("fine", e_fine), ("hit", e_hit)):
                st = C.component_stats(e, m)
                for comp in C.COMPONENTS:
                    summary.append(dict(
                        column=name, direction=dname, component=comp,
                        reference=label, unit=C.UNITS[comp],
                        n_steps=int(np.median(n_steps[m])) if m.any() else 0,
                        wall_s=round(wall, 2), **st[comp]))
            # the growth curve is stored column-wise rather than row-wise: 50
            # checkpoints x 5 components x 3 directions x 6 columns is 4,500
            # rows per network and 3.2 million over the grid, which is a table
            # nobody can open. `aggregate.py` unrolls what the figures need.
            g = {"nominal_fraction": [(k + 1) / C.N_CHECKPOINTS
                                      for k in range(C.N_CHECKPOINTS)],
                 "z_fraction": [], "components": {}}
            stats = [C.component_stats(cp_err[:, k, :], m)
                     for k in range(C.N_CHECKPOINTS)]
            g["z_fraction"] = [float(np.median(frac[m, k])) if m.any()
                               else float("nan")
                               for k in range(C.N_CHECKPOINTS)]
            for comp in C.COMPONENTS:
                g["components"][comp] = {
                    "unit": C.UNITS[comp],
                    "median_abs": [s[comp]["median_abs"] for s in stats],
                    "p95_abs": [s[comp]["p95_abs"] for s in stats]}
            growth.append(dict(column=name, direction=dname, **g))
        print("   %-14s N=%d..%d  %d tracks  far-plane max_xy median %.4g um "
              "(%.1f s)" % (name, int(n_steps.min()), int(n_steps.max()),
                            len(idx),
                            float(np.median(np.abs(e_fine[:, :2]).max(axis=1)
                                            * 1e3)), wall), flush=True)

    os.makedirs(os.path.join(out_dir, "chains"), exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "chains", tag + ".npz"),
                        columns=np.array([c[0] for c in C.COLUMNS]),
                        column_keys=np.array([C.COLUMN_KEYS[c[0]]
                                              for c in C.COLUMNS]),
                        components=np.array(["dx_mm", "dy_mm", "dtx", "dty"]),
                        DIRECTION=DIR.astype(np.int8), **saved)
    return summary, growth


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default=RESULTS)
    ap.add_argument("--tracks", default=os.path.join(RESULTS,
                                                     "chain_tracks.npz"))
    ap.add_argument("--columns", default=None,
                    help="comma-separated column names, for the timing run")
    ap.add_argument("--n-per-direction", type=int, default=None,
                    help="override every column's track count (timing run)")
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--no-single-step", action="store_true")
    ap.add_argument("--progress", type=int, default=None)
    a = ap.parse_args(argv)

    width, depth, q, mode, seed = C.parse_tag(a.tag)
    if a.columns or a.n_per_direction:
        want = a.columns.split(",") if a.columns else None
        C.COLUMNS = tuple((n, d, a.n_per_direction or k) for n, d, k in C.COLUMNS
                          if want is None or n in want)
    t0 = time.time()
    print("%s  (%d x %d, q = %d, %s, seed %d)"
          % (a.tag, depth, width, q, mode, seed), flush=True)
    model, data = load_model(a.tag, q, width, depth)

    single, checks = ([], {}) if a.no_single_step else \
        single_step(model, data, a.tag, a.out, check=not a.no_check)
    if single:
        print("   single step: test-split max_xy median %.4g um"
              % [r for r in single if r["stratum"] == "all"
                 and r["direction"] == "all"
                 and r["component"] == "max_xy"][0]["median_abs"], flush=True)

    tracks = np.load(a.tracks)
    summary, growth = chains(model, data, tracks, a.tag, a.out,
                             progress=a.progress)

    rec = {"tag": a.tag, "width": width, "depth": depth, "q": q, "mode": mode,
           "seed": seed, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
           "host": os.uname().nodename, "wall_s": round(time.time() - t0, 1),
           "n_checkpoints": C.N_CHECKPOINTS,
           "columns": [{"name": n, "nominal_dz_mm": d, "n_per_direction": k}
                       for n, d, k in C.COLUMNS],
           "single_step_check": checks, "single_step": single,
           "chained": summary, "growth": growth}
    os.makedirs(os.path.join(a.out, "records"), exist_ok=True)
    with open(os.path.join(a.out, "records", a.tag + ".json"), "w") as f:
        json.dump(rec, f)
    print("done in %.1f s -> results/records/%s.json"
          % (rec["wall_s"], a.tag), flush=True)
    return rec


if __name__ == "__main__":
    main()
