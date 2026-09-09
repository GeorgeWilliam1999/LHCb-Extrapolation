#!/usr/bin/env python
"""C5.2 - score three extrapolators against the particle's actual next hit.

For every step in `results/hit_set.npz` - start state measured on one sensor
plane, end state measured on the next - three predictions are made from the
start state alone and compared with the **simulated end state**, which carries
multiple scattering and energy loss that none of the three model:

1. **the fine reference** - `rk6_rows` at 0.1 mm on the MagUp map, i.e. exact
   field-only propagation. Its miss against the real hit is not its error: it
   is the **material floor**, the part of the step no field-only extrapolator
   of any kind can predict. Everything else in the table is read against it.
2. **the exact q = 8 Gauss-Legendre scheme** - `../C2_Exact_scheme_table`'s
   root-finder, driven to a residual of 1e-9 in mm and mrad. This is the
   ceiling a discrete-time network is trying to reach.
3. **the grid networks** - the trained checkpoints of
   `../C3_Step_size_and_stage_grid` at q = 8, both the physics arm and the data
   twin, several architectures. Per reporting stratum the **seed with the best
   validation error in the matching grid stratum** is used, and the matching is
   by median |dz| in log space and written into the meta json, so the choice is
   never a free hand.

The straight line is scored too, unasked, because it is the null step and it
costs nothing.

Two errors are reported for each predictor:

* `vs_hit` - against the simulated end state. This is the number the folder
  exists for.
* `vs_reference` - against the fine reference's own answer from the same start
  state. This is the field-only error, the quantity `../C3_Step_size_and_stage_grid`
  measured, and it is what says whether a comparison against hits can resolve
  the predictor at all: once `vs_reference` is well below the material floor,
  `vs_hit` is the floor and nothing else.

    PYTHONNOUSERSITE=1 python score_hits.py

writes `results/hit_table.csv`, `results/hit_rows.npz` (gitignored) and
`results/hit_scoring_meta.json`.
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import argparse
import csv
import json
import sys
import time

import numpy as np

import use_shared                                     # noqa: F401
from _shared.reference import (field_md5, field_path, gauss_legendre,
                               make_field, rk6_rows)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
GRID = os.path.abspath(os.path.join(HERE, "..", "C3_Step_size_and_stage_grid"))
SCHEME_DIR = os.path.abspath(os.path.join(HERE, "..", "C2_Exact_scheme_table"))

FIELD = "up"
RK6_STEP = 0.1
SCHEME_Q = 8
GRID_Q = 8
# depth x width. 4x64 and 8x128 are the two the plan names; 2x256 is the twin's
# own best architecture at short steps and 8x256 the physics arm's at the
# crossing (`../C3_Step_size_and_stage_grid/results/reading_best_architecture.csv`),
# so all four together bracket the grid rather than sampling it once.
ARCHS = ((4, 64), (8, 128), (2, 256), (8, 256))
MODES = ("physics", "data")
SEEDS = (0, 1, 2)
GRID_STRATA = ("0.05-0.2 mm", "0.5-2 mm", "5-20 mm", "50-200 mm",
               "500-2000 mm", "full crossing")
GRID_STRATUM_DZ = {"0.05-0.2 mm": 0.1, "0.5-2 mm": 1.0, "5-20 mm": 10.1,
                   "50-200 mm": 99.8, "500-2000 mm": 998.7,
                   "full crossing": 5175.0}


def endpoint_error_um(pred_xy, truth_xy):
    """The house measure: the larger of |dx| and |dy|, in microns."""
    return np.abs(np.asarray(pred_xy) - np.asarray(truth_xy)).max(axis=1) * 1e3


# ------------------------------------------------------- the seed selection --
def val_errors(path):
    """(width, depth, mode, q, seed, stratum) -> the validation median in um."""
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["split"] != "val" or r["direction"] != "all":
                continue
            out[(int(r["width"]), int(r["depth"]), r["mode"], int(r["q"]),
                 int(r["seed"]), r["stratum"])] = float(r["median"])
    return out


def match_stratum(abs_dz_mm):
    """The grid stratum whose median |dz| is nearest in log space."""
    return min(GRID_STRATA,
               key=lambda s: abs(np.log10(GRID_STRATUM_DZ[s])
                                 - np.log10(max(abs_dz_mm, 1e-6))))


# ------------------------------------------------------------ the predictors --
def run_reference(S0, z0, z1, stratum, field, names):
    """The fine reference, one batch per stratum so the march stays tight."""
    out = np.empty_like(S0)
    for si in range(len(names)):
        m = stratum == si
        if not m.any():
            continue
        t0 = time.time()
        out[m] = rk6_rows(S0[m], z0[m], z1[m], step=RK6_STEP, field=field)
        print("   reference %-14s %6d rows  %7.1f s"
              % (names[si], int(m.sum()), time.time() - t0))
    return out


def run_scheme(S0, z0, dz, q, field):
    """The exact q-stage scheme, one root solve per row."""
    sys.path.insert(0, SCHEME_DIR)
    from exact_solver import solve_state                     # noqa: E402
    tab = gauss_legendre(q)                # verified on construction
    out = np.empty_like(S0)
    ok = np.zeros(len(S0), bool)
    res = np.zeros(len(S0))
    t0 = time.time()
    for i in range(len(S0)):
        _, S1, conv, _, r = solve_state(S0[i], z0[i], dz[i], tab, field)
        out[i] = S1
        ok[i] = conv
        res[i] = r
        if (i + 1) % 2000 == 0:
            print("   scheme %d / %d  (%.1f s)"
                  % (i + 1, len(S0), time.time() - t0))
    return out, ok, res


def load_grid_models(q, archs, modes, seeds, field):
    """The trained checkpoints, rebuilt on the grid's own dataset constants."""
    sys.path.insert(0, GRID)
    import torch
    from grid_model import build_grid_model                  # noqa: E402
    data = np.load(os.path.join(GRID, "results", "grid_q%02d.npz" % q))
    models = {}
    for (d, w) in archs:
        for mode in modes:
            for s in seeds:
                tag = "w%d_d%d_q%02d_%s_s%d" % (w, d, q, mode, s)
                ckpt = os.path.join(GRID, "results", tag + ".pt")
                if not os.path.exists(ckpt):
                    print("   missing checkpoint %s - skipped" % tag)
                    continue
                m = build_grid_model(data, width=w, depth=d, field=field)
                m.load_state_dict(torch.load(ckpt, weights_only=True))
                m.eval()
                models[(d, w, mode, s)] = m
    extra_mean = np.asarray(data["extra_mean"])
    extra_scale = np.asarray(data["extra_scale"])
    return models, extra_mean, extra_scale


def predict_grid(model, S0, z0, dz, extra_mean, extra_scale):
    import torch
    extra = np.stack([(z0 - extra_mean[0]) / extra_scale[0],
                      (dz - extra_mean[1]) / extra_scale[1]], axis=1)
    with torch.no_grad():
        out = model(torch.as_tensor(S0), torch.as_tensor(extra)).numpy()
    S1 = np.column_stack([out[:, -1, 0], out[:, -1, 1],
                          out[:, -1, 2], out[:, -1, 3], S0[:, 4]])
    return S1


# ------------------------------------------------------------- the reporting --
def rows_for(name, arch, mode, seed_col, err_hit, err_ref, d):
    """One csv row per (stratum, momentum band, direction) group."""
    out = []
    strata = [("all", np.ones(len(err_hit), bool))]
    strata += [(n, d["STRATUM"] == i)
               for i, n in enumerate(d["stratum_names"])]
    bands = [("all", np.ones(len(err_hit), bool))]
    bands += [(n, (d["P"] >= lo) & (d["P"] < hi))
              for n, (lo, hi) in zip(d["p_band_names"], d["p_band_edges"])]
    dirs = [("all", np.ones(len(err_hit), bool)),
            ("forward", d["DIRECTION"] == 1),
            ("backward", d["DIRECTION"] == -1)]
    for sn, sm in strata:
        for bn, bm in bands:
            for dn, dm in dirs:
                m = sm & bm & dm
                if m.sum() < 5:
                    continue
                out.append({
                    "predictor": name, "arch": arch, "mode": mode,
                    "seed": seed_col, "stratum": sn, "p_band": bn,
                    "direction": dn, "n": int(m.sum()),
                    "median_abs_dz_mm": float(np.median(np.abs(d["dz"][m]))),
                    "vs_hit_med_um": float(np.median(err_hit[m])),
                    "vs_hit_p95_um": float(np.quantile(err_hit[m], 0.95)),
                    "vs_reference_med_um": float(np.median(err_ref[m])),
                    "vs_reference_p95_um": float(np.quantile(err_ref[m], 0.95)),
                })
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--set", default=os.path.join(RESULTS, "hit_set.npz"))
    ap.add_argument("--out", default=RESULTS)
    ap.add_argument("--skip-scheme", action="store_true")
    a = ap.parse_args(argv)
    t_all = time.time()

    z = np.load(a.set, allow_pickle=True)
    d = {k: z[k] for k in z.files}
    d["stratum_names"] = [str(s) for s in d["stratum_names"]]
    d["p_band_names"] = [str(s) for s in d["p_band_names"]]
    S0, HIT = d["S0"], d["HIT"]
    z0, z1, dz = d["z0"], d["z1"], d["dz"]
    n = len(S0)
    print("%d rows" % n)

    field = make_field(FIELD)

    print("1. the fine reference (the material floor)")
    REF = run_reference(S0, z0, z1, d["STRATUM"], field, d["stratum_names"])

    print("2. the exact q = %d scheme" % SCHEME_Q)
    if a.skip_scheme:
        SCH, sch_ok, sch_res = REF.copy(), np.zeros(n, bool), np.zeros(n)
    else:
        SCH, sch_ok, sch_res = run_scheme(S0, z0, dz, SCHEME_Q, field)

    print("3. the grid networks at q = %d" % GRID_Q)
    models, extra_mean, extra_scale = load_grid_models(
        GRID_Q, ARCHS, MODES, SEEDS, field)
    preds = {k: predict_grid(m, S0, z0, dz, extra_mean, extra_scale)
             for k, m in models.items()}

    STRAIGHT = np.column_stack([S0[:, 0] + S0[:, 2] * dz,
                                S0[:, 1] + S0[:, 3] * dz,
                                S0[:, 2], S0[:, 3], S0[:, 4]])

    # ------------------------------------------------------- the seed choice --
    val = val_errors(os.path.join(GRID, "results", "error_vs_dz_q.csv"))
    dz_med = {sn: float(np.median(np.abs(dz[d["STRATUM"] == i])))
              for i, sn in enumerate(d["stratum_names"])
              if (d["STRATUM"] == i).any()}
    matched = {sn: match_stratum(v) for sn, v in dz_med.items()}
    choice = {}
    for (dep, w) in ARCHS:
        for mode in MODES:
            for sn, gs in matched.items():
                cand = [(val[(w, dep, mode, GRID_Q, s, gs)], s) for s in SEEDS
                        if (w, dep, mode, GRID_Q, s, gs) in val
                        and (dep, w, mode, s) in preds]
                if cand:
                    choice[(dep, w, mode, sn)] = min(cand)[1]

    # ----------------------------------------------------------- the scoring --
    ref_hit = endpoint_error_um(REF[:, :2], HIT[:, :2])
    table = []
    table += rows_for("fine reference (material floor)", "-", "-", "-",
                      ref_hit, np.zeros(n), d)
    table += rows_for("straight line", "-", "-", "-",
                      endpoint_error_um(STRAIGHT[:, :2], HIT[:, :2]),
                      endpoint_error_um(STRAIGHT[:, :2], REF[:, :2]), d)
    if not a.skip_scheme:
        table += rows_for("exact scheme q=%d" % SCHEME_Q, "-", "-", "-",
                          endpoint_error_um(SCH[:, :2], HIT[:, :2]),
                          endpoint_error_um(SCH[:, :2], REF[:, :2]), d)

    # every seed, so the selection is auditable ...
    for (dep, w, mode, s), P in sorted(preds.items()):
        table += rows_for("network (every seed)", "%dx%d" % (dep, w), mode,
                          str(s), endpoint_error_um(P[:, :2], HIT[:, :2]),
                          endpoint_error_um(P[:, :2], REF[:, :2]), d)
    # ... and the selected one, stratum by stratum
    for (dep, w) in ARCHS:
        for mode in MODES:
            hit = np.full(n, np.nan)
            ref = np.full(n, np.nan)
            for i, sn in enumerate(d["stratum_names"]):
                s = choice.get((dep, w, mode, sn))
                if s is None:
                    continue
                m = d["STRATUM"] == i
                P = preds[(dep, w, mode, s)]
                hit[m] = endpoint_error_um(P[m][:, :2], HIT[m][:, :2])
                ref[m] = endpoint_error_um(P[m][:, :2], REF[m][:, :2])
            ok = np.isfinite(hit)
            if not ok.any():
                continue
            sub = {k: (v[ok] if isinstance(v, np.ndarray) and v.shape[:1] == (n,)
                       else v) for k, v in d.items()}
            table += rows_for("network (best val seed)", "%dx%d" % (dep, w),
                              mode, "per stratum", hit[ok], ref[ok], sub)

    fields = list(table[0].keys())
    with open(os.path.join(a.out, "hit_table.csv"), "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=fields)
        wtr.writeheader()
        wtr.writerows(table)

    np.savez_compressed(
        os.path.join(a.out, "hit_rows.npz"),
        REF=REF, SCHEME=SCH, STRAIGHT=STRAIGHT, HIT=HIT, S0=S0,
        dz=dz, STRATUM=d["STRATUM"], P=d["P"], DIRECTION=d["DIRECTION"],
        scheme_converged=sch_ok, scheme_residual=sch_res,
        **{"pred_%dx%d_%s_s%d" % (dep, w, mode, s): P
           for (dep, w, mode, s), P in preds.items()})

    meta = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_rows": int(n),
        "field": {"which": FIELD, "file": field_path(FIELD),
                  "md5": field_md5(FIELD)},
        "reference": {"integrator": "fp64 RK6 (Butcher, 7 stages)",
                      "step_mm": RK6_STEP,
                      "role": "field-only propagation from the start hit; its "
                              "miss against the real hit IS the material floor"},
        "scheme": {"q": SCHEME_Q, "solver": "C2_Exact_scheme_table/exact_solver.py",
                   "converged_fraction": float(sch_ok.mean()),
                   "max_residual": float(sch_res.max())},
        "networks": {"q": GRID_Q,
                     "architectures": ["%dx%d" % (d_, w_) for (d_, w_) in ARCHS],
                     "modes": list(MODES), "seeds": list(SEEDS),
                     "checkpoints_loaded": len(models)},
        "seed_selection": {
            "rule": "the seed with the lowest validation median in the grid "
                    "stratum whose median |dz| is nearest in log space",
            "stratum_match": matched,
            "median_abs_dz_mm": dz_med,
            "chosen": {"%dx%d %s %s" % (dep, w, mode, sn): int(s)
                       for (dep, w, mode, sn), s in sorted(choice.items())},
        },
        "label_caveat": "the end state is the particle's own next MCHit state; "
                        "the v2 rows' Y column (MagDown field-only) is never "
                        "read",
        "wall_s": round(time.time() - t_all, 1),
    }
    with open(os.path.join(a.out, "hit_scoring_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("%d table rows -> results/hit_table.csv  (%.1f s)"
          % (len(table), meta["wall_s"]))


if __name__ == "__main__":
    main()
