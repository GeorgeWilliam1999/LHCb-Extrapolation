#!/usr/bin/env python
"""C2.2. Solve the exact scheme on the v3 test states, for one stage count q.

One invocation = one q. It walks the six |dz| strata of
`../Magnet_tracks_dataset/results/magnet_tracks_v3.npz`, solves the q-stage
Gauss-Legendre equations exactly on every test state of each (no network
anywhere), and writes one row per solve to
`results/scheme_rows_q<qq>.npz`.

The field is the **MagUp** map, passed explicitly to the solver: it is the
polarity the sample was simulated with and the polarity the dataset's labels
`Y` were built with (`../Magnet_tracks_dataset/README.md`). Passing it rather
than inheriting a default is the point of `exact_solver.py`.

    PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python \\
        run_scheme_grid.py --q 8 [--cap 2000] [--split test]

`--q all` runs the whole grid in one process. Each q writes its own file, so a
partially finished grid is resumed by rerunning the missing q values (or with
`--skip-existing`), and the ten files can equally be produced by ten farm jobs
(`condor/jobs.txt`).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import use_shared  # noqa: F401  (puts _shared on sys.path)
from _shared.prepare import STRATUM_NAMES, load_magnet_tracks
from _shared.reference import (field_md5, field_path, gauss_legendre,
                               RK6_STEP)
from exact_solver import RESIDUAL_TOL, field_for, solve_state

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
DATASET = os.path.abspath(os.path.join(
    HERE, "..", "Magnet_tracks_dataset", "results", "magnet_tracks_v3.npz"))

# The stage counts of the sweep: every even q from 2 to 20. Gauss-Legendre has
# classical order 2q, so this spans nominal orders 4 to 40.
QS = (2, 4, 6, 8, 10, 12, 14, 16, 18, 20)

# The polarity of the sample and of the dataset's labels.
FIELD = "up"


def run_one_q(q, split="test", cap=2000, dataset=DATASET, field_which=FIELD,
              verbose=True):
    """Solve every test state of every stratum at this q; write the row file."""
    tab = gauss_legendre(q)                 # verified on construction
    fld = field_for(field_which)

    cols = {k: [] for k in ("stratum", "direction", "abs_dz_mm", "P",
                            "converged", "n_eval", "residual", "err_um",
                            "slope_err_mrad", "wall_s")}
    t_start = time.time()
    for si, name in enumerate(STRATUM_NAMES):
        d = load_magnet_tracks(dataset, split=split, stratum=si)
        X, Y = d["X"], d["Y"]
        n = len(X) if cap is None or cap >= len(X) else cap
        t0 = time.time()
        for i in range(n):
            S0 = np.ascontiguousarray(X[i, :5])
            z0, dz = float(X[i, 5]), float(X[i, 6])
            ti = time.perf_counter()
            _, S1, ok, nev, res = solve_state(S0, z0, dz, tab, fld)
            wall = time.perf_counter() - ti
            # The house measure (`_shared/evaluate.score_against_reference`):
            # worst of |dx| and |dy| in microns, and worst of |dtx| and |dty|
            # in milliradians, against the fine RK6 reference.
            cols["stratum"].append(si)
            cols["direction"].append(int(np.sign(dz)))
            cols["abs_dz_mm"].append(abs(dz))
            cols["P"].append(float(d["P"][i]))
            cols["converged"].append(bool(ok))
            cols["n_eval"].append(int(nev))
            cols["residual"].append(res)
            cols["err_um"].append(float(np.abs(S1[:2] - Y[i, :2]).max() * 1e3))
            cols["slope_err_mrad"].append(
                float(np.abs(S1[2:4] - Y[i, 2:4]).max() * 1e3))
            cols["wall_s"].append(wall)
        if verbose:
            sel = slice(-n, None)
            e = np.array(cols["err_um"][sel])
            print("  q=%-3d %-14s n=%-5d conv=%.4f  median %.4g um  (%.0f s)"
                  % (q, name, n, np.mean(cols["converged"][sel]),
                     np.median(e), time.time() - t0), flush=True)

    out = {k: np.array(v) for k, v in cols.items()}
    out["q"] = np.array(q)
    out["split"] = np.array(split)
    out["field"] = np.array(field_which)
    out["residual_tol"] = np.array(RESIDUAL_TOL)
    out["total_wall_s"] = np.array(time.time() - t_start)
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "scheme_rows_q%02d.npz" % q)
    np.savez_compressed(path, **out)
    if verbose:
        print("q=%-3d written: %s  (%.0f s total)"
              % (q, os.path.basename(path), float(out["total_wall_s"])),
              flush=True)
    return path


def write_meta(qs, split, cap, field_which, wall_s):
    meta = {
        "kind": "exact_scheme_table_rows",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "what": "the exact q-stage Gauss-Legendre scheme solved on the v3 "
                "test states, no network anywhere",
        "dataset": DATASET,
        "split": split,
        "cap_per_stratum": cap,
        "strata": list(STRATUM_NAMES),
        "q_values": list(qs),
        "solver": {
            "module": "Exact_scheme_table/exact_solver.py",
            "generalised_from": "../Simple_first_pass/exact_scheme.py solve_leg",
            "method": "scipy.optimize.root, method='hybr', numerical jacobian",
            "unknowns_per_state": "4 * q (q/p held fixed)",
            "residual_units": "mm, mm, mrad, mrad",
            "residual_tol": RESIDUAL_TOL,
            "step_tol_passed_to_root": 1e-12,
            "initial_guess": "straight line from the input state at the nodes",
            "convergence_judged_on": "the residual, not the solver's flag",
            "nodes": "z0 + c_j * dz, per sample z0 and dz",
        },
        "field": {"which": field_which, "file": field_path(field_which),
                  "md5": field_md5(field_which)},
        "reference": {"integrator": "fp64 RK6 (Butcher, 7 stages, order 6)",
                      "step_mm": RK6_STEP,
                      "floor_um": 5e-5,
                      "source": "the dataset's own Y column"},
        "measure": {"endpoint_um": "max(|dx|, |dy|) * 1e3",
                    "slope_mrad": "max(|dtx|, |dty|) * 1e3"},
        "wall_s": round(wall_s, 1),
    }
    path = os.path.join(RESULTS, "scheme_grid_meta.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=1)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--q", default="all",
                    help="a stage count, or 'all' for %s" % (QS,))
    ap.add_argument("--split", default="test")
    ap.add_argument("--cap", type=int, default=2000,
                    help="states per stratum (the test split holds 2000)")
    ap.add_argument("--field", default=FIELD, choices=("up", "down"))
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args()

    qs = QS if a.q == "all" else (int(a.q),)
    t0 = time.time()
    for q in qs:
        path = os.path.join(RESULTS, "scheme_rows_q%02d.npz" % q)
        if a.skip_existing and os.path.exists(path):
            print("q=%-3d already done, skipped" % q, flush=True)
            continue
        run_one_q(q, split=a.split, cap=a.cap, field_which=a.field)
    if a.q == "all":
        print("meta:", write_meta(qs, a.split, a.cap, a.field, time.time() - t0))
    print("total %.0f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
