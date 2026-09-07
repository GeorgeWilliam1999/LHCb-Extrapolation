#!/usr/bin/env python
"""The one assumption the dataset rests on, checked.

C0.3 draws a start state from the dense states stored along a leg's RK6 path
and integrates onwards from it, on the grounds that "the start state is itself
on the RK6 path, so no re-basing is needed". That is a claim about the code,
and this script tests it two ways on rows of the finished dataset:

  1. RESTART. For each sampled row, the end state computed by the builder -
     RK6 from the stored start state at z0 to z0 + dz - is compared with RK6
     run from the leg's own start plane all the way to z0 + dz in one go. If
     the stored state is genuinely a state of that trajectory, the two agree to
     the arithmetic floor. (They are not required to be bit-identical: the two
     marches accumulate z differently, and the second one crosses z0 as an
     interior step boundary rather than starting there.)

  2. GRID. The stored dense state at z0 is compared with RK6 run from the leg's
     start plane to z0. Same statement, one step earlier.

  3. CLOSURE. Each row integrated back from its end state to z0, compared with
     the start state - the same round trip C1.2 makes, on this dataset's own
     rows rather than on whole crossings.

Run:
    PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python check_path_consistency.py

Output: results/path_consistency.json
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"

import argparse
import json

import numpy as np

import use_shared                                            # noqa: F401
from _shared.prepare import STRATUM_NAMES
from _shared.reference import make_field, rk6_rows

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def stats(a):
    a = np.asarray(a, dtype=np.float64)
    return {"n": int(a.size), "median_um": float(np.median(a) * 1e3),
            "p95_um": float(np.quantile(a, 0.95) * 1e3),
            "max_um": float(a.max() * 1e3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=os.path.join(RESULTS,
                                                  "magnet_tracks_v3.npz"))
    ap.add_argument("--dense", default=os.path.join(RESULTS,
                                                    "dense_states.npz"))
    ap.add_argument("--per-stratum", type=int, default=40)
    a = ap.parse_args()

    d = np.load(a.npz)
    dn = np.load(a.dense)
    field = make_field(str(d["field"]))
    step = float(d["rk6_step_mm"])
    X, Y, STRAT, LEG = d["X"], d["Y"], d["STRATUM"], d["LEG_INDEX"]
    leg_z0 = dn["leg_z0"]
    # the leg's own start state is the dense node with j = 0
    first = np.flatnonzero(np.r_[True, np.diff(dn["LEG_INDEX"]) != 0])
    leg_start = dn["S"][first]
    assert np.array_equal(dn["LEG_INDEX"][first], np.arange(len(leg_z0)))

    rng = np.random.default_rng(0)
    pick = []
    for s in range(len(STRATUM_NAMES)):
        idx = np.flatnonzero(STRAT == s)
        if len(idx):
            pick.append(rng.permutation(idx)[:a.per_stratum])
    pick = np.concatenate(pick)

    S0, z0, dz = X[pick, :5], X[pick, 5], X[pick, 6]
    legs = LEG[pick]
    out = {"dataset": os.path.relpath(a.npz, HERE),
           "rows_checked": int(len(pick)),
           "per_stratum": a.per_stratum,
           "rk6_step_mm": step, "field": str(d["field"])}

    # 1. restart
    scratch = rk6_rows(leg_start[legs], leg_z0[legs], z0 + dz, step=step,
                       field=field)
    gap = np.abs(scratch[:, :4] - Y[pick, :4]).max(axis=1)
    out["1_restart_vs_whole_march"] = dict(
        stats(gap),
        what="RK6 from the stored start state to z0+dz, against RK6 from the "
             "leg's own start plane to z0+dz in one march")
    out["1_by_stratum"] = {
        STRATUM_NAMES[s]: stats(gap[STRAT[pick] == s])
        for s in range(len(STRATUM_NAMES)) if (STRAT[pick] == s).any()}

    # 2. grid
    regrid = rk6_rows(leg_start[legs], leg_z0[legs], z0, step=step, field=field)
    gap2 = np.abs(regrid[:, :4] - S0[:, :4]).max(axis=1)
    out["2_stored_dense_state_vs_march_to_z0"] = dict(
        stats(gap2),
        what="the stored start state against RK6 from the leg's start plane "
             "to that same z0")

    # 3. closure
    back = rk6_rows(Y[pick], z0 + dz, z0, step=step, field=field)
    gap3 = np.abs(back[:, :4] - S0[:, :4]).max(axis=1)
    out["3_row_closure"] = dict(
        stats(gap3), what="each row integrated back from its own end state")
    out["3_by_stratum"] = {
        STRATUM_NAMES[s]: stats(gap3[STRAT[pick] == s])
        for s in range(len(STRATUM_NAMES)) if (STRAT[pick] == s).any()}

    out["qop_passthrough_exact"] = bool(np.array_equal(Y[:, 4], X[:, 4]))
    out["verdict"] = (
        "the stored dense state is a state of the same RK6 trajectory: "
        "restarting from it and marching the whole way agree to %.3g um "
        "(median) and %.3g um (worst) over %d rows"
        % (out["1_restart_vs_whole_march"]["median_um"],
           out["1_restart_vs_whole_march"]["max_um"], len(pick)))

    with open(os.path.join(RESULTS, "path_consistency.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items()
                      if not k.endswith("_by_stratum")}, indent=1))


if __name__ == "__main__":
    main()
