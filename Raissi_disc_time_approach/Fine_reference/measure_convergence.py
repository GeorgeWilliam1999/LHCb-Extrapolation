#!/usr/bin/env python
"""C1.2 / C1.3 - how fine the reference has to be, and what it costs.

The question this answers is not "is RK6 order six" (`check_tableau.py` settles
that) but "on the LHCb field map, at which step does the answer stop moving,
and by how much does it still move there". The map is trilinear on a 100 mm
grid, so the field is continuous but its derivative jumps at every cell face:
a magnet crossing passes about fifty of them. A scheme of order six cannot show
order six through a kink, so the convergence must flatten somewhere, and the
level it flattens at IS the reference's own accuracy - the number every result
built on this reference is quoted against.

Four measurements, all on the same 200 magnet-crossing legs:

  1. THE STEP SEQUENCE. The endpoint at 0.8, 0.4, 0.2, 0.1 and 0.05 mm.
     Reported as the successive-halving differences |S(h) - S(h/2)| and as the
     difference from the finest step. On a clean order-p method each halving
     divides the difference by 2^p; the observed ratio says where that stops.

  2. THE SMOOTH-FIELD CONTROL. The same ladder with the trilinear map replaced
     by the smooth analytic field of `check_tableau.py`, on the same legs. If
     the flattening in 1 is the map's C0-ness and not the scheme or the
     arithmetic, this control must sit far below it, at the fp64 floor.

  3. FORWARD-THEN-BACK CLOSURE at 0.1 mm: integrate to the far plane and back,
     and compare with the start state. A round trip is not an accuracy test
     (both halves share the same truncation error) but it is a floor: the
     reference cannot be trusted below the level at which it fails to undo
     itself.

  4. RK4 AT 5 mm AND AT 1 mm against RK6 at 0.05 mm, on the same legs. The 5 mm
     RK4 engine is what every label in the training set and every experiment so
     far was built with, so this is the number that says how much the fine
     reference actually moves the ground.

Plus C1.3: the wall time per track at 0.1 mm across the crossing, and the batch
size that fits in memory.

Run:
    PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python measure_convergence.py

Outputs:
    results/reference_convergence.csv          the summary table
    results/reference_convergence_per_leg.csv  one row per leg per measurement
    results/reference_convergence_meta.json    the legs, the cost, the verdict
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"

import csv
import json
import platform
import resource
import sys
import time

import numpy as np

import use_shared                                            # noqa: F401
from _shared.prepare import P_BANDS, magnet_leg_rows, p_band_index
from _shared.reference import (RK6_STEP, field_bounds, field_md5, field_path,
                               make_field, rk4_rows, rk6_dense_rows, rk6_rows)

from check_tableau import SmoothField

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

FIELD = "up"                                # see ../Magnet_tracks_dataset/check_polarity.py
CROSS_CHECK_FIELD = "down"                  # the same verdict on the other polarity
CROSS_CHECK_STEPS = (0.1, 0.05)

STEPS = (0.8, 0.4, 0.2, 0.1, 0.05)          # mm, the ladder
CLOSURE_STEP = 0.1                          # mm
RK4_STEPS = (5.0, 1.0)                      # mm, the incumbent engine
SMOOTH_STEPS = (0.8, 0.2, 0.05)             # mm, the control ladder
N_PER_CELL = 20                             # legs per (momentum band, direction)
SEED = 20260906


# ------------------------------------------------------------------ legs ----
def choose_legs(field, verbose=True):
    """200 cross-magnet legs: 20 per momentum band per direction, in the map.

    The momentum stratification is deliberate. The bend across the magnet goes
    as 1/p, so an unstratified draw would be almost all stiff tracks and would
    measure the reference on the easy half of the problem.
    """
    sel = magnet_leg_rows(verbose=verbose)
    P = sel["P"]
    band = p_band_index(P)
    rng = np.random.default_rng(SEED)
    lo, hi = field_bounds(field)

    # A candidate pool per cell, all screened in one batch. Whether a leg
    # leaves the field map is a metre-scale question, so a 20 mm RK6 pass
    # settles it; the legs that survive are then integrated properly.
    n_cand = 4 * N_PER_CELL
    cand, cell_of = [], []
    for b in range(len(P_BANDS)):
        for d in (1, -1):
            pool = np.flatnonzero((band == b) & (sel["DIRECTION"] == d))
            rng.shuffle(pool)
            take = pool[:n_cand]
            cand.extend(take)
            cell_of.extend([(b, d)] * len(take))
    cand = np.array(cand)
    _, Sg, valid = rk6_dense_rows(sel["S0"][cand], sel["z0"][cand],
                                  sel["z1"][cand], sample_mm=20.0, step=20.0,
                                  field=field)
    out = ((Sg[:, :, 0] < lo[0]) | (Sg[:, :, 0] > hi[0])
           | (Sg[:, :, 1] < lo[1]) | (Sg[:, :, 1] > hi[1]))
    ok = ~((out & valid).any(axis=1)) & np.isfinite(
        np.where(valid[:, :, None], Sg, 0.0)).all(axis=(1, 2))
    if verbose:
        print("  screened %d candidates, %d stay inside the map"
              % (len(cand), int(ok.sum())))

    picked, cells = [], []
    for b in range(len(P_BANDS)):
        for d in (1, -1):
            m = np.array([c == (b, d) for c in cell_of]) & ok
            taken = cand[m][:N_PER_CELL]
            cells.append({"p_band_GeV": list(P_BANDS[b]), "direction": d,
                          "candidates": int(sum(1 for c in cell_of
                                                if c == (b, d))),
                          "in_map": int(m.sum()), "taken": int(len(taken))})
            picked.extend(taken.tolist())
            if verbose:
                print("  band %g-%g GeV, dir %+d: %d legs"
                      % (P_BANDS[b][0], P_BANDS[b][1], d, len(taken)))
    idx = np.array(picked)
    legs = {k: sel[k][idx] for k in ("S0", "z0", "z1", "DIRECTION", "P", "ETA",
                                     "PID", "EVT", "MCKEY")}
    legs["BAND"] = band[idx]
    legs["cells"] = cells
    return legs


# ------------------------------------------------------------- measuring ----
def endpoint_gap(a, b):
    """(position gap in um, slope gap in urad) per row, worst component."""
    pos = np.abs(a[:, :2] - b[:, :2]).max(axis=1) * 1e3
    slope = np.abs(a[:, 2:4] - b[:, 2:4]).max(axis=1) * 1e6
    return pos, slope


def summarise(pos, slope, mask=None):
    m = np.ones(len(pos), dtype=bool) if mask is None else mask
    if not m.any():
        return None
    return {
        "n": int(m.sum()),
        "pos_med_um": float(np.median(pos[m])),
        "pos_p95_um": float(np.quantile(pos[m], 0.95)),
        "pos_max_um": float(pos[m].max()),
        "slope_med_urad": float(np.median(slope[m])),
        "slope_p95_urad": float(np.quantile(slope[m], 0.95)),
        "slope_max_urad": float(slope[m].max()),
    }


def rows_for(measurement, step, ref_step, integrator, fieldname,
             pos, slope, legs):
    """The summary rows: all legs, then per momentum band, then per direction."""
    out = []
    groups = [("all", "all", None)]
    for b in range(len(P_BANDS)):
        groups.append(("p_band", "%g-%g GeV" % P_BANDS[b], legs["BAND"] == b))
    for d, nm in ((1, "UT -> SciFi"), (-1, "SciFi -> UT")):
        groups.append(("direction", nm, legs["DIRECTION"] == d))
    for kind, name, mask in groups:
        s = summarise(pos, slope, mask)
        if s is None:
            continue
        out.append(dict(measurement=measurement, step_mm=step,
                        reference_step_mm=ref_step, integrator=integrator,
                        field=fieldname, group_kind=kind, group=name, **s))
    return out


def main():
    os.makedirs(RESULTS, exist_ok=True)
    t_start = time.time()
    field = make_field(FIELD)
    smooth = SmoothField()

    print("[1] choosing the legs")
    legs = choose_legs(field)
    S0, z0, z1 = legs["S0"], legs["z0"], legs["z1"]
    n = len(S0)
    span = np.abs(z1 - z0)
    print("  %d legs, |dz| %.1f - %.1f mm (median %.1f)"
          % (n, span.min(), span.max(), np.median(span)))

    summary, per_leg = [], []
    timing = {}

    def note(measurement, step, ref_step, integrator, fieldname, a, b):
        pos, slope = endpoint_gap(a, b)
        summary.extend(rows_for(measurement, step, ref_step, integrator,
                                fieldname, pos, slope, legs))
        for i in range(n):
            per_leg.append(dict(
                leg=i, EVT=int(legs["EVT"][i]), MCKEY=int(legs["MCKEY"][i]),
                P_GeV=float(legs["P"][i]), ETA=float(legs["ETA"][i]),
                direction=int(legs["DIRECTION"][i]),
                p_band="%g-%g" % P_BANDS[legs["BAND"][i]],
                z0=float(z0[i]), z1=float(z1[i]),
                measurement=measurement, step_mm=step,
                reference_step_mm=ref_step, integrator=integrator,
                field=fieldname, pos_um=float(pos[i]),
                slope_urad=float(slope[i])))
        return pos

    # ---- 1. the step ladder on the real map --------------------------------
    print("[2] the step ladder on the v8r1.%s map" % FIELD)
    ends = {}
    for h in STEPS:
        t0 = time.time()
        ends[h] = rk6_rows(S0, z0, z1, step=h, field=field)
        timing["rk6_%g_mm_wall_s" % h] = round(time.time() - t0, 2)
        print("  step %5.2f mm: %.1f s  (%.4f s per track)"
              % (h, time.time() - t0, (time.time() - t0) / n))
    for a, b in zip(STEPS[:-1], STEPS[1:]):
        pos = note("step halving", a, b, "RK6", "v8r1." + FIELD, ends[a], ends[b])
        print("    |S(%.2f) - S(%.2f)| median %.4g um" % (a, b, np.median(pos)))
    for h in STEPS[:-1]:
        note("difference from the finest step", h, STEPS[-1], "RK6",
             "v8r1." + FIELD, ends[h], ends[STEPS[-1]])

    # ---- 2. the smooth-field control ---------------------------------------
    print("[3] the smooth-field control")
    sends = {h: rk6_rows(S0, z0, z1, step=h, field=smooth) for h in SMOOTH_STEPS}
    for a, b in zip(SMOOTH_STEPS[:-1], SMOOTH_STEPS[1:]):
        pos = note("step halving", a, b, "RK6", "smooth analytic",
                   sends[a], sends[b])
        print("    smooth |S(%.2f) - S(%.2f)| median %.4g um"
              % (a, b, np.median(pos)))
    note("difference from the finest step", SMOOTH_STEPS[0], SMOOTH_STEPS[-1],
         "RK6", "smooth analytic", sends[SMOOTH_STEPS[0]],
         sends[SMOOTH_STEPS[-1]])

    # ---- 3. forward-then-back closure --------------------------------------
    print("[4] forward-then-back closure at %g mm" % CLOSURE_STEP)
    t0 = time.time()
    back = rk6_rows(ends[CLOSURE_STEP], z1, z0, step=CLOSURE_STEP, field=field)
    timing["closure_wall_s"] = round(time.time() - t0, 2)
    pos = note("forward-then-back closure", CLOSURE_STEP, CLOSURE_STEP, "RK6",
               "v8r1." + FIELD, back, S0)
    print("  median %.4g um, p95 %.4g um, worst %.4g um"
          % (np.median(pos), np.quantile(pos, 0.95), pos.max()))

    # ---- 4. RK4, the incumbent engine --------------------------------------
    print("[5] RK4 against the fine reference")
    for h in RK4_STEPS:
        t0 = time.time()
        r4 = rk4_rows(S0, z0, z1, step=h, field=field)
        timing["rk4_%g_mm_wall_s" % h] = round(time.time() - t0, 2)
        pos = note("RK4 against RK6 at %g mm" % STEPS[-1], h, STEPS[-1], "RK4",
                   "v8r1." + FIELD, r4, ends[STEPS[-1]])
        print("  RK4 %g mm: median %.4g um, p95 %.4g um"
              % (h, np.median(pos), np.quantile(pos, 0.95)))

    # ---- the other polarity, to show the verdict is not about the sign ------
    print("[5b] the same rungs on the v8r1.%s map" % CROSS_CHECK_FIELD)
    other = make_field(CROSS_CHECK_FIELD)
    oends = {h: rk6_rows(S0, z0, z1, step=h, field=other)
             for h in CROSS_CHECK_STEPS}
    pos = note("step halving", CROSS_CHECK_STEPS[0], CROSS_CHECK_STEPS[1],
               "RK6", "v8r1." + CROSS_CHECK_FIELD,
               oends[CROSS_CHECK_STEPS[0]], oends[CROSS_CHECK_STEPS[1]])
    print("    |S(%.2f) - S(%.2f)| median %.4g um"
          % (CROSS_CHECK_STEPS[0], CROSS_CHECK_STEPS[1], np.median(pos)))
    oback = rk6_rows(oends[CLOSURE_STEP], z1, z0, step=CLOSURE_STEP, field=other)
    pos = note("forward-then-back closure", CLOSURE_STEP, CLOSURE_STEP, "RK6",
               "v8r1." + CROSS_CHECK_FIELD, oback, S0)
    print("    closure median %.4g um" % np.median(pos))

    # ---- C1.3 cost ----------------------------------------------------------
    print("[6] cost")
    per_track = timing["rk6_%g_mm_wall_s" % RK6_STEP] / n
    batch = {}
    for nb in (200, 1000, 5000, 20000):
        S = np.repeat(S0[:1], nb, axis=0)
        t0 = time.time()
        rk6_rows(S, z0[0], z0[0] + 20.0, step=RK6_STEP, field=field)
        dt = time.time() - t0
        batch[nb] = {"wall_s_for_20mm": round(dt, 3),
                     "extrapolated_s_per_track_for_the_crossing":
                         round(dt / nb * float(np.median(span)) / 20.0, 4)}
        print("  batch %6d: %.3f s for 20 mm -> %.4f s per track for a "
              "%.0f mm crossing" % (nb, dt, batch[nb][
                  "extrapolated_s_per_track_for_the_crossing"],
                  float(np.median(span))))
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    # ---- write --------------------------------------------------------------
    cols = ["measurement", "step_mm", "reference_step_mm", "integrator",
            "field", "group_kind", "group", "n", "pos_med_um", "pos_p95_um",
            "pos_max_um", "slope_med_urad", "slope_p95_urad", "slope_max_urad"]
    with open(os.path.join(RESULTS, "reference_convergence.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(summary)
    pcols = ["leg", "EVT", "MCKEY", "P_GeV", "ETA", "direction", "p_band",
             "z0", "z1", "measurement", "step_mm", "reference_step_mm",
             "integrator", "field", "pos_um", "slope_urad"]
    with open(os.path.join(RESULTS, "reference_convergence_per_leg.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=pcols)
        w.writeheader()
        w.writerows(per_leg)

    def med(measurement, step, fieldname="v8r1." + FIELD):
        for r in summary:
            if (r["measurement"] == measurement and r["step_mm"] == step
                    and r["field"] == fieldname and r["group"] == "all"):
                return r["pos_med_um"]
        return None

    ladder = [(a, med("step halving", a)) for a in STEPS[:-1]]
    ratios = [ladder[i][1] / ladder[i + 1][1] for i in range(len(ladder) - 1)]
    meta = {
        "what": "C1.2 / C1.3 - the fine reference's convergence and cost",
        "node": platform.node(), "python": sys.version.split()[0],
        "numpy": np.__version__,
        "field": {"which": FIELD, "file": field_path(FIELD),
                  "md5": field_md5(FIELD),
                  "grid": "trilinear on a 100 mm cell, so C0 in the derivative"},
        "legs": {"n": n, "per_cell": N_PER_CELL, "seed": SEED,
                 "cells": legs["cells"],
                 "span_mm": {"min": float(span.min()), "max": float(span.max()),
                             "median": float(np.median(span))},
                 "selection": "magnet_leg_rows() then a coarse in-map check"},
        "step_ladder_mm": list(STEPS),
        "successive_halving_median_um": {str(a): b for a, b in ladder},
        "successive_halving_ratios": [round(r, 3) for r in ratios],
        "ideal_ratio_for_order_6": 64,
        "smooth_control_median_um": {
            "%g vs %g" % (a, b): med("step halving", a, "smooth analytic")
            for a, b in zip(SMOOTH_STEPS[:-1], SMOOTH_STEPS[1:])},
        "closure_median_um": med("forward-then-back closure", CLOSURE_STEP),
        "cross_check_other_polarity": {
            "field": "v8r1." + CROSS_CHECK_FIELD,
            "step_halving_%g_vs_%g_median_um" % CROSS_CHECK_STEPS:
                med("step halving", CROSS_CHECK_STEPS[0],
                    "v8r1." + CROSS_CHECK_FIELD),
            "closure_median_um": med("forward-then-back closure", CLOSURE_STEP,
                                     "v8r1." + CROSS_CHECK_FIELD),
            "why": "the map's grid, not its sign, sets the floor; these legs "
                   "were chosen on the %s map so this is the same 200 start "
                   "states integrated through the other polarity"
                   % FIELD},
        "rk4_vs_rk6_median_um": {
            str(h): med("RK4 against RK6 at %g mm" % STEPS[-1], h)
            for h in RK4_STEPS},
        "cost": {"wall_s": timing,
                 "seconds_per_track_at_%g_mm" % RK6_STEP: round(per_track, 4),
                 "crossing_mm": float(np.median(span)),
                 "batch": batch,
                 "peak_rss_MB": round(peak_mb, 1),
                 "note": "one thread; the field call is the whole cost, and it "
                         "amortises over the batch, so the per-track figure "
                         "falls until the batch stops fitting in cache"},
        "wall_s": round(time.time() - t_start, 1),
    }
    with open(os.path.join(RESULTS, "reference_convergence_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps({k: meta[k] for k in
                      ("successive_halving_median_um",
                       "successive_halving_ratios",
                       "smooth_control_median_um", "closure_median_um",
                       "rk4_vs_rk6_median_um")}, indent=1))
    print("done in %.0f s" % meta["wall_s"])


if __name__ == "__main__":
    main()
