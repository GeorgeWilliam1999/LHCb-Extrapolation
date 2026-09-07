#!/usr/bin/env python
"""C2.3. Aggregate the per-solve rows into the two tables the study reports.

Reads every `results/scheme_rows_q*.npz` written by `run_scheme_grid.py` and
writes

  * `results/scheme_table.csv` - one row per (q, stratum, direction): how many
    states, what fraction of them the root-finder actually solved, the endpoint
    error against the fine RK6 reference in microns (median, p95, mean), the
    slope error in milliradians, the median number of residual evaluations one
    solve cost, and the wall time of the cell;
  * `results/straight_line_table.csv` - the same strata and directions with the
    magnet switched off: the end state obtained by carrying the input state
    along its own slopes. It needs no solver, so it has no convergence or cost
    columns; it is the null step every scheme has to beat.

`direction` is `both` (the pooled row), `+1` (dz > 0) and `-1` (dz < 0).

The error statistics are taken over the **converged** solves of a cell, and
`n` / `n_converged` / `converged_frac` say how many that was. A cell whose
solves did not converge is not silently averaged in.

    PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python make_tables.py
"""
from __future__ import annotations

import csv
import glob
import json
import os

import numpy as np

import use_shared  # noqa: F401  (puts _shared on sys.path)
from _shared.prepare import STRATUM_NAMES, load_magnet_tracks

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
DATASET = os.path.abspath(os.path.join(
    HERE, "..", "Magnet_tracks_dataset", "results", "magnet_tracks_v3.npz"))

DIRECTIONS = (("both", None), ("+1", 1), ("-1", -1))

# The fine reference's own floor on a full crossing (../Fine_reference, C1.4).
REFERENCE_FLOOR_UM = 5e-5

# The v8r1 map is trilinear on a cubic grid of this pitch; a step longer than
# one cell crosses a face, where the field's derivative jumps.
FIELD_CELL_MM = 100.0

SCHEME_FIELDS = ["q", "stratum", "direction", "n", "n_converged",
                 "converged_frac", "median_abs_dz_mm", "endpoint_med_um",
                 "endpoint_p95_um", "endpoint_mean_um", "slope_err_med_mrad",
                 "median_n_eval", "wall_s"]

LINE_FIELDS = ["stratum", "direction", "n", "median_abs_dz_mm",
               "endpoint_med_um", "endpoint_p95_um", "endpoint_mean_um",
               "slope_err_med_mrad"]


def _stats(err, slope):
    if not len(err):
        return dict(endpoint_med_um="", endpoint_p95_um="",
                    endpoint_mean_um="", slope_err_med_mrad="")
    return dict(endpoint_med_um=float(np.median(err)),
                endpoint_p95_um=float(np.quantile(err, 0.95)),
                endpoint_mean_um=float(err.mean()),
                slope_err_med_mrad=float(np.median(slope)))


def scheme_rows():
    rows = []
    for path in sorted(glob.glob(os.path.join(RESULTS, "scheme_rows_q*.npz"))):
        d = np.load(path, allow_pickle=False)
        q = int(d["q"])
        for si, name in enumerate(STRATUM_NAMES):
            in_st = d["stratum"] == si
            for label, sign in DIRECTIONS:
                m = in_st if sign is None else in_st & (d["direction"] == sign)
                conv = m & d["converged"]
                r = dict(q=q, stratum=name, direction=label,
                         n=int(m.sum()), n_converged=int(conv.sum()),
                         converged_frac=float(conv.sum() / max(1, m.sum())),
                         median_abs_dz_mm=float(np.median(d["abs_dz_mm"][m])),
                         median_n_eval=float(np.median(d["n_eval"][m])),
                         wall_s=float(d["wall_s"][m].sum()))
                r.update(_stats(d["err_um"][conv], d["slope_err_mrad"][conv]))
                rows.append(r)
    return rows


def straight_line_rows(split="test", cap=2000):
    """The magnet switched off, on the same states the scheme was solved on."""
    rows = []
    for si, name in enumerate(STRATUM_NAMES):
        d = load_magnet_tracks(DATASET, split=split, stratum=si)
        X, Y = d["X"][:cap], d["Y"][:cap]
        dz = X[:, 6]
        end = X[:, :2] + X[:, 2:4] * dz[:, None]          # the null step
        err = np.abs(end - Y[:, :2]).max(axis=1) * 1e3    # um
        slope = np.abs(X[:, 2:4] - Y[:, 2:4]).max(axis=1) * 1e3   # mrad
        for label, sign in DIRECTIONS:
            m = (np.ones(len(X), bool) if sign is None
                 else np.sign(dz) == sign)
            r = dict(stratum=name, direction=label, n=int(m.sum()),
                     median_abs_dz_mm=float(np.median(np.abs(dz[m]))))
            r.update(_stats(err[m], slope[m]))
            rows.append(r)
    return rows


def dz_slope(dz, errs, floor):
    """How the error grows with the step length, at fixed q.

    A q-stage Gauss-Legendre step of length h has classical local error
    O(h^(2q+1)) on a *smooth* problem, so a log-log plot of endpoint error
    against |dz| should have slope 2q + 1. That is the measurement that can
    actually see the classical order here: comparing two stage counts at one
    step length cannot, because the error constant changes with q as well.

    Only points above `floor` are fitted - below it the number is the
    reference's own arithmetic, not the scheme's error.
    """
    dz = np.asarray(dz, float)
    errs = np.asarray(errs, float)
    live = errs > floor
    if live.sum() < 2:
        return dict(n_points=int(live.sum()), measured_slope="")
    slope = np.polyfit(np.log10(dz[live]), np.log10(errs[live]), 1)[0]
    return dict(n_points=int(live.sum()), measured_slope=float(slope))


def reading(srows, lrows, floor=REFERENCE_FLOOR_UM):
    """C2.4: the few numbers the README's reading is built on."""
    qs = sorted({r["q"] for r in srows})
    pooled = {(r["q"], r["stratum"]): r for r in srows
              if r["direction"] == "both"}
    line = {r["stratum"]: r for r in lrows if r["direction"] == "both"}
    dzmed = {s: line[s]["median_abs_dz_mm"] for s in STRATUM_NAMES}

    # --- per stratum: does the error move with q at all? -------------------
    per_stratum = []
    for s in STRATUM_NAMES:
        e2 = pooled[(qs[0], s)]["endpoint_med_um"]
        e20 = pooled[(qs[-1], s)]["endpoint_med_um"]
        p2 = pooled[(qs[0], s)]["endpoint_p95_um"]
        p20 = pooled[(qs[-1], s)]["endpoint_p95_um"]
        best = min(qs, key=lambda q: pooled[(q, s)]["endpoint_med_um"])
        gain = e2 / e20 if e20 > 0 else float("inf")
        per_stratum.append(dict(
            stratum=s, median_abs_dz_mm=dzmed[s],
            field_cells_crossed=dzmed[s] / FIELD_CELL_MM,
            med_q2_um=e2, med_q20_um=e20, gain_med=gain,
            p95_q2_um=p2, p95_q20_um=p20,
            gain_p95=p2 / p20 if p20 > 0 else float("inf"),
            best_q=best, best_med_um=pooled[(best, s)]["endpoint_med_um"],
            straight_line_um=line[s]["endpoint_med_um"],
            beats_straight_line_by=line[s]["endpoint_med_um"] / e20
            if e20 > 0 else float("inf"),
            # a median that barely moves over the whole q range is not the
            # scheme's error at all - it is what it is being measured against
            verdict=("at the reference floor - the scheme's own error is "
                     "smaller than the reference can resolve" if gain < 3
                     else "resolved - the scheme's own error is being measured"),
        ))

    # --- per q: the step-length slope, against the classical 2q + 1 --------
    order = []
    for q in qs:
        fit = dz_slope([dzmed[s] for s in STRATUM_NAMES],
                       [pooled[(q, s)]["endpoint_med_um"] for s in STRATUM_NAMES],
                       floor)
        order.append(dict(q=q, classical_local_order=2 * q + 1, **fit))

    ceiling = []
    for s in STRATUM_NAMES:
        for q in qs:
            r = pooled[(q, s)]
            ceiling.append(dict(stratum=s, q=q, n=r["n"],
                                converged_frac=r["converged_frac"],
                                med_um=r["endpoint_med_um"],
                                p95_um=r["endpoint_p95_um"],
                                slope_med_mrad=r["slope_err_med_mrad"],
                                median_n_eval=r["median_n_eval"],
                                evals_per_cell=r["median_n_eval"] * r["n"],
                                wall_s=r["wall_s"]))
    ceiling = [c for c in ceiling if c["q"] in (8, 20)]

    full = STRATUM_NAMES[-1]
    tail = [q for q in qs if q >= 8]
    plateau = [pooled[(q, full)]["endpoint_med_um"] for q in tail]
    headline = {
        "reference_floor_um": floor,
        "field_cell_mm": FIELD_CELL_MM,
        "states_per_cell": pooled[(qs[0], STRATUM_NAMES[0])]["n"],
        "cells_in_the_grid": len(qs) * len(STRATUM_NAMES),
        "solves": sum(r["n"] for r in srows if r["direction"] == "both"),
        "converged_frac_min": min(r["converged_frac"] for r in srows),
        "full_crossing_q8_med_um": pooled[(8, full)]["endpoint_med_um"],
        "full_crossing_q8_p95_um": pooled[(8, full)]["endpoint_p95_um"],
        "full_crossing_best_q": min(qs, key=lambda q: pooled[(q, full)]["endpoint_med_um"]),
        "full_crossing_best_med_um": min(pooled[(q, full)]["endpoint_med_um"] for q in qs),
        "full_crossing_plateau_q_ge_8_um": [round(v, 2) for v in plateau],
        "full_crossing_plateau_range_um": [round(min(plateau), 2), round(max(plateau), 2)],
        "straight_line_full_crossing_um": line[full]["endpoint_med_um"],
        "grid_wall_s": sum(r["wall_s"] for r in srows if r["direction"] == "both"),
    }
    return dict(headline=headline, per_stratum=per_stratum,
                step_length_order=order, ceiling_row=ceiling)


def _write(path, fields, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("wrote", os.path.relpath(path, HERE), "(%d rows)" % len(rows))


def main():
    srows = scheme_rows()
    assert srows, "no results/scheme_rows_q*.npz - run run_scheme_grid.py first"
    _write(os.path.join(RESULTS, "scheme_table.csv"), SCHEME_FIELDS, srows)
    lrows = straight_line_rows()
    _write(os.path.join(RESULTS, "straight_line_table.csv"), LINE_FIELDS, lrows)

    # the q x stratum grid of pooled medians, for the README and the message
    qs = sorted({r["q"] for r in srows})
    grid = {q: {r["stratum"]: r["endpoint_med_um"] for r in srows
                if r["q"] == q and r["direction"] == "both"} for q in qs}
    line = {r["stratum"]: r["endpoint_med_um"] for r in lrows
            if r["direction"] == "both"}
    hdr = "%-4s" % "q" + "".join("%16s" % s for s in STRATUM_NAMES)
    print("\nendpoint error, median micron, both directions pooled")
    print(hdr)
    for q in qs:
        print("%-4d" % q + "".join("%16.4g" % grid[q][s]
                                   for s in STRATUM_NAMES))
    print("%-4s" % "line" + "".join("%16.4g" % line[s] for s in STRATUM_NAMES))

    with open(os.path.join(RESULTS, "summary_grid.json"), "w") as f:
        json.dump({"median_um": {str(q): grid[q] for q in qs},
                   "straight_line_median_um": line,
                   "strata": list(STRATUM_NAMES)}, f, indent=1)

    rd = reading(srows, lrows)
    with open(os.path.join(RESULTS, "reading.json"), "w") as f:
        json.dump(rd, f, indent=1)

    print("\nper stratum: does the median move with q at all?")
    print("%-15s %11s %8s %11s %11s %8s  %s"
          % ("stratum", "|dz| [mm]", "cells", "q=2 [um]", "q=20 [um]",
             "gain", "verdict"))
    for o in rd["per_stratum"]:
        print("%-15s %11.4g %8.3g %11.4g %11.4g %8.1f  %s"
              % (o["stratum"], o["median_abs_dz_mm"], o["field_cells_crossed"],
                 o["med_q2_um"], o["med_q20_um"], o["gain_med"], o["verdict"]))

    print("\nerror against step length at fixed q (classical local order 2q+1)")
    print("%-4s %10s %10s %8s" % ("q", "2q+1", "measured", "points"))
    for o in rd["step_length_order"]:
        m = o["measured_slope"]
        print("%-4d %10d %10s %8d"
              % (o["q"], o["classical_local_order"],
                 "%.2f" % m if m != "" else "-", o["n_points"]))

    print("\nwrote results/reading.json and results/summary_grid.json")


if __name__ == "__main__":
    main()
