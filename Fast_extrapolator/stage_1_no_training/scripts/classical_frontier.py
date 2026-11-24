"""Family 0: what classical integrators give at each cost, with no network.

It answers two questions before any training:

  - the frontier: for a given number of field reads (and compiled time), what
    is the smallest error that a classical integrator reaches? A network must
    beat this frontier, and a tuned classical integrator can be the answer;
  - the floor: for a layout of N steps and q Gauss-Legendre stages, what is the
    error of the exact scheme? No network on that layout can be better, so the
    floor says which layouts can reach 1 um at all.

Methods, on the linear map (the field of LHCb) and on the cubic map (the
standard of the project), 13,447 test tracks each:

    RK4, equal steps                        4 reads a step
    Nystrom, equal steps                    2 reads a step and 1
    RK4 and Nystrom on a step plan          steps of unequal length (step_plans.py);
                                            the beta of a plan is chosen on 2,000
                                            training tracks
    the HLT2 adaptive Nystrom (STEP)        its tolerance scanned, 0.1 um to 100 um
    the HLT2 adaptive Cash-Karp             its two tolerances scaled together
    the Allen adaptive Nystrom              its tolerance scanned (linear map only)
    sweeps of the Gauss-Legendre stage map  from the straight line, 1 to 3 sweeps:
                                            the physics layer without its network

A fixed-step method runs in single precision, with the field read of LHCb on
the linear map; the copies of the LHCb tools run as their originals do.

    classical_frontier.py choose_plans   results/chosen_plans.csv
    classical_frontier.py accuracy       results/classical_accuracy.csv
    classical_frontier.py floor          results/floor_of_the_layouts.csv
    classical_frontier.py time CORE      results/classical_timing.csv
    classical_frontier.py table          results/classical_frontier.csv
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastex import bench  # noqa: E402
from step_plans import BETAS, STEPS, file_of_plan  # noqa: E402

RESULTS = os.path.join(HERE, "results")
MAPS = ("linear", "cubic")
FAST = {"linear": ("float", "rec"), "cubic": ("float", "cubic")}        # fixed-step methods
EXACT = {"linear": ("double", "rec"), "cubic": ("double", "cubic")}      # the Rec tools
TOLERANCES_UM = (0.1, 0.3, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
SCALES_OF_CASH_KARP = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
LAYOUTS = [(steps, stages) for steps in (1, 2, 4, 8, 16, 32) for stages in (2, 4, 8)]


def row(label, family, method, of_map, config, tracks=None, **more):
    precision, reader = config
    return dict(label=label, family=family, method=method, precision=precision, reader=reader,
                tracks=tracks or of_map, map_of_the_field=of_map, **more)


def plan_rows(of_map, tracks, betas_of=None):
    """The rows of the step plans: every beta (to choose), or the chosen one."""
    rows = []
    for integrator, name in (("rk4_plan", "RK4"), ("nystrom_plan", "Nystrom")):
        for steps in STEPS:
            betas = BETAS if betas_of is None else [betas_of[(of_map, integrator, steps)]]
            for beta in betas:
                rows.append(row("%s on a step plan, %d steps" % (name, steps),
                                "%s, step plan" % name,
                                "%s:%s" % (integrator, file_of_plan(of_map, steps, beta)), of_map,
                                FAST[of_map], tracks=tracks, steps=steps, beta=beta,
                                integrator=integrator))
    return rows


def chosen_betas():
    import pandas as pd
    chosen = pd.read_csv(os.path.join(RESULTS, "chosen_plans.csv"))
    return {(r["map_of_the_field"], r["integrator"], int(r["steps"])): float(r["beta"])
            for _, r in chosen.iterrows()}


def methods():
    rows = []
    betas = chosen_betas()
    for of_map in MAPS:
        for steps in (4, 6, 8, 10, 11, 12, 16, 19, 20, 22, 24, 29, 31, 32, 33, 34, 36, 48, 64):
            rows.append(row("RK4, %d equal steps" % steps, "RK4, equal steps", "rk4:%d" % steps,
                            of_map, FAST[of_map], steps=steps))
        for steps in (6, 8, 10, 11, 12, 16, 19, 20, 22, 24, 29, 31, 32, 33, 34, 36, 48, 64):
            rows.append(row("Nystrom, %d equal steps" % steps, "Nystrom, equal steps",
                            "nystrom:%d" % steps, of_map, FAST[of_map], steps=steps))
        rows += plan_rows(of_map, of_map, betas)
        for um in TOLERANCES_UM:
            rows.append(row("HLT2 adaptive Nystrom (STEP), tolerance %g um" % um,
                            "HLT2 adaptive Nystrom", "rec_step:tolerance=%g" % (um * 1e-3), of_map,
                            EXACT[of_map], tolerance_um=um))
        for scale in SCALES_OF_CASH_KARP:
            rows.append(row("HLT2 adaptive Cash-Karp, tolerances x %g" % scale,
                            "HLT2 adaptive Cash-Karp",
                            "rec_cash_karp:tolerance=%g:relative_tolerance_tx=%g"
                            % (1e-3 * scale, 5e-5 * scale), of_map, EXACT[of_map],
                            tolerance_um=scale))
        if of_map == "linear":
            for um in TOLERANCES_UM:
                rows.append(row("Allen adaptive Nystrom, tolerance %g um" % um,
                                "Allen adaptive Nystrom",
                                "allen_adaptive_nystrom:tolerance=%g" % (um * 1e-3), of_map,
                                ("float", "allen"), tolerance_um=um))
        for steps in (2, 4, 8, 16):
            for stages in (2, 4):
                for sweeps in (1, 2, 3):
                    rows.append(row("Gauss-Legendre, %d steps x %d stages, %d sweeps"
                                    % (steps, stages, sweeps), "Gauss-Legendre sweeps",
                                    "sweeps:%d:%d:%d" % (steps, stages, sweeps), of_map,
                                    FAST[of_map], steps=steps, stages=stages, sweeps=sweeps))
    return rows


def choose_plans():
    """Every beta of every plan on the training tracks; the beta with the
    lowest median error is the chosen one."""
    rows = []
    for of_map in MAPS:
        for m in plan_rows(of_map, of_map + "_training"):
            rows.append(bench.score(m, report=lambda text: None))
    bench.write(os.path.join(RESULTS, "plans_on_the_training_tracks.csv"), rows)
    best = {}
    for r in rows:
        key = (r["map_of_the_field"], r["integrator"], r["steps"])
        if key not in best or r["radial_median_um"] < best[key]["radial_median_um"]:
            best[key] = r
    chosen = [dict(map_of_the_field=k[0], integrator=k[1], steps=k[2], beta=r["beta"],
                   training_radial_median_um=r["radial_median_um"],
                   training_radial_95_um=r["radial_95_um"]) for k, r in sorted(best.items())]
    bench.write(os.path.join(RESULTS, "chosen_plans.csv"), chosen)
    for c in chosen:
        print(c)


def floor():
    """The exact scheme of every layout: the stage map swept to convergence
    (40 sweeps; 60 sweeps as the check of convergence), in double precision."""
    rows = []
    for of_map in MAPS:
        reader = "linear" if of_map == "linear" else "cubic"
        for steps, stages in LAYOUTS:
            found = {}
            for sweeps in (40, 60):
                m = row("Exact Gauss-Legendre scheme, %d steps x %d stages" % (steps, stages),
                        "exact scheme", "sweeps:%d:%d:%d" % (steps, stages, sweeps), of_map,
                        ("double", reader), steps=steps, stages=stages, sweeps=sweeps)
                found[sweeps] = bench.score(m, report=lambda text: None)
            r = found[60]
            r["median_change_from_40_to_60_sweeps_um"] = abs(
                found[60]["radial_median_um"] - found[40]["radial_median_um"])
            r["percentile_95_change_from_40_to_60_sweeps_um"] = abs(
                found[60]["radial_95_um"] - found[40]["radial_95_um"])
            r["converged"] = bool(np.isfinite(r["radial_95_um"])
                                  and r["percentile_95_change_from_40_to_60_sweeps_um"]
                                  < 1e-3 * max(r["radial_95_um"], 1e-6))
            r["field_reads_of_one_sweep"] = steps * stages
            rows.append(r)
            print("%-6s %2d steps x %d stages: median %10.4g um, 95 %% %10.4g um, worst band %10.4g um"
                  "  %s" % (of_map, steps, stages, r["radial_median_um"], r["radial_95_um"],
                            r["worst_band_median_um"], "" if r["converged"] else "NOT CONVERGED"),
                  flush=True)
    bench.write(os.path.join(RESULTS, "floor_of_the_layouts.csv"), rows)


def table():
    both = bench.join(os.path.join(RESULTS, "classical_accuracy.csv"),
                      os.path.join(RESULTS, "classical_timing.csv"))
    both.to_csv(os.path.join(RESULTS, "classical_frontier.csv"), index=False)
    print(both[["label", "map_of_the_field", "field_reads_mean", "ns_lowest", "radial_median_um",
                "radial_95_um", "worst_band_median_um"]].to_string())


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "choose_plans":
        choose_plans()
    elif command == "accuracy":
        bench.accuracy(methods(), os.path.join(RESULTS, "classical_accuracy.csv"))
    elif command == "floor":
        floor()
    elif command == "time":
        bench.time(methods(), sys.argv[2], os.path.join(RESULTS, "classical_timing.csv"))
    elif command == "table":
        table()
    else:
        raise SystemExit("choose_plans, accuracy, floor, time CORE or table")
