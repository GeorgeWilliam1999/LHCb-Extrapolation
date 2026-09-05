#!/usr/bin/env python
"""A4.4 - collect the 13 farm runs and set them beside the magnet-down baseline.

    results/summary.csv     one row per run (the json each training wrote)
    results/up_vs_down.csv  the comparison table: the label-free (physics) runs
                            and their supervised twins on each polarity, the
                            exact-scheme ceiling on each, and the straight line

Three magnet-down comparators, because they are not interchangeable:

  * `../Network_size_and_seed_study` (A1) at width 50, depth 4 - the closest
    match to these runs: same architecture, same ten-seed protocol, same
    single-thread arithmetic. This is the comparator the verdict rests on.
  * `../One_step_network_v2` - the original three-seed baseline, run with four
    BLAS threads, which shifts the optimiser's path in the fourth digit.
  * the exact-scheme ceiling, measured by `ceiling_up.py` on both polarities,
    on both the 32 momentum-stratified legs of `../Simple_first_pass` and the
    frozen leg's own test split. The test-split number is the like-for-like one,
    since it is the population the networks are scored on; the down value it
    measures (22.5 um) reproduces
    `../Stage_count_sweep/results/scheme_ceiling_same_population_q08.json` to the
    last digit, which is the cross-check that the solver copied into
    `ceiling_up.py` is the shared one.

The 29 um figure in `../Simple_first_pass/results/scheme_error_vs_q.csv` is not
used at all: it was measured on the v1 self-generated training population, while
everything here stands on the v2 official-sample population.
"""
from __future__ import annotations

import csv
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
V2 = os.path.join(HERE, "..", "One_step_network_v2", "results", "summary.csv")
A1 = os.path.join(HERE, "..", "Network_size_and_seed_study", "results",
                  "summary.csv")

FIELDS = ["tag", "mode", "seed", "field", "q", "width", "depth", "restarts",
          "final_loss", "converged", "wall_s",
          "train_med_um", "val_med_um", "test_med_um", "test_p95_um",
          "test_stage_med_um", "test_slope_med_mrad", "test_rho_median",
          "straight_med_um", "n_test"]


def collect():
    rows = []
    for p in sorted(glob.glob(os.path.join(RES, "up_*.json"))):
        with open(p) as f:
            r = json.load(f)
        rows.append({
            "tag": r["tag"], "mode": r["mode"], "seed": r["seed"],
            "field": r["field"], "q": r["q"], "width": r["width"],
            "depth": r["depth"], "restarts": r["restarts"],
            "final_loss": r["final_loss"], "converged": r["converged"],
            "wall_s": r["wall_s"],
            "train_med_um": r["train"]["endpoint_med_um"],
            "val_med_um": r["val"]["endpoint_med_um"],
            "test_med_um": r["test"]["endpoint_med_um"],
            "test_p95_um": r["test"]["endpoint_p95_um"],
            "test_stage_med_um": r["test"]["stage_med_um"],
            "test_slope_med_mrad": r["test"]["slope_med_mrad"],
            "test_rho_median": r["test"]["rho_median"],
            "straight_med_um": r["test"]["straight_med_um"],
            "n_test": r["test"]["n"],
        })
    rows.sort(key=lambda r: (r["mode"], r["seed"]))
    with open(os.path.join(RES, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows


def down_rows():
    with open(os.path.abspath(V2)) as f:
        return list(csv.DictReader(f))


def band(vals):
    v = np.asarray(sorted(vals), dtype=float)
    return float(np.median(v)), float(v.min()), float(v.max()), len(v)


def a1_rows(width=50, depth=4):
    """The A1 down-field runs at this architecture, if that experiment is there."""
    p = os.path.abspath(A1)
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return [r for r in csv.DictReader(f)
                if int(r["width"]) == width and int(r["depth"]) == depth]


def main():
    rows = collect()
    dn = down_rows()
    a1 = a1_rows()
    with open(os.path.join(RES, "ceiling_summary.json")) as f:
        c = json.load(f)
    ceil_test = c["on_the_test_states"]      # the like-for-like comparator
    ceil_strat = c["per_field"]              # the 32 momentum-stratified legs

    out = []

    def add(what, polarity, vals, note=""):
        med, lo, hi, n = band(vals)
        out.append({"quantity": what, "field": polarity,
                    "median_um": round(med, 1), "min_um": round(lo, 1),
                    "max_um": round(hi, 1), "n_seeds": n, "note": note})

    for mode, label in (("physics", "network, physics loss (label-free)"),
                        ("data", "network, data loss (supervised twin)")):
        up = [r for r in rows if r["mode"] == mode]
        if up:
            add(label, "up", [r["test_med_um"] for r in up],
                "this experiment; converged %d/%d; restarts %d-%d"
                % (sum(bool(r["converged"]) for r in up), len(up),
                   min(r["restarts"] for r in up),
                   max(r["restarts"] for r in up)))
        v = [r for r in a1 if r["mode"] == mode]
        if v:
            add(label, "down", [float(r["test_endpoint_med_um"]) for r in v],
                "A1 Network_size_and_seed_study 50x4; converged %d/%d"
                % (sum(r["converged"] == "True" for r in v), len(v)))
            vc = [r for r in v if r["converged"] == "True"]
            if vc and len(vc) != len(v):
                add(label + ", converged only", "down",
                    [float(r["test_endpoint_med_um"]) for r in vc],
                    "A1 50x4, the runs that confirmed")
        d = [r for r in dn if r["mode"] == mode]
        add(label, "down", [float(r["endpoint_med_um"]) for r in d],
            "One_step_network_v2 baseline, 3 seeds, 4 BLAS threads")

    for pol in ("up", "down"):
        out.append({"quantity": "exact-scheme ceiling, on the test states",
                    "field": pol,
                    "median_um": round(ceil_test[pol]["endpoint_med_um"], 1),
                    "min_um": round(ceil_test[pol]["endpoint_med_um"], 1),
                    "max_um": round(ceil_test[pol]["endpoint_p95_um"], 1),
                    "n_seeds": ceil_test[pol]["n"],
                    "note": "q=8, all test states; max_um column is the p95; "
                            "solved %.0f%%" % (100 * ceil_test[pol]["converged_frac"])})
    for pol in ("up", "down"):
        out.append({"quantity": "exact-scheme ceiling, 32 stratified legs",
                    "field": pol,
                    "median_um": round(ceil_strat[pol]["median_err_um"], 1),
                    "min_um": round(ceil_strat[pol]["median_err_um"], 1),
                    "max_um": round(ceil_strat[pol]["worst_err_um"], 1),
                    "n_seeds": ceil_strat[pol]["n"],
                    "note": "the Simple_first_pass population; max_um is the "
                            "worst state; solved %d/%d"
                            % (ceil_strat[pol]["n_converged"], ceil_strat[pol]["n"])})

    if rows:
        s_up = rows[0]["straight_med_um"]
        out.append({"quantity": "straight line (no bending at all)",
                    "field": "up", "median_um": round(s_up, 1),
                    "min_um": round(s_up, 1), "max_um": round(s_up, 1),
                    "n_seeds": 1, "note": "the trivial baseline"})
    s_dn = float(dn[0]["straight_med_um"])
    out.append({"quantity": "straight line (no bending at all)", "field": "down",
                "median_um": round(s_dn, 1), "min_um": round(s_dn, 1),
                "max_um": round(s_dn, 1), "n_seeds": 1,
                "note": "the trivial baseline"})

    # how far above its OWN ceiling each polarity's label-free network sits
    for pol, src in (("up", [r for r in out if r["field"] == "up"
                             and r["quantity"].startswith("network, physics")]),
                     ("down", [r for r in out if r["field"] == "down"
                               and r["quantity"].startswith("network, physics")])):
        if not src:
            continue
        n = src[0]
        cl = ceil_test[pol]["endpoint_med_um"]
        out.append({"quantity": "physics network / its own ceiling",
                    "field": pol,
                    "median_um": round(n["median_um"] / cl, 2),
                    "min_um": round(n["min_um"] / cl, 2),
                    "max_um": round(n["max_um"] / cl, 2),
                    "n_seeds": n["n_seeds"],
                    "note": "a RATIO, not micrometres; %s" % n["note"]})

    with open(os.path.join(RES, "up_vs_down.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    print("%-48s %-5s %10s %10s %10s %6s  %s"
          % ("quantity", "field", "median", "min", "max", "n", "note"))
    for r in out:
        print("%-48s %-5s %10.1f %10.1f %10.1f %6d  %s"
              % (r["quantity"], r["field"], r["median_um"], r["min_um"],
                 r["max_um"], r["n_seeds"], r["note"]))
    print("\n%d runs collected; converged %d"
          % (len(rows), sum(bool(r["converged"]) for r in rows)))


if __name__ == "__main__":
    main()
