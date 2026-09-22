#!/usr/bin/env python
"""D2 - the comparison table: every (N, q) chain against every comparator.

Joins ../D1_Chain_grid/results/chain_table.csv (the networks, the straight
line and the material floor, written by D1's aggregate.py) with the
exact-scheme records here and the twin, into

    results/comparison_table.csv    one row per (N, q): the chain vs RK6, vs the
                                    real SciFi state, the exact scheme vs RK6,
                                    the straight line, the material floor, and
                                    the ratios that read them
    results/comparison_summary.json the best chain, the best exact scheme, the
                                    twin, the floors

Computes only ratios of numbers already recorded.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
D1 = os.path.join(HERE, "..", "D1_Chain_grid", "results", "chain_table.csv")


def main():
    if not os.path.exists(D1):
        raise SystemExit("run ../D1_Chain_grid/aggregate.py first")
    with open(D1) as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "test"]
    by = {}
    for r in rows:
        by.setdefault((int(r["N"]), int(r["q"])), {})[r["comparator"]] = r
    exact = {}
    for p in glob.glob(os.path.join(RESULTS, "exact_N*_q*.json")):
        m = re.search(r"N(\d+)_q(\d+)", p)
        with open(p) as f:
            exact[(int(m.group(1)), int(m.group(2)))] = json.load(f)
    twin = None
    tp = os.path.join(RESULTS, "twin", "twin_scores.json")
    if os.path.exists(tp):
        with open(tp) as f:
            twin = json.load(f)
    out = []
    for (N, q), comps in sorted(by.items()):
        net = float(comps["vs_rk6_endpoint"]["pos_med_um"])
        real = float(comps["vs_real_scifi_state"]["pos_med_um"])
        straight = float(comps["straight_line_vs_rk6_endpoint"]["pos_med_um"])
        floor = float(comps["rk6_truth_vs_real_scifi_state"]["pos_med_um"])
        ex = exact.get((N, q))
        exv = float(ex["endpoint_pos_med_um"]) if ex else float("nan")
        out.append({"N": N, "q": q, "dz_mm": comps["vs_rk6_endpoint"]["dz_mm"],
                    "network_vs_rk6_med_um": net,
                    "network_vs_rk6_p95_um": comps["vs_rk6_endpoint"]["pos_p95_um"],
                    "exact_scheme_vs_rk6_med_um": exv,
                    "network_over_exact": net / exv if ex and exv > 0 else "",
                    "network_vs_real_scifi_med_um": real,
                    "material_floor_med_um": floor,
                    "network_real_over_floor": real / floor if floor > 0 else "",
                    "straight_line_vs_rk6_med_um": straight,
                    "network_over_straight": net / straight if straight > 0 else "",
                    "all_legs_converged": comps["vs_rk6_endpoint"]["all_legs_converged"],
                    "total_restarts": comps["vs_rk6_endpoint"]["total_restarts"],
                    "total_train_wall_s": comps["vs_rk6_endpoint"]["total_train_wall_s"]})
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "comparison_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    best = min(out, key=lambda r: r["network_vs_rk6_med_um"])
    best_ex = min((r for r in out if r["exact_scheme_vs_rk6_med_um"] == r["exact_scheme_vs_rk6_med_um"]),
                  key=lambda r: r["exact_scheme_vs_rk6_med_um"], default=None)
    summary = {"chains": len(out),
               "best_network": {k: best[k] for k in ("N", "q", "network_vs_rk6_med_um",
                                                     "network_vs_rk6_p95_um",
                                                     "network_vs_real_scifi_med_um")},
               "best_exact_scheme": ({k: best_ex[k] for k in ("N", "q", "exact_scheme_vs_rk6_med_um")}
                                     if best_ex else None),
               "twin_test": twin["test"] if twin else None,
               "material_floor_med_um": out[0]["material_floor_med_um"],
               "straight_line_med_um": out[0]["straight_line_vs_rk6_med_um"]}
    with open(os.path.join(RESULTS, "comparison_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
