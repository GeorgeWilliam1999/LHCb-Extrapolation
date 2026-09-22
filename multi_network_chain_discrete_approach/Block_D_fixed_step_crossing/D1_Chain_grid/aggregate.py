#!/usr/bin/env python
"""D1 - the chain records -> the tables.

Reads every results/N<NNN>_q<qq>/chain.json (and, when present, the exact-scheme
records and the twin of ../D2_Comparators) and writes:

    results/chain_table.csv     one row per (N, q, split, comparator): median,
                                p95 and slope error, convergence, cost
    results/table_<comparator>_<split>.csv
                                the 5 x 20 table error(N, q), medians in um
                                (comparators: vs_rk6_endpoint,
                                 vs_real_scifi_state, exact_scheme)
    results/per_leg.csv         one row per (N, q, leg): the leg's own-step
                                error and the chain's error at its plane
    results/growth.csv          one row per (N, q, plane): the chain's error
                                along the crossing
    results/by_p_band.csv       the endpoint errors per momentum band
    results/status.csv          which chains are finished
    results/components.csv      one row per (N, q, split, comparator, component):
                                median, p95, mean and bias of |dx|, |dy|, |dtx|,
                                |dty|; the q/p check
    results/table_<comparator>_<split>_<component>_med.csv
                                the 5 x 20 tables per component

Computes nothing new: every number is copied from a record.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
D2 = os.path.join(HERE, "..", "D2_Comparators", "results")
N_VALUES = (1, 4, 16, 64, 128)
QS = tuple(range(1, 21))
COMPARATORS = ("vs_rk6_endpoint", "vs_real_scifi_state",
               "straight_line_vs_rk6_endpoint", "rk6_truth_vs_real_scifi_state")
COMPONENTS = (("x", "um"), ("y", "um"), ("tx", "mrad"), ("ty", "mrad"))


def write_csv(path, rows, fields=None):
    if not rows:
        return
    fields = fields or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def load_chains():
    chains = {}
    for p in sorted(glob.glob(os.path.join(RESULTS, "N*_q*", "chain.json"))):
        m = re.search(r"N(\d+)_q(\d+)", p)
        with open(p) as f:
            chains[(int(m.group(1)), int(m.group(2)))] = json.load(f)
    return chains


def load_exact():
    out = {}
    for p in glob.glob(os.path.join(D2, "exact_N*_q*.json")):
        m = re.search(r"N(\d+)_q(\d+)", p)
        with open(p) as f:
            out[(int(m.group(1)), int(m.group(2)))] = json.load(f)
    return out


def pivot(values, fmt="%.6g"):
    """values: {(N, q): number} -> rows of the 5 x 20 table."""
    rows = []
    for N in N_VALUES:
        r = {"N": N}
        for q in QS:
            v = values.get((N, q))
            r["q%02d" % q] = (fmt % v) if v is not None else ""
        rows.append(r)
    return rows


def main():
    chains = load_chains()
    exact = load_exact()
    twin = None
    tp = os.path.join(D2, "twin", "twin_scores.json")
    if os.path.exists(tp):
        with open(tp) as f:
            twin = json.load(f)

    table, per_leg, growth, bands, status, comps = [], [], [], [], [], []
    for N in N_VALUES:
        for q in QS:
            c = chains.get((N, q))
            status.append({"N": N, "q": q, "done": c is not None,
                           "all_legs_converged": c["all_legs_converged"] if c else "",
                           "total_restarts": c["total_restarts"] if c else "",
                           "total_train_wall_s": c["total_train_wall_s"] if c else ""})
            if c is None:
                continue
            for split in ("val", "test"):
                for comp in COMPARATORS:
                    s = c[split][comp]
                    table.append({"N": N, "q": q, "dz_mm": c["dz_mm"], "split": split,
                                  "comparator": comp, "pos_med_um": s["pos_med_um"],
                                  "pos_p95_um": s["pos_p95_um"],
                                  "slope_med_mrad": s["slope_med_mrad"], "n": s["n"],
                                  "all_legs_converged": c["all_legs_converged"],
                                  "total_restarts": c["total_restarts"],
                                  "total_train_wall_s": c["total_train_wall_s"],
                                  "n_parameters_per_leg": c["n_parameters_per_leg"]})
                    cs = s.get("components")
                    if cs:
                        for name, unit in COMPONENTS:
                            comps.append({"N": N, "q": q, "split": split, "comparator": comp,
                                          "component": name, "unit": unit,
                                          "med": cs["%s_med_%s" % (name, unit)],
                                          "p95": cs["%s_p95_%s" % (name, unit)],
                                          "mean": cs["%s_mean_%s" % (name, unit)],
                                          "bias": cs["%s_bias_%s" % (name, unit)],
                                          "qop_max_abs_change": cs.get("qop_max_abs_change", ""),
                                          "qop_passthrough_max_abs_change":
                                              c[split].get("qop_passthrough_max_abs_change", "")})
                    if "by_p_band" in s:
                        for band, v in s["by_p_band"].items():
                            bands.append({"N": N, "q": q, "split": split, "comparator": comp,
                                          "p_band": band, **v})
                pp = c[split]["per_plane"]
                for k, z in enumerate(c["planes"]):
                    row = {"N": N, "q": q, "split": split, "plane": k, "z_mm": z,
                           "pos_med_um": pp["pos_med_um"][k],
                           "pos_p95_um": pp["pos_p95_um"][k],
                           "slope_med_mrad": pp["slope_med_mrad"][k]}
                    for name, unit in COMPONENTS:
                        key = "%s_med_%s" % (name, unit)
                        row[key] = pp[key][k] if key in pp else ""
                    growth.append(row)
            for r in c["legs"]:
                per_leg.append({"N": N, "q": q, **r})
            e = exact.get((N, q))
            if e and e.get("components"):
                for name, unit in COMPONENTS:
                    cs = e["components"]
                    comps.append({"N": N, "q": q, "split": "test",
                                  "comparator": "exact_scheme_vs_rk6_endpoint",
                                  "component": name, "unit": unit,
                                  "med": cs["%s_med_%s" % (name, unit)],
                                  "p95": cs["%s_p95_%s" % (name, unit)],
                                  "mean": cs["%s_mean_%s" % (name, unit)],
                                  "bias": cs["%s_bias_%s" % (name, unit)],
                                  "qop_max_abs_change": cs.get("qop_max_abs_change", ""),
                                  "qop_passthrough_max_abs_change": ""})
            if e:
                table.append({"N": N, "q": q, "dz_mm": c["dz_mm"], "split": "test",
                              "comparator": "exact_scheme_vs_rk6_endpoint",
                              "pos_med_um": e["endpoint_pos_med_um"],
                              "pos_p95_um": e["endpoint_pos_p95_um"],
                              "slope_med_mrad": e["endpoint_slope_med_mrad"], "n": e["n"],
                              "all_legs_converged": "", "total_restarts": "",
                              "total_train_wall_s": e["wall_s"], "n_parameters_per_leg": ""})
    write_csv(os.path.join(RESULTS, "chain_table.csv"), table)
    write_csv(os.path.join(RESULTS, "per_leg.csv"), per_leg)
    write_csv(os.path.join(RESULTS, "growth.csv"), growth)
    write_csv(os.path.join(RESULTS, "by_p_band.csv"), bands)
    write_csv(os.path.join(RESULTS, "status.csv"), status)
    for split in ("val", "test"):
        for comp in ("vs_rk6_endpoint", "vs_real_scifi_state"):
            for stat in ("pos_med_um", "pos_p95_um"):
                vals = {(r["N"], r["q"]): r[stat] for r in table
                        if r["split"] == split and r["comparator"] == comp}
                write_csv(os.path.join(RESULTS, "table_%s_%s_%s.csv" % (comp, split, stat)),
                          pivot(vals))
    write_csv(os.path.join(RESULTS, "components.csv"), comps)
    for comp in ("vs_rk6_endpoint", "vs_real_scifi_state", "exact_scheme_vs_rk6_endpoint"):
        for name, unit in COMPONENTS:
            vals = {(r["N"], r["q"]): r["med"] for r in comps
                    if r["split"] == "test" and r["comparator"] == comp and r["component"] == name}
            if vals:
                write_csv(os.path.join(RESULTS, "table_%s_test_%s_med.csv" % (comp, name)),
                          pivot(vals))
    if exact:
        write_csv(os.path.join(RESULTS, "table_exact_scheme_test_pos_med_um.csv"),
                  pivot({k: v["endpoint_pos_med_um"] for k, v in exact.items()}))
    done = sum(1 for s in status if s["done"])
    print("%d / %d chains done; %d exact-scheme records; twin %s"
          % (done, len(status), len(exact), "present" if twin else "absent"))
    if twin:
        print("twin test vs RK6: %.4g um" % twin["test"]["vs_rk6_endpoint"]["pos_med_um"])
    return table


if __name__ == "__main__":
    main()
