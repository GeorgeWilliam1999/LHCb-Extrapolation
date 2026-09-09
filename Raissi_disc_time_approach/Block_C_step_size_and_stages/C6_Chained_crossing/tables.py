#!/usr/bin/env python
"""C6.4a - the long tables become the display tables and the ranking.

Nothing here re-scores anything: it reads `results/chained_components.csv`,
`results/single_step_components.csv` and `results/reference_rows.csv`, which
`aggregate.py` builds out of the per-network records, plus
`../C3_Step_size_and_stage_grid/results/table_cells.csv` for the single-step
reference column. It can therefore be run while the cluster is still draining,
and re-run when it has finished.

| output | what it is |
|---|---|
| `results/chained_table_<D>x<W>.csv` | rows q = 2 … 20, columns the six step lengths, cells `median [min-max over seeds]` of max(\|dx\|, \|dy\|) at the far plane against the fine reference, physics arm and data twin |
| `results/chained_table_<D>x<W>_<component>.csv` | the same for x and y in µm and tx and ty in mrad |
| `results/chained_table_<D>x<W>_hit.csv` | the same max(\|dx\|, \|dy\|) table against the particle's real far-plane hit |
| `results/single_step_table_<D>x<W>_<component>.csv` | the single step on the v3 test split, per component, rows q, columns the six strata |
| `results/ranking.csv` | every network's best chained far-plane median over the six columns, and the column that reached it |
| `results/ranking_extremes.csv` | the best three and worst three of each arm - the twelve networks C6.4's figures are drawn for |

Every display table carries four reference rows beneath the network rows:

* **single step, full crossing** - the same architecture's own single-step cell
  from `../C3_Step_size_and_stage_grid` C4, at each q. It is what the chain has to
  beat, and the "full crossing" column of the chained table is the same
  measurement on this experiment's 1,000 tracks rather than the grid's 2,000
  test rows, so the two should agree to the population difference and no more.
* **straight line** - the null step, measured on exactly these tracks. Chaining
  it changes nothing, so one row serves every column.
* **material floor** - the particle's real far-plane hit against the fine
  reference on the same tracks: the part of a crossing no field-only method can
  predict (`../C5_MC_hit_comparison`).
* **reference floor** - 5e-5 µm, the fine reference's own accuracy over a
  crossing (`../C1_Fine_reference` C1.4).

    PYTHONNOUSERSITE=1 python tables.py
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import argparse
import csv
import json
from collections import defaultdict

import numpy as np

import use_shared                                        # noqa: F401
import chain as C

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GRID = os.path.join(ROOT, "C3_Step_size_and_stage_grid")
RESULTS = os.path.join(HERE, "results")
ARCHS = [(d, w) for d in (2, 4, 8) for w in (32, 64, 128, 256)]
QS = tuple(range(2, 21, 2))
ARMS = (("physics", "physics"), ("data", "twin"))
COLUMN_NAMES = [c[0] for c in C.COLUMNS]


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def cell(vals):
    """'median [min-max]' over the seeds of one cell, or '' if empty."""
    v = [x for x in vals if np.isfinite(x)]
    if not v:
        return ""
    if len(v) == 1:
        return "%.3g [%.3g-%.3g]" % (v[0], v[0], v[0])
    return "%.3g [%.3g-%.3g]" % (np.median(v), min(v), max(v))


def grid_single_step(stratum="full crossing"):
    """(depth, width, mode, q) -> (median, min, max) from the grid's C4 table."""
    out = {}
    for r in read_csv(os.path.join(GRID, "results", "table_cells.csv")):
        if (r["split"] != "test" or r["direction"] != "all"
                or r["stratum"] != stratum):
            continue
        out[(int(r["depth"]), int(r["width"]), r["mode"], int(r["q"]))] = (
            float(r["median_um"]), float(r["min_um"]), float(r["max_um"]),
            float(r["straight_um"]), int(r["n_parameters"]),
            int(r["multiply_adds"]))
    return out


def reference_block(refs, component, reference, columns):
    """The four reference rows of a display table, as (label, arm, cells)."""
    def look(predictor, column):
        for r in refs:
            if (r["predictor"] == predictor and r["column"] == column
                    and r["direction"] == "all" and r["component"] == component
                    and r["reference"] == reference):
                return float(r["median_abs"])
        return float("nan")
    rows = []
    for predictor in ("straight line", "material floor", "reference floor"):
        if predictor == "material floor" and reference == "hit":
            continue           # the floor is defined against the fine reference
        cells = []
        for col in columns:
            v = look(predictor, col)
            cells.append("" if not np.isfinite(v) else "%.4g" % v)
        rows.append((predictor, predictor, cells))
    return rows


def write_table(path, header, network_rows, extra_rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in network_rows + extra_rows:
            w.writerow(r)


def chained_tables(rows, refs, grid, out_dir):
    by = defaultdict(list)
    for r in rows:
        by[(int(r["depth"]), int(r["width"]), r["mode"], int(r["q"]),
            r["column"], r["component"], r["reference"])].append(
                float(r["median_abs"]))
    made = []
    for component in C.COMPONENTS:
        for reference in ("fine", "hit"):
            if component != "max_xy" and reference == "hit":
                continue
            for (d, w) in ARCHS:
                net = []
                for q in QS:
                    for mode, arm in ARMS:
                        cells = [cell(by.get((d, w, mode, q, col, component,
                                              reference), []))
                                 for col in COLUMN_NAMES]
                        if not any(cells):
                            continue
                        net.append(["q=%d" % q, arm] + cells)
                if not net:
                    continue
                extra = []
                if component == "max_xy":
                    for q in QS:
                        for mode, arm in ARMS:
                            g = grid.get((d, w, mode, q))
                            if g is None:
                                continue
                            c = ["" for _ in COLUMN_NAMES]
                            c[COLUMN_NAMES.index("full crossing")] = \
                                "%.3g [%.3g-%.3g]" % g[:3]
                            extra.append(["single step (grid C4) q=%d" % q,
                                          arm + ", single step"] + c)
                extra += [[lab, arm] + c
                          for lab, arm, c in reference_block(
                              refs, component, reference, COLUMN_NAMES)]
                suffix = ("" if (component == "max_xy" and reference == "fine")
                          else ("_hit" if reference == "hit"
                                else "_" + component))
                path = os.path.join(out_dir,
                                    "chained_table_%dx%d%s.csv" % (d, w, suffix))
                write_table(path, ["row", "arm"] + COLUMN_NAMES, net, extra)
                made.append(os.path.basename(path))
    return made


def single_step_tables(rows, grid_by_stratum, out_dir):
    strata = ["0.05-0.2 mm", "0.5-2 mm", "5-20 mm", "50-200 mm",
              "500-2000 mm", "full crossing"]
    by = defaultdict(list)
    for r in rows:
        if r["direction"] != "all":
            continue
        by[(int(r["depth"]), int(r["width"]), r["mode"], int(r["q"]),
            r["stratum"], r["component"])].append(float(r["median_abs"]))
    made = []
    for component in C.COMPONENTS:
        for (d, w) in ARCHS:
            net = []
            for q in QS:
                for mode, arm in ARMS:
                    cells = [cell(by.get((d, w, mode, q, s, component), []))
                             for s in strata]
                    if not any(cells):
                        continue
                    net.append(["q=%d" % q, arm] + cells)
            if not net:
                continue
            extra = []
            if component == "max_xy":
                sl = ["%.4g" % grid_by_stratum[s] if s in grid_by_stratum
                      else "" for s in strata]
                extra.append(["straight line", "straight line"] + sl)
                extra.append(["reference floor", "reference floor"]
                             + ["%.4g" % C.REFERENCE_FLOOR_UM for _ in strata])
            path = os.path.join(
                out_dir, "single_step_table_%dx%d_%s.csv" % (d, w, component))
            write_table(path, ["row", "arm"] + strata, net, extra)
            made.append(os.path.basename(path))
    return made


def ranking(rows, grid, out_dir):
    best = {}
    per_column = defaultdict(dict)
    for r in rows:
        if (r["direction"] != "all" or r["component"] != "max_xy"
                or r["reference"] != "fine"):
            continue
        tag, v = r["tag"], float(r["median_abs"])
        per_column[tag][r["column"]] = v
        cur = best.get(tag)
        if cur is None or v < cur[0]:
            best[tag] = (v, r["column"], r)
    out = []
    for tag, (v, col, r) in best.items():
        d, w, q = int(r["depth"]), int(r["width"]), int(r["q"])
        g = grid.get((d, w, r["mode"], q))
        row = dict(tag=tag, width=w, depth=d, arch="%dx%d" % (d, w), q=q,
                   mode=r["mode"], seed=int(r["seed"]),
                   best_column=col, best_median_um=v,
                   single_step_full_crossing_um=(g[0] if g else float("nan")),
                   n_parameters=(g[4] if g else 0),
                   multiply_adds=(g[5] if g else 0))
        for c in COLUMN_NAMES:
            row["col_" + c.replace(" ", "_")] = per_column[tag].get(
                c, float("nan"))
        out.append(row)
    out.sort(key=lambda r: r["best_median_um"])
    for i, r in enumerate(out):
        r["rank"] = i + 1
    cols = (["rank", "tag", "arch", "width", "depth", "q", "mode", "seed",
             "best_column", "best_median_um", "single_step_full_crossing_um",
             "n_parameters", "multiply_adds"]
            + ["col_" + c.replace(" ", "_") for c in COLUMN_NAMES])
    with open(os.path.join(out_dir, "ranking.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in out:
            w.writerow([("%.6g" % r[c]) if isinstance(r[c], float) else r[c]
                        for c in cols])

    extremes = []
    for mode, arm in ARMS:
        arm_rows = [r for r in out if r["mode"] == mode]
        for label, sel in (("best", arm_rows[:3]),
                           ("worst", arm_rows[-3:][::-1])):
            for i, r in enumerate(sel):
                extremes.append(dict(arm=arm, group=label, place=i + 1, **r))
    with open(os.path.join(out_dir, "ranking_extremes.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "group", "place"] + cols)
        for r in extremes:
            w.writerow([r["arm"], r["group"], r["place"]]
                       + [("%.6g" % r[c]) if isinstance(r[c], float) else r[c]
                          for c in cols])
    return out, extremes


def component_reading(chained, out_dir):
    """Which component dominates, and how much of it is bias rather than spread.

    One row per (architecture, arm, column, component), pooling the ten stage
    counts and three seeds: the median of the cell medians, and the median of
    |mean| / rms - the fraction of the error that is a systematic offset of the
    whole population rather than scatter within it.
    """
    by = defaultdict(lambda: defaultdict(list))
    for r in chained:
        if r["direction"] != "all" or r["reference"] != "fine":
            continue
        k = ("%sx%s" % (r["depth"], r["width"]), r["mode"], r["column"],
             r["component"])
        by[k]["median"].append(float(r["median_abs"]))
        by[k]["p95"].append(float(r["p95_abs"]))
        m, rms = abs(float(r["mean"])), float(r["rms"])
        if rms > 0:
            by[k]["bias_over_rms"].append(m / rms)
        med = float(r["median_abs"])
        if med > 0:
            by[k]["bias_over_median"].append(m / med)
            by[k]["p95_over_median"].append(float(r["p95_abs"]) / med)
    rows = []
    for (arch, mode, col, comp), v in by.items():
        rows.append(dict(arch=arch, mode=mode, column=col, component=comp,
                         unit=C.UNITS[comp],
                         median_abs=float(np.median(v["median"])),
                         p95_abs=float(np.median(v["p95"])),
                         bias_over_rms=(float(np.median(v["bias_over_rms"]))
                                        if v["bias_over_rms"]
                                        and comp != "max_xy"
                                        else float("nan")),
                         bias_over_median=(float(np.median(
                             v["bias_over_median"]))
                             if v["bias_over_median"] and comp != "max_xy"
                             else float("nan")),
                         p95_over_median=float(np.median(
                             v["p95_over_median"]))
                         if v["p95_over_median"] else float("nan"),
                         n_cells=len(v["median"])))
    with open(os.path.join(out_dir, "component_reading.csv"), "w",
              newline="") as f:
        wr = csv.writer(f)
        cols = ["arch", "mode", "column", "component", "unit", "median_abs",
                "p95_abs", "bias_over_rms", "bias_over_median",
                "p95_over_median", "n_cells"]
        wr.writerow(cols)
        for r in sorted(rows, key=lambda r: (r["arch"], r["mode"], r["column"],
                                             r["component"])):
            wr.writerow([("%.6g" % r[c]) if isinstance(r[c], float) else r[c]
                         for c in cols])
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=RESULTS)
    a = ap.parse_args(argv)

    chained = read_csv(os.path.join(a.results, "chained_components.csv"))
    single = read_csv(os.path.join(a.results, "single_step_components.csv"))
    refs = read_csv(os.path.join(a.results, "reference_rows.csv"))
    grid = grid_single_step()
    straight_by_stratum = {}
    for r in read_csv(os.path.join(GRID, "results", "table_cells.csv")):
        if r["split"] == "test" and r["direction"] == "all":
            straight_by_stratum[r["stratum"]] = float(r["straight_um"])

    made = chained_tables(chained, refs, grid, a.results)
    made += single_step_tables(single, straight_by_stratum, a.results)
    rank, extremes = ranking(chained, grid, a.results)
    comps = component_reading(chained, a.results)
    print("component_reading.csv %d rows" % len(comps))

    print("%d display tables written" % len(made))
    print("ranking over %d networks" % len(rank))
    for mode, arm in ARMS:
        rows = [r for r in rank if r["mode"] == mode]
        if not rows:
            continue
        print("  %-7s best : " % arm + ", ".join(
            "%s %s %.4g um (%s)" % (r["arch"], "q=%d" % r["q"],
                                    r["best_median_um"], r["best_column"])
            for r in rows[:3]))
        print("  %-7s worst: " % arm + ", ".join(
            "%s %s %.4g um (%s)" % (r["arch"], "q=%d" % r["q"],
                                    r["best_median_um"], r["best_column"])
            for r in rows[-3:][::-1]))
    with open(os.path.join(a.results, "tables_meta.json"), "w") as f:
        json.dump({"tables": made, "n_ranked": len(rank),
                   "n_chained_rows": len(chained),
                   "n_single_step_rows": len(single)}, f, indent=1)


if __name__ == "__main__":
    main()
