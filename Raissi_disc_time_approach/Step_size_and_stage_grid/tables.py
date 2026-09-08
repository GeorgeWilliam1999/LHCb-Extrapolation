#!/usr/bin/env python
"""C4.1 / C4.3 - the error(dz, q) tables and the numbers the readings rest on.

`aggregate_grid.py` (C3.5) already turns the 720 run records into
`results/error_vs_dz_q.csv`, one row per (run, split, stratum, direction). This
script does not re-score anything either: it reads that csv, the exact scheme's
own table from `../Exact_scheme_table` and the straight line's, and writes

**the tables**

* `results/table_cells.csv` - the tidy master every figure and every reading
  below is computed from. One row per (width, depth, mode, q, stratum,
  direction, split): the median over the seeds that confirmed, the min and max
  over those seeds, the p95, the slope error, rho, the straight line on exactly
  those rows, the number of seeds present and a `pending` flag when fewer than
  three landed.
* `results/table_{depth}x{width}.csv` - the display table per architecture,
  both directions pooled: rows q = 2 ... 20, columns the six |dz| strata, the
  physics arm and the data twin one under the other as
  `median [min-max]` in um, then the exact scheme at every q, the straight
  line, and the fine reference's own floor as a note row.
* `results/table_{depth}x{width}_by_direction.csv` - the same, split into
  forward (dz > 0) and backward (dz < 0).
* `results/pending_cells.csv` - the grid points whose record had not landed.

**the readings** (one csv each, so every number in the README is traceable)

| file | reading |
|---|---|
| `results/reading_q_sensitivity.csv` | (a) in which columns q moves the answer at all |
| `results/reading_dz_slope.csv` | (b) the log-log slope of the floor against \|dz\| |
| `results/reading_ratio_straight.csv` | (c) network / straight line per stratum |
| `results/reading_physics_vs_twin.csv` | (d) the label-free loss against supervision |
| `results/reading_best_architecture.csv` | (e) which architecture wins per column |
| `results/reading_direction.csv` | (f) forward against backward |
| `results/reading_p95.csv` | (g) the tails |
| `results/reading_cost.csv` | (h) error against parameters and multiply-adds |

    PYTHONNOUSERSITE=1 python tables.py
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import argparse
import csv
import glob
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
SCHEME = os.path.join(HERE, "..", "Exact_scheme_table", "results")

STRATA = ("0.05-0.2 mm", "0.5-2 mm", "5-20 mm", "50-200 mm", "500-2000 mm",
          "full crossing")
MODES = ("physics", "data")
MODE_LABEL = {"physics": "physics", "data": "twin"}
WIDTHS = (32, 64, 128, 256)
DEPTHS = (2, 4, 8)
QS = tuple(range(2, 21, 2))
SEEDS = (0, 1, 2)

# ../Fine_reference C1.4: the worst case over 200 momentum-stratified whole
# crossings, of both the step-halving and the closure measurements. It is a
# floor on a 5.2 m crossing; over a 0.1 mm step the march takes one step
# instead of ~52,000 and the floor is orders of magnitude lower.
REFERENCE_FLOOR_UM = 5e-5


# ------------------------------------------------------------ the ingredients --
def read_long(path):
    """`results/error_vs_dz_q.csv` as a list of dicts, typed."""
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            for k in ("width", "depth", "q", "seed", "n"):
                r[k] = int(r[k])
            for k in ("median", "p95", "slope", "rho", "straight_line"):
                r[k] = float(r[k]) if r[k] not in ("", "None") else np.nan
            r["converged"] = str(r["converged"]).lower() in ("true", "1")
            out.append(r)
    return out


def read_scheme():
    """(q, stratum) -> the exact scheme's median in um, both directions pooled."""
    out = {}
    with open(os.path.join(SCHEME, "scheme_table.csv")) as f:
        for r in csv.DictReader(f):
            if r["direction"] != "both":
                continue
            out[(int(r["q"]), r["stratum"])] = float(r["endpoint_med_um"])
    return out


def read_straight_line():
    """stratum -> the straight line's median, p95 and median |dz|, pooled."""
    out = {}
    with open(os.path.join(SCHEME, "straight_line_table.csv")) as f:
        for r in csv.DictReader(f):
            if r["direction"] != "both":
                continue
            out[r["stratum"]] = {
                "median_um": float(r["endpoint_med_um"]),
                "p95_um": float(r["endpoint_p95_um"]),
                "abs_dz_mm": float(r["median_abs_dz_mm"]),
            }
    return out


def dz_medians():
    """stratum -> median |dz| in mm, from C3.1's own check."""
    with open(os.path.join(RESULTS, "scale_check.json")) as f:
        check = json.load(f)
    return {k: v["abs_dz_mm"]["median"] for k, v in check["per_stratum"].items()}


def read_parameters():
    """(width, depth, q) -> the parameter count the run itself reported."""
    out = {}
    with open(os.path.join(RESULTS, "summary.csv")) as f:
        for r in csv.DictReader(f):
            out[(int(r["width"]), int(r["depth"]), int(r["q"]))] = \
                int(r["n_parameters"])
    return out


def multiply_adds(width, depth, q, n_in=7):
    """One forward pass of the MLP, in multiply-accumulate operations.

    The network is `_shared/model.OneStepNetwork`: `depth` hidden layers of
    `width` with tanh, then a linear layer to 4*(q+1) outputs, from
    5 state inputs plus the 2 normalised extras. Biases and the tanh are not
    counted - they are additions and transcendentals, not multiply-adds - and
    neither is the residual wrapper's field integral, which is 16 field lookups
    per sample whatever the architecture and is reported separately.
    """
    return (n_in * width + (depth - 1) * width * width
            + width * 4 * (q + 1))


def missing_tags():
    """The grid points with no record on disk, and the cells they belong to."""
    have = {os.path.basename(p)[:-5]
            for p in glob.glob(os.path.join(RESULTS, "w*_d*_q*_s*.json"))}
    miss = []
    for w in WIDTHS:
        for d in DEPTHS:
            for q in QS:
                for m in MODES:
                    for s in SEEDS:
                        tag = "w%d_d%d_q%02d_%s_s%d" % (w, d, q, m, s)
                        if tag not in have:
                            miss.append({"tag": tag, "width": w, "depth": d,
                                         "q": q, "mode": m, "seed": s})
    return miss


# ------------------------------------------------------------------- the cells --
def build_cells(rows, dz_med, params):
    """The tidy master: one row per (arch, mode, q, stratum, direction, split).

    A cell is the **median over the seeds that confirmed**; `n_seeds` says how
    many that was and `pending` is true when fewer than the three the grid
    plans for are present, so a thin cell can never pass for a full one.
    """
    bucket = {}
    for r in rows:
        key = (r["width"], r["depth"], r["mode"], r["q"], r["stratum"],
               r["direction"], r["split"])
        bucket.setdefault(key, []).append(r)
    cells = []
    for key, rs in sorted(bucket.items()):
        w, d, mode, q, stratum, direction, split = key
        ok = [r for r in rs if r["converged"]]
        use = ok if ok else rs
        med = np.array([r["median"] for r in use])
        cells.append({
            "width": w, "depth": d, "arch": "%dx%d" % (d, w), "mode": mode,
            "q": q, "stratum": stratum, "direction": direction, "split": split,
            "median_um": float(np.median(med)),
            "min_um": float(med.min()), "max_um": float(med.max()),
            "p95_um": float(np.median([r["p95"] for r in use])),
            "slope_mrad": float(np.median([r["slope"] for r in use])),
            "rho": float(np.median([r["rho"] for r in use])),
            "straight_um": float(np.median([r["straight_line"] for r in use])),
            "n_rows": int(np.median([r["n"] for r in use])),
            "n_seeds": len(use), "n_confirmed": len(ok),
            "pending": len(use) < len(SEEDS),
            "abs_dz_mm": dz_med.get(stratum, np.nan),
            "n_parameters": params.get((w, d, q), 0),
            "multiply_adds": multiply_adds(w, d, q),
        })
    return cells


def index(cells, split="test", direction="all"):
    """(arch, mode, q, stratum) -> cell, for one split and direction."""
    return {(c["arch"], c["mode"], c["q"], c["stratum"]): c for c in cells
            if c["split"] == split and c["direction"] == direction}


# ------------------------------------------------------- the display tables --
def fmt(v):
    if v is None or not np.isfinite(v):
        return ""
    return "%.3g" % v


def cell_text(c):
    if c is None:
        return "pending"
    body = "%s [%s-%s]" % (fmt(c["median_um"]), fmt(c["min_um"]),
                           fmt(c["max_um"]))
    if c["pending"]:
        body += " (%d/%d seeds)" % (c["n_seeds"], len(SEEDS))
    return body


def write_architecture_table(path, idx, arch, scheme, straight, dz_med):
    header = ["row", "arm"] + list(STRATA)
    out = [header]
    for q in QS:
        for mode in MODES:
            row = ["q=%d" % q, MODE_LABEL[mode]]
            for s in STRATA:
                row.append(cell_text(idx.get((arch, mode, q, s))))
            out.append(row)
    for q in QS:
        out.append(["exact scheme q=%d" % q, "exact scheme"]
                   + [fmt(scheme.get((q, s))) for s in STRATA])
    out.append(["straight line", "straight line"]
               + [fmt(straight[s]["median_um"]) for s in STRATA])
    out.append(["median |dz| [mm]", "note"]
               + [fmt(dz_med[s]) for s in STRATA])
    out.append(["fine reference floor [um]", "note"]
               + ["%.0e over a whole crossing; far lower per short step"
                  % REFERENCE_FLOOR_UM] + [""] * (len(STRATA) - 1))
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(out)


def write_direction_table(path, cells, arch, split="test"):
    idx = {(c["mode"], c["q"], c["stratum"], c["direction"]): c for c in cells
           if c["split"] == split and c["arch"] == arch}
    header = ["row", "arm", "direction"] + list(STRATA)
    out = [header]
    for q in QS:
        for mode in MODES:
            for d in ("forward", "backward"):
                out.append(["q=%d" % q, MODE_LABEL[mode], d]
                           + [cell_text(idx.get((mode, q, s, d)))
                              for s in STRATA])
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(out)


# ------------------------------------------------------------- the readings --
def reading_q_sensitivity(idx, scheme, straight):
    """(a) does q move the answer, and where does it stop moving.

    The control is the seed scatter inside a cell: if the spread over q is no
    larger than the spread over three seeds at one q, q is not doing anything
    that the optimiser's own noise does not.
    """
    rows = []
    archs = sorted({k[0] for k in idx})
    for arch in archs:
        for mode in MODES:
            for s in STRATA:
                cs = [idx.get((arch, mode, q, s)) for q in QS]
                cs = [c for c in cs if c is not None]
                if not cs:
                    continue
                med = np.array([c["median_um"] for c in cs])
                qs = np.array([c["q"] for c in cs])
                within = np.median([c["max_um"] / c["min_um"] for c in cs
                                    if c["min_um"] > 0])
                best = med.min()
                plateau = int(qs[np.flatnonzero(med <= 1.5 * best)[0]])
                rows.append({
                    "arch": arch, "mode": mode, "stratum": s,
                    "q2": med[qs == 2][0] if (qs == 2).any() else np.nan,
                    "q4": med[qs == 4][0] if (qs == 4).any() else np.nan,
                    "q_best": int(qs[med.argmin()]), "best_um": best,
                    "worst_um": med.max(),
                    "spread_over_q": med.max() / med.min(),
                    "seed_spread_within_cell": within,
                    "q_matters": bool(med.max() / med.min() > 2.0 * within),
                    "plateau_from_q": plateau,
                    "scheme_q2_um": scheme.get((2, s)),
                    "scheme_q8_um": scheme.get((8, s)),
                    "straight_um": straight[s]["median_um"],
                })
    return rows


def reading_dz_slope(idx):
    """(b) the log-log slope of the error against |dz| at fixed q.

    The residual scale is kappa |qop| I_B |dz| / 2 with I_B itself proportional
    to |dz|, so a network that lands a fixed fraction of one residual scale away
    from the truth has an error going as |dz|^2. The fit is over the five strata
    below the full crossing (the crossing is a different population: whole legs
    rather than a log-uniform draw) and, for comparison, over all six.
    """
    rows = []
    for arch in sorted({k[0] for k in idx}):
        for mode in MODES:
            for q in QS:
                cs = [idx.get((arch, mode, q, s)) for s in STRATA]
                if any(c is None for c in cs):
                    continue
                x = np.log10([c["abs_dz_mm"] for c in cs])
                y = np.log10([c["median_um"] for c in cs])
                sl5 = np.polyfit(x[:5], y[:5], 1)[0]
                sl6 = np.polyfit(x, y, 1)[0]
                rows.append({"arch": arch, "mode": mode, "q": q,
                             "slope_strata_0_4": sl5, "slope_all_six": sl6})
    return rows


def reading_ratio_straight(idx):
    """(c) the network against the straight line, per stratum."""
    rows = []
    for arch in sorted({k[0] for k in idx}):
        for mode in MODES:
            for s in STRATA:
                cs = [idx.get((arch, mode, q, s)) for q in QS]
                cs = [c for c in cs if c is not None]
                if not cs:
                    continue
                ratio = np.array([c["median_um"] / c["straight_um"]
                                  for c in cs])
                rows.append({"arch": arch, "mode": mode, "stratum": s,
                             "abs_dz_mm": cs[0]["abs_dz_mm"],
                             "straight_um": cs[0]["straight_um"],
                             "ratio_median_over_q": float(np.median(ratio)),
                             "ratio_best_q": float(ratio.min()),
                             "q_of_best": int(cs[int(ratio.argmin())]["q"]),
                             "best_um": float(min(c["median_um"] for c in cs))})
    return rows


def reading_physics_vs_twin(idx):
    """(d) the label-free loss against supervision, cell by cell."""
    rows = []
    for arch in sorted({k[0] for k in idx}):
        for q in QS:
            for s in STRATA:
                p = idx.get((arch, "physics", q, s))
                t = idx.get((arch, "data", q, s))
                if p is None or t is None:
                    continue
                rows.append({"arch": arch, "q": q, "stratum": s,
                             "physics_um": p["median_um"],
                             "twin_um": t["median_um"],
                             "physics_over_twin": p["median_um"] / t["median_um"],
                             "pending": p["pending"] or t["pending"]})
    return rows


def reading_best_architecture(idx):
    """(e) which architecture wins each column, and what depth 8 costs."""
    rows = []
    for mode in MODES:
        for s in STRATA:
            cand = [(c["median_um"], c) for k, c in idx.items()
                    if k[1] == mode and k[3] == s]
            if not cand:
                continue
            cand.sort(key=lambda t: t[0])
            v, c = cand[0]
            # the best of each depth, at the same stratum and mode
            by_depth = {}
            for _, cc in cand:
                by_depth.setdefault(cc["depth"], cc["median_um"])
                by_depth[cc["depth"]] = min(by_depth[cc["depth"]],
                                            cc["median_um"])
            rows.append({
                "mode": mode, "stratum": s, "best_arch": c["arch"],
                "best_q": c["q"], "best_um": v,
                "best_n_parameters": c["n_parameters"],
                "best_depth2_um": by_depth.get(2), "best_depth4_um": by_depth.get(4),
                "best_depth8_um": by_depth.get(8),
                "depth8_over_best": by_depth.get(8, np.nan) / v,
                "median_over_all_cells_um": float(np.median([x for x, _ in cand])),
            })
    return rows


def reading_direction(cells, split="test"):
    """(f) forward against backward, the way C2 asked it of the exact scheme."""
    idx = {(c["arch"], c["mode"], c["q"], c["stratum"], c["direction"]): c
           for c in cells if c["split"] == split}
    rows = []
    for arch in sorted({k[0] for k in idx}):
        for mode in MODES:
            for q in QS:
                for s in STRATA:
                    f = idx.get((arch, mode, q, s, "forward"))
                    b = idx.get((arch, mode, q, s, "backward"))
                    if f is None or b is None:
                        continue
                    rows.append({"arch": arch, "mode": mode, "q": q,
                                 "stratum": s,
                                 "forward_um": f["median_um"],
                                 "backward_um": b["median_um"],
                                 "backward_over_forward":
                                     b["median_um"] / f["median_um"]})
    return rows


def reading_p95(idx, straight):
    """(g) the tails: p95 against the median, and against the straight line."""
    rows = []
    for arch in sorted({k[0] for k in idx}):
        for mode in MODES:
            for q in QS:
                for s in STRATA:
                    c = idx.get((arch, mode, q, s))
                    if c is None:
                        continue
                    rows.append({"arch": arch, "mode": mode, "q": q,
                                 "stratum": s, "median_um": c["median_um"],
                                 "p95_um": c["p95_um"],
                                 "p95_over_median": c["p95_um"] / c["median_um"],
                                 "straight_p95_um": straight[s]["p95_um"],
                                 "p95_over_straight_p95":
                                     c["p95_um"] / straight[s]["p95_um"]})
    return rows


def reading_cost(idx):
    """(h) error against parameters and multiply-adds, at every cell."""
    rows = []
    for (arch, mode, q, s), c in sorted(idx.items()):
        rows.append({"arch": arch, "mode": mode, "q": q, "stratum": s,
                     "n_parameters": c["n_parameters"],
                     "multiply_adds": c["multiply_adds"],
                     "median_um": c["median_um"],
                     "straight_um": c["straight_um"],
                     "gain_over_straight": c["straight_um"] / c["median_um"],
                     "gain_per_kparam": (c["straight_um"] / c["median_um"])
                                        / max(c["n_parameters"], 1) * 1e3})
    return rows


def write_rows(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# -------------------------------------------------------------------- main --
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", default="test", choices=("val", "test"))
    a = ap.parse_args(argv)

    rows = read_long(os.path.join(RESULTS, "error_vs_dz_q.csv"))
    scheme = read_scheme()
    straight = read_straight_line()
    dz_med = dz_medians()
    params = read_parameters()

    cells = build_cells(rows, dz_med, params)
    write_rows(os.path.join(RESULTS, "table_cells.csv"), cells)
    idx = index(cells, split=a.split, direction="all")

    archs = sorted({c["arch"] for c in cells},
                   key=lambda s: (int(s.split("x")[0]), int(s.split("x")[1])))
    for arch in archs:
        d, w = arch.split("x")
        write_architecture_table(
            os.path.join(RESULTS, "table_%sx%s.csv" % (d, w)),
            idx, arch, scheme, straight, dz_med)
        write_direction_table(
            os.path.join(RESULTS, "table_%sx%s_by_direction.csv" % (d, w)),
            cells, arch, split=a.split)

    miss = missing_tags()
    for m in miss:
        m["cell"] = "%dx%d %s q=%d" % (m["depth"], m["width"], m["mode"], m["q"])
        c = idx.get(("%dx%d" % (m["depth"], m["width"]), m["mode"], m["q"],
                     "full crossing"))
        m["seeds_present_in_cell"] = c["n_seeds"] if c else 0
    write_rows(os.path.join(RESULTS, "pending_cells.csv"), miss)

    write_rows(os.path.join(RESULTS, "reading_q_sensitivity.csv"),
               reading_q_sensitivity(idx, scheme, straight))
    write_rows(os.path.join(RESULTS, "reading_dz_slope.csv"),
               reading_dz_slope(idx))
    write_rows(os.path.join(RESULTS, "reading_ratio_straight.csv"),
               reading_ratio_straight(idx))
    write_rows(os.path.join(RESULTS, "reading_physics_vs_twin.csv"),
               reading_physics_vs_twin(idx))
    write_rows(os.path.join(RESULTS, "reading_best_architecture.csv"),
               reading_best_architecture(idx))
    write_rows(os.path.join(RESULTS, "reading_direction.csv"),
               reading_direction(cells, split=a.split))
    write_rows(os.path.join(RESULTS, "reading_p95.csv"),
               reading_p95(idx, straight))
    write_rows(os.path.join(RESULTS, "reading_cost.csv"), reading_cost(idx))

    print("%d cells -> results/table_cells.csv" % len(cells))
    print("%d architecture tables (+ by-direction)" % len(archs))
    print("%d grid points still missing a record" % len(miss))
    for m in miss:
        print("   pending: %s  (%d of %d seeds present in its cell)"
              % (m["tag"], m["seeds_present_in_cell"], len(SEEDS)))
    return cells


if __name__ == "__main__":
    main()
