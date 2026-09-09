#!/usr/bin/env python
"""C3.5 - turn the grid's run records into the two tables step C4 reads.

Every run of the grid writes `results/<tag>.json`, and `train_grid.py` has
already put the per-stratum and per-direction scores inside it, so this script
reads json only - it never loads a checkpoint and never re-scores anything.
That matters: it means the tables can be built while the grid is still
draining, on whatever has landed so far, and the numbers cannot drift from the
ones the run itself reported.

Two files come out.

`results/summary.csv` - one row per run, the whole-split view:

    tag, width, depth, q, mode, seed, n_train, n_parameters, converged,
    restarts, final_loss, wall_s, and for val and test the endpoint median and
    p95 in um, the stage median, the endpoint slope median in mrad, rho and
    the straight-line median on the same rows.

`results/error_vs_dz_q.csv` - the long table, one row per
(run, split, stratum, direction):

    width, depth, q, mode, seed, split, stratum, direction, converged,
    median, p95, slope, rho, straight_line, n, tag

with `stratum` the name (`all` for the whole split) and `direction` one of
`all`, `forward` (dz > 0), `backward` (dz < 0). `median` and `p95` are the
endpoint position error in um, `slope` the endpoint slope error in mrad, `rho`
the median of the agreed scalar relative error, and `straight_line` the
straight-line endpoint median on exactly those rows - the baseline the cell has
to be read against. Runs that did not confirm are kept, with `converged` false,
so C4 can decide what to do with them rather than being handed a filtered set.

    PYTHONNOUSERSITE=1 python aggregate_grid.py
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import argparse
import csv
import glob
import json
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
TAG_RE = re.compile(r"^w(\d+)_d(\d+)_q(\d+)_(physics|data)_s(\d+)$")

SUMMARY_FIELDS = [
    "tag", "width", "depth", "q", "mode", "seed", "n_train", "n_parameters",
    "converged", "restarts", "final_loss", "wall_s",
    "val_median_um", "val_p95_um", "val_stage_um", "val_slope_mrad",
    "val_rho", "val_straight_um",
    "test_median_um", "test_p95_um", "test_stage_um", "test_slope_mrad",
    "test_rho", "test_straight_um",
]
LONG_FIELDS = [
    "width", "depth", "q", "mode", "seed", "split", "stratum", "direction",
    "converged", "median", "p95", "slope", "rho", "straight_line", "n", "tag",
]


def parse_tag(tag):
    m = TAG_RE.match(tag)
    if not m:
        return None
    return {"width": int(m.group(1)), "depth": int(m.group(2)),
            "q": int(m.group(3)), "mode": m.group(4), "seed": int(m.group(5))}


def load_records(pattern):
    out = []
    for path in sorted(glob.glob(pattern)):
        tag = os.path.splitext(os.path.basename(path))[0]
        meta = parse_tag(tag)
        if meta is None:
            continue
        try:
            with open(path) as f:
                rec = json.load(f)
        except (ValueError, OSError):
            continue
        if "test" not in rec:
            continue
        out.append((tag, meta, rec))
    return out


def summary_row(tag, meta, rec):
    row = {"tag": tag, **meta,
           "n_train": rec.get("n_train"),
           "n_parameters": rec.get("n_parameters"),
           "converged": rec.get("converged"),
           "restarts": rec.get("restarts"),
           "final_loss": rec.get("final_loss"),
           "wall_s": rec.get("wall_s")}
    for split in ("val", "test"):
        s = rec.get(split, {})
        row["%s_median_um" % split] = s.get("endpoint_med_um")
        row["%s_p95_um" % split] = s.get("endpoint_p95_um")
        row["%s_stage_um" % split] = s.get("stage_med_um")
        row["%s_slope_mrad" % split] = s.get("slope_med_mrad")
        row["%s_rho" % split] = s.get("rho_median")
        row["%s_straight_um" % split] = s.get("straight_med_um")
    return row


def long_rows(tag, meta, rec):
    rows = []
    for r in rec.get("by_stratum", []):
        rows.append({
            **meta,
            "split": r["split"],
            "stratum": r["stratum_name"],
            "direction": r["direction"],
            "converged": rec.get("converged"),
            "median": r["endpoint_med_um"],
            "p95": r["endpoint_p95_um"],
            "slope": r["slope_med_mrad"],
            "rho": r["rho_median"],
            "straight_line": r["straight_med_um"],
            "n": r["n"],
            "tag": tag,
        })
    return rows


def write_csv(path, fields, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=RESULTS)
    ap.add_argument("--glob", default="w*_d*_q*_s*.json")
    a = ap.parse_args(argv)

    recs = load_records(os.path.join(a.results, a.glob))
    if not recs:
        print("no run records matched %s" % os.path.join(a.results, a.glob))
        return
    summary = [summary_row(*r) for r in recs]
    long = [row for r in recs for row in long_rows(*r)]
    write_csv(os.path.join(a.results, "summary.csv"), SUMMARY_FIELDS, summary)
    write_csv(os.path.join(a.results, "error_vs_dz_q.csv"), LONG_FIELDS, long)

    done = sum(1 for r in summary if r["converged"])
    print("%d run records -> results/summary.csv (%d confirmed, %d not) and "
          "results/error_vs_dz_q.csv (%d rows)"
          % (len(summary), done, len(summary) - done, len(long)))
    arch = sorted({(r["width"], r["depth"]) for r in summary})
    qs = sorted({r["q"] for r in summary})
    print("architectures present: %s" % ", ".join("%dx%d" % (d, w)
                                                  for w, d in arch))
    print("q present: %s" % ", ".join(str(q) for q in qs))
    return summary, long


if __name__ == "__main__":
    main()
