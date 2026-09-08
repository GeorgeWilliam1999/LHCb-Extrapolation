#!/usr/bin/env python
"""C6.1 / C6.2 - the per-network records become the three long tables.

`chain_one.py` writes one json per network into `results/records/`. This script
reads json only - it never loads a checkpoint and never re-scores anything - so
it can be run while the cluster is still draining and its numbers cannot drift
from the ones each job reported.

| output | one row per |
|---|---|
| `results/single_step_components.csv` | (network, stratum, direction, component) on the v3 test split, single step |
| `results/chained_components.csv` | (network, column, direction, component, reference) at the far plane |
| `results/chain_growth.csv` | (network, column, direction, checkpoint) along the chain |
| `results/reference_rows.csv` | the three things every table is read against |
| `results/landed.csv` | which of the 720 records are on disk |

**`chain_growth.csv` carries `max_xy` only.** Fifty checkpoints x five
components x three directions x six columns x 720 networks is 3.2 million rows;
the file the growth figures need is the max(|dx|, |dy|) one, and the four
signed components at every checkpoint stay in `results/records/<tag>.json` and
in `results/chains/<tag>.npz`.

The three reference rows, computed here once from `results/chain_tracks.npz`:

* **the straight line**, chained the same way. Chaining it changes nothing -
  each sub-step keeps tx and ty and adds tx*step to x, and the sub-steps sum to
  dz - so the chained straight line *is* the single straight step, and one row
  serves every column. It is the null extrapolator: what you get by ignoring
  the magnet.
* **the material floor**: the particle's real far-plane hit against the fine
  reference from the same start state, on exactly these tracks. No field-only
  method of any kind can predict it (`../MC_hit_comparison`).
* **the reference floor**, 5e-5 um over a crossing (`../Fine_reference` C1.4):
  the level below which the truth itself is not a number.

    PYTHONNOUSERSITE=1 python aggregate.py
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import argparse
import csv
import glob
import json

import numpy as np

import use_shared                                        # noqa: F401
import chain as C

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
WIDTHS = (32, 64, 128, 256)
DEPTHS = (2, 4, 8)
QS = tuple(range(2, 21, 2))
MODES = ("physics", "data")
SEEDS = (0, 1, 2)
GROWTH_COMPONENT = "max_xy"
GROWTH_DIRECTIONS = ("forward", "backward")


def all_tags():
    return ["w%d_d%d_q%02d_%s_s%d" % (w, d, q, m, s)
            for w in WIDTHS for d in DEPTHS for q in QS
            for m in MODES for s in SEEDS]


def reference_rows(tracks_npz, out):
    """The straight line, the material floor and the reference floor."""
    t = np.load(tracks_npz)
    S0, dz = np.asarray(t["S0"]), np.asarray(t["dz"])
    REF, HIT, DIR = np.asarray(t["REF"]), np.asarray(t["HIT"]), \
        np.asarray(t["DIRECTION"])
    rows = []
    for name, n_per in (("0.1 mm", 250),) + tuple(
            (c[0], c[2]) for c in C.COLUMNS if c[0] != "0.1 mm"):
        idx = C.column_tracks(DIR, n_per)
        straight = C.straight_endpoint(S0[idx], dz[idx])
        e_fine = C.component_errors(straight, REF[idx])
        e_hit = C.component_errors(straight, HIT[idx])
        e_floor = C.component_errors(HIT[idx], REF[idx])
        d = DIR[idx]
        for dname, dval in (("all", None), ("forward", 1), ("backward", -1)):
            m = np.ones(len(idx), bool) if dval is None else (d == dval)
            for predictor, e, ref in (("straight line", e_fine, "fine"),
                                      ("straight line", e_hit, "hit"),
                                      ("material floor", e_floor, "fine")):
                st = C.component_stats(e, m)
                for comp in C.COMPONENTS:
                    rows.append(dict(predictor=predictor, column=name,
                                     direction=dname, component=comp,
                                     reference=ref, unit=C.UNITS[comp],
                                     **st[comp]))
            for comp in C.COMPONENTS:
                rows.append(dict(predictor="reference floor", column=name,
                                 direction=dname, component=comp,
                                 reference="fine", unit=C.UNITS[comp],
                                 median_abs=C.REFERENCE_FLOOR_UM
                                 if comp in ("x", "y", "max_xy") else
                                 float("nan"),
                                 p95_abs=float("nan"), mean=float("nan"),
                                 rms=float("nan"), n=int(m.sum())))
    write_csv(out, rows, ["predictor", "column", "direction", "component",
                          "reference", "unit", "median_abs", "p95_abs",
                          "mean", "rms", "n"])
    return rows


def growth_exponent(fractions, medians, lo=0.1, hi=1.0):
    """The power law the error follows along the chain, or None.

    log10(median) fitted against log10(fraction of the crossing walked), over
    the checkpoints from a tenth of the crossing on. The exponent is the
    reading: independent per-step errors adding in quadrature give **0.5**, a
    bias that repeats identically at every step accumulates coherently and
    gives **1**, and a coherent error in the *slope* integrates into a position
    error going as the square of the distance, which gives **2**.
    """
    x, y = [], []
    for fr, v in zip(fractions, medians):
        if lo <= fr <= hi and v > 0 and np.isfinite(v) and np.isfinite(fr):
            x.append(np.log10(fr))
            y.append(np.log10(v))
    if len(x) < 5 or max(x) - min(x) < 1e-9:
        return None
    slope, _ = np.polyfit(np.array(x), np.array(y), 1)
    return float(slope), len(x)


def write_csv(path, rows, cols):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([("%.6g" % r[c]) if isinstance(r[c], float) else r[c]
                        for c in cols])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=RESULTS)
    a = ap.parse_args(argv)
    rec_dir = os.path.join(a.results, "records")

    single, chained, growth, landed, exps = [], [], [], [], []
    have = set()
    for path in sorted(glob.glob(os.path.join(rec_dir, "*.json"))):
        with open(path) as f:
            rec = json.load(f)
        tag = rec["tag"]
        have.add(tag)
        who = {k: rec[k] for k in ("tag", "width", "depth", "q", "mode", "seed")}
        landed.append(dict(wall_s=rec["wall_s"], host=rec.get("host", ""),
                           created=rec.get("created", ""), **who))
        for r in rec["single_step"]:
            single.append(dict(**who, **r))
        for r in rec["chained"]:
            chained.append(dict(**who, **{k: v for k, v in r.items()
                                          if k != "wall_s"}))
        for g in rec["growth"]:
            if g["direction"] not in GROWTH_DIRECTIONS:
                continue
            c = g["components"][GROWTH_COMPONENT]
            for k in range(len(g["nominal_fraction"])):
                growth.append(dict(
                    tag=tag, column=g["column"], direction=g["direction"],
                    checkpoint_z_fraction=g["z_fraction"][k],
                    component=GROWTH_COMPONENT,
                    median_abs=c["median_abs"][k], p95_abs=c["p95_abs"][k]))
            for comp, cc in g["components"].items():
                e = growth_exponent(g["z_fraction"], cc["median_abs"])
                if e is None:
                    continue
                exps.append(dict(**who, column=g["column"],
                                 direction=g["direction"], component=comp,
                                 unit=cc["unit"], exponent=e[0],
                                 n_checkpoints=e[1]))

    write_csv(os.path.join(a.results, "single_step_components.csv"), single,
              ["tag", "width", "depth", "q", "mode", "seed", "stratum",
               "direction", "component", "unit", "median_abs", "p95_abs",
               "mean", "rms", "n"])
    write_csv(os.path.join(a.results, "chained_components.csv"), chained,
              ["tag", "width", "depth", "q", "mode", "seed", "column",
               "direction", "component", "reference", "unit", "median_abs",
               "p95_abs", "mean", "rms", "n", "n_steps"])
    write_csv(os.path.join(a.results, "chain_growth.csv"), growth,
              ["tag", "column", "direction", "checkpoint_z_fraction",
               "component", "median_abs", "p95_abs"])
    write_csv(os.path.join(a.results, "growth_exponent.csv"), exps,
              ["tag", "width", "depth", "q", "mode", "seed", "column",
               "direction", "component", "unit", "exponent", "n_checkpoints"])
    write_csv(os.path.join(a.results, "landed.csv"), landed,
              ["tag", "width", "depth", "q", "mode", "seed", "wall_s", "host",
               "created"])
    reference_rows(os.path.join(a.results, "chain_tracks.npz"),
                   os.path.join(a.results, "reference_rows.csv"))

    tags = all_tags()
    missing = [t for t in tags if t not in have]
    with open(os.path.join(a.results, "pending.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag"])
        for t in missing:
            w.writerow([t])
    print("%d / %d records landed; %d pending -> results/pending.csv"
          % (len(have), len(tags), len(missing)))
    print("single_step_components.csv %d rows, chained_components.csv %d rows, "
          "chain_growth.csv %d rows, growth_exponent.csv %d rows"
          % (len(single), len(chained), len(growth), len(exps)))


if __name__ == "__main__":
    main()
