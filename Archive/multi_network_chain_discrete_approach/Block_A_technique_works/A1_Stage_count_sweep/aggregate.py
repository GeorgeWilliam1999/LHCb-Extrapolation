#!/usr/bin/env python
"""Collect the 24 training runs into the two tables the analysis reads.

    PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python aggregate.py

Reads every `results/q<qq>_<mode>_s<seed>.json` written by the shared
`train.py`, plus two measurements of the exact scheme's own error: the leg-B
rows of `../../Block_0_first_pass/S1_Simple_first_pass/results/scheme_error_vs_q.csv` (a
momentum-stratified sample of 32 legs) and, on this experiment's own 2018 test
states, `results/scheme_ceiling_same_population_q<qq>.json` from
`measure_scheme_ceiling.py`. The second is the one to compare the networks
against; the first is carried so the two can be seen not to agree, and why.
It writes:

    results/summary.csv          one row per run, as recorded - nothing pooled
    results/error_vs_stages.csv  one row per q: the median over seeds and the
                                 seed spread for each loss, beside both
                                 measurements of the exact-scheme ceiling

Nothing is recomputed from the model weights here; the numbers are the ones
`train.py` scored and recorded, so the table cannot drift from the runs.

Two rules about unfinished work, both deliberate:

  * a run whose json is missing is simply absent from `summary.csv`, and the
    script says which ones are missing;
  * a run whose json says `converged = false` is written into `summary.csv`
    with that flag intact, and is EXCLUDED from the medians in
    `error_vs_stages.csv`. Pooling a run that is still descending with runs
    that have stalled would quietly report a floor that nobody measured. The
    per-q columns `n_seeds_used` and `n_seeds_unconverged` say what happened.

`--include-unconverged` overrides the exclusion, for looking at partial
results while the farm is still draining; it stamps the CSV so that a table
made that way cannot be mistaken for the final one.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
SCHEME_CSV = os.path.join(HERE, "..", "..", "Block_0_first_pass", "S1_Simple_first_pass", "results",
                          "scheme_error_vs_q.csv")

Q_VALUES = (2, 4, 8, 16)
MODES = ("physics", "data")
SEEDS = (0, 1, 2)
TAG_RE = re.compile(r"^q(\d+)_(physics|data)_s(\d+)$")

SUMMARY_FIELDS = [
    "q", "mode", "seed",
    "test_endpoint_med_um", "test_endpoint_p95_um", "test_stage_med_um",
    "test_slope_med_mrad", "test_rho_median",
    "restarts", "converged", "wall_s", "final_loss",
    "val_endpoint_med_um", "train_endpoint_med_um",
    "straight_med_um", "n_test", "tag",
]


# ------------------------------------------------------------ the runs ------
def load_runs():
    """Every run json in results/, newest-wins, keyed by (q, mode, seed)."""
    runs, bad = {}, []
    for path in sorted(glob.glob(os.path.join(RESULTS, "q*_s*.json"))):
        tag = os.path.splitext(os.path.basename(path))[0]
        m = TAG_RE.match(tag)
        if not m:
            continue
        try:
            rec = json.load(open(path))
        except (ValueError, OSError) as exc:        # a json still being written
            bad.append((tag, str(exc)))
            continue
        runs[(int(m.group(1)), m.group(2), int(m.group(3)))] = rec
    return runs, bad


def summary_row(key, rec):
    q, mode, seed = key
    t, v, tr = rec["test"], rec["val"], rec["train"]
    return {
        "q": q, "mode": mode, "seed": seed,
        "test_endpoint_med_um": t["endpoint_med_um"],
        "test_endpoint_p95_um": t["endpoint_p95_um"],
        "test_stage_med_um": t["stage_med_um"],
        "test_slope_med_mrad": t["slope_med_mrad"],
        "test_rho_median": t["rho_median"],
        "restarts": rec["restarts"],
        "converged": rec["converged"],
        "wall_s": rec["wall_s"],
        "final_loss": rec["final_loss"],
        "val_endpoint_med_um": v["endpoint_med_um"],
        "train_endpoint_med_um": tr["endpoint_med_um"],
        "straight_med_um": t["straight_med_um"],
        "n_test": t["n"],
        "tag": rec["tag"],
    }


# ---------------------------------------------------------- the ceiling -----
def scheme_ceiling(leg="B"):
    """The exact scheme's own endpoint error on the magnet crossing, in um.

    `../../Block_0_first_pass/S1_Simple_first_pass` solved the same Gauss-Legendre scheme directly with
    a root-finder, so this is the error of the equations the physics loss asks
    the network to satisfy - the floor under anything trained on them. Leg B is
    the magnet crossing, the leg this experiment's frozen leg is drawn from.
    The file records millimetres.
    """
    out = {}
    with open(SCHEME_CSV) as f:
        for row in csv.DictReader(f):
            if row["leg"] == leg:
                out[int(row["q"])] = {
                    "median_um": float(row["median_err_mm"]) * 1e3,
                    "worst_um": float(row["worst_err_mm"]) * 1e3,
                    "n": int(row["n"]),
                    "converged_frac": float(row["converged_frac"]),
                }
    return out


def scheme_ceiling_same_population():
    """The same scheme solved on THIS experiment's own test states.

    `measure_scheme_ceiling.py` writes one json per q. This is the like-for-like
    ceiling: same states, same fp64 RK4 reference, same max(|dx|, |dy|) measure
    as the networks are scored with.
    """
    out = {}
    for path in sorted(glob.glob(os.path.join(
            RESULTS, "scheme_ceiling_same_population_q*.json"))):
        rec = json.load(open(path))
        out[int(rec["q"])] = rec
    return out


# ------------------------------------------------------------- the tables ---
def median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def error_vs_stages(runs, ceiling, ceiling_same, include_unconverged=False):
    rows = []
    for q in Q_VALUES:
        row = {"q": q}
        for mode in MODES:
            got = [(s, runs[(q, mode, s)]) for s in SEEDS if (q, mode, s) in runs]
            used = [r for _, r in got
                    if include_unconverged or r["converged"]]
            end = [r["test"]["endpoint_med_um"] for r in used]
            stage = [r["test"]["stage_med_um"] for r in used]
            p95 = [r["test"]["endpoint_p95_um"] for r in used]
            row["%s_endpoint_med_um" % mode] = median(end) if end else ""
            row["%s_endpoint_min_um" % mode] = min(end) if end else ""
            row["%s_endpoint_max_um" % mode] = max(end) if end else ""
            row["%s_stage_med_um" % mode] = median(stage) if stage else ""
            row["%s_p95_med_um" % mode] = median(p95) if p95 else ""
            row["%s_n_seeds_used" % mode] = len(used)
            row["%s_n_seeds_unconverged" % mode] = sum(
                1 for _, r in got if not r["converged"])
            row["%s_n_seeds_missing" % mode] = len(SEEDS) - len(got)
        c = ceiling.get(q, {})
        row["scheme_stratified_endpoint_med_um"] = c.get("median_um", "")
        row["scheme_stratified_endpoint_worst_um"] = c.get("worst_um", "")
        cs = ceiling_same.get(q, {})
        row["scheme_endpoint_med_um"] = cs.get("endpoint_med_um", "")
        row["scheme_endpoint_p95_um"] = cs.get("endpoint_p95_um", "")
        row["scheme_stage_med_um"] = cs.get("stage_med_um", "")
        row["scheme_n"] = cs.get("n", "")
        row["scheme_converged_frac"] = cs.get("converged_frac", "")
        # the do-nothing reference: the straight line through the same leg,
        # identical for every run, so any one of them carries it
        straight = [r["test"]["straight_med_um"] for k, r in runs.items()
                    if k[0] == q]
        row["straight_med_um"] = straight[0] if straight else ""
        rows.append(row)
    return rows


def write_csv(path, rows, fields, banner=None):
    with open(path, "w", newline="") as f:
        if banner:
            f.write("# %s\n" % banner)
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("wrote %s (%d rows)" % (path, len(rows)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--include-unconverged", action="store_true",
                    help="pool runs that have not converged (partial results "
                         "only; the CSV is stamped)")
    a = ap.parse_args()

    runs, bad = load_runs()
    missing = [(q, m, s) for q in Q_VALUES for m in MODES for s in SEEDS
               if (q, m, s) not in runs]
    unconv = sorted(k for k, r in runs.items() if not r["converged"])

    print("runs found: %d of %d" % (len(runs), len(Q_VALUES) * len(MODES) * len(SEEDS)))
    if bad:
        print("unreadable (probably still being written): %s"
              % ", ".join(t for t, _ in bad))
    if missing:
        print("still missing: " + ", ".join("q%d %s s%d" % k for k in missing))
    if unconv:
        print("NOT CONVERGED: " + ", ".join("q%d %s s%d" % k for k in unconv))

    rows = [summary_row(k, runs[k]) for k in sorted(runs)]
    write_csv(os.path.join(RESULTS, "summary.csv"), rows, SUMMARY_FIELDS)

    ceiling = scheme_ceiling("B")
    ceiling_same = scheme_ceiling_same_population()
    if not ceiling_same:
        print("no same-population ceiling yet: run measure_scheme_ceiling.py")
    evs = error_vs_stages(runs, ceiling, ceiling_same, a.include_unconverged)
    fields = ["q"]
    for mode in MODES:
        fields += ["%s_endpoint_med_um" % mode, "%s_endpoint_min_um" % mode,
                   "%s_endpoint_max_um" % mode, "%s_stage_med_um" % mode,
                   "%s_p95_med_um" % mode, "%s_n_seeds_used" % mode,
                   "%s_n_seeds_unconverged" % mode, "%s_n_seeds_missing" % mode]
    fields += ["scheme_endpoint_med_um", "scheme_endpoint_p95_um",
               "scheme_stage_med_um", "scheme_n", "scheme_converged_frac",
               "scheme_stratified_endpoint_med_um",
               "scheme_stratified_endpoint_worst_um",
               "straight_med_um"]
    banner = ("medians over converged seeds only; unconverged runs excluded"
              if not a.include_unconverged else
              "PARTIAL: unconverged runs POOLED IN (--include-unconverged)")
    write_csv(os.path.join(RESULTS, "error_vs_stages.csv"), evs, fields, banner)

    # a small readable echo, so a farm-side run says something useful
    print()
    print("%-4s %-28s %-28s %12s %12s"
          % ("q", "physics med [min-max] um", "data twin med [min-max] um",
             "ceiling um", "(stratified)"))
    for r in evs:
        def fmt(mode):
            if r["%s_endpoint_med_um" % mode] == "":
                return "-"
            return "%.0f [%.0f-%.0f] (%d seeds)" % (
                r["%s_endpoint_med_um" % mode], r["%s_endpoint_min_um" % mode],
                r["%s_endpoint_max_um" % mode], r["%s_n_seeds_used" % mode])
        ceil = r["scheme_endpoint_med_um"]
        strat = r["scheme_stratified_endpoint_med_um"]
        print("%-4d %-28s %-28s %12s %12s"
              % (r["q"], fmt("physics"), fmt("data"),
                 ("%.0f" % ceil) if ceil != "" else "-",
                 ("%.0f" % strat) if strat != "" else "-"))


if __name__ == "__main__":
    main()
