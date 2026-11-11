#!/usr/bin/env python
"""G2 - has each run stopped improving, and what is its headline number?

The rule is the one fixed before the runs, imported from Block F's
`compare_to_blockE.py` so there is only one copy of it: a run has plateaued AS
OF THE LATEST ROUND when the median validation error of the last ten rounds is
no more than 5 percent below the median of the ten before, at each of the last
three rounds (`plateaued_now`). `plateau` reports the first round the rule ever
held, which is a record rather than a verdict - on the earlier runs the rule
held early and the error then started falling again - so the verdict column is
`plateaued_now` and the first-hold round is printed beside it for information.

The lesson behind it (2026-09-18): the training loss went flat while the
validation error was still falling 5-24 percent per ten rounds in eleven runs
out of eleven. Convergence is never called from the loss.

The headline is the MEDIAN of the last ten rounds with its round-to-round
spread, not the final checkpoint: a single checkpoint wanders by 10-20 percent
and one number would be reading noise.

DUPLICATED ROWS. A run that two jobs have been training at once (it happens
when a resubmission is made while the first job is alive) has the same round
number written twice in `rounds.csv` and the same restart number twice in
`history.csv`, by two processes interleaved. Both files are de-duplicated here
BY NUMBER, keeping the last row written for each round (or restart) and then
sorting by number; what was dropped is printed and recorded in the CSV
(`dup_rounds`, `dup_restarts`). Any run with duplicates is a run whose history
after the first duplicate is interleaved, and is marked in the output.

Run:  PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python convergence.py [--runs DIR] [--out DIR]
Outputs: <out>/results/convergence_check.csv, <out>/figures/convergence_check.png
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"

import sys           # noqa: E402
# importing a script out of a read-only folder must not leave a .pyc behind in it
sys.dont_write_bytecode = True

import argparse          # noqa: E402
import csv               # noqa: E402
import importlib.util    # noqa: E402

import numpy as np       # noqa: E402

import run_discovery as rd   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
F2 = os.path.abspath(os.path.join(HERE, "..", "..", "Block_F_reweighted_loss", "F2_Analysis"))
DEFAULT_RUNS = os.path.join(HERE, "..", "G1_Training", "results", "p03-08", "full")
METRIC = "val_z1_pos_med_um"


def _f2(name):
    spec = importlib.util.spec_from_file_location("f2_" + name, os.path.join(F2, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_RULE = _f2("compare_to_blockE")
plateau = _RULE.plateau
plateaued_now = _RULE.plateaued_now
WINDOW = _RULE.WINDOW                 # 10 rounds
PLATEAU_TOL = _RULE.PLATEAU_TOL       # 5 percent
PLATEAU_HOLD = _RULE.PLATEAU_HOLD     # for 3 rounds running


def dedup(rows, key):
    """Rows de-duplicated by an integer key, last write kept, sorted by key.

    Returns (rows, the duplicated key values in order).
    """
    seen, dups = {}, []
    for r in rows:
        k = int(float(r[key]))
        if k in seen:
            dups.append(k)
        seen[k] = r
    return [seen[k] for k in sorted(seen)], sorted(set(dups))


def read_rounds(run):
    p = os.path.join(run, "rounds.csv")
    if not os.path.exists(p):
        return None, [], 0
    with open(p) as f:
        raw = list(csv.DictReader(f))
    rows, dups = dedup(raw, "round")
    return rows, dups, len(raw)


def read_history_dups(run):
    p = os.path.join(run, "history.csv")
    if not os.path.exists(p):
        return [], 0
    with open(p) as f:
        raw = list(csv.DictReader(f))
    _, dups = dedup(raw, "restart")
    return dups, len(raw)


def ten_round_medians(v, window=WINDOW):
    """The rolling median of each block of `window` rounds, at each round end."""
    return [(r, float(np.median(v[r - window:r]))) for r in range(window, len(v) + 1)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", default=DEFAULT_RUNS)
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--only", default=None, help="comma-separated run folders, e.g. N064_q02")
    a = ap.parse_args(argv)
    # a run being watched during training has rounds.csv long before it has
    # chain_states.npz, so the states are not required here
    nets = rd.require(rd.find_runs(a.runs, a.only, require_states=False), a.runs)
    print(rd.describe(nets, a.runs))
    print("\nthe rule: the median validation error (%s) of the last %d rounds no more than "
          "%.0f%% below the median of the %d before, at each of the last %d rounds"
          % (METRIC, WINDOW, 100 * PLATEAU_TOL, WINDOW, PLATEAU_HOLD))

    rows, series = [], {}
    for n in nets:
        rr, dups, n_raw = read_rounds(n["path"])
        if rr is None or len(rr) < 2:
            print("  %-40s no rounds.csv yet" % rd.short_label(n))
            continue
        hdups, h_raw = read_history_dups(n["path"])
        v = np.array([float(x[METRIC]) for x in rr])
        series[n["tag"]] = (n, v)
        last, prev = v[-WINDOW:], v[-2 * WINDOW:-WINDOW]
        rows.append(dict(
            N=n["N"], q=n["q"], dz_mm=round(n["dz_mm"], 1), rounds=len(v),
            restarts=n["restarts"], checkpoint=int(n["checkpoint"]),
            plateaued_now=int(plateaued_now(v)), first_plateau_round=plateau(v) or -1,
            headline_val_med_um=float(np.median(last)),
            headline_spread_pct=float(100 * (last.max() - last.min()) / 2 / np.median(last)),
            headline_min_um=float(last.min()), headline_max_um=float(last.max()),
            last10_over_prev10=float(np.median(last) / np.median(prev)) if len(prev) == WINDOW else float("nan"),
            last10_vs_prev10_pct=float(100 * (np.median(last) - np.median(prev)) / np.median(prev))
            if len(prev) == WINDOW else float("nan"),
            final_round_val_um=float(v[-1]),
            dup_rounds=" ".join(str(d) for d in dups), n_dup_rounds=len(dups), rounds_rows_read=n_raw,
            dup_restarts_from=(min(hdups) if hdups else -1), n_dup_restarts=len(hdups),
            history_rows_read=h_raw,
            interleaved=int(bool(dups or hdups))))
        if dups or hdups:
            print("\n  %s: two jobs have been writing this run." % rd.short_label(n))
            if dups:
                print("    rounds.csv: %d of %d rows were repeats of rounds %s; the last row written "
                      "for each round was kept" % (len(dups), n_raw, ", ".join(str(d) for d in dups)))
            if hdups:
                print("    history.csv: %d of %d rows repeated restart numbers, from %d on; treat this "
                      "run's history after that point as interleaved"
                      % (len(hdups), h_raw, min(hdups)))

    if not rows:
        raise SystemExit("no run has a rounds.csv with at least two rounds yet")

    res, figs = os.path.join(a.out, "results"), os.path.join(a.out, "figures")
    os.makedirs(res, exist_ok=True)
    os.makedirs(figs, exist_ok=True)
    with open(os.path.join(res, "convergence_check.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\nvalidation error at the first SciFi plane, per round [um]; the headline is the median of "
          "the last %d rounds" % WINDOW)
    print("%-32s %7s %9s %9s  %-24s %-22s" % (
        "network", "rounds", "flat now", "first hold", "headline (last %d rounds)" % WINDOW,
        "last %d over previous %d" % (WINDOW, WINDOW)))
    for r in rows:
        n = [x for x in nets if x["N"] == r["N"] and x["q"] == r["q"]][0]
        print("%-32s %7d %9s %9s  %8.1f +- %-4.0f%%        %6.3f  (%+5.1f%%)%s" % (
            rd.short_label(n).split(" (")[0], r["rounds"],
            "yes" if r["plateaued_now"] else "no",
            r["first_plateau_round"] if r["first_plateau_round"] > 0 else "-",
            r["headline_val_med_um"], r["headline_spread_pct"],
            r["last10_over_prev10"], r["last10_vs_prev10_pct"],
            "   [rows de-duplicated]" if r["interleaved"] else ""))
        if not r["plateaued_now"]:
            print("      still improving: it is a checkpoint, not a result")

    print("\nthe %d-round medians, round by round:" % WINDOW)
    for tag, (n, v) in series.items():
        med = ten_round_medians(v)
        print("  %-32s %s" % (rd.short_label(n).split(" (")[0],
                              "  ".join("r%d %.0f" % (r, m) for r, m in med[-8:])))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, len(series), figsize=(5.0 * len(series), 4.0), squeeze=False)
    for a_, (tag, (n, v)) in zip(ax[0], series.items()):
        a_.plot(np.arange(1, len(v) + 1), v, "-o", ms=3, lw=1.0, color="#d62728", label="each round")
        med = ten_round_medians(v)
        if med:
            a_.plot([r for r, _ in med], [m for _, m in med], "-", lw=2.0, color="#08306b",
                    label="median of the last %d rounds" % WINDOW)
        r = [x for x in rows if x["N"] == n["N"] and x["q"] == n["q"]][0]
        a_.set_title("%s\n%s" % (rd.short_label(n),
                                 "stopped falling" if r["plateaued_now"] else "still falling"), fontsize=9)
        a_.set_xlabel("training round", fontsize=8)
        a_.set_yscale("log")
        a_.grid(alpha=0.3, which="both")
        a_.legend(fontsize=7)
    ax[0, 0].set_ylabel("validation error at z1 [um]", fontsize=9)
    fig.suptitle("Validation error round by round, and the %d-round median the headline is read from\n%s"
                 % (WINDOW, rd.study_text(nets)), fontsize=11)
    fig.tight_layout()
    out_png = os.path.join(figs, "convergence_check.png")
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print("\nwrote %s, %s" % (os.path.join(res, "convergence_check.csv"), out_png))
    return rows


if __name__ == "__main__":
    main()
