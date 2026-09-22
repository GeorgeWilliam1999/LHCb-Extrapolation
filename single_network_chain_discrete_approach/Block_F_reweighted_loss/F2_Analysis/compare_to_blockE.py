#!/usr/bin/env python
"""F2 - Block F against Block E, on one rule applied to both.

The rule is fixed before the runs so neither side is favoured. A run has
PLATEAUED once the median validation error of its last WINDOW rounds has stopped
falling - no more than PLATEAU_TOL below the median of the WINDOW before it, for
PLATEAU_HOLD rounds running (see `plateau`). Its headline is that median, with
the round-to-round spread beside it, because a single final checkpoint wanders
by 10-20 percent and a single number would be reading noise.

Applied to Block E's own logs the rule says NONE of its sixteen runs had
plateaued when they were stopped by hand on 2026-09-18: all eleven with enough
rounds were still improving, by 5 to 24 percent per ten rounds, and eleven
falls out of eleven has probability 0.0005 if the wandering were pure noise. So
Block E's numbers are where its training was stopped, not where it converges.

Every finished run is also scored on the test tracks the way Block E's analysis
scores them: the RADIAL distance at z1 between the network's endpoint and the
RK6 endpoint, overall and in the 10-50 GeV band that matters.

Run:     PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python compare_to_blockE.py
Outputs: results/headline.csv, figures/headline.png
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"

import argparse   # noqa: E402
import csv        # noqa: E402
import glob       # noqa: E402
import json       # noqa: E402

import numpy as np   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
E = os.path.join(HERE, "..", "..", "Block_E_single_network_chain")
F = os.path.join(HERE, "..", "F1_Training", "results")
TRACKS = os.path.join(E, "E0_Track_dataset", "results", "tracks.npz")
SPLIT = "test"
WINDOW = 10
PLATEAU_TOL = 0.05
PLATEAU_HOLD = 3
BAND = (10.0, 50.0)


def plateau(v, window=WINDOW, tol=PLATEAU_TOL, hold=PLATEAU_HOLD):
    """(the round it plateaued at, or None) for a series of per-round errors.

    The rule is ONE-SIDED and has to hold for `hold` consecutive rounds: the
    median of the last `window` rounds is no more than `tol` BELOW the median of
    the `window` before it. A two-sided "within 5 percent" rule was tried first
    and is useless here - the round-to-round error wanders by 8-23 percent, so
    ten-round medians rarely land within 5 percent of each other even when the
    run has long stopped improving. What we need to know is whether it is still
    getting better, not whether it is steady, and a single crossing can happen
    by chance, hence `hold`.
    """
    seen = 0
    for r in range(2 * window, len(v) + 1):
        a, b = np.median(v[r - window:r]), np.median(v[r - 2 * window:r - window])
        seen = seen + 1 if a >= (1.0 - tol) * b else 0
        if seen >= hold:
            return r
    return None


def plateaued_now(v, window=WINDOW, tol=PLATEAU_TOL, hold=PLATEAU_HOLD):
    """Has the error stopped falling AS OF THE LATEST ROUND?

    `plateau` reports the first round the rule held, which is a record, not a
    verdict: on Block E's extended runs the rule held early and then the error
    started falling again (N = 128, q = 8 held at round 28 and was then falling
    13 percent per ten rounds). The question for stopping a run is whether the
    condition holds at each of the last `hold` rounds.
    """
    if len(v) < 2 * window + hold - 1:
        return False
    for r in range(len(v) - hold + 1, len(v) + 1):
        a, b = np.median(v[r - window:r]), np.median(v[r - 2 * window:r - window])
        if a < (1.0 - tol) * b:
            return False
    return True


def read_run(run, label):
    rounds = os.path.join(run, "rounds.csv")
    if not os.path.exists(rounds):
        return None
    rows = list(csv.DictReader(open(rounds)))
    v = np.array([float(x["val_z1_pos_med_um"]) for x in rows])
    if len(v) < 2:
        return None
    last = v[-WINDOW:]
    prev = v[-2 * WINDOW:-WINDOW]
    r = dict(label=label, run=os.path.relpath(run, HERE), rounds=len(v),
             plateau_round=plateau(v) or -1,
             plateaued_now=bool(plateaued_now(v)),
             last10_vs_prev10_pct=(float(100 * (np.median(v[-WINDOW:]) - np.median(prev))
                                         / np.median(prev)) if len(prev) == WINDOW else float("nan")),
             val_med_um=float(np.median(last)), val_min_um=float(last.min()),
             val_max_um=float(last.max()),
             val_spread_pct=float(100 * (last.max() - last.min()) / 2 / np.median(last)))
    rec = os.path.join(run, "record.json")
    if os.path.exists(rec):
        j = json.load(open(rec))
        r.update(N=j["N"], q=j["q"], restarts=j["restarts"], final_loss=j["final_loss"],
                 weighting=j.get("weighting", "blockE"), train_wall_h=j["train_wall_s"] / 3600.0)
    return r


def score(run, D, r):
    """The radial test error at z1, overall and in the band."""
    st = os.path.join(run, "chain_states.npz")
    if not (os.path.exists(st) and "N" in r):
        return r
    n_max = int(D["n_max"])
    end = np.load(st)["%s_states" % SPLIT][:, -1]
    truth = np.asarray(D["%s_truth" % SPLIT])[:, n_max]
    P = np.asarray(D["%s_P" % SPLIT])
    rad = np.hypot(end[:, 0] - truth[:, 0], end[:, 1] - truth[:, 1]) * 1e3
    m = (P >= BAND[0]) & (P < BAND[1])
    r.update(test_radial_med_um=float(np.median(rad)),
             test_radial_p95_um=float(np.quantile(rad, 0.95)),
             test_radial_band_med_um=float(np.median(rad[m])),
             test_radial_soft_med_um=float(np.median(rad[P < 5])),
             n_test=int(len(rad)))
    return r


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", default=None, help="restrict to one N_q, e.g. N064_q02")
    a = ap.parse_args(argv)
    D = np.load(TRACKS)
    runs = [(p, "Block E") for p in sorted(glob.glob(
        os.path.join(E, "E1_Network_grid", "results", "N[0-9][0-9][0-9]_q[0-9][0-9]")))]
    runs += [(p, "Block F: " + os.path.basename(os.path.dirname(p)))
             for p in sorted(glob.glob(os.path.join(F, "*", "N[0-9][0-9][0-9]_q[0-9][0-9]")))]
    rows = []
    for path, label in runs:
        if a.only and a.only not in path:
            continue
        r = read_run(path, label)
        if r:
            rows.append(score(path, D, r))
    if not rows:
        raise SystemExit("no runs found")
    keys = sorted({k for r in rows for k in r})
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "headline.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    print("%-16s %5s %4s %7s %8s  %-22s  %-34s" % (
        "run", "N", "q", "rounds", "flat now", "validation, last 10 rounds",
        "test radial at z1 [um]"))
    print("%-16s %5s %4s %7s %8s  %-22s  %-34s" % (
        "", "", "", "", "", "median   spread   trend", "all      10-50 GeV   <5 GeV"))
    for r in sorted(rows, key=lambda x: (x.get("N", 0), x.get("q", 0), x["label"])):
        print("%-16s %5s %4s %7d %8s  %6.1f  +-%4.0f%%  %+5.0f%%  %8s %8s %8s" % (
            r["label"], r.get("N", "?"), r.get("q", "?"), r["rounds"],
            "yes" if r["plateaued_now"] else "no",
            r["val_med_um"], r["val_spread_pct"], r["last10_vs_prev10_pct"],
            "%.1f" % r["test_radial_med_um"] if "test_radial_med_um" in r else "-",
            "%.1f" % r["test_radial_band_med_um"] if "test_radial_band_med_um" in r else "-",
            "%.1f" % r["test_radial_soft_med_um"] if "test_radial_soft_med_um" in r else "-"))
    print("\nwrote results/headline.csv")
    return rows


if __name__ == "__main__":
    main()
