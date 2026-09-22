#!/usr/bin/env python
"""G1 - the farm job list for Block G: the moved momentum window at three step counts.

Block F's `F1_Training/make_jobs.py` with the momentum window added to every
job line and to the run folder. The grid is the same three settings Block E's
and Block F's analyses rest on, so each has a direct counterpart:

    N = 256 q = 16    dz = 20.2 mm
    N = 128 q = 8     dz = 40.5 mm
    N = 64  q = 2     dz = 80.9 mm    the case-study setting

The window goes into the run folder as well as the job line
(`--out results/p03-08` -> `results/p03-08/full/N064_q02/`), so a second window
can never be written into a first window's checkpoint, and `resubmit.py` can
tell the two apart in the queue.

Every job is resumable (train_windowed.py checkpoints after every restart), so
a job that is evicted or runs past its slot is sent back with `resubmit.py`.
The slowest goes first.

    python make_jobs.py --p-lo 3 --p-hi 8              # the three runs, window 3-8 GeV
    python make_jobs.py --p-lo 3 --p-hi 8 --settings 64:2
    cd .. && condor_submit condor/jobs.sub
"""
from __future__ import annotations

import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))       # .../G1_Training/condor
ROOT = os.path.dirname(HERE)                            # .../G1_Training
SETTINGS = ((256, 16), (128, 8), (64, 2))
P_LO, P_HI = 3.0, 8.0                                   # the anchor window (George 2026-09-21)

SUB = """universe                = vanilla
executable              = {wrapper}
arguments               = $(args)

output                  = condor/logs/$(Cluster).$(Process).out
error                   = condor/logs/$(Cluster).$(Process).err
log                     = condor/logs/$(Cluster).log

request_cpus            = 1
request_memory          = {memory}
should_transfer_files   = NO
getenv                  = False
+UseOS                  = "el9"
+JobCategory            = "{category}"

queue args from {jobs}
"""


def edge(p):
    """One side of the window as it appears in a folder name: 3 -> `03`, 7.5 -> `07p50`."""
    return "%02d" % int(round(p)) if float(p).is_integer() else ("%05.2f" % p).replace(".", "p")


def window_tag(p_lo, p_hi):
    """The folder name for a window: 3-8 GeV -> `p03-08`."""
    return "p%s-%s" % (edge(p_lo), edge(p_hi))


def out_for(p_lo, p_hi):
    return os.path.join("results", window_tag(p_lo, p_hi))


def parse_settings(text):
    if not text:
        return SETTINGS
    out = []
    for part in text.split(","):
        N, q = part.split(":")
        out.append((int(N), int(q)))
    return tuple(out)


def job_lines(p_lo=P_LO, p_hi=P_HI, weighting="full", settings=SETTINGS, round_restarts=25,
              out_dir=None, round_cap=40, outer_cap=1000):
    out_dir = out_for(p_lo, p_hi) if out_dir is None else out_dir
    return ["--N %d --q %d --weighting %s --p-lo %g --p-hi %g --out %s --round-restarts %d "
            "--round-cap %d --outer-cap %d"
            % (N, q, weighting, p_lo, p_hi, out_dir, round_restarts, round_cap, outer_cap)
            for N, q in settings]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--p-lo", type=float, default=P_LO, help="GeV, bottom of the window")
    ap.add_argument("--p-hi", type=float, default=P_HI, help="GeV, top of the window")
    ap.add_argument("--weighting", default="full")
    ap.add_argument("--settings", default=None, help='e.g. "64:2,128:8"; default all three')
    ap.add_argument("--round-restarts", type=int, default=25)
    ap.add_argument("--round-cap", type=int, default=40)
    ap.add_argument("--outer-cap", type=int, default=1000)
    ap.add_argument("--memory-mb", type=int, default=8192)
    ap.add_argument("--category", default="medium")
    ap.add_argument("--out", default=None,
                    help="run root (default: results/p<lo>-<hi>, the window in the path)")
    ap.add_argument("--jobs", default="condor/jobs.txt")
    ap.add_argument("--sub", default="condor/jobs.sub")
    a = ap.parse_args(argv)
    if not 0 < a.p_lo < a.p_hi:
        raise SystemExit("the window needs 0 < --p-lo < --p-hi (got %g and %g)" % (a.p_lo, a.p_hi))
    out_dir = a.out or out_for(a.p_lo, a.p_hi)
    lines = job_lines(a.p_lo, a.p_hi, a.weighting, parse_settings(a.settings), a.round_restarts,
                      out_dir=out_dir, round_cap=a.round_cap, outer_cap=a.outer_cap)
    with open(os.path.join(ROOT, a.jobs), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(ROOT, a.sub), "w") as f:
        f.write(SUB.format(wrapper=os.path.join(HERE, "wrapper.sh"),
                           memory=a.memory_mb, category=a.category, jobs=a.jobs))
    print("%d jobs, loss window %g-%g GeV, runs under %s/"
          % (len(lines), a.p_lo, a.p_hi, out_dir))
    print("-> %s; submit with: cd %s && condor_submit %s" % (a.jobs, ROOT, a.sub))
    for line in lines:
        print("   " + line)
    return lines


if __name__ == "__main__":
    main()
