#!/usr/bin/env python
"""F1 - the farm job list for Block F: the reweighted loss at three step counts.

The first grid is the SAME weighting at the three settings Block E's analysis
rests on, so each has a direct counterpart to compare with:

    N = 64  q = 2     the case-study setting (Block E: 124 um val, +-13%)
    N = 128 q = 8     (Block E: 162 um, +-9%)
    N = 256 q = 16    (Block E: 173 um, +-18%)

Every job is resumable (train_weighted.py checkpoints after every restart), so
a job that is evicted or runs past its slot is sent back with `resubmit.py`.
The slowest goes first.

    python make_jobs.py                        # the three `full` runs
    python make_jobs.py --weighting no_lever --settings 64:2   # one ablation
    condor_submit condor/jobs.sub
"""
from __future__ import annotations

import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SETTINGS = ((256, 16), (128, 8), (64, 2))

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


def parse_settings(text):
    if not text:
        return SETTINGS
    out = []
    for part in text.split(","):
        N, q = part.split(":")
        out.append((int(N), int(q)))
    return tuple(out)


def job_lines(weighting="full", settings=SETTINGS, round_restarts=25, out_dir="results",
              round_cap=40, outer_cap=1000):
    return ["--N %d --q %d --weighting %s --out %s --round-restarts %d --round-cap %d "
            "--outer-cap %d" % (N, q, weighting, out_dir, round_restarts, round_cap, outer_cap)
            for N, q in settings]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--weighting", default="full")
    ap.add_argument("--settings", default=None, help='e.g. "64:2,128:8"; default all three')
    ap.add_argument("--round-restarts", type=int, default=25)
    ap.add_argument("--round-cap", type=int, default=40)
    ap.add_argument("--outer-cap", type=int, default=1000)
    ap.add_argument("--memory-mb", type=int, default=8192)
    ap.add_argument("--category", default="medium")
    ap.add_argument("--jobs", default="condor/jobs.txt")
    ap.add_argument("--sub", default="condor/jobs.sub")
    a = ap.parse_args(argv)
    lines = job_lines(a.weighting, parse_settings(a.settings), a.round_restarts,
                      round_cap=a.round_cap, outer_cap=a.outer_cap)
    with open(os.path.join(HERE, a.jobs), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(HERE, a.sub), "w") as f:
        f.write(SUB.format(wrapper=os.path.join(HERE, "condor", "wrapper.sh"),
                           memory=a.memory_mb, category=a.category, jobs=a.jobs))
    print("%d jobs -> %s; submit with: cd %s && condor_submit %s" % (len(lines), a.jobs, HERE, a.sub))
    for line in lines:
        print("   " + line)
    return lines


if __name__ == "__main__":
    main()
