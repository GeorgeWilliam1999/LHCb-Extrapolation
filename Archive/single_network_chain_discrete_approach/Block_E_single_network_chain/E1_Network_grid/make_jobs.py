#!/usr/bin/env python
"""E1 - the 16-line farm job list: one job per (N, q) network.

    N in {2, 64, 128, 256}   x   q in {2, 4, 8, 16}   = 16 networks

Every job is resumable (train_network.py checkpoints after every restart), so a
job that is evicted or runs past its slot is sent back with `resubmit.py`.
The slowest models (q = 16 on the short steps) go first.

    python make_jobs.py --round-restarts 25      # writes the list; does NOT submit
    condor_submit condor/jobs.sub
"""
from __future__ import annotations

import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))
N_VALUES = (2, 64, 128, 256)
QS = (2, 4, 8, 16)

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


def job_lines(round_restarts=None, out_dir="results"):
    extra = "" if round_restarts is None else " --round-restarts %d" % round_restarts
    lines = ["--N %d --q %d --out %s%s" % (N, q, out_dir, extra) for N in N_VALUES for q in QS]
    lines.sort(key=lambda s: (-int(s.split()[3]), -int(s.split()[1])))
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--round-restarts", type=int, default=None)
    ap.add_argument("--memory-mb", type=int, default=8192)
    ap.add_argument("--category", default="medium")
    ap.add_argument("--jobs", default="condor/jobs.txt")
    ap.add_argument("--sub", default="condor/jobs.sub")
    a = ap.parse_args(argv)
    lines = job_lines(a.round_restarts)
    with open(os.path.join(HERE, a.jobs), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(HERE, a.sub), "w") as f:
        f.write(SUB.format(wrapper=os.path.join(HERE, "condor", "wrapper.sh"),
                           memory=a.memory_mb, category=a.category, jobs=a.jobs))
    print("%d jobs -> %s; submit with: cd %s && condor_submit %s" % (len(lines), a.jobs, HERE, a.sub))
    return lines


if __name__ == "__main__":
    main()
