#!/usr/bin/env python
"""D1 - the 100-line job list: one farm job per (N, q) chain.

    N in {1, 4, 16, 64, 128}   x   q in {1, ..., 20}   = 100 chains,
    one job each, N networks trained in series inside the job.

`queue args from` does not skip comment lines, so the list holds argument
lines only. 4 GB is far more than a 2x128 network needs; it is the farm's
standard slot. The long chains (N = 128) are resumable, so a job that is
evicted or runs past the slot's time is simply resubmitted with
`resubmit_chain.py` and carries on from its last finished restart.

    python make_jobs.py && condor_submit condor/jobs_chain.sub
"""
from __future__ import annotations

import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))
N_VALUES = (1, 4, 16, 64, 128)
QS = tuple(range(1, 21))

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


def job_lines(out_dir="results", order="long-first"):
    lines = []
    for N in N_VALUES:
        for q in QS:
            lines.append("--N %d --q %d --out %s" % (N, q, out_dir))
    if order == "long-first":       # the N = 128 chains take longest: queue them first
        lines.sort(key=lambda s: -int(s.split()[1]))
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--memory-mb", type=int, default=4096)
    ap.add_argument("--category", default="medium")
    ap.add_argument("--jobs", default="condor/jobs_chain.txt")
    ap.add_argument("--sub", default="condor/jobs_chain.sub")
    a = ap.parse_args(argv)
    lines = job_lines()
    with open(os.path.join(HERE, a.jobs), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(HERE, a.sub), "w") as f:
        f.write(SUB.format(wrapper=os.path.join(HERE, "condor", "wrapper_chain.sh"),
                           memory=a.memory_mb, category=a.category, jobs=a.jobs))
    print("%d jobs -> %s; submit with: cd %s && condor_submit %s"
          % (len(lines), a.jobs, HERE, a.sub))
    return lines


if __name__ == "__main__":
    main()
