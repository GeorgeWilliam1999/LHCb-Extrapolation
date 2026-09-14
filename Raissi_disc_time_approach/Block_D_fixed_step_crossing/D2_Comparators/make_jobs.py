#!/usr/bin/env python
"""D2 - the job list for the exact-scheme chains: one farm job per (N, q).

    python make_jobs.py && condor_submit condor/jobs_exact.sub
"""
from __future__ import annotations

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
request_memory          = 2048
should_transfer_files   = NO
getenv                  = False
+UseOS                  = "el9"
+JobCategory            = "medium"

queue args from condor/jobs_exact.txt
"""


def main():
    lines = ["--N %d --q %d" % (N, q) for N in N_VALUES for q in QS]
    lines.sort(key=lambda s: -int(s.split()[1]) * 100 - int(s.split()[3]))
    os.makedirs(os.path.join(HERE, "condor", "logs"), exist_ok=True)
    with open(os.path.join(HERE, "condor", "jobs_exact.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(HERE, "condor", "jobs_exact.sub"), "w") as f:
        f.write(SUB.format(wrapper=os.path.join(HERE, "condor", "wrapper_exact.sh")))
    print("%d jobs; submit with: cd %s && condor_submit condor/jobs_exact.sub" % (len(lines), HERE))


if __name__ == "__main__":
    main()
