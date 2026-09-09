#!/usr/bin/env python
"""C6.3b - the 720-line job list and its submit file, one job per network.

One job chains one trained network across the magnet at every step length and
scores its single step per component: `chain_one.py --tag <tag>`. The 720 tags
are `../C3_Step_size_and_stage_grid`'s own,

    width in {32, 64, 128, 256} x depth in {2, 4, 8} x q in {2, 4, ..., 20}
    x mode in {physics, data} x seed in {0, 1, 2}

**No comment lines in the job list.** `queue args from <file>` does not skip
`#`; a comment line is submitted as a job and fails on the worker node
(`../../_shared/condor/README.md`, gotcha of 2026-09-05). The generated `.txt`
holds argument lines and nothing else.

**One CPU, 4 GB.** The job holds one network (at most 484,180 fp64 parameters),
the field map's three 81x81x146 grids, the 1,000-track list and one batch of at
most 1,000 states; there is no optimiser history, which is what made the
training jobs heavy. `measure_timing.py` records the measured peak.

    PYTHONNOUSERSITE=1 python make_jobs.py
    condor_submit condor/jobs_chain.sub
"""
from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# wrapper.sh resolves its first argument against Raissi_disc_time_approach/,
# so the job line must carry this folder's path *relative to that root*, not
# just its basename: since the experiments were grouped into block folders the
# two differ (Block_C_step_size_and_stages/C6_Chained_crossing vs
# C6_Chained_crossing) and a basename would send the job to a missing path.
import use_shared                                    # noqa: E402,F401
EXPERIMENT = os.path.relpath(HERE, use_shared.SHARED_ROOT)
WIDTHS = (32, 64, 128, 256)
DEPTHS = (2, 4, 8)
QS = tuple(range(2, 21, 2))
MODES = ("physics", "data")
SEEDS = (0, 1, 2)

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


def tags():
    return ["w%d_d%d_q%02d_%s_s%d" % (w, d, q, m, s)
            for w in WIDTHS for d in DEPTHS for q in QS
            for m in MODES for s in SEEDS]


def job_lines(tag_list, extra=""):
    return ["%s --tag %s%s" % (EXPERIMENT, t, extra) for t in tag_list]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--memory-mb", type=int, default=4096)
    ap.add_argument("--category", default="medium")
    ap.add_argument("--jobs", default="condor/jobs_chain.txt")
    ap.add_argument("--sub", default="condor/jobs_chain.sub")
    ap.add_argument("--tags", default=None,
                    help="a file of tags, one per line; default = all 720")
    a = ap.parse_args(argv)

    if a.tags:
        with open(a.tags) as f:
            tag_list = [ln.strip() for ln in f if ln.strip()]
    else:
        tag_list = tags()
    lines = job_lines(tag_list)
    jobs, sub = os.path.join(HERE, a.jobs), os.path.join(HERE, a.sub)
    os.makedirs(os.path.dirname(jobs), exist_ok=True)
    with open(jobs, "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(sub, "w") as f:
        f.write(SUB.format(
            wrapper=os.path.join(HERE, "condor", "wrapper_chain.sh"),
            memory="%d" % a.memory_mb, category=a.category, jobs=a.jobs))
    try:
        with open(os.path.join(HERE, "results", "timing.json")) as f:
            t = json.load(f)["decision"]
    except (OSError, ValueError, KeyError):
        t = None
    print("%d jobs -> %s (%d MB each, timing decision %s)"
          % (len(lines), a.jobs, a.memory_mb, t))
    print("submit with:  cd %s && condor_submit %s" % (HERE, a.sub))
    return lines


if __name__ == "__main__":
    main()
