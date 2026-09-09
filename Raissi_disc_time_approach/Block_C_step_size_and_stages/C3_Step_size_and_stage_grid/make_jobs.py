#!/usr/bin/env python
"""C3.4 - write the 720-line job list and its submit file.

The grid is

    width  in {32, 64, 128, 256}
    depth  in {2, 4, 8}
    q      in {2, 4, 6, 8, 10, 12, 14, 16, 18, 20}
    mode   in {physics, data}
    seed   in {0, 1, 2}

= 4 x 3 x 10 x 2 x 3 = 720 runs, one job each, one cluster, tagged
`w<width>_d<depth>_q<qq>_<mode>_s<seed>`.

Two details of the submit file are worth stating because they are easy to get
wrong.

**No comment lines in the job list.** `queue args from <file>` does not skip
`#`; a comment line is submitted as a job and fails on the worker node with an
argument error (`../../_shared/condor/README.md`, gotcha of 2026-09-05). The
generated `.txt` therefore contains argument lines and nothing else.

**Memory is per job.** L-BFGS with `history_size 120` keeps 240 vectors the
length of the parameter list, so the 8 x 256 network at q = 20 (about 480,000
parameters) carries roughly a gigabyte of optimiser history alone, on top of
the field map, the dataset and the autograd graph, while the 2 x 32 networks
need a small fraction of that. The C3.3 timing run measured the peak resident
set of exactly that heaviest point on a farm slot at **2.28 GB** on the 4,002
training rows the grid uses, so 4 GB covers every job and the 8 GB the plan
allowed for is not needed - `--big-memory-mb` therefore defaults to the same
4 GB. (It is not idle machinery: the same measurement at 16,002 rows peaked at
4.04 GB, so a larger training set would have needed it.) The list is written
with the line's `request_memory` as its **first** field and the arguments as
the rest, and the submit file reads both with `queue mem, args from`. Memory
comes first because HTCondor splits a multi-variable queue line at the first
run of whitespace and gives the remainder of the line to the last variable -
put the arguments first and every job asks for "--data" megabytes and the
submit is rejected (checked with `condor_submit -dry-run`). `--big-memory-mb`
sets the value used for the architectures the timing run showed need it;
everything else gets `--memory-mb`.

    PYTHONNOUSERSITE=1 python make_jobs.py
    condor_submit condor/jobs_grid.sub
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
BIG = ((256, 8),)          # architectures given the larger memory request

SUB = """universe                = vanilla
executable              = {wrapper}
arguments               = $(args)

output                  = condor/logs/$(Cluster).$(Process).out
error                   = condor/logs/$(Cluster).$(Process).err
log                     = condor/logs/$(Cluster).log

request_cpus            = 1
request_memory          = $(mem)
should_transfer_files   = NO
getenv                  = False
+UseOS                  = "el9"
+JobCategory            = "{category}"

queue mem, args from {jobs}
"""


def job_lines(memory_mb, big_memory_mb, out_dir="results"):
    """One line per run: that line's request_memory, then the arguments.

    `--n-train` is deliberately absent. Every `results/grid_q<qq>.npz` was
    built at the row count the timing run chose, so the trainer takes the
    dataset whole and every point of the grid sees the same rows; passing a
    number here as well would only open the door to a run that quietly trains
    on a different subset. The count actually used is recorded in each run's
    own json as `n_train`.
    """
    lines = []
    for width in WIDTHS:
        for depth in DEPTHS:
            for q in QS:
                for mode in MODES:
                    for seed in SEEDS:
                        tag = "w%d_d%d_q%02d_%s_s%d" % (width, depth, q, mode,
                                                        seed)
                        args = ("%s --data results/grid_q%02d.npz --mode %s "
                                "--seed %d --q %d --width %d --depth %d "
                                "--out %s --tag %s"
                                % (EXPERIMENT, q, mode, seed, q, width, depth,
                                   out_dir, tag))
                        mem = (big_memory_mb if (width, depth) in BIG
                               else memory_mb)
                        lines.append("%d %s" % (mem, args))
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--memory-mb", type=int, default=4096)
    ap.add_argument("--big-memory-mb", type=int, default=4096)
    ap.add_argument("--category", default="medium")
    ap.add_argument("--jobs", default="condor/jobs_grid.txt")
    ap.add_argument("--sub", default="condor/jobs_grid.sub")
    a = ap.parse_args(argv)

    try:
        with open(os.path.join(HERE, "results", "timing.json")) as f:
            n_train = int(json.load(f)["chosen_n_train"])
    except (OSError, ValueError, KeyError):
        n_train = None

    lines = job_lines(a.memory_mb, a.big_memory_mb)
    jobs = os.path.join(HERE, a.jobs)
    sub = os.path.join(HERE, a.sub)
    os.makedirs(os.path.dirname(jobs), exist_ok=True)
    with open(jobs, "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(sub, "w") as f:
        f.write(SUB.format(
            wrapper=os.path.join(HERE, "condor", "wrapper_grid.sh"),
            category=a.category, jobs=a.jobs))
    print("%d jobs -> %s (datasets built at n_train %s, %d MB, %d MB for %s)"
          % (len(lines), a.jobs, n_train, a.memory_mb, a.big_memory_mb,
             ", ".join("%dx%d" % (d, w) for w, d in BIG)))
    print("submit with:  cd %s && condor_submit %s" % (HERE, a.sub))
    return lines


if __name__ == "__main__":
    main()
