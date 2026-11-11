#!/usr/bin/env python
"""Send the Block E networks that have not finished back to the farm to continue.

A network is finished when its `results/N<NNN>_q<qq>/record.json` exists.
Anything else is resumed by running the identical command line again. A
network whose (N, q) is idle or running in the queue is not resubmitted (two
jobs writing one checkpoint would corrupt it).

    python resubmit.py --round-restarts 25            # list only
    python resubmit.py --round-restarts 25 --submit   # one pass
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
from make_jobs import SUB, job_lines                      # noqa: E402


WRAPPER = os.path.join(HERE, "condor", "wrapper.sh")


def queued():
    """The (N, q) of the jobs of THIS block that are in the queue.

    The executable has to be matched as well as the arguments: Block F runs the
    same (N, q) through its own wrapper, and matching on --N/--q alone would let
    one block's job stop the other's from being resubmitted.
    """
    try:
        out = subprocess.run(["condor_q", "-af", "Cmd", "Args"],
                             capture_output=True, text=True).stdout
    except FileNotFoundError:
        return set()
    keys = set()
    for line in out.splitlines():
        if WRAPPER not in line:
            continue
        m = re.search(r"--N\s+(\d+)\s+--q\s+(\d+)", line)
        if m:
            keys.add((int(m.group(1)), int(m.group(2))))
    return keys


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--round-restarts", type=int, default=None)
    ap.add_argument("--memory-mb", type=int, default=8192)
    ap.add_argument("--category", default="medium")
    ap.add_argument("--jobs-from", default=os.path.join(HERE, "condor", "jobs_active.txt"),
                    help="the job lines to keep alive (default: condor/jobs_active.txt if it exists, "
                         "else the standard 16)")
    a = ap.parse_args(argv)
    inq = queued()
    lines = (open(a.jobs_from).read().split("\n") if os.path.exists(a.jobs_from)
             else job_lines(a.round_restarts))
    todo = []
    for line in [x for x in lines if x.strip()]:
        m = re.search(r"--N\s+(\d+)\s+--q\s+(\d+)", line)
        N, q = int(m.group(1)), int(m.group(2))
        prog = os.path.join(HERE, "results", "N%03d_q%02d" % (N, q), "progress.json")
        # A run is finished when its progress file says "done" AND it has used
        # the restart budget of THIS job line. `record.json` alone is not
        # enough: a run waiting to be extended still carries the record of its
        # previous stopping point, and a line with a larger --outer-cap is a
        # request to train it further.
        cap = re.search(r"--outer-cap\s+(\d+)", line)
        cap = int(cap.group(1)) if cap else 400
        if os.path.exists(prog):
            pr = json.load(open(prog))
            if pr.get("phase") == "done" and pr.get("restart", 0) >= cap:
                continue
        if (N, q) in inq:
            continue
        todo.append(line)
    print("%d networks unfinished and not queued" % len(todo))
    if not todo:
        return []
    rnd = 1
    while os.path.exists(os.path.join(HERE, "condor", "jobs_round%d.txt" % rnd)):
        rnd += 1
    jobs, sub = "condor/jobs_round%d.txt" % rnd, "condor/jobs_round%d.sub" % rnd
    with open(os.path.join(HERE, jobs), "w") as f:
        f.write("\n".join(todo) + "\n")
    with open(os.path.join(HERE, sub), "w") as f:
        f.write(SUB.format(wrapper=os.path.join(HERE, "condor", "wrapper.sh"),
                           memory=a.memory_mb, category=a.category, jobs=jobs))
    if a.submit:
        subprocess.run(["condor_submit", sub], cwd=HERE, check=True)
    else:
        print("not submitted: %s" % sub)
    return todo


if __name__ == "__main__":
    main()
