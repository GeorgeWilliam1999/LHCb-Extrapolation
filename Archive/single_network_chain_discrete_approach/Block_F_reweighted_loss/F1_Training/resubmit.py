#!/usr/bin/env python
"""Send the Block F runs that have not finished back to the farm to continue.

Block E's `resubmit.py` with the run folder and the queue match extended by the
weighting, so an ablation can never be mistaken for the `full` run of the same
(N, q) - two jobs writing one checkpoint would corrupt it.

A run is finished when its `progress.json` says "done" AND it has used the
restart budget of THIS job line. `record.json` alone is not enough: a run
waiting to be extended still carries the record of its previous stopping point,
and a line with a larger --outer-cap is a request to train it further.

    python resubmit.py            # list only
    python resubmit.py --submit   # one pass
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
from make_jobs import SUB, job_lines                      # noqa: E402

KEY = re.compile(r"--N\s+(\d+)\s+--q\s+(\d+)")
WEIGHTING = re.compile(r"--weighting\s+(\w+)")


def key_of(line, default_weighting="full"):
    m = KEY.search(line)
    if not m:
        return None
    w = WEIGHTING.search(line)
    return (int(m.group(1)), int(m.group(2)), w.group(1) if w else default_weighting)


WRAPPER = os.path.join(HERE, "condor", "wrapper.sh")


def queued():
    """The (N, q, weighting) of the jobs of THIS block that are in the queue.

    The executable has to be matched as well as the arguments: Block E runs the
    same (N, q) through its own wrapper and passes no --weighting, so matching
    on --N/--q alone would read its jobs as this block's `full` runs.
    """
    try:
        out = subprocess.run(["condor_q", "-af", "Cmd", "Args"],
                             capture_output=True, text=True).stdout
    except FileNotFoundError:
        return set()
    return {k for k in (key_of(line) for line in out.splitlines() if WRAPPER in line) if k}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--memory-mb", type=int, default=8192)
    ap.add_argument("--category", default="medium")
    ap.add_argument("--jobs-from", default=os.path.join(HERE, "condor", "jobs_active.txt"),
                    help="the job lines to keep alive (default: condor/jobs_active.txt if it "
                         "exists, else the standard three `full` runs)")
    a = ap.parse_args(argv)
    inq = queued()
    lines = (open(a.jobs_from).read().split("\n") if os.path.exists(a.jobs_from) else job_lines())
    todo = []
    for line in [x for x in lines if x.strip()]:
        k = key_of(line)
        if k is None:
            continue
        N, q, w = k
        prog = os.path.join(HERE, "results", w, "N%03d_q%02d" % (N, q), "progress.json")
        cap = re.search(r"--outer-cap\s+(\d+)", line)
        cap = int(cap.group(1)) if cap else 400
        if os.path.exists(prog):
            pr = json.load(open(prog))
            if pr.get("phase") == "done" and pr.get("restart", 0) >= cap:
                continue
        if k in inq:
            continue
        todo.append(line)
    print("%d runs unfinished and not queued" % len(todo))
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
