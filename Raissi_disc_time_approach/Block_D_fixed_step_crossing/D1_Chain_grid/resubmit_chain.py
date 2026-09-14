#!/usr/bin/env python
"""Send the chains that have not finished back to the farm to continue.

A chain job is finished when its `chain.json` exists. Anything else - a job
evicted mid-leg, one that ran past the slot's time, one whose node died - is
resumed by running the identical command line again: `train_chain.py` skips
the legs already trained and applied, and `_shared/train.py` picks a
half-trained leg up from its last restart.

Idempotent: a chain whose (N, q) is idle or running in the queue is not
resubmitted (two jobs writing one checkpoint would corrupt it). Safe to run
on a schedule until every chain.json exists.

    python resubmit_chain.py --submit      # one pass
    python resubmit_chain.py               # list only
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
from make_jobs import SUB, job_lines                      # noqa: E402


def queued():
    try:
        out = subprocess.run(["condor_q", "-af", "Args"], capture_output=True,
                             text=True).stdout
    except FileNotFoundError:
        return set()
    keys = set()
    for line in out.splitlines():
        m = re.search(r"--N\s+(\d+)\s+--q\s+(\d+)", line)
        if m:
            keys.add((int(m.group(1)), int(m.group(2))))
    return keys


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--round", type=int, default=None)
    ap.add_argument("--memory-mb", type=int, default=4096)
    ap.add_argument("--category", default="medium")
    a = ap.parse_args(argv)
    inq = queued()
    todo = []
    for line in job_lines():
        m = re.search(r"--N\s+(\d+)\s+--q\s+(\d+)", line)
        N, q = int(m.group(1)), int(m.group(2))
        if os.path.exists(os.path.join(HERE, "results", "N%03d_q%02d" % (N, q), "chain.json")):
            continue
        if (N, q) in inq:
            continue
        todo.append(line)
    rnd = a.round
    if rnd is None:
        rnd = 1
        while os.path.exists(os.path.join(HERE, "condor", "jobs_chain_round%d.txt" % rnd)):
            rnd += 1
    print("%d chains unfinished and not queued" % len(todo))
    if not todo:
        return []
    jobs = "condor/jobs_chain_round%d.txt" % rnd
    sub = "condor/jobs_chain_round%d.sub" % rnd
    with open(os.path.join(HERE, jobs), "w") as f:
        f.write("\n".join(todo) + "\n")
    with open(os.path.join(HERE, sub), "w") as f:
        f.write(SUB.format(wrapper=os.path.join(HERE, "condor", "wrapper_chain.sh"),
                           memory=a.memory_mb, category=a.category, jobs=jobs))
    print("wrote %s" % sub)
    if a.submit:
        r = subprocess.run(["condor_submit", sub], cwd=HERE, capture_output=True, text=True)
        print(r.stdout.strip() or r.stderr.strip())
    return todo


if __name__ == "__main__":
    main()
