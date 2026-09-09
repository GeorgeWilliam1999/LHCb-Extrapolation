#!/usr/bin/env python
"""C6.3c - send the networks whose record has not landed back to the farm.

A chained job is stateless: it reads a checkpoint and the track list and writes
one record. There is nothing to resume, so a failure is simply re-run. This
script picks up every tag with no `results/records/<tag>.json` **and** no job
of its own already idle or running in the queue - two jobs writing one record
would interleave - and writes its own numbered round files.

It is idempotent and safe to run on a schedule: a pass run while a round is
still draining selects nothing, and a pass run after more records land selects
exactly the ones still missing.

    PYTHONNOUSERSITE=1 python resubmit.py            # show what would be sent
    PYTHONNOUSERSITE=1 python resubmit.py --submit   # send it
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import argparse
import glob
import re
import subprocess

import make_jobs

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def queued_tags():
    """Tags with a job of their own idle (1) or running (2) in the queue."""
    try:
        out = subprocess.run(
            ["condor_q", "-af", "JobStatus", "Args"],
            capture_output=True, text=True, timeout=120).stdout
    except (OSError, subprocess.SubprocessError):
        print("condor_q not available - assuming an empty queue")
        return set()
    tags = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 2 or parts[0] not in ("1", "2"):
            continue
        m = re.search(r"--tag\s+(\S+)", line)
        if m:
            tags.add(m.group(1))
    return tags


def next_round(pattern="condor/jobs_chain_round%d.txt"):
    n = 1
    while os.path.exists(os.path.join(HERE, pattern % n)):
        n += 1
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--round", type=int, default=None)
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--memory-mb", type=int, default=4096)
    ap.add_argument("--category", default="medium")
    a = ap.parse_args(argv)

    have = {os.path.basename(p)[:-5]
            for p in glob.glob(os.path.join(RESULTS, "records", "*.json"))}
    inq = queued_tags()
    missing = [t for t in make_jobs.tags() if t not in have and t not in inq]
    print("%d records on disk, %d tags in the queue, %d to resubmit"
          % (len(have), len(inq), len(missing)))
    if not missing:
        return []

    n = a.round or next_round()
    jobs = "condor/jobs_chain_round%d.txt" % n
    sub = "condor/jobs_chain_round%d.sub" % n
    lines = make_jobs.job_lines(missing)
    with open(os.path.join(HERE, jobs), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(HERE, sub), "w") as f:
        f.write(make_jobs.SUB.format(
            wrapper=os.path.join(HERE, "condor", "wrapper_chain.sh"),
            memory="%d" % a.memory_mb, category=a.category, jobs=jobs))
    print("round %d: %d jobs -> %s" % (n, len(lines), jobs))
    if a.submit:
        r = subprocess.run(["condor_submit", sub], cwd=HERE,
                           capture_output=True, text=True)
        print(r.stdout.strip() or r.stderr.strip())
    else:
        print("dry run; add --submit to send it")
    return lines


if __name__ == "__main__":
    main()
