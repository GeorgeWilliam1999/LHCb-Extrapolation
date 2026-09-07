#!/usr/bin/env python
"""Send the grid runs that did not confirm back to the farm to continue.

The pattern is `../General_leg_network/resubmit_residual.py`'s, for the same
reason. The shared stall criterion - two consecutive restarts each improving
the loss by less than 1% - can fire while the endpoint medians are still moving
by more than 1%; the confirmation pass then re-stalls at once, the run is
recorded `converged = false`, and it stops even though it was still improving.
That is harness issue A1, and the fix is not to change the criterion (which
would break comparability with every earlier experiment) but to hand the run
back its own checkpoint: `train_grid.py` inherits the shared trainer's
checkpoint-every-restart, so submitting the identical command line again
resumes with a fresh optimiser and another confirmation attempt.

A run still in the queue is never included - two processes writing one
checkpoint would corrupt it - and neither is one whose json says it hit the
restart cap rather than failing to confirm, unless `--include-capped` is given:
a capped run has not stalled at all and resuming it will simply cap again
unless the cap is also raised.

    python resubmit_grid.py --round 2            # write the list
    python resubmit_grid.py --round 2 --submit   # write it and submit it
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_SUB = os.path.join(HERE, "condor", "jobs_grid.sub")
BASE_TXT = os.path.join(HERE, "condor", "jobs_grid.txt")


def running_tags():
    """Tags of this user's jobs that are still in the queue."""
    try:
        out = subprocess.run(["condor_q", "-af", "Args"], capture_output=True,
                             text=True).stdout
    except FileNotFoundError:
        return set()
    return set(re.findall(r"--tag\s+(\S+)", out))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--outer-cap", type=int, default=400)
    ap.add_argument("--include-capped", action="store_true")
    ap.add_argument("--submit", action="store_true")
    a = ap.parse_args(argv)

    with open(BASE_TXT) as f:
        lines = {}
        for line in f:
            line = line.strip()
            if not line:
                continue
            mem, _, args = line.partition(" ")
            m = re.search(r"--tag\s+(\S+)", args)
            if m:
                lines[m.group(1)] = (args.strip(), mem.strip())

    busy = running_tags()
    todo, skipped = [], {"converged": 0, "in_queue": 0, "capped": 0,
                         "not_in_list": 0}
    for jf in sorted(glob.glob(os.path.join(HERE, "results",
                                            "w*_d*_q*_s*.json"))):
        tag = os.path.splitext(os.path.basename(jf))[0]
        try:
            with open(jf) as f:
                j = json.load(f)
        except (ValueError, OSError):
            continue
        if j.get("converged"):
            skipped["converged"] += 1
            continue
        if tag in busy:
            skipped["in_queue"] += 1
            continue
        if tag not in lines:
            skipped["not_in_list"] += 1
            continue
        capped = j.get("restarts", 0) >= j.get("outer_cap", a.outer_cap)
        if capped and not a.include_capped:
            skipped["capped"] += 1
            continue
        args, mem = lines[tag]
        cap = a.outer_cap if not capped else max(a.outer_cap,
                                                 j.get("restarts", 0) + 100)
        todo.append("%s %s --outer-cap %d" % (mem, args, cap))

    print("skipped: %s" % json.dumps(skipped))
    if not todo:
        print("nothing to resubmit")
        return
    txt = os.path.join(HERE, "condor", "jobs_grid_round%d.txt" % a.round)
    sub = os.path.join(HERE, "condor", "jobs_grid_round%d.sub" % a.round)
    with open(txt, "w") as f:
        f.write("\n".join(todo) + "\n")
    with open(BASE_SUB) as f:
        s = f.read()
    s = s.replace("queue mem, args from condor/jobs_grid.txt",
                  "queue mem, args from condor/jobs_grid_round%d.txt" % a.round)
    with open(sub, "w") as f:
        f.write(s)
    print("%d runs to continue -> %s" % (len(todo), os.path.basename(txt)))
    if a.submit:
        r = subprocess.run(["condor_submit", sub], cwd=HERE,
                           capture_output=True, text=True)
        print(r.stdout.strip() or r.stderr.strip())


if __name__ == "__main__":
    main()
