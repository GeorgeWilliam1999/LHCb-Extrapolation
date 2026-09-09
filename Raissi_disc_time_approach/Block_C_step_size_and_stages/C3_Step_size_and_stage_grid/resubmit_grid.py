#!/usr/bin/env python
"""Send the grid runs that did not confirm back to the farm to continue.

The pattern is `../../Block_A_technique_works/A3a_General_leg_network/resubmit_residual.py`'s. A run that
stops with `converged = false` has not failed - it has run out of the harness's
patience - and `train_grid.py` inherits the shared trainer's
checkpoint-every-restart, so submitting the identical command line again picks
the run up from its own checkpoint and carries on.

Since `../../_shared/train.py` was changed on 2026-09-07 a failed confirmation no
longer ends the run: it drops back to the stall phase and the stall/confirm
cycle repeats until a confirmation holds or `--outer-cap` is reached, and
`resume_phase` reads that position back out of the history. So a run resumed by
this script now continues *training* where it previously only re-ran the
confirmation it had just failed. Records written before that change resume the
same way, because the phase is derived from their history rather than stored.

**This script is idempotent and safe to run on a schedule.** It picks up only
records whose json says `converged = false` and whose tag is not currently
idle or running in the queue - two processes writing one checkpoint would
corrupt it - so running it again while a round is still draining selects
nothing, and running it after more records land selects exactly the new ones.
Each pass writes its own numbered round files; `--round` defaults to the next
number not yet used. A run that hit the restart cap rather than failing to
confirm is skipped unless `--include-capped` is given, since resuming it
without also raising the cap would simply cap again.

    python resubmit_grid.py --submit             # one pass; safe to repeat
    python resubmit_grid.py                      # write the list, submit nothing
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
    ap.add_argument("--round", type=int, default=None,
                    help="round number for the generated files; default: the "
                         "next one not yet used")
    ap.add_argument("--outer-cap", type=int, default=400)
    ap.add_argument("--include-capped", action="store_true")
    ap.add_argument("--submit", action="store_true")
    a = ap.parse_args(argv)
    if a.round is None:
        used = [int(m.group(1)) for m in
                (re.search(r"jobs_grid_round(\d+)\.txt$", p) for p in
                 glob.glob(os.path.join(HERE, "condor",
                                        "jobs_grid_round*.txt"))) if m]
        a.round = max(used) + 1 if used else 2

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
