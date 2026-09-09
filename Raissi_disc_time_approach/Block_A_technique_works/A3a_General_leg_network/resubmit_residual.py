#!/usr/bin/env python
"""Send the residual runs that did not confirm back to the farm to continue.

`resubmit_unconverged.py` for the residual arm: same reasoning, same mechanism,
different job list and submit file (it must use `condor/wrapper_residual.sh`,
which starts `train_residual.py` rather than the shared `train.py`).

The reason it is needed is the harness issue A1 found on this dataset: the
shared stall criterion - two consecutive restarts each improving the loss by
less than 1% - fires while the endpoint medians are still moving by more than
1%, so the confirmation pass re-stalls at once, the run is recorded
`converged = false`, and it stops even though it was still improving.
`train_residual.py` inherits the shared trainer's checkpoint-every-restart, so
submitting the identical command line again resumes from the checkpoint with a
fresh optimiser and another confirmation attempt.

    python resubmit_residual.py --round 2 [--submit]

writes `condor/jobs_residual_round<N>.txt` and `.sub`. Runs still in the queue
are never included: two processes writing one checkpoint would corrupt it.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_SUB = os.path.join(HERE, "condor", "jobs_residual.sub")


def running_tags():
    try:
        out = subprocess.run(["condor_q", "-af", "Args"], capture_output=True,
                             text=True).stdout
    except FileNotFoundError:
        return set()
    return set(re.findall(r"--tag\s+(\S+)", out))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--outer-cap", type=int, default=400)
    ap.add_argument("--submit", action="store_true")
    a = ap.parse_args()

    with open(os.path.join(HERE, "condor", "jobs_residual.txt")) as f:
        lines = {re.search(r"--tag\s+(\S+)", l).group(1): l.strip()
                 for l in f if l.strip()}
    busy = running_tags()
    todo = []
    for jf in sorted(glob.glob(os.path.join(HERE, "results",
                                            "residual_w*_s*.json"))):
        tag = os.path.splitext(os.path.basename(jf))[0]
        with open(jf) as f:
            j = json.load(f)
        if j["converged"] or tag in busy or tag not in lines:
            continue
        todo.append("%s --outer-cap %d" % (lines[tag], a.outer_cap))
    if not todo:
        print("nothing to resubmit")
        return
    txt = os.path.join(HERE, "condor", "jobs_residual_round%d.txt" % a.round)
    sub = os.path.join(HERE, "condor", "jobs_residual_round%d.sub" % a.round)
    with open(txt, "w") as f:
        f.write("\n".join(todo) + "\n")
    with open(BASE_SUB) as f:
        s = f.read()
    s = s.replace("queue args from condor/jobs_residual.txt",
                  "queue args from condor/jobs_residual_round%d.txt" % a.round)
    with open(sub, "w") as f:
        f.write(s)
    print("%d runs to continue -> %s" % (len(todo), os.path.basename(txt)))
    if a.submit:
        r = subprocess.run(["condor_submit", sub], cwd=HERE,
                           capture_output=True, text=True)
        print(r.stdout.strip() or r.stderr.strip())


if __name__ == "__main__":
    main()
