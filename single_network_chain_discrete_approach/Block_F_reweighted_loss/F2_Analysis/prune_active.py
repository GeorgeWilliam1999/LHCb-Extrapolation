#!/usr/bin/env python
"""Take a run out of the farm's keep-alive list once it is REALLY finished.

CHANGED 2026-09-19. The first version asked for BOTH the trainer's own rule
(converged and confirmed) AND the plateau rule. At N >= 64 the trainer's rule
never fires: with the rescaled loss every restart still cuts the loss by more
than 1 percent, 500 restarts after the validation error stopped moving, so all
nine N >= 64 runs at q <= 8 ran into their restart cap instead. The loss is not
the measure of being finished here; the validation error is. A run is now done
when `plateaued_now` holds - the error has stopped falling as of its latest
rounds - which also catches a run that flattened and then started improving
again (N = 128 q = 8 and N = 256 q = 2 did exactly that).

The notes below describe the first version and are kept for the record.

`condor/jobs_active.txt` is what each keeper resubmits. Without this, a run that
stops on the trainer's own rule is sent straight back by the keeper and trains
on to its restart cap, and a run that is still improving could be left to stop
early. Both are decided here, by one definition of finished that needs BOTH:

  * the trainer's own rule fired - the refreshed states taught nothing new AND
    the confirmation pass with a fresh optimiser held (`converged` and
    `confirmed` in progress.json). A confirmation that did NOT hold means the
    fresh optimiser either kept improving the loss or moved the validation
    median, so the run is not finished;
  * the plateau rule of `compare_to_blockE.py` fired - the median validation
    error of the last ten rounds has stopped falling.

They answer different questions - "has this round stopped teaching it anything"
and "has the error stopped falling over tens of rounds" - and a run is only
done when both say so. Runs that meet neither, or only one, stay in the list.

    python prune_active.py --block E           # Block E's extensions
    python prune_active.py --block F           # the Block F runs
    python prune_active.py --block E --write   # actually rewrite the list
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re

import numpy as np

from compare_to_blockE import plateaued_now

HERE = os.path.dirname(os.path.abspath(__file__))
BLOCKS = {
    "E": dict(root=os.path.join(HERE, "..", "..", "Block_E_single_network_chain", "E1_Network_grid"),
              layout="results/N%03d_q%02d"),
    "F": dict(root=os.path.join(HERE, "..", "F1_Training"),
              layout="results/%(w)s/N%(N)03d_q%(q)02d"),
}


def run_dir(block, root, line):
    N = int(re.search(r"--N\s+(\d+)", line).group(1))
    q = int(re.search(r"--q\s+(\d+)", line).group(1))
    w = re.search(r"--weighting\s+(\w+)", line)
    if block == "E":
        return os.path.join(root, "results", "N%03d_q%02d" % (N, q)), (N, q)
    return (os.path.join(root, "results", w.group(1) if w else "full", "N%03d_q%02d" % (N, q)),
            (N, q))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--block", choices=("E", "F"), required=True)
    ap.add_argument("--write", action="store_true", help="rewrite jobs_active.txt")
    a = ap.parse_args(argv)
    root = BLOCKS[a.block]["root"]
    active = os.path.join(root, "condor", "jobs_active.txt")
    if not os.path.exists(active):
        print("no jobs_active.txt at %s" % active)
        return []
    lines = [x for x in open(active).read().split("\n") if x.strip()]
    keep, drop = [], []
    for line in lines:
        d, (N, q) = run_dir(a.block, root, line)
        prog = os.path.join(d, "progress.json")
        rounds = os.path.join(d, "rounds.csv")
        if not (os.path.exists(prog) and os.path.exists(rounds)):
            keep.append(line)
            continue
        p = json.load(open(prog))
        v = np.array([float(x["val_z1_pos_med_um"]) for x in csv.DictReader(open(rounds))])
        if plateaued_now(v):
            drop.append((line, N, q, p["round"]))
        else:
            keep.append(line)
            print("  keep  N=%3d q=%2d (round %2d): the validation error is still falling"
                  % (N, q, p["round"]))
    for line, N, q, rnd in drop:
        print("  DONE  N=%3d q=%2d (round %2d): the validation error has stopped falling"
              % (N, q, rnd))
    print("%d of %d runs still to train" % (len(keep), len(lines)))
    if a.write and drop:
        with open(active, "w") as f:
            f.write("\n".join(keep) + ("\n" if keep else ""))
        print("rewrote %s" % active)
    elif drop:
        print("not written (pass --write)")
    return keep


if __name__ == "__main__":
    main()
