#!/usr/bin/env python
"""Take a run out of the farm's keep-alive list once its error has stopped falling.

`condor/jobs_active.txt` is what the keeper resubmits. Without this, a run that
stops on the trainer's own rule is sent straight back and trains on to its
restart cap, and a run that is still improving is left to stop early.

The rule is NOT rewritten here. `plateaued_now` is IMPORTED from
`Block_F_reweighted_loss/F2_Analysis/compare_to_blockE.py`, so both blocks are
judged by the same function and it cannot drift into two versions: the median
validation error of the last ten rounds is no more than 5 percent below the
median of the ten before, for three rounds running, as of the latest round.

The trainer's own rule (two restarts under 1 percent, then a confirmation pass)
is deliberately not required as well. At N >= 64 with the rescaled loss it never
fires: every restart still cuts the loss by more than 1 percent hundreds of
restarts after the validation error stopped moving. The loss is not the measure
of being finished here; the validation error is (Block F, 2026-09-19).

    python prune_active.py             # say what would be dropped
    python prune_active.py --write     # actually rewrite the list
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))       # .../G1_Training/condor
ROOT = os.path.dirname(HERE)                            # .../G1_Training
BLOCK_F_F2 = os.path.abspath(os.path.join(
    ROOT, "..", "..", "Block_F_reweighted_loss", "F2_Analysis"))
sys.path.insert(0, HERE)
sys.path.insert(0, BLOCK_F_F2)

from compare_to_blockE import plateaued_now             # noqa: E402  (imported, never copied)
from resubmit import key_of, run_dir                    # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--active", default=os.path.join(HERE, "jobs_active.txt"))
    ap.add_argument("--write", action="store_true", help="rewrite the keep-alive list")
    a = ap.parse_args(argv)
    if not os.path.exists(a.active):
        print("no keep-alive list at %s" % a.active)
        return []
    lines = [x for x in open(a.active).read().split("\n") if x.strip()]
    keep, drop = [], []
    for line in lines:
        k = key_of(line)
        if k is None:
            continue
        N, q, _w, p_lo, p_hi = k
        rounds = os.path.join(run_dir(line), "rounds.csv")
        if not os.path.exists(rounds):
            keep.append(line)
            continue
        rows = list(csv.DictReader(open(rounds)))
        v = np.array([float(x["val_z1_pos_med_um"]) for x in rows])
        if plateaued_now(v):
            drop.append((line, N, q, p_lo, p_hi, len(v)))
        else:
            keep.append(line)
            print("  keep  N = %3d, q = %2d, window %g-%g GeV (round %2d): "
                  "the validation error is still falling" % (N, q, p_lo, p_hi, len(v)))
    for line, N, q, p_lo, p_hi, rnd in drop:
        print("  DONE  N = %3d, q = %2d, window %g-%g GeV (round %2d): "
              "the validation error has stopped falling" % (N, q, p_lo, p_hi, rnd))
    print("%d of %d runs still to train" % (len(keep), len(lines)))
    if a.write and drop:
        with open(a.active, "w") as f:
            f.write("\n".join(keep) + ("\n" if keep else ""))
        print("rewrote %s" % a.active)
    elif drop:
        print("not written (pass --write)")
    return keep


if __name__ == "__main__":
    main()
