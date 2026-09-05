#!/usr/bin/env python
"""Write the job list that continues the runs the 150-restart cap cut short.

`_shared/train.py` counts the stall phase and the confirmation pass against the
same `--outer-cap`, so a physics run that stalls near restart 150 hits the cap
before it can be confirmed and is recorded `converged = false` although it is
about to converge (found by the stage-count study on 2026-09-05; the shared
default has since been raised to 400, but the jobs of cluster 5781153 were
already running with 150).  Runs are resumable: submitting the identical line
with `--outer-cap 400` continues from the checkpoint and finishes the
confirmation in a restart or two.

This reads results/summary.csv and re-emits, with `--outer-cap 400` appended,
every unconverged run whose restart count is within `--near` of its cap.  Runs
that stopped well short of the cap are NOT re-emitted: those failed the
confirmation on their own merits (the fresh optimiser moved the endpoint
medians by more than 1%), which is a result, not a truncation.

    python write_continuation_list.py            -> condor/jobs_continuation.txt
"""
import argparse
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDER = "Network_size_and_seed_study"
DATA = "results/frozen_leg_q08.npz"
LINE = ("{folder} --data {data} --mode {mode} --seed {seed} --q 8 "
        "--width {width} --depth {depth} --out results --tag {tag} "
        "--outer-cap {cap}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--near", type=int, default=8,
                    help="restarts within this many of the cap counts as truncated. "
                         "A run can leave the stall phase a few restarts short "
                         "of the cap and then spend the remainder in the "
                         "confirmation pass, so the window is wider than 2.")
    ap.add_argument("--cap", type=int, default=400)
    ap.add_argument("--out", default="condor/jobs_continuation.txt")
    a = ap.parse_args()

    df = pd.read_csv(os.path.join(HERE, "results", "summary.csv"))
    cut = df[(~df["converged"]) & (df["restarts"] >= df["outer_cap"] - a.near)]
    lines = [LINE.format(folder=FOLDER, data=DATA, mode=r["mode"],
                         seed=int(r["seed"]), width=int(r["width"]),
                         depth=int(r["depth"]), tag=r["tag"], cap=a.cap)
             for _, r in cut.sort_values("tag").iterrows()]
    path = os.path.join(HERE, a.out)
    with open(path, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    print("%d truncated runs of %d unconverged -> %s"
          % (len(lines), int((~df["converged"]).sum()), path))
    for line in lines:
        print("  " + line.split("--tag ")[1])


if __name__ == "__main__":
    main()
