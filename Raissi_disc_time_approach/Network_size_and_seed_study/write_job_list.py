#!/usr/bin/env python
"""Write the HTCondor job list for the size-and-seed grid.

One line per run: widths {32, 50, 100, 200} x depths {2, 4, 6} x seeds 0..9,
physics loss, q = 8, on the frozen-leg dataset built by build_dataset.py.
That is 120 jobs, submitted as one cluster.  The optional --mode/--widths/
--depths switches write the data-twin list for the winning architecture
(sub-goal A2.2) into the same file format.

    python write_job_list.py                       -> condor/jobs.txt (120 lines)
    python write_job_list.py --mode data --widths 50 --depths 4 \
        --out condor/jobs_data_twin.txt            -> 10 lines
"""
import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDER = "Network_size_and_seed_study"
DATA = "results/frozen_leg_q08.npz"

LINE = ("{folder} --data {data} --mode {mode} --seed {seed} --q 8 "
        "--width {width} --depth {depth} --out results --tag {tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="physics", choices=("physics", "data"))
    ap.add_argument("--widths", type=int, nargs="+", default=[32, 50, 100, 200])
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 4, 6])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--out", default="condor/jobs.txt")
    a = ap.parse_args()

    lines = []
    for width in a.widths:
        for depth in a.depths:
            for seed in a.seeds:
                tag = "w%03d_d%d_s%d" % (width, depth, seed)
                if a.mode == "data":
                    tag = "data_" + tag
                lines.append(LINE.format(folder=FOLDER, data=DATA, mode=a.mode,
                                         seed=seed, width=width, depth=depth,
                                         tag=tag))
    path = os.path.join(HERE, a.out)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote %d lines to %s" % (len(lines), path))


if __name__ == "__main__":
    main()
