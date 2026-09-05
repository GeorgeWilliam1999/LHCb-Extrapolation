#!/usr/bin/env python
"""Build the frozen-leg dataset this study trains on.

One call to the shared builder with the baseline settings (q = 8, MagDown,
2000 training states, the baseline split seed, the fiducial requirement), so
the grid below differs from `../One_step_network_v2` only in the network size
and the seed.  The split counts are asserted against the verified baseline:
1979 / 2062 / 2018 states after the fiducial cut.

    PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python build_dataset.py

Writes results/frozen_leg_q08.npz (+ _meta.json).
"""
import os

import use_shared  # noqa: F401  (puts _shared on sys.path)
from _shared.prepare import frozen_leg_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "frozen_leg_q08.npz")
EXPECTED = {"train": 1979, "val": 2062, "test": 2018}


def main():
    d = frozen_leg_dataset(q=8, field="down", n_train=2000, seed=20260718,
                           rebase_mm=60.0, fiducial=True, out_npz=OUT)
    counts = {s: int(len(d["%s_S" % s])) for s in ("train", "val", "test")}
    print("counts:", counts)
    assert counts == EXPECTED, "counts %s != baseline %s" % (counts, EXPECTED)
    print("counts match the verified baseline; wrote", OUT)


if __name__ == "__main__":
    main()
