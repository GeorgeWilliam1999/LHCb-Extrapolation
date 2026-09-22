#!/usr/bin/env python
"""Rescore finished chains from their states.npz with metrics.chain_scores,
rewriting the val/test blocks of chain.json in place (the legs, the cost and
everything else in the record are kept). Idempotent. Needed only for chains
whose chain.json predates the per-component scoring of 2026-09-14.

    python score_components.py            # every chain in results/
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"

import argparse
import glob
import json

import numpy as np

import use_shared                                        # noqa: F401
from _shared.reference import make_field
from metrics import chain_scores
from train_chain import DEFAULT_DATA, SPLITS

HERE = os.path.dirname(os.path.abspath(__file__))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--force", action="store_true", help="rescore even if components exist")
    a = ap.parse_args(argv)
    D = {k: v for k, v in np.load(os.path.abspath(a.data), allow_pickle=False).items()}
    fld = make_field(str(D["field"]))
    n_done = 0
    for p in sorted(glob.glob(os.path.join(a.results, "N*_q*", "chain.json"))):
        with open(p) as f:
            rec = json.load(f)
        if not a.force and "components" in rec.get("test", {}).get("vs_rk6_endpoint", {}):
            continue
        st = np.load(os.path.join(os.path.dirname(p), "states.npz"))
        states = {s: st["%s_states" % s] for s in SPLITS}
        rec.update(chain_scores(states, D, int(rec["N"]), fld))
        with open(p, "w") as f:
            json.dump(rec, f, indent=1)
        n_done += 1
        print("rescored", os.path.relpath(p, a.results))
    print("%d chains rescored" % n_done)


if __name__ == "__main__":
    main()
