#!/usr/bin/env python
"""G2 - endpoint error(dz, q) at the SciFi plane, one table per state component.

Two momentum ranges: 5-30 GeV (the range George asked for on 2026-09-21) and
3-8 GeV (the range this study's loss window is aimed at).

These are ENDPOINT errors: each network is applied N times from the track's
real state on the last UT plane, and its state on the first SciFi plane
(z1 = 7,826 mm) is compared with the RK6 track carried there. They are NOT
single-step errors (one application from an RK6 state); those are in
G3_Analysis/results/error_qdz_single_step.csv.

Per cell, over the test tracks in the range: the median |error|, the 95th
percentile, and the signed median. dx, dy in um; dtx, dty in mrad. Every cell
is one network, named by N, q and dz; a network whose record is a checkpoint of
a run still training is marked.

Nothing is recomputed here. Block F's `error_tables_by_component.py` already
does exactly this, and its two module constants (HERE, where results/ and
figures/ are written; RUNS, the folder of run folders) are the only things that
tie it to Block F. This script imports it by path, points those two here, and
calls its `main` once per range - so the tables are the same code, not a
retyping of it, and nothing under Block F is touched or written to.

Run:  PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python error_tables_by_component.py [--runs DIR] [--out DIR]
Outputs: <out>/results/error_by_component_5-30GeV.csv and _3-8GeV.csv,
         <out>/figures/error_by_component_5-30GeV.png and _3-8GeV.png
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"

import sys           # noqa: E402
# importing a script out of a read-only folder must not leave a .pyc behind in it
sys.dont_write_bytecode = True

import argparse          # noqa: E402
import importlib.util    # noqa: E402

import run_discovery as rd   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
F2 = os.path.abspath(os.path.join(HERE, "..", "..", "Block_F_reweighted_loss", "F2_Analysis"))
DEFAULT_RUNS = os.path.join(HERE, "..", "G1_Training", "results", "p03-08", "full")
RANGES = ((5.0, 30.0), (3.0, 8.0))


def load_f2(name):
    spec = importlib.util.spec_from_file_location("f2_" + name, os.path.join(F2, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", default=DEFAULT_RUNS)
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--only", default=None, help="comma-separated run folders, e.g. N064_q02")
    a = ap.parse_args(argv)
    nets = rd.require(rd.find_runs(a.runs, a.only), a.runs)
    print(rd.describe(nets, a.runs))
    if a.only:
        raise SystemExit("--only is not supported here: the imported table script reads every run "
                         "folder under --runs. Point --runs at a folder holding just the run you want.")
    os.makedirs(os.path.join(a.out, "results"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "figures"), exist_ok=True)
    mod = load_f2("error_tables_by_component")
    mod.HERE = os.path.abspath(a.out)
    mod.RUNS = os.path.abspath(a.runs)
    for lo, hi in RANGES:
        print("\n" + "=" * 100)
        print("%s, loss window %s" % (rd.STUDY, rd.window_text(nets)))
        mod.main(["--p-lo", "%g" % lo, "--p-hi", "%g" % hi])


if __name__ == "__main__":
    main()
