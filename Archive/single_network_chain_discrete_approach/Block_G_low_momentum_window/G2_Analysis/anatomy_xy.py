#!/usr/bin/env python
"""G2 - the two-slope anatomy of the endpoint error, run for each network.

Where the endpoint error comes from. At every one of the N steps the network is
given the SAME state the RK6 reference is given, so the step's own error is
isolated: a position error dx, dy and a slope error dtx, dty. A slope error
made at step k is still being carried at the end, over the lever arm
(z1 - z_end of that step), so the endpoint error it predicts is
sum_k (dx_k + dtx_k * lever_k) in x and the same in y, combined radially. What
fraction of the real endpoint error that sum explains says whether the error is
made in the positions or in the slopes, and whether the steps' errors add up
coherently or partly cancel.

Nothing is recomputed here. Block F's `anatomy_xy.py` already takes a run
folder (`--run`) and a label (`--label`) on the command line, which is all this
needs; this script imports it by path, points its output folder here, and calls
its `main` once per run. Nothing under Block F is touched or written to.

Two things to read with care, both inherited unchanged from the imported code:
  * the band medians in the output (`final_band_med_um`, `local_step_band`) are
    for 10-50 GeV, a fixed reference band, NOT for this study's loss window;
  * `local_step` is a SINGLE-STEP error (one application from the reference
    state), while `final_med_um` is the ENDPOINT error after the full chain.

About 30-40 s a network.

Run:  PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python anatomy_xy.py [--runs DIR] [--out DIR]
Outputs: <out>/results/anatomy_xy_<N064_q02>.json, one per network
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
    os.makedirs(os.path.join(a.out, "results"), exist_ok=True)
    mod = load_f2("anatomy_xy")
    mod.HERE = os.path.abspath(a.out)
    for n in nets:
        print("\n=== %s (%s; loss window %s) ===" % (rd.label(n), rd.STUDY, rd.window_text([n])))
        mod.main(["--run", n["path"], "--label", n["tag"]])
    print("\nwrote " + ", ".join(os.path.join(a.out, "results", "anatomy_xy_%s.json" % n["tag"])
                                 for n in nets))


if __name__ == "__main__":
    main()
