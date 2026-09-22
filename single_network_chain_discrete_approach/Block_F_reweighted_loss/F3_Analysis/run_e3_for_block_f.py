#!/usr/bin/env python
"""F3 - every Block E analysis (E3) reproduced for the Block F networks.

George (2026-09-20): "reproduce all the analysis plots from block E for Block F".

E3's scripts are NOT copied. Each is imported from
`../../Block_E_single_network_chain/E3_Analysis/` and run with three of its
module constants pointed elsewhere, so the plots are the same code on
different runs:

    HERE   -> this folder, so results/ and figures/ are written here (and the
              scripts that read results/convergence.csv to pick the best
              network read the Block F one)
    E1     -> runs/ (a view: runs/results/N<NNN>_q<qq> are symlinks to the
              FINISHED runs in F1_Training/results/full, rebuilt on every call,
              because the E3 scripts glob run folders and expect a record in
              each; a run still training has none)
    E2, BD, TRACKS stay as imported: the exact scheme, Block D and the tracks
              are the same comparators for both blocks

Four E3 scripts lay their figures out over the N x q grid (every N assumed to
have every q); Block F has three runs on a diagonal, so those four are
replaced by `grid_free_panels.py` here: the same computation and per-panel
drawing, one panel or one series per run. The
over-training check evaluates Block E's loss on the network, which for Block
F is not the trained loss; its chain-error test/train ratio is what to read.
E3's figure titles say "Block E" in their source; the runner wraps
matplotlib's suptitle and set_title so that those read "Block F" here, and
changes nothing else about the drawing.

Run:     PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python run_e3_for_block_f.py [--only tables,along_z]
Outputs: results/*.csv, figures/*.png (the E3 names), results/run_log.json
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"

import argparse          # noqa: E402
import importlib.util    # noqa: E402
import json              # noqa: E402
import sys               # noqa: E402
import time              # noqa: E402
import traceback         # noqa: E402

import glob              # noqa: E402

import matplotlib        # noqa: E402
matplotlib.use("Agg")
import matplotlib.axes   # noqa: E402
import matplotlib.figure # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
E3 = os.path.abspath(os.path.join(HERE, "..", "..", "Block_E_single_network_chain", "E3_Analysis"))
RUNS = os.path.join(HERE, "runs")
FULL = os.path.abspath(os.path.join(HERE, "..", "F1_Training", "results", "full"))


def build_view():
    """runs/results/<run> -> the finished Block F runs, and nothing else."""
    view = os.path.join(RUNS, "results")
    os.makedirs(view, exist_ok=True)
    for old in glob.glob(os.path.join(view, "N*_q*")):
        os.unlink(old)
    done = []
    for run in sorted(glob.glob(os.path.join(FULL, "N[0-9][0-9][0-9]_q[0-9][0-9]"))):
        if os.path.exists(os.path.join(run, "record.json")) and os.path.exists(os.path.join(run, "chain_states.npz")):
            os.symlink(os.path.relpath(run, view), os.path.join(view, os.path.basename(run)))
            prog = json.load(open(os.path.join(run, "progress.json")))
            tag = os.path.basename(run)
            if prog.get("phase") != "done":
                # a run that hit its cap wrote a record, then the keeper reopened
                # it with --extend: the record is a CHECKPOINT, and the weights
                # on disk are ahead of it
                tag += " (CHECKPOINT at restart %d: the run is at %d and still training)" % (
                    json.load(open(os.path.join(run, "record.json")))["restarts"], prog.get("restart", -1))
            done.append(tag)
    return done


def relabel_titles():
    """E3's titles say "Block E"; the same drawing here is of Block F."""
    for cls, name in ((matplotlib.figure.Figure, "suptitle"), (matplotlib.axes.Axes, "set_title")):
        orig = getattr(cls, name)
        if getattr(orig, "_relabelled", False):
            continue

        def wrapped(self, t, *a, _orig=orig, **k):
            return _orig(self, t.replace("Block E", "Block F") if isinstance(t, str) else t, *a, **k)
        wrapped._relabelled = True
        setattr(cls, name, wrapped)
# the order matters: convergence.csv is what the case study and the anatomy
# read to pick the best network
ORDER = ("convergence", "tables", "single_step_tables", "along_z", "evaluate_splits",
         "errors_vs_momentum", "case_study", "case_study_3d", "error_anatomy")
GRID_BOUND = {"convergence": "grid_free_panels", "errors_vs_momentum": "grid_free_panels",
              "along_z": "grid_free_panels", "single_step_tables": "grid_free_panels"}


def load_e3(name):
    if E3 not in sys.path:
        sys.path.insert(0, E3)
    spec = importlib.util.spec_from_file_location("e3_" + name, os.path.join(E3, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def point_at_block_f(mod):
    mod.HERE = HERE
    if hasattr(mod, "E1"):
        # two conventions in E3: E1 = ".../E1_Network_grid" (then joined with
        # "results") or E1 = ".../E1_Network_grid/results"
        mod.E1 = RUNS if mod.E1.rstrip("/").endswith("E1_Network_grid") else os.path.join(RUNS, "results")
    if hasattr(mod, "CASE"):
        mod.CASE = None
    return mod


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", default=None, help="comma-separated subset of: " + ", ".join(ORDER))
    a = ap.parse_args(argv)
    todo = a.only.split(",") if a.only else list(ORDER)
    log = {"finished_runs": build_view()}
    print("finished Block F runs in the view: %s" % ", ".join(log["finished_runs"]), flush=True)
    relabel_titles()
    os.chdir(HERE)
    for name in todo:
        t0 = time.time()
        try:
            if name in GRID_BOUND:
                spec = importlib.util.spec_from_file_location("grid_free_panels",
                                                              os.path.join(HERE, "grid_free_panels.py"))
                gf = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(gf)
                getattr(gf, name)()
            else:
                mod = point_at_block_f(load_e3(name))
                mod.main()
            log[name] = dict(ok=True, wall_s=round(time.time() - t0, 1))
            print("OK   %-22s %.0f s" % (name, time.time() - t0), flush=True)
        except Exception as e:      # noqa: BLE001 - report and carry on
            log[name] = dict(ok=False, error="%s: %s" % (type(e).__name__, e),
                             trace=traceback.format_exc().splitlines()[-4:], wall_s=round(time.time() - t0, 1))
            print("FAIL %-22s %s: %s" % (name, type(e).__name__, e), flush=True)
    with open(os.path.join(HERE, "results", "run_log.json"), "w") as f:
        json.dump(log, f, indent=1)
    return log


if __name__ == "__main__":
    main()
