#!/usr/bin/env python
"""F4 - redraw the F3 figures the write-up uses, without the block letter.

The write-up names a network by what it is (N steps, q stages, dz mm) and never
by the folder the study lives in. Four of the figures the page wants are drawn
by F3's runner, which wraps matplotlib so that E3's hardcoded "Block E" reads
"Block F" here. This script does the same trick the other way: it wraps
`Figure.suptitle` and `Axes.set_title` so that "Block E: " / "Block F: " is
dropped from the title altogether, then runs the same drawing code on the same
runs. It is modelled on `../F3_Analysis/run_e3_for_block_f.py`, and like that
runner it IMPORTS the analysis modules rather than copying or editing them.

Nothing under `_shared/`, Block E, F0, F1, F2 or F3's `results/` is written.
The analysis modules write their CSVs beside their figures, so this script
points their `HERE` at a scratch folder (`results/figure_rebuild/` under this
folder) seeded with a copy of F3's results, lets them write there, and then
copies only the finished figures into `../F3_Analysis/figures_writeup/`.

Figures produced (all on the three finished runs, 1,452 test tracks):

  convergence_grid.png            training loss and validation error per round
  error_vs_z.png                  how the endpoint error grows along the crossing
  single_step.png                 one step from an RK6 state on every start plane
  error_vs_p_{x,y,tx,ty}.png      endpoint error against momentum, per component
  case_study_components_x0_maps.png   endpoint error against momentum and
                                  starting x, for the validation-best network

Run:  PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python make_figures.py
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"

import sys               # noqa: E402
sys.dont_write_bytecode = True

# this folder holds a file called numbers.py, which shadows the standard-library
# module of that name that numpy imports; take the folder off the import path
_HERE0 = os.path.dirname(os.path.abspath(__file__))
for _p in ("", ".", _HERE0):
    while _p in sys.path:
        sys.path.remove(_p)

import argparse          # noqa: E402
import glob              # noqa: E402
import importlib.util    # noqa: E402
import shutil            # noqa: E402
import time              # noqa: E402
import traceback         # noqa: E402

import matplotlib        # noqa: E402
matplotlib.use("Agg")
import matplotlib.axes    # noqa: E402
import matplotlib.figure  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BLOCK = os.path.abspath(os.path.join(HERE, ".."))
F3 = os.path.join(BLOCK, "F3_Analysis")
E3 = os.path.abspath(os.path.join(BLOCK, "..", "Block_E_single_network_chain", "E3_Analysis"))
VIEW = os.path.join(F3, "runs")                  # F3's symlink view of the finished runs
OUT = os.path.join(F3, "figures_writeup")
WORK = os.path.join(HERE, "results", "figure_rebuild")

# the block letter, however it is spelled in a title, comes out
STRIP = (("Block E case study in 3D", "Case study in 3D"),
         ("Block F case study in 3D", "Case study in 3D"),
         ("Block E: ", ""), ("Block F: ", ""), ("Block D: ", ""),
         ("Block E", ""), ("Block F", ""))

# name -> the module that draws it; "grid_free" means F3's own per-run drawing
JOBS = (("convergence", "grid_free"),
        ("along_z", "grid_free"),
        ("single_step_tables", "grid_free"),
        ("errors_vs_momentum", "grid_free"),
        ("case_study_3d", "e3"))

WANTED = ("convergence_grid.png", "error_vs_z.png", "single_step.png",
          "error_vs_p_x.png", "error_vs_p_y.png", "error_vs_p_tx.png", "error_vs_p_ty.png",
          "case_study_components_x0_maps.png")


def strip_block_letters():
    """Wrap matplotlib's title setters so no figure carries a block letter."""
    def clean(t):
        if not isinstance(t, str):
            return t
        for a, b in STRIP:
            t = t.replace(a, b)
        return t.lstrip(": ").strip()

    for cls, name in ((matplotlib.figure.Figure, "suptitle"), (matplotlib.axes.Axes, "set_title")):
        orig = getattr(cls, name)
        if getattr(orig, "_block_letters_stripped", False):
            continue

        def wrapped(self, t, *a, _orig=orig, **k):
            return _orig(self, clean(t), *a, **k)
        wrapped._block_letters_stripped = True
        setattr(cls, name, wrapped)


def seed_work():
    """A scratch copy of F3's results, so the drawing code writes nothing real."""
    os.makedirs(os.path.join(WORK, "results"), exist_ok=True)
    os.makedirs(os.path.join(WORK, "figures"), exist_ok=True)
    for src in glob.glob(os.path.join(F3, "results", "*")):
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(WORK, "results", os.path.basename(src)))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", default=None, help="comma-separated subset of: "
                                                 + ", ".join(n for n, _ in JOBS))
    a = ap.parse_args(argv)
    todo = a.only.split(",") if a.only else [n for n, _ in JOBS]

    seed_work()
    strip_block_letters()
    if E3 not in sys.path:
        sys.path.insert(0, E3)
    os.chdir(F3)                                  # the drawing code expects F3's cwd

    for name, kind in JOBS:
        if name not in todo:
            continue
        t0 = time.time()
        try:
            if kind == "grid_free":
                gf = load(os.path.join(F3, "grid_free_panels.py"), "grid_free_panels")
                gf.HERE = WORK
                gf.RUNS = os.path.join(VIEW, "results")
                getattr(gf, name)()
            else:
                mod = load(os.path.join(E3, name + ".py"), "e3_" + name)
                mod.HERE = WORK
                if hasattr(mod, "E1"):
                    mod.E1 = (VIEW if mod.E1.rstrip("/").endswith("E1_Network_grid")
                              else os.path.join(VIEW, "results"))
                if hasattr(mod, "CASE"):
                    mod.CASE = None
                mod.main()
            print("OK   %-22s %.0f s" % (name, time.time() - t0), flush=True)
        except Exception as e:                    # noqa: BLE001 - report and carry on
            print("FAIL %-22s %s: %s" % (name, type(e).__name__, e), flush=True)
            print("     " + " | ".join(traceback.format_exc().splitlines()[-3:]), flush=True)

    os.makedirs(OUT, exist_ok=True)
    made = []
    for fn in WANTED:
        src = os.path.join(WORK, "figures", fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(OUT, fn))
            made.append(fn)
        else:
            print("MISSING %s" % fn, flush=True)
    print("\nwrote %d figures into %s:\n  %s" % (len(made), OUT, "\n  ".join(made)))
    return made


if __name__ == "__main__":
    main()
