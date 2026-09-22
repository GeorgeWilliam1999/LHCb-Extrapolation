#!/usr/bin/env python
"""E4 - regenerate every E3 figure for the write-up, without the block label in its title.

George's standing instruction (2026-09-21): an output names a network by what it is -
step count, stage count, step length - never by a block letter. Several E3 figures carry
"Block E" in their suptitle or in an axis title, written into the E3 source.

E3's scripts are NOT copied and NOT edited. Each is imported from `../E3_Analysis/` and run
with two of its module constants pointed elsewhere, exactly as
`../../Block_F_reweighted_loss/F3_Analysis/run_e3_for_block_f.py` does for Block F:

    HERE -> `regen/` beside this file, so the regenerated results/ and figures/ land there
            and nothing in E3_Analysis/results is touched
    E1   -> `runs_stopped_2026-09-18/` (a view: runs_stopped_2026-09-18/results/N<NNN>_q<qq>
            are symlinks to E1_Network_grid/results/N<NNN>_q<qq>/stopped_2026-09-18/)

    E2, BD and TRACKS stay as imported.

**Why the snapshot and not the run folders.** The E3 tables and figures on disk describe the
checkpoints of 2026-09-18, when training was stopped by hand; the run folders now hold the
networks extended past that point, so running E3's scripts against them gives different
(better) numbers. Pointing at `stopped_2026-09-18/` reproduces the published E3 numbers, so
the write-up's figures and its tables describe the same networks. `check_against_e3()`
verifies that cell by cell and prints any disagreement.

matplotlib's `Figure.suptitle` and `Axes.set_title` are wrapped so that
"Block E: <sentence>" becomes "<Sentence>" and "Block E case study" becomes "The case study";
nothing else about the drawing changes.

The figures are then copied to `../E3_Analysis/figures_writeup/`, which is what the Notion
page links to.

Run:     PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python make_figures.py [--only tables,along_z]
Outputs: regen/results/*.csv, regen/figures/*.png, ../E3_Analysis/figures_writeup/*.png,
         regen/run_log.json
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"

import argparse          # noqa: E402
import csv               # noqa: E402
import glob              # noqa: E402
import importlib.util    # noqa: E402
import json              # noqa: E402
import re                # noqa: E402
import shutil            # noqa: E402
import sys               # noqa: E402
import time              # noqa: E402
import traceback         # noqa: E402

import matplotlib        # noqa: E402
matplotlib.use("Agg")
import matplotlib.axes    # noqa: E402
import matplotlib.figure  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
E3 = os.path.abspath(os.path.join(HERE, "..", "E3_Analysis"))
E1_RESULTS = os.path.abspath(os.path.join(HERE, "..", "E1_Network_grid", "results"))
VIEW = os.path.join(HERE, "runs_stopped_2026-09-18")
REGEN = os.path.join(HERE, "regen")
OUT_FIGS = os.path.join(E3, "figures_writeup")
SNAPSHOT = "stopped_2026-09-18"

ORDER = ("convergence", "tables", "single_step_tables", "along_z", "evaluate_splits",
         "errors_vs_momentum", "case_study", "case_study_3d", "error_anatomy")


# ------------------------------------------------------------------ the view --
def build_view():
    """runs_stopped_2026-09-18/results/N<NNN>_q<qq> -> each run's 2026-09-18 snapshot."""
    view = os.path.join(VIEW, "results")
    os.makedirs(view, exist_ok=True)
    for old in glob.glob(os.path.join(view, "N*_q*")):
        os.unlink(old)
    made = []
    for run in sorted(glob.glob(os.path.join(E1_RESULTS, "N[0-9][0-9][0-9]_q[0-9][0-9]"))):
        snap = os.path.join(run, SNAPSHOT)
        need = ("record.json", "network.pt", "scale.json", "chain_states.npz",
                "history.csv", "rounds.csv")
        missing = [f for f in need if not os.path.exists(os.path.join(snap, f))]
        if missing:
            raise SystemExit("%s: snapshot incomplete, missing %s" % (run, ", ".join(missing)))
        os.symlink(os.path.relpath(snap, view), os.path.join(view, os.path.basename(run)))
        made.append(os.path.basename(run))
    return made


# -------------------------------------------------------- the title wrapper --
def _clean(t):
    if not isinstance(t, str):
        return t
    t = t.replace("Block E case study in 3D", "The case study in three dimensions")
    t = t.replace("Block E case study", "The case study")
    new = re.sub(r"^Block E:\s*", "", t)
    if new != t and new:
        new = new[0].upper() + new[1:]
    return new


def relabel_titles():
    for cls, name in ((matplotlib.figure.Figure, "suptitle"), (matplotlib.axes.Axes, "set_title")):
        orig = getattr(cls, name)
        if getattr(orig, "_relabelled", False):
            continue

        def wrapped(self, t, *a, _orig=orig, **k):
            return _orig(self, _clean(t), *a, **k)
        wrapped._relabelled = True
        setattr(cls, name, wrapped)


# ------------------------------------------------------------ running E3's code --
def load_e3(name):
    if E3 not in sys.path:
        sys.path.insert(0, E3)
    spec = importlib.util.spec_from_file_location("e3_" + name, os.path.join(E3, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def point_at_snapshot(mod):
    mod.HERE = REGEN
    if hasattr(mod, "E1"):
        # two conventions in E3: E1 = ".../E1_Network_grid" (joined with "results" later)
        # or E1 = ".../E1_Network_grid/results"
        mod.E1 = VIEW if mod.E1.rstrip("/").endswith("E1_Network_grid") else os.path.join(VIEW, "results")
    return mod


# ------------------------------------------------- did we reproduce E3's tables? --
CHECK = {
    "error_qdz_chain.csv": ("N", "q", ["med_um", "p95_um", "exact_med_um", "block_d_med_um"]),
    "error_qdz_single_step.csv": ("N", "q", ["step_med_um", "step_p95_um", "chain_med_um"]),
    "cost_accuracy.csv": ("N", "q", ["med_um", "n_parameters"]),
    "tails.csv": ("N", "q", ["med_um", "p95_um", "p99_um"]),
}


def check_against_e3(tol=1e-9):
    """Cell-by-cell comparison of the regenerated tables with E3's published ones."""
    out = {}
    for name, (k1, k2, cols) in CHECK.items():
        a = os.path.join(REGEN, "results", name)
        b = os.path.join(E3, "results", name)
        if not (os.path.exists(a) and os.path.exists(b)):
            continue
        ra = {(r[k1], r[k2]): r for r in csv.DictReader(open(a))}
        rb = {(r[k1], r[k2]): r for r in csv.DictReader(open(b))}
        bad, n = [], 0
        for key in sorted(set(ra) & set(rb)):
            for c in cols:
                x, y = ra[key].get(c), rb[key].get(c)
                if x is None or y is None:
                    continue
                n += 1
                try:
                    fx, fy = float(x), float(y)
                except ValueError:
                    continue
                if fx != fx and fy != fy:       # both nan
                    continue
                if abs(fx - fy) > tol * max(1.0, abs(fy)):
                    bad.append(dict(N=key[0], q=key[1], column=c, regenerated=fx, e3=fy))
        out[name] = dict(cells=n, mismatches=bad)
        print("check %-30s %4d cells, %d mismatches" % (name, n, len(bad)), flush=True)
        for m in bad[:8]:
            print("      N=%s q=%s %s: %.6g vs E3 %.6g" % (m["N"], m["q"], m["column"],
                                                           m["regenerated"], m["e3"]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", default=None, help="comma-separated subset of: " + ", ".join(ORDER))
    a = ap.parse_args(argv)
    todo = a.only.split(",") if a.only else list(ORDER)
    os.makedirs(os.path.join(REGEN, "results"), exist_ok=True)
    os.makedirs(os.path.join(REGEN, "figures"), exist_ok=True)
    os.makedirs(OUT_FIGS, exist_ok=True)
    log = {"snapshot": SNAPSHOT, "runs_in_the_view": build_view(), "ran": {}}
    print("view: %d runs at their %s checkpoint" % (len(log["runs_in_the_view"]), SNAPSHOT), flush=True)
    relabel_titles()
    os.chdir(REGEN)
    for name in todo:
        t0 = time.time()
        try:
            mod = point_at_snapshot(load_e3(name))
            mod.main()
            log["ran"][name] = dict(ok=True, wall_s=round(time.time() - t0, 1))
            print("OK   %-22s %.0f s" % (name, time.time() - t0), flush=True)
        except Exception as e:                      # noqa: BLE001 - report and carry on
            log["ran"][name] = dict(ok=False, error="%s: %s" % (type(e).__name__, e),
                                    trace=traceback.format_exc().splitlines()[-4:],
                                    wall_s=round(time.time() - t0, 1))
            print("FAIL %-22s %s: %s" % (name, type(e).__name__, e), flush=True)
    copied = []
    for png in sorted(glob.glob(os.path.join(REGEN, "figures", "*.png"))):
        shutil.copy2(png, os.path.join(OUT_FIGS, os.path.basename(png)))
        copied.append(os.path.basename(png))
    log["figures_copied_to_figures_writeup"] = copied
    log["reproduction_check"] = check_against_e3()
    with open(os.path.join(REGEN, "run_log.json"), "w") as f:
        json.dump(log, f, indent=1)
    print("copied %d figures to %s" % (len(copied), OUT_FIGS))
    return log


if __name__ == "__main__":
    main()
