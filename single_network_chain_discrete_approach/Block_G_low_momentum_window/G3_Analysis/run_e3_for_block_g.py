#!/usr/bin/env python
"""G3 - the whole Block E analysis set (E3) reproduced for the Block G networks.

The same pattern as `Block_F_reweighted_loss/F3_Analysis/run_e3_for_block_f.py`,
and for the same reason: E3's scripts are NOT copied. Each is imported from
`../../Block_E_single_network_chain/E3_Analysis/` and run with a few of its
module constants pointed elsewhere, so the plots are the same code on different
runs:

    HERE   -> the output folder, so results/ and figures/ are written there
              (and the scripts that read results/convergence.csv to pick the
              network to study in depth read this block's one)
    E1     -> <out>/runs/ (a view: <out>/runs/results/N<NNN>_q<qq> are symlinks
              to the runs that have a record and stored chain states, rebuilt on
              every call, because the E3 scripts glob run folders and expect a
              record in each)
    BD     -> <out>/comparators/ (a view holding ONLY the exact-scheme states,
              so the exact scheme at the same N and q is still measured while
              the OTHER study's trained chains are not: George asked for this
              block in isolation, with no comparison rows against another
              block's networks). TRACKS and the straight-line and material-floor
              comparators stay as imported.

Four E3 scripts lay their figures out over the full N x q grid (every N assumed
to have every q). This study has three runs on a diagonal, so those four come
from `../../Block_F_reweighted_loss/F3_Analysis/grid_free_panels.py`, imported
with its own HERE and RUNS pointed here - the same computation and the same
per-panel drawing, one panel or one series per run.

WORDING. E3's and F3's figure titles and printed tables carry block letters in
their source ("Block E: ...", "Block F: ..."). An output a human reads must not
(George, 2026-09-21), so this runner wraps matplotlib's suptitle and set_title
and the built-in print, and rewrites the CSV headers and cells it produces, so
that those read "one network per step length, chained; loss window <lo>-<hi>
GeV" instead - the window taken from each run's scale.json, never typed in. The
drawing itself is not changed. `--no-relabel` turns the rewriting off, for
checking what the imported code said.

Run:  PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python run_e3_for_block_g.py [--runs DIR] [--out DIR] [--only tables,along_z]
Outputs: <out>/results/*.csv, <out>/figures/*.png (the E3 names), <out>/results/run_log.json
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"

# importing a script out of a read-only folder must not leave a .pyc behind in it
import sys           # noqa: E402
sys.dont_write_bytecode = True

import argparse          # noqa: E402
import builtins          # noqa: E402
import csv               # noqa: E402
import glob              # noqa: E402
import importlib.util    # noqa: E402
import inspect           # noqa: E402
import json              # noqa: E402
import textwrap          # noqa: E402
import time              # noqa: E402
import traceback         # noqa: E402

import matplotlib        # noqa: E402
matplotlib.use("Agg")
import matplotlib.axes    # noqa: E402
import matplotlib.figure  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
G2 = os.path.abspath(os.path.join(HERE, "..", "G2_Analysis"))
if G2 not in sys.path:
    sys.path.insert(0, G2)
import run_discovery as rd   # noqa: E402

E3 = os.path.abspath(os.path.join(HERE, "..", "..", "Block_E_single_network_chain", "E3_Analysis"))
F3 = os.path.abspath(os.path.join(HERE, "..", "..", "Block_F_reweighted_loss", "F3_Analysis"))
BD_SRC = os.path.abspath(os.path.join(HERE, "..", "..", "..", "multi_network_chain_discrete_approach",
                                      "Block_D_fixed_step_crossing"))
DEFAULT_RUNS = os.path.join(HERE, "..", "G1_Training", "results", "p03-08", "full")

# the order matters: convergence.csv is what the in-depth case study and the
# error anatomy read to pick the network to study
ORDER = ("convergence", "tables", "single_step_tables", "along_z", "evaluate_splits",
         "errors_vs_momentum", "case_study", "case_study_3d", "error_anatomy")
GRID_BOUND = ("convergence", "errors_vs_momentum", "along_z", "single_step_tables")


def build_view(out, runs_dir):
    """<out>/runs/results/<run> -> the runs that can be analysed, and nothing else."""
    view = os.path.join(out, "runs", "results")
    os.makedirs(view, exist_ok=True)
    for old in glob.glob(os.path.join(view, "N*_q*")):
        os.unlink(old)
    found = []
    for r in rd.find_runs(runs_dir):
        os.symlink(os.path.relpath(r["path"], view), os.path.join(view, r["tag"]))
        found.append(rd.label(r))
    return found


def build_comparator_view(out):
    """<out>/comparators/ -> only the exact-scheme states of the shared comparator set.

    The E3 scripts look for two things under this folder: `D2_Comparators/
    results/exact_N<NNN>_q<qq>_states.npz` (the q-stage collocation scheme
    chained without a network - a numerical-scheme comparator, kept) and
    `D1_Chain_grid/results/N<NNN>_q<qq>/states.npz` (another study's trained
    chains - a comparison against other networks, dropped). Linking only the
    first keeps the one and removes the other without touching either folder.
    """
    view = os.path.join(out, "comparators")
    os.makedirs(view, exist_ok=True)
    link = os.path.join(view, "D2_Comparators")
    if os.path.islink(link):
        os.unlink(link)
    if not os.path.exists(link):
        os.symlink(os.path.join(BD_SRC, "D2_Comparators"), link)
    return view


def text_map(study):
    """(what to look for, what to write instead, what means it is already done).

    Longest pattern first, because they are applied in order. The third entry
    guards against a second pass over the same text (the CSVs are rewritten on
    every run): a rule is skipped when its result is already there.
    """
    return ((u"Block D: one network per step, same N and q",
             u"a separate network for each step, same N and q", None),
            (u"Block D's one-network-per-step chain", u"a separate network for each step", None),
            (u"Block D", u"a separate network per step", None),
            (u"Block E case study in 3D", study + u", in depth in 3D", None),
            (u"Block E case study", study + u", in depth", None),
            (u"Block E", study, None),
            (u"Block F", study, None),
            # the E3 titles that say this mean the error after the whole chain;
            # an output has to say which kind of error it is showing
            (u"error at the SciFi plane", u"endpoint error at the SciFi plane, after the full chain",
             u"endpoint error at the SciFi plane"))


TITLE_WRAP = 110         # characters; the study phrase is longer than "Block E" was


def relabel(study):
    """Wrap matplotlib's titles and the built-in print; return the undo."""
    subs = text_map(study)

    def fix(t):
        if not isinstance(t, str):
            return t
        for a, b, done in subs:
            if a in t and not (done and done in t):
                t = t.replace(a, b)
        return t

    def fix_title(t):
        """The same, and then folded to a width that fits the canvas.

        The replacement wording is much longer than the block letter it
        replaces, and a long single-line title is drawn straight off the edge
        of the figure, so a title over TITLE_WRAP characters is wrapped. Only
        titles are wrapped; printed text is not.
        """
        t = fix(t)
        if isinstance(t, str) and len(t) > TITLE_WRAP:
            t = "\n".join(textwrap.fill(line, TITLE_WRAP) for line in t.split("\n"))
        return t

    undo = []
    for cls, name in ((matplotlib.figure.Figure, "suptitle"), (matplotlib.axes.Axes, "set_title")):
        orig = getattr(cls, name)

        def wrapped(self, t, *a, _orig=orig, **k):
            return _orig(self, fix_title(t), *a, **k)
        setattr(cls, name, wrapped)
        undo.append((cls, name, orig))
    orig_print = builtins.print

    def printed(*a, **k):
        return orig_print(*[fix(x) for x in a], **k)
    builtins.print = printed
    undo.append((builtins, "print", orig_print))

    def restore():
        for obj, name, orig in undo:
            setattr(obj, name, orig)
    return restore, fix


def sanitise_csvs(out, fix):
    """Rewrite the CSVs this runner produced so no header or cell carries a block letter.

    A column of the imported tables holds the other study's chains, which this
    block does not draw; it is empty here, so it is dropped. Anything else is
    reworded, never removed.
    """
    renames = {"block_d_med_um": "separate_network_per_step_med_um"}
    touched = []
    for p in sorted(glob.glob(os.path.join(out, "results", "*.csv"))):
        with open(p, newline="") as f:
            rows = list(csv.reader(f))
        if not rows:
            continue
        head, body = rows[0], rows[1:]
        drop = {i for i, c in enumerate(head)
                if c in renames and all(r[i].strip().lower() in ("", "nan", "-") for r in body if i < len(r))}
        new_head = [renames.get(c, fix(c)) for i, c in enumerate(head) if i not in drop]
        new_body = [[fix(v) for i, v in enumerate(r) if i not in drop] for r in body]
        if [new_head] + new_body == rows:
            continue
        with open(p, "w", newline="") as f:
            csv.writer(f).writerows([new_head] + new_body)
        touched.append(os.path.basename(p))
    return touched


def load_module(folder, name, as_name):
    if folder not in sys.path:
        sys.path.insert(0, folder)
    spec = importlib.util.spec_from_file_location(as_name, os.path.join(folder, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def point_here(mod, out, view, comparators):
    mod.HERE = out
    if hasattr(mod, "E1"):
        # two conventions in E3: E1 = ".../E1_Network_grid" (then joined with
        # "results") or E1 = ".../E1_Network_grid/results"
        mod.E1 = (os.path.join(out, "runs") if mod.E1.rstrip("/").endswith("E1_Network_grid")
                  else view)
    if hasattr(mod, "BD"):
        mod.BD = comparators
    if hasattr(mod, "CASE"):
        mod.CASE = None
    return mod


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", default=DEFAULT_RUNS, help="folder holding the N<NNN>_q<qq> run folders")
    ap.add_argument("--out", default=HERE, help="where runs/, results/ and figures/ are written")
    ap.add_argument("--only", default=None, help="comma-separated subset of: " + ", ".join(ORDER))
    ap.add_argument("--case", default=None, metavar="N,q",
                    help="the network the in-depth study, the 3D study and the error anatomy use "
                         "(e.g. 64,2). The default is the imported code's own choice: the best "
                         "mean validation error over the last eight rounds, which can be a run "
                         "that is still training.")
    ap.add_argument("--no-relabel", action="store_true",
                    help="leave the imported code's own wording (block letters and all) in place")
    a = ap.parse_args(argv)
    out = os.path.abspath(a.out)
    todo = a.only.split(",") if a.only else list(ORDER)
    unknown = [t for t in todo if t not in ORDER]
    if unknown:
        raise SystemExit("unknown analysis %s; choose from %s" % (", ".join(unknown), ", ".join(ORDER)))

    nets = rd.require(rd.find_runs(a.runs), a.runs)
    study = rd.study_text(nets)
    os.makedirs(os.path.join(out, "results"), exist_ok=True)
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    view = build_view(out, a.runs)
    comparators = build_comparator_view(out)
    print(rd.describe(nets, a.runs))
    print("networks in the view: %s" % "; ".join(view))
    print("output: %s" % out)
    log = {"study": study, "runs_dir": os.path.abspath(a.runs), "out": out,
           "networks": view, "relabelled": not a.no_relabel}

    case_argv = []
    if a.case:
        try:
            cn, cq = [x.strip() for x in a.case.split(",")]
            case_argv = ["--N", str(int(cn)), "--q", str(int(cq))]
        except ValueError:
            raise SystemExit("--case wants N,q - e.g. --case 64,2")
        log["case_network"] = "N = %s, q = %s" % (cn, cq)

    restore, fix = (lambda: None, lambda t: t) if a.no_relabel else relabel(study)
    cwd = os.getcwd()
    os.chdir(out)
    try:
        for name in todo:
            t0 = time.time()
            try:
                if name in GRID_BOUND:
                    gf = load_module(F3, "grid_free_panels", "g3_grid_free_panels")
                    gf.HERE = out
                    gf.RUNS = os.path.join(out, "runs", "results")
                    getattr(gf, name)()
                else:
                    mod = point_here(load_module(E3, name, "g3_" + name), out,
                                     os.path.join(out, "runs", "results"), comparators)
                    # some E3 scripts take their own command line (--N/--q, which
                    # default to the network picked from results/convergence.csv);
                    # give them an empty one rather than this runner's flags
                    if "argv" in inspect.signature(mod.main).parameters:
                        mod.main(case_argv)
                    else:
                        mod.main()
                log[name] = dict(ok=True, wall_s=round(time.time() - t0, 1))
                print("OK   %-22s %.0f s" % (name, time.time() - t0), flush=True)
            except Exception as e:      # noqa: BLE001 - report and carry on
                log[name] = dict(ok=False, error="%s: %s" % (type(e).__name__, e),
                                 trace=traceback.format_exc().splitlines()[-4:],
                                 wall_s=round(time.time() - t0, 1))
                print("FAIL %-22s %s: %s" % (name, type(e).__name__, e), flush=True)
    finally:
        restore()
        os.chdir(cwd)

    reworded = [] if a.no_relabel else sanitise_csvs(out, fix)
    # a partial re-run (--only) keeps what the earlier runs recorded
    log_path = os.path.join(out, "results", "run_log.json")
    if os.path.exists(log_path):
        try:
            with open(log_path) as f:
                previous = json.load(f)
            reworded = sorted(set(reworded) | set(previous.get("csvs_reworded", [])))
            previous.update(log)
            log = previous
        except ValueError:
            pass
    log["csvs_reworded"] = reworded
    log["written"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=1)
    ok = [k for k in todo if log.get(k, {}).get("ok")]
    print("\n%d of %d analyses ran: %s" % (len(ok), len(todo), ", ".join(ok)))
    bad = [k for k in todo if not log.get(k, {}).get("ok")]
    if bad:
        print("did not run: %s" % ", ".join("%s (%s)" % (k, log[k]["error"]) for k in bad))
    print("wrote %s" % os.path.join(out, "results", "run_log.json"))
    return log


if __name__ == "__main__":
    main()
