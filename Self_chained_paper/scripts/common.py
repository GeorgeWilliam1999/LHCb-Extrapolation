#!/usr/bin/env python
"""Self_chained_paper - the loaders, conventions and statistics the paper uses.

This module is the single place where the paper's conventions live.  Nothing
here writes into the experiment folders; every path below is opened read-only.

CONVENTIONS THE PAPER FIXES (and which differ from the existing analysis code)

1.  SLOPES ARE DIMENSIONLESS.  t_x = dx/dz and t_y = dy/dz are pure numbers.
    Every slope error this module returns is the raw difference (e.g. 7.3e-7).
    It is NEVER multiplied by 1e3 and no key or column is ever called "mrad".
    The experiment modules `E1_Network_grid/metrics.py`, `E3_Analysis/*.py` and
    `F2_Analysis/*.py` all apply a x1e3 and label the result "mrad"; their slope
    functions are deliberately NOT imported here.  Positions are in micrometres
    (mm x 1e3), which is fine and is what every key with `_um` carries.

2.  MOMENTUM BANDS.  Edges [0, 3, 8, 20, 50, inf) GeV labelled "<3", "3-8",
    "8-20", "20-50", ">50", plus the extra band "10-50 (loss window)" which is
    reported alongside them wherever per-band numbers are given.  Band
    membership always comes from the per-split `P` array of tracks.npz (GeV),
    never from the stored `PBAND` column (whose edges are the old
    1/2/5/10/25/200 GeV set).

3.  THE REFERENCE is the RK6 truth state stored in tracks.npz: `<split>_truth`
    of shape (n, 257, 5), the field-only order-6 Runge-Kutta track at 0.1 mm
    sampled on the 257-plane grid z0 + k L/256.  A network with N steps has its
    step k end on plane index (k+1) * (256 // N); the endpoint is plane 256,
    i.e. z1 = 7826.0 mm.

4.  STATISTICS.  median |.|, p95 of |.|, signed median, 68 % half-width
    (q84 - q16)/2, RMS, radial = hypot(dx, dy), max-metric = max(|dx|, |dy|).
    Every key names its metric.

Import it as

    import common as C
    D = C.load_tracks()
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"

import csv          # noqa: E402
import json         # noqa: E402
import sys          # noqa: E402

sys.dont_write_bytecode = True          # never leave a .pyc in a read-only folder

import numpy as np  # noqa: E402

# --------------------------------------------------------------------- paths --
HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.abspath(os.path.join(HERE, ".."))
RESULTS = os.path.join(PAPER, "results")
CACHE = os.path.join(RESULTS, "cache")
REPO = os.path.abspath(os.path.join(PAPER, ".."))

S = os.path.join(REPO, "single_network_chain_discrete_approach")
SHARED = os.path.join(S, "_shared")
E = os.path.join(S, "Block_E_single_network_chain")
F = os.path.join(S, "Block_F_reweighted_loss")
G = os.path.join(S, "Block_G_low_momentum_window")
RVT = os.path.join(S, "reference_vs_truth")

E0_RESULTS = os.path.join(E, "E0_Track_dataset", "results")
TRACKS = os.path.join(E0_RESULTS, "tracks.npz")
TRACKS_META = os.path.join(E0_RESULTS, "tracks_meta.json")
BUILD_TRACKS_LOG = os.path.join(E, "E0_Track_dataset", "build_tracks.log")

E1 = os.path.join(E, "E1_Network_grid")
E1_RUNS = os.path.join(E1, "results")
E2_RESULTS = os.path.join(E, "E2_Comparators", "results")
E3_RESULTS = os.path.join(E, "E3_Analysis", "results")

F0 = os.path.join(F, "F0_Weighting")
F0_RESULTS = os.path.join(F0, "results")
F1_RUNS = os.path.join(F, "F1_Training", "results", "full")
F2 = os.path.join(F, "F2_Analysis")
F2_RESULTS = os.path.join(F2, "results")
F3_RESULTS = os.path.join(F, "F3_Analysis", "results")
F4_RESULTS = os.path.join(F, "F4_Writeup", "results")

MNC = os.path.join(REPO, "multi_network_chain_discrete_approach")
C1 = os.path.join(MNC, "Block_C_step_size_and_stages", "C1_Fine_reference")
C1_RESULTS = os.path.join(C1, "results")

GATES_JSON = os.path.join(REPO, "Data_generation_exploration", "Data", "results", "gates.json")

SNAPSHOT = "stopped_2026-09-18"          # the checkpoint folder inside each E1 run

# ------------------------------------------------------------- the geometry --
Z0_MM = 2648.2
Z1_MM = 7826.0
L_MM = 5177.8
N_MAX = 256                              # the 257-plane grid is z0 + k L/256


def dz_mm(N):
    """The step length of an N-step chain [mm]."""
    return L_MM / float(N)


def plane_index(N, k):
    """The index into the 257-plane grid of step k's START plane of an N-chain."""
    return int(k) * (N_MAX // int(N))


# ------------------------------------------------------------- the run grid --
GRID = [(N, q) for N in (2, 64, 128, 256) for q in (2, 4, 8, 16)]
PAIRS = [(64, 2), (128, 8), (256, 16)]   # the three matched pooled / reweighted pairs

SPLITS = ("train", "val", "test")
SPLIT_SIZES = {"train": 11567, "val": 1463, "test": 1452}


def run_tag(N, q):
    return "N%03d_q%02d" % (int(N), int(q))


def run_key(N, q):
    return "N=%d,q=%d" % (int(N), int(q))


def pooled_run(N, q, snapshot=False):
    """The folder of a pooled-loss (Block E) run, extended or the 18 Sept snapshot."""
    p = os.path.join(E1_RUNS, run_tag(N, q))
    return os.path.join(p, SNAPSHOT) if snapshot else p


def reweighted_run(N, q):
    """The folder of a reweighted-loss (Block F) run."""
    return os.path.join(F1_RUNS, run_tag(N, q))


# ------------------------------------------------------------ the house palette --
# paper 1's plot scripts (single_network_chain_discrete_approach/data_supplement/
# make_figures.py and the Block 0/A plot scripts that share the same surface).
SURFACE = "#fcfcfb"
TEXT1 = "#0b0b0b"
TEXT2 = "#52514e"
BLUE = "#2a78d6"
GREEN = "#008300"
MAGENTA = "#e87ba4"
NEUTRAL = "#c9c8c2"
PALETTE = dict(SURFACE=SURFACE, TEXT1=TEXT1, TEXT2=TEXT2, BLUE=BLUE,
               GREEN=GREEN, MAGENTA=MAGENTA, NEUTRAL=NEUTRAL)


# --------------------------------------------------------------- the bands --
BAND_EDGES = (0.0, 3.0, 8.0, 20.0, 50.0, float("inf"))
BAND_LABELS = ("<3", "3-8", "8-20", "20-50", ">50")
LOSS_WINDOW = (10.0, 50.0)
LOSS_WINDOW_LABEL = "10-50 (loss window)"
ALL_BAND_LABELS = tuple(BAND_LABELS) + (LOSS_WINDOW_LABEL,)


def band_masks(P, include_all=False):
    """[(label, mask)] for the five bands plus the loss window, from P in GeV.

    The loss window overlaps the "8-20" and "20-50" bands on purpose: it is
    reported ALONGSIDE the partition, not as part of it.  `include_all` puts an
    "all" row first.
    """
    P = np.asarray(P, dtype=float)
    out = []
    if include_all:
        out.append(("all", np.ones(P.shape, dtype=bool)))
    for lo, hi, label in zip(BAND_EDGES[:-1], BAND_EDGES[1:], BAND_LABELS):
        out.append((label, (P >= lo) & (P < hi)))
    out.append((LOSS_WINDOW_LABEL, (P >= LOSS_WINDOW[0]) & (P < LOSS_WINDOW[1])))
    return out


def band_counts(P):
    """{label: n} for the five bands plus the loss window."""
    return {label: int(m.sum()) for label, m in band_masks(P)}


# --------------------------------------------------------------- statistics --
def med_abs(v):
    """median |v|."""
    v = np.asarray(v, dtype=float)
    return float(np.median(np.abs(v))) if v.size else float("nan")


def p95_abs(v):
    """95th percentile of |v|."""
    v = np.asarray(v, dtype=float)
    return float(np.quantile(np.abs(v), 0.95)) if v.size else float("nan")


def signed_median(v):
    """median of the signed v (not of |v|)."""
    v = np.asarray(v, dtype=float)
    return float(np.median(v)) if v.size else float("nan")


def halfwidth68(v):
    """(q84 - q16) / 2 of the signed v: the 68 % half-width, tail-proof."""
    v = np.asarray(v, dtype=float)
    if v.size == 0:
        return float("nan")
    lo, hi = np.quantile(v, [0.16, 0.84])
    return float((hi - lo) / 2.0)


def rms(v):
    """sqrt(mean(v^2)) of the signed v: tail-pulled, quoted beside the median."""
    v = np.asarray(v, dtype=float)
    return float(np.sqrt(np.mean(v ** 2))) if v.size else float("nan")


def median(v):
    v = np.asarray(v, dtype=float)
    return float(np.median(v)) if v.size else float("nan")


def quantile(v, p):
    v = np.asarray(v, dtype=float)
    return float(np.quantile(v, p)) if v.size else float("nan")


def signed_stats(v, unit):
    """The five statistics of a SIGNED deviation, with the unit in every key.

    `unit` is "um" for a position and "slope" for a dimensionless slope.
    """
    return {
        "n": int(np.asarray(v).size),
        "med_abs_%s" % unit: med_abs(v),
        "p95_abs_%s" % unit: p95_abs(v),
        "signed_med_%s" % unit: signed_median(v),
        "hw68_%s" % unit: halfwidth68(v),
        "rms_%s" % unit: rms(v),
    }


def magnitude_stats(r, unit="um", above_um=1000.0):
    """The statistics of a non-negative magnitude (radial or max-metric)."""
    r = np.asarray(r, dtype=float)
    out = {
        "n": int(r.size),
        "med_%s" % unit: median(r),
        "p95_%s" % unit: quantile(r, 0.95),
        "p99_%s" % unit: quantile(r, 0.99),
        "mean_%s" % unit: float(r.mean()) if r.size else float("nan"),
        "max_%s" % unit: float(r.max()) if r.size else float("nan"),
    }
    if unit == "um":
        out["frac_above_1mm"] = float((r > above_um).mean()) if r.size else float("nan")
        out["n_above_1mm"] = int((r > above_um).sum())
    return out


# ------------------------------------------------------------- error arrays --
COMPONENTS = (("x", 0, "um"), ("y", 1, "um"), ("tx", 2, "slope"), ("ty", 3, "slope"))


def deltas(pred, truth):
    """Signed per-component deviations, in the paper's units.

    pred, truth   (n, >= 4) state arrays in mm and dimensionless slopes.
    Returns {"x": um, "y": um, "tx": dimensionless, "ty": dimensionless,
             "radial_um": hypot(dx, dy), "max_um": max(|dx|, |dy|)}.
    """
    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)
    dx = (pred[:, 0] - truth[:, 0]) * 1e3          # um
    dy = (pred[:, 1] - truth[:, 1]) * 1e3          # um
    dtx = pred[:, 2] - truth[:, 2]                 # dimensionless, NOT x1e3
    dty = pred[:, 3] - truth[:, 3]                 # dimensionless, NOT x1e3
    return {"x": dx, "y": dy, "tx": dtx, "ty": dty,
            "radial_um": np.hypot(dx, dy),
            "max_um": np.maximum(np.abs(dx), np.abs(dy)),
            "slope_max": np.maximum(np.abs(dtx), np.abs(dty))}


def component_block(d, mask=None):
    """Per-component signed statistics of a `deltas` dict, optionally masked."""
    out = {}
    for name, _, unit in COMPONENTS:
        v = d[name] if mask is None else d[name][mask]
        for k, val in signed_stats(v, unit).items():
            if k == "n":
                out["n"] = val
            else:
                out["%s_%s" % (name, k)] = val
    return out


# ----------------------------------------------------------------- loaders --
_TRACKS_CACHE = {}


def load_tracks():
    """tracks.npz, memoised.  Read-only; the file is never rewritten."""
    if "D" not in _TRACKS_CACHE:
        _TRACKS_CACHE["D"] = np.load(TRACKS, allow_pickle=False)
    return _TRACKS_CACHE["D"]


def tracks_meta():
    with open(TRACKS_META) as f:
        return json.load(f)


def split_arrays(split):
    """The arrays of one split of tracks.npz, as a dict.

    Keys, all with the split prefix stripped:
      S0            (n, 5)        the real last-UT state carried to z0 with RK6
      truth         (n, 257, 5)   the field-only RK6 track on the plane grid
      truth_zpost   (n, 5)        that reference carried on to the particle's
                                  own first SciFi plane z_post
      S_pre, S_post (n, 5)        the particle's REAL Geant4 states at z_pre, z_post
      z_pre, z_post (n,)          those planes [mm]
      P             (n,)          |p| at the particle's origin [GeV]
      ETA, PID, EVT, MCKEY, PBAND (n,)
    """
    D = load_tracks()
    keys = ("S0", "truth", "truth_zpost", "S_pre", "S_post", "z_pre", "z_post",
            "P", "ETA", "PID", "EVT", "MCKEY", "PBAND")
    return {k: np.asarray(D["%s_%s" % (split, k)]) for k in keys}


def concat_splits(keys, splits=SPLITS):
    """One array per key, the three splits concatenated in train/val/test order."""
    D = load_tracks()
    return {k: np.concatenate([np.asarray(D["%s_%s" % (s, k)]) for s in splits]) for k in keys}


def read_csv(path):
    """A CSV as a list of dicts (strings), read-only."""
    with open(path) as f:
        return list(csv.DictReader(f))


def read_json(path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------- the de-duplication rule --
def dedup(rows, key):
    """Rows de-duplicated by an integer column, LAST row written kept, sorted.

    Exactly the rule of `Block_G_low_momentum_window/G2_Analysis/convergence.py`
    (`dedup`), which is the rule the write-up's convergence table uses.  It
    exists because two farm jobs trained the N = 256, q = 16 reweighted run at
    the same time from restart 1,198 on: that run's `history.csv` holds 302
    repeated restart numbers and its `rounds.csv` 13 repeated round numbers,
    written by two interleaved processes.  The last row written for a number is
    the one kept.

    Returns (rows_sorted_by_key, sorted_unique_duplicated_keys).
    """
    seen, dups = {}, []
    for r in rows:
        k = int(float(r[key]))
        if k in seen:
            dups.append(k)
        seen[k] = r
    return [seen[k] for k in sorted(seen)], sorted(set(dups))


def load_rounds(run_dir):
    """rounds.csv de-duplicated by round.  Returns (rows, dups, n_raw_rows).

    Columns: round, source, n_states, per_plane, restarts, loss_first, loss_last,
    train_end_moved_med_um, val_z1_pos_med_um (max-metric on the validation
    split), val_z1_x_med_um, val_z1_y_med_um, wall_s.
    """
    p = os.path.join(run_dir, "rounds.csv")
    raw = read_csv(p)
    rows, dups = dedup(raw, "round")
    return rows, dups, len(raw)


def load_history(run_dir):
    """history.csv de-duplicated by restart.  Returns (rows, dups, n_raw_rows).

    Columns: restart, round, in_round, phase, loss_before, loss_after, gain,
    factor, factor_renewed, n_iter, func_evals, early_stop, wall_s.
    """
    p = os.path.join(run_dir, "history.csv")
    raw = read_csv(p)
    rows, dups = dedup(raw, "restart")
    return rows, dups, len(raw)


def load_record(run_dir):
    """record.json: N, q, dz_mm, seed, width, depth, field, n_parameters,
    n_train_tracks, states_per_round, round_restarts, outer_cap, round_cap,
    rounds, restarts, converged, confirmed, hit_cap, early_stop_restarts,
    final_loss, train_wall_s, carry_us_per_track, and the `val` / `test` score
    blocks written by `E1_Network_grid/metrics.chain_scores`."""
    return read_json(os.path.join(run_dir, "record.json"))


def load_scale(run_dir):
    """scale.json: N, q, L, z0, dz, seed, width, depth, y_floor_rel, field,
    in_scale (5), states_per_round, n_train_tracks, tracks; and for a
    reweighted run an extra `weighting` block with D_ref, lev_ref, i_bar, z1,
    L, mode, clamp, p_lo, p_hi, rolloff, w_floor."""
    return read_json(os.path.join(run_dir, "scale.json"))


def load_chain_states(run_dir, split="test"):
    """chain_states.npz -> (n, N+1, 5).

    The file holds exactly two arrays, `val_states` (1463, N+1, 5) and
    `test_states` (1452, N+1, 5): the network's state on EVERY plane of the
    chain, plane 0 being the track's own state at z0 and plane N the endpoint
    at z1.  The fifth component q/p is carried through unchanged.
    """
    with np.load(os.path.join(run_dir, "chain_states.npz")) as z:
        return np.asarray(z["%s_states" % split])


def chain_states_keys(run_dir):
    with np.load(os.path.join(run_dir, "chain_states.npz")) as z:
        return {k: list(np.asarray(z[k]).shape) for k in z.files}


def load_exact(N, q):
    """E2_Comparators/results/exact_N<NNN>_q<qq>.json, or None if not built.

    The JSON stores, for the 1,452 test tracks, the endpoint error of the EXACT
    collocation scheme chained N times against the RK6 truth:
      endpoint_pos_med_um / endpoint_pos_p95_um   max(|dx|, |dy|) [um]
      endpoint_slope_med_mrad                     max(|dtx|, |dty|) x 1e3
      components.{x,y}_{med,p95,mean,bias}_um     positions [um]
      components.{tx,ty}_{med,p95,mean,bias}_mrad slopes x 1e3
      per_plane.*                                 the same along the chain
    This loader returns the raw JSON plus `*_slope` copies of every slope entry
    divided by 1e3, so the paper never quotes a "mrad".  Only N = 2 and N = 256
    were run (`E2_Comparators/README.md`); the N = 64 and N = 128 ceilings live
    in `E3_Analysis/results/error_qdz_chain.csv` (`exact_med_um`).
    """
    p = os.path.join(E2_RESULTS, "exact_%s.json" % run_tag(N, q))
    if not os.path.exists(p):
        return None
    j = read_json(p)
    j["endpoint_slope_med_slope"] = j["endpoint_slope_med_mrad"] / 1e3
    for k in list(j.get("components", {})):
        if k.endswith("_mrad"):
            j["components"][k[:-5] + "_slope"] = j["components"][k] / 1e3
    return j


# ------------------------------------------------------ the plateau rule --
PLATEAU_WINDOW = 10
PLATEAU_TOL = 0.05
PLATEAU_HOLD = 3


def plateaued_now(v, window=PLATEAU_WINDOW, tol=PLATEAU_TOL, hold=PLATEAU_HOLD):
    """Has the validation error stopped falling as of the latest round?

    The rule of section 3.7 of the reweighted-loss write-up (`F4_Writeup/
    page.md`), fixed before any reweighted training in
    `F2_Analysis/compare_to_blockE.py`: the median of the last `window` rounds
    is no more than `tol` BELOW the median of the `window` before it, and that
    must hold at each of the last `hold` rounds.
    """
    v = np.asarray(v, dtype=float)
    if len(v) < 2 * window + hold - 1:
        return False
    for r in range(len(v) - hold + 1, len(v) + 1):
        a, b = np.median(v[r - window:r]), np.median(v[r - 2 * window:r - window])
        if a < (1.0 - tol) * b:
            return False
    return True


def first_plateau_round(v, window=PLATEAU_WINDOW, tol=PLATEAU_TOL, hold=PLATEAU_HOLD):
    """The first round the rule ever held (a record, not a verdict), or None."""
    v = np.asarray(v, dtype=float)
    seen = 0
    for r in range(2 * window, len(v) + 1):
        a, b = np.median(v[r - window:r]), np.median(v[r - 2 * window:r - window])
        seen = seen + 1 if a >= (1.0 - tol) * b else 0
        if seen >= hold:
            return r
    return None


def validation_headline(v, window=PLATEAU_WINDOW):
    """The median of the last `window` rounds with its (max - min)/median spread."""
    v = np.asarray(v, dtype=float)
    last = v[-window:]
    prev = v[-2 * window:-window]
    med = float(np.median(last))
    return dict(
        headline_val_med_um=med,
        headline_spread_pct=float(100.0 * (last.max() - last.min()) / 2.0 / med),
        headline_full_range_over_median_pct=float(100.0 * (last.max() - last.min()) / med),
        headline_min_um=float(last.min()), headline_max_um=float(last.max()),
        last10_vs_prev10_pct=(float(100.0 * (med - np.median(prev)) / np.median(prev))
                              if len(prev) == window else float("nan")),
        rounds_used=int(len(last)))


# ------------------------------------------------- the experiment machinery --
def add_experiment_paths():
    """Put the read-only experiment modules on sys.path (imports only).

    `E1_Network_grid` brings `chain_network` (the network class, `load_network`,
    `step_outputs`, `carry`) and, through its own `use_shared.py`, the `_shared`
    package (`reference.rk6_rows`, `irk.verify_tableau`, `model.physics_loss`).
    `F0_Weighting` brings `weighted_loss.py`.  Nothing in either folder is
    written to; `sys.dont_write_bytecode` is set at the top of this module so
    not even a .pyc is left behind.
    """
    for p in (E1, F0):
        if p not in sys.path:
            sys.path.insert(0, p)
    import use_shared                                  # noqa: F401
    return True


def load_weighted_loss():
    """Import `F0_Weighting/weighted_loss.py` read-only, by file location."""
    import importlib.util
    add_experiment_paths()
    spec = importlib.util.spec_from_file_location(
        "f0_weighted_loss", os.path.join(F0, "weighted_loss.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rel(path):
    """A path relative to the repository root, for the `source` strings."""
    return os.path.relpath(path, REPO)


def source(*paths):
    """The `source` string of a JSON entry: the files it was computed from."""
    return " ; ".join(rel(p) for p in paths)


def write_csv(path, rows, fieldnames=None):
    """Write one of the paper's tab_*.csv tables."""
    if not rows:
        return path
    fieldnames = fieldnames or list(rows[0].keys())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


def jsonable(o):
    """numpy -> plain python, so json.dump never chokes."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, float) and (o != o or o in (float("inf"), float("-inf"))):
        return None if o != o else ("inf" if o > 0 else "-inf")
    return o
