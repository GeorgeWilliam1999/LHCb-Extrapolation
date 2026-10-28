"""What every script of this experiment shares: the runs, the tracks, the
single-step errors, the chain with a correction hook, and the scores.

Units, everywhere in this experiment:
    positions x, y      micrometres (um)
    slopes tx, ty       microradians (urad), i.e. 1e-6
    q/p                 the package's unit, 0.299792458 * charge / (momentum in GeV)

The signed single-step error of a step is
    e = (network one-step state from the reference state on the start plane)
        - (reference state on the end plane),
on the 64 steps of 80.903 mm between the 65 standard planes (every 4th of the
257 planes of the track set). The reference is the sixth-order method at
0.1 mm stored with the tracks, so "reference one step" from the reference
start state is the reference state on the next plane.

Reads the store and the field map. Writes only to results/ and figures/ beside
this file, and to the cache folder outside the repository.
"""
from __future__ import annotations

import csv
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")
CACHE = "/data/bfys/gscriven/rkpinn_analysis_cache/what_is_the_per_step_bias"
STORE = "/data/bfys/gscriven/rkpinn_store"
TRACKS_KEY = "12a8d35c3165"
NUMBER_OF_STEPS = 64
SNAPSHOTS_AVERAGED = 5

from rkpinn.equation_of_motion.field_map import load_field_map            # noqa: E402
from rkpinn.equation_of_motion.lhcb import (                              # noqa: E402
    DifferentiableLhcbEquationOfMotion, LhcbEquationOfMotion)
from rkpinn.integrators.gauss_legendre_tableau import gauss_legendre_tableau  # noqa: E402
from rkpinn.run_record.configuration import read_configuration            # noqa: E402
from rkpinn.run_record.store import Store                                 # noqa: E402
from rkpinn.track_data.load_tracks import load_tracks                     # noqa: E402
from rkpinn.training.checkpoints import RunFolder                         # noqa: E402
from rkpinn.training.round_trainer import build_run                       # noqa: E402

RUNS = [("straight line, per-track scale", "25dfa17ef168"),
        ("straight line, one global scale", "9cef9c160b7d"),
        ("direct states", "49e5e2450eaa")]
SHORT = {"25dfa17ef168": "straight line per-track",
         "9cef9c160b7d": "straight line global",
         "49e5e2450eaa": "direct states"}

BANDS = [("< 3 GeV", 0.0, 3.0), ("3-8 GeV", 3.0, 8.0), ("8-20 GeV", 8.0, 20.0),
         ("20-50 GeV", 20.0, 50.0), ("> 50 GeV", 50.0, np.inf)]
COMPONENTS = ("x", "y", "tx", "ty")
TO_REPORTED = np.array([1e3, 1e3, 1e6, 1e6])       # mm -> um, slope -> urad

store = Store(STORE)
tracks = load_tracks(store, TRACKS_KEY)
FIELD = load_field_map("v8r1_up")
EQUATION = LhcbEquationOfMotion(FIELD)
DIFFERENTIABLE = DifferentiableLhcbEquationOfMotion(FIELD)
TABLEAU = gauss_legendre_tableau(2)
PLANE_INDEX = tracks.planes_a_step_starts_on(NUMBER_OF_STEPS)
PLANE_INDEX = np.append(PLANE_INDEX, tracks.number_of_steps_between_planes)   # 65 planes
PLANES_MM = tracks.planes_mm[PLANE_INDEX]
STEP_MM = float(PLANES_MM[1] - PLANES_MM[0])
START_PLANES_MM = PLANES_MM[:-1]


def split(name):
    return tracks.split(name)


def on_planes(name):
    """Reference states on the 65 standard planes, shape (n, 65, 5)."""
    return split(name).reference_states_on_planes[:, PLANE_INDEX]


def band_masks(momentum_gev):
    return [(label, (momentum_gev >= lo) & (momentum_gev < hi)) for label, lo, hi in BANDS]


def write_rows(file, rows, folder=RESULTS):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, file), "w", newline="") as handle:
        writer = csv.DictWriter(handle, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("written: results/%s" % file, flush=True)


# ------------------------------------------------------------------ runs
_runs = {}


def run_of(key):
    if key not in _runs:
        folder = RunFolder(store, key)
        configuration = read_configuration(folder.path("configuration.yaml"))
        _runs[key] = (folder, configuration, build_run(configuration, store, folder.constants()))
    return _runs[key]


def last_round(key):
    return int(run_of(key)[0].state()["rounds_done"])


def network_of(key, rounds=None):
    """The network of a run with the weights of one round, or the mean of the
    weights of several rounds (rounds a list)."""
    folder, _, run = run_of(key)
    rounds = [last_round(key)] if rounds is None else list(np.atleast_1d(rounds))
    dicts = [folder.weights_of_round(int(r)) for r in rounds]
    mean = {name: sum(d[name] for d in dicts) / len(dicts) for name in dicts[0]}
    run.network.load_state_dict(mean)
    run.network.eval()
    return run.network


# ------------------------------------------------------------------ one step
def stages_and_end(network, states, start_planes_mm, rows_at_a_time=65536):
    """The network's stages (n, 2, 4) and summed end state (n, 4) for states
    (n, 5) on their own start planes (n,). numpy in, numpy out."""
    states = np.asarray(states, dtype=np.float64)
    planes = np.broadcast_to(np.asarray(start_planes_mm, dtype=np.float64), (len(states),))
    stages = np.empty((len(states), 2, 4))
    ends = np.empty((len(states), 4))
    with torch.no_grad():
        for i in range(0, len(states), rows_at_a_time):
            step = network.predict(torch.as_tensor(states[i:i + rows_at_a_time]),
                                   torch.as_tensor(planes[i:i + rows_at_a_time].copy()))
            stages[i:i + rows_at_a_time] = step.stage_states[:, 0].numpy()
            ends[i:i + rows_at_a_time] = step.end_states[:, 0].numpy()
    return stages, ends


def single_step_errors(network, name, step_function=None):
    """Signed errors e (n, 64, 4) in um / urad, from the reference state on each
    start plane. step_function(states, planes) -> end states (n, 4) replaces
    the plain network step when given."""
    ref = on_planes(name)
    n = len(ref)
    states = ref[:, :-1].reshape(-1, 5)
    planes = np.broadcast_to(START_PLANES_MM, (n, NUMBER_OF_STEPS)).reshape(-1)
    if step_function is None:
        ends = stages_and_end(network, states, planes)[1]
    else:
        ends = step_function(states, planes)
    e = ends.reshape(n, NUMBER_OF_STEPS, 4) - ref[:, 1:, :4]
    return e * TO_REPORTED


def step_features(name):
    """(x, y, tx, ty, q/p, z) of the start of every step, shape (n, 64, 6)."""
    ref = on_planes(name)
    z = np.broadcast_to(START_PLANES_MM, ref.shape[:1] + (NUMBER_OF_STEPS,))
    return np.concatenate([ref[:, :-1], z[..., None]], axis=2)


def cached_single_step_errors(key, name, round_number=None):
    """e of one run, split and round, cached in CACHE."""
    round_number = last_round(key) if round_number is None else int(round_number)
    file = os.path.join(CACHE, "e_%s_%s_round_%04d.npy" % (key, name, round_number))
    if not os.path.exists(file):
        os.makedirs(CACHE, exist_ok=True)
        e = single_step_errors(network_of(key, [round_number]), name)
        np.save(file, e)
    return np.load(file)


# ------------------------------------------------------------------ chain
def chain(step_function, start_states):
    """Chain step_function(states (n,5), planes (n,)) -> ends (n,4) over the 64
    steps. Returns the states on the 65 planes, shape (n, 65, 5)."""
    current = np.asarray(start_states, dtype=np.float64).copy()
    out = np.empty((len(current), NUMBER_OF_STEPS + 1, 5))
    out[:, 0] = current
    for k in range(NUMBER_OF_STEPS):
        ends = step_function(current, np.full(len(current), START_PLANES_MM[k]))
        current = np.concatenate([ends, current[:, 4:5]], axis=1)
        out[:, k + 1] = current
    return out


def plain_step(network):
    return lambda states, planes: stages_and_end(network, states, planes)[1]


# ------------------------------------------------------------------ scores
def lag_one_autocorrelation(errors):
    """errors (n, S): the lag-1 autocorrelation along the steps, per track."""
    centred = errors - errors.mean(axis=1, keepdims=True)
    return (centred[:, :-1] * centred[:, 1:]).sum(axis=1) / (centred ** 2).sum(axis=1)


def endpoint_scores(final_states, prefix=""):
    """The scores of a variant from its states on the last plane (test, n x 5)."""
    test = split("test")
    d = (np.asarray(final_states)[:, :4] - test.reference_end_state[:, :4]) * TO_REPORTED
    radial = np.hypot(d[:, 0], d[:, 1])
    row = {prefix + "endpoint_radial_median_um": float(np.median(radial))}
    for label, m in band_masks(test.momentum_gev):
        row[prefix + "median_um_" + label] = float(np.median(radial[m]))
    for i, c in enumerate(("x", "y")):
        row[prefix + "mean_signed_%s_um" % c] = float(d[:, i].mean())
        row[prefix + "se_signed_%s_um" % c] = float(d[:, i].std(ddof=1) / np.sqrt(len(d)))
    return row, d


def lag_one_scores(e, prefix=""):
    return {prefix + "lag1_%s_median" % c: float(np.median(lag_one_autocorrelation(e[..., i])))
            for i, c in enumerate(("x", "y"))}


def bend_sign_of_steps(name):
    """+1 where the reference slope in x grows over the step, -1 where it falls; (n, 64)."""
    ref = on_planes(name)
    return np.sign(ref[:, 1:, 2] - ref[:, :-1, 2])


def bend_sign_of_tracks(name):
    """The bend direction of the whole crossing, as in the resolution thread:
    sign(tx on the last plane - tx on the first plane); (n,)."""
    s = split(name)
    return np.sign(s.reference_end_state[:, 2] - s.start_state[:, 2])


def setup_threads():
    torch.set_num_threads(int(os.environ.get("THREADS", "4")))
