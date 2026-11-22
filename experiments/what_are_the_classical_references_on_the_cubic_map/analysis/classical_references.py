"""The network-free references of the cubic track set, on its test split.

    classical_references.py rk4      classic fourth-order Runge-Kutta, N = 2 .. 256
    classical_references.py exact W  the exact Gauss-Legendre scheme, N = 2 .. 256,
                                     q = 2, 4, 8, 16, on W worker processes

Every error is radial, sqrt(dx^2 + dy^2) in micrometres, at the last plane
(7826.0 mm), against the sixth-order reference at 0.1 mm stored with the
tracks, over the 13,447 test tracks of 5558be31abeb (v8r1_up_cubic).

The exact scheme is reached by Newton sweeps of the whole track from the
forward-Euler path (evaluation/newton_sweep.py), until the largest residual of
the stage equations is below 1e-10. Where the store holds the root finder's
exact states of the same cell, the script also gives the largest end-state
difference between the two, as a check.

The per-track errors are kept in CACHE; the statistics go to results/*.csv.
Reads the store; writes nothing to it.
"""
from __future__ import annotations

import csv
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
STORE = "/data/bfys/gscriven/rkpinn_store"
CACHE = "/data/bfys/gscriven/rkpinn_analysis_cache/what_are_the_classical_references_on_the_cubic_map"
RESULTS = os.path.join(HERE, "results")
TRACKS_KEY = "5558be31abeb"
FIELD_MAP = "v8r1_up_cubic"
STEPS = [2, 4, 8, 16, 32, 64, 128, 256]
STAGES = [2, 4, 8, 16]
BANDS = {"all momenta": (0.0, np.inf), "below 3 GeV": (0.0, 3.0), "3 to 8 GeV": (3.0, 8.0),
         "8 to 20 GeV": (8.0, 20.0), "20 to 50 GeV": (20.0, 50.0), "above 50 GeV": (50.0, np.inf)}
QUANTILES = (0.16, 0.50, 0.84)
RESIDUAL_TOLERANCE = 1e-10
MOST_SWEEPS = 8
TRACKS_A_SHARE = 500          # at N q <= 1024; fewer above, to keep a worker near 5 GB

from rkpinn.equation_of_motion.field_map import load_field_map               # noqa: E402
from rkpinn.equation_of_motion.lhcb import (DifferentiableLhcbEquationOfMotion,  # noqa: E402
                                            LhcbEquationOfMotion)
from rkpinn.evaluation.newton_sweep import forward_euler_path, one_sweep     # noqa: E402
from rkpinn.integrators.gauss_legendre_tableau import gauss_legendre_tableau  # noqa: E402
from rkpinn.predicted_track.track_layout import TrackLayout                  # noqa: E402
from rkpinn.run_record.store import NotInTheStore, Store                     # noqa: E402
from rkpinn.track_data.exact_states import load_exact_states                 # noqa: E402
from rkpinn.track_data.load_tracks import load_tracks                        # noqa: E402

store = Store(STORE)
tracks = load_tracks(store, TRACKS_KEY)
assert tracks.description["settings"]["field_map"] == FIELD_MAP
test = tracks.test
reference = test.reference_states_on_planes
momentum = test.momentum_gev


def radial_um(predicted, wanted):
    d = (np.asarray(predicted)[..., :2] - np.asarray(wanted)[..., :2]) * 1e3
    return np.hypot(d[..., 0], d[..., 1])


def statistics(endpoint):
    rows = []
    for band, (low, high) in BANDS.items():
        e = endpoint[(momentum >= low) & (momentum < high)]
        row = dict(momentum_band=band, number_of_tracks=int(len(e)))
        for qu, v in zip(QUANTILES, np.quantile(e, QUANTILES)):
            row["endpoint_radial_p%02d_micrometres" % round(qu * 100)] = float(v)
        rows.append(row)
    return rows


def cached(name, compute):
    path = os.path.join(CACHE, name + ".npz")
    if os.path.isfile(path):
        with np.load(path) as loaded:
            return {k: loaded[k] for k in loaded.files}
    arrays = compute()
    os.makedirs(CACHE, exist_ok=True)
    np.savez(path, **arrays)
    return arrays


def write_rows(file, rows):
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, file), "w", newline="") as handle:
        writer = csv.DictWriter(handle, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rk4():
    equation = LhcbEquationOfMotion(load_field_map(FIELD_MAP))
    rows = []
    for N in STEPS:
        h, starts = tracks.length_mm / N, tracks.start_planes_mm(N)

        def compute():
            started = time.time()
            S = np.array(test.start_state, dtype=np.float64)
            for z in starts:
                k1 = equation.rates(S, z)
                k2 = equation.rates(S + 0.5 * h * k1, z + 0.5 * h)
                k3 = equation.rates(S + 0.5 * h * k2, z + 0.5 * h)
                k4 = equation.rates(S + h * k3, z + h)
                S = S + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            return dict(endpoint=radial_um(S, reference[:, -1]),
                        seconds=np.array(time.time() - started))

        a = cached("rk4_%03d_steps" % N, compute)
        for row in statistics(a["endpoint"]):
            rows.append(dict(scheme="classic fourth-order Runge-Kutta", number_of_steps=N,
                             number_of_stages=4, step_length_mm=h,
                             field_lookups_per_track=4 * N, **row))
        print("rk4 %d: %.4f um" % (N, np.median(a["endpoint"])), flush=True)
    write_rows("rk4.csv", rows)


_differentiable = None


def _sweep_one_share(arguments):
    global _differentiable
    N, q, share = arguments
    torch.set_num_threads(1)
    if _differentiable is None:
        _differentiable = DifferentiableLhcbEquationOfMotion(load_field_map(FIELD_MAP))
    layout = TrackLayout(tracks.first_plane_mm, tracks.last_plane_mm, N, gauss_legendre_tableau(q))
    track = forward_euler_path(_differentiable, test.start_state[share], layout)
    residual, sweeps = np.inf, 0
    while residual > RESIDUAL_TOLERANCE and sweeps < MOST_SWEEPS:
        sweep = one_sweep(track, _differentiable)
        track, residual, sweeps = sweep.track, sweep.largest_residual_after, sweeps + 1
    return np.asarray(track.final_state()), residual, sweeps


def exact(workers):
    rows = []
    for q in STAGES:
        for N in STEPS:
            def compute():
                started = time.time()
                shares = np.array_split(np.arange(test.number_of_tracks),
                                        int(np.ceil(test.number_of_tracks * max(1, N * q / 1024)
                                                    / TRACKS_A_SHARE)))
                with Pool(workers) as pool:
                    parts = pool.map(_sweep_one_share, [(N, q, s) for s in shares], chunksize=1)
                final = np.concatenate([p[0] for p in parts])
                return dict(endpoint=radial_um(final, reference[:, -1]), final=final,
                            largest_residual=np.array(max(p[1] for p in parts)),
                            most_sweeps=np.array(max(p[2] for p in parts)),
                            seconds=np.array(time.time() - started), workers=np.array(workers))

            a = cached("exact_%03d_steps_%02d_stages" % (N, q), compute)
            try:
                root = load_exact_states(store, TRACKS_KEY, "test", N, q)[0][:, -1]
                against_root = float(np.abs(a["final"][:, :2] - root[:, :2]).max() * 1e3)
            except NotInTheStore:
                against_root = ""
            for row in statistics(a["endpoint"]):
                rows.append(dict(scheme="exact Gauss-Legendre scheme", number_of_steps=N,
                                 number_of_stages=q, step_length_mm=tracks.length_mm / N,
                                 field_lookups_per_sweep_per_track=N * q,
                                 sweeps=int(a["most_sweeps"]),
                                 largest_residual=float(a["largest_residual"]),
                                 largest_difference_to_the_root_finder_micrometres=against_root,
                                 **row))
            print("exact %d/%d: %.5f um, %d sweeps, residual %.1e, %.0f s, root diff %s"
                  % (N, q, np.median(a["endpoint"]), a["most_sweeps"], a["largest_residual"],
                     a["seconds"], against_root), flush=True)
            write_rows("exact_scheme.csv", rows)


if __name__ == "__main__":
    torch.set_num_threads(1)
    if len(sys.argv) > 1 and sys.argv[1] == "rk4":
        rk4()
    elif len(sys.argv) > 1 and sys.argv[1] == "exact":
        exact(int(sys.argv[2]))
    else:
        raise SystemExit(__doc__)
