"""The exact scheme's states of a track set, written once into the store.

For one split of a track set, one number of steps and one number of stages,
the exact collocation scheme is chained across the crossing from every start
state, and the state after every step is kept.

Why they are kept in the store. The root finder rounds differently on
different processors, so the same solve gives answers that differ in the last
digits from machine to machine (measured on 2026-09-28: up to 4.5e-13 mm).
George ruled on 2026-09-28 that the data is kept in a single master store. So
the states are solved once, on one machine, and everything reads them from
here. The record beside each file names the machine and its processor.

To write the exact states of the project, from `RK_Pinn_module`:

    PYTHONNOUSERSITE=1 PYTHONPATH=src /data/bfys/gscriven/conda/envs/TE/bin/python \\
        -m rkpinn.track_data.exact_states \\
        --store /data/bfys/gscriven/rkpinn_store --tracks <key> --split test \\
        --number-of-steps 2 64 128 256 --number-of-stages 2 4 8 16 --workers 16

Gates: tests/test_exact_states_in_the_store.py.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from rkpinn.equation_of_motion.field_map import load_field_map
from rkpinn.equation_of_motion.lhcb import LhcbEquationOfMotion
from rkpinn.integrators import exact_collocation
from rkpinn.integrators.gauss_legendre_tableau import gauss_legendre_tableau
from rkpinn.run_record import manifest
from rkpinn.run_record.provenance import provenance
from rkpinn.run_record.store import NotInTheStore, Store, key_of_content
from rkpinn.track_data.load_tracks import TrackSet, load_tracks


def _solve_one_share(arguments):
    """The exact chain for one share of the start states. Run in a worker."""
    (start_states, field_map, first_plane_mm, length_mm, number_of_steps,
     number_of_stages) = arguments
    equation = LhcbEquationOfMotion(load_field_map(field_map))
    solved = exact_collocation.solve_whole_track(
        equation, start_states, first_plane_mm, length_mm, number_of_steps,
        gauss_legendre_tableau(number_of_stages))
    return solved.states, solved.solves_not_converged, solved.residual_evaluations


def solve_exact_states(tracks: TrackSet, split: str, number_of_steps: int,
                       number_of_stages: int, workers: int, most_tracks=None):
    """The exact scheme's states, shape (n, number_of_steps + 1, 5), with the
    number of solves that did not converge and of residual evaluations."""
    tracks.planes_a_step_starts_on(number_of_steps)        # refuses steps off the planes
    start_states = tracks.split(split).start_state[:most_tracks]
    field_map = tracks.description["settings"]["field_map"]
    shares = np.array_split(np.arange(len(start_states)),
                            max(1, min(workers, len(start_states))))
    jobs = [(start_states[share], field_map, tracks.first_plane_mm, tracks.length_mm,
             number_of_steps, number_of_stages) for share in shares]
    if workers <= 1 or len(jobs) == 1:
        parts = [_solve_one_share(job) for job in jobs]
    else:
        with Pool(len(jobs)) as pool:
            parts = pool.map(_solve_one_share, jobs)
    states = np.concatenate([part[0] for part in parts], axis=0)
    return states, int(sum(p[1] for p in parts)), int(sum(p[2] for p in parts))


def write_exact_states(store: Store, tracks_key: str, split: str, number_of_steps: int,
                       number_of_stages: int, workers: int,
                       allow_uncommitted_changes: bool = False, report=print,
                       record=None) -> str:
    """Solve and write one file of exact states. Returns the key of its content.

    record  the provenance, when it was taken already at the start of a
            command that writes several files; None takes it now.
    """
    if record is None:
        record = provenance(allow_uncommitted_changes)
    record = dict(record, created=time.strftime("%Y-%m-%d %H:%M:%S"))
    file = store.file_of_exact_states(tracks_key, split, number_of_steps, number_of_stages)
    store.refuse_if_present(file)
    tracks = load_tracks(store, tracks_key)
    started = time.time()
    states, not_converged, evaluations = solve_exact_states(
        tracks, split, number_of_steps, number_of_stages, workers)
    seconds = time.time() - started
    step_length = tracks.length_mm / number_of_steps
    arrays = {"states": states,
              "planes_mm": tracks.first_plane_mm + np.arange(number_of_steps + 1) * step_length}
    key = key_of_content(arrays)
    os.makedirs(os.path.dirname(file), exist_ok=True)
    np.savez_compressed(file, **arrays)
    description = {
        "what": "the exact collocation scheme chained across the crossing from "
                "every start state of the split; the state after every step",
        "tracks_key": tracks_key, "split": split,
        "number_of_steps": number_of_steps, "number_of_stages": number_of_stages,
        "step_length_mm": step_length, "number_of_tracks": int(len(states)),
        "key_of_content": key,
        "arrays": {"states": "shape (tracks, steps + 1, 5); entry 0 is the start state",
                   "planes_mm": "z of the plane of every entry"},
        "solver": {
            "root_finder": "scipy.optimize.root, method hybr",
            "step_tolerance_of_the_root_finder":
                exact_collocation.STEP_TOLERANCE_OF_THE_ROOT_FINDER,
            "tolerance_of_the_residual": exact_collocation.TOLERANCE_OF_THE_RESIDUAL,
            "scale_of_the_residual": exact_collocation.SCALE_OF_THE_RESIDUAL.tolist(),
            "first_guess": "the input state carried along its own slopes"},
        "solves_not_converged": not_converged,
        "residual_evaluations_per_track": evaluations / max(1, len(states)),
        "cost": {"seconds": round(seconds, 1), "workers": workers},
        "provenance": record,
    }
    with open(store.description_of_exact_states(
            tracks_key, split, number_of_steps, number_of_stages), "w") as handle:
        json.dump(description, handle, indent=1)
    manifest.add_exact_states(store, {
        "tracks_key": tracks_key, "split": split, "number_of_steps": number_of_steps,
        "number_of_stages": number_of_stages, "step_length_mm": step_length,
        "number_of_tracks": int(len(states)), "solves_not_converged": not_converged,
        "key_of_content": key, "created": record["created"],
        "package_version": record["package_version"], "commit": record["commit"],
        "traceable_to_the_commit": record["traceable_to_the_commit"],
        "machine": record["machine"], "processor": record["processor"],
    })
    report("%d steps, %d stages, %s: %d tracks, %d solves not converged, %.0f s"
           % (number_of_steps, number_of_stages, split, len(states), not_converged, seconds))
    return key


def load_exact_states(store: Store, tracks_key: str, split: str, number_of_steps: int,
                      number_of_stages: int, check_the_key: bool = True):
    """(states, planes_mm, description) of one file of exact states."""
    file = store.file_of_exact_states(tracks_key, split, number_of_steps, number_of_stages)
    if not os.path.isfile(file):
        raise NotInTheStore(
            "the store holds no exact states for the tracks %r, split %s, %d steps "
            "and %d stages" % (tracks_key, split, number_of_steps, number_of_stages))
    with np.load(file, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    with open(store.description_of_exact_states(
            tracks_key, split, number_of_steps, number_of_stages)) as handle:
        description = json.load(handle)
    if check_the_key and key_of_content(arrays) != description["key_of_content"]:
        raise ValueError("the exact states in %s no longer have the content their "
                         "record describes" % file)
    return arrays["states"], arrays["planes_mm"], description


def main(arguments=None):
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    parser = argparse.ArgumentParser(
        description="Write the exact scheme's states of a track set into the store.")
    parser.add_argument("--store", required=True)
    parser.add_argument("--tracks", required=True, help="the key of the track set")
    parser.add_argument("--split", required=True)
    parser.add_argument("--number-of-steps", type=int, nargs="+", required=True)
    parser.add_argument("--number-of-stages", type=int, nargs="+", required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--allow-uncommitted-changes", action="store_true")
    a = parser.parse_args(arguments)
    store = Store(a.store)
    record = provenance(a.allow_uncommitted_changes)     # once, before any work
    for number_of_steps in a.number_of_steps:
        for number_of_stages in a.number_of_stages:
            file = store.file_of_exact_states(
                a.tracks, a.split, number_of_steps, number_of_stages)
            if os.path.exists(file):
                print("%d steps, %d stages: already in the store, left as it is"
                      % (number_of_steps, number_of_stages), flush=True)
                continue
            write_exact_states(store, a.tracks, a.split, number_of_steps,
                               number_of_stages, a.workers, a.allow_uncommitted_changes,
                               report=lambda text: print(text, flush=True),
                               record=record)


if __name__ == "__main__":
    main()
