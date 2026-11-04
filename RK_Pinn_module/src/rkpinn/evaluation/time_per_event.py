"""The time per event: the wall time to carry one event of tracks across the
crossing, on one CPU thread, for the methods the one-shot path study compares.

    chain network           a trained stage network applied N times
    path network            a trained path network, one call
    path network + 1 sweep  the same, then one Newton sweep (newton_sweep.py)
    Euler + 1 sweep         the forward-Euler path, then one sweep
    Euler + 2 sweeps        the same with a second sweep
    RK4                     classic fourth-order Runge-Kutta, N steps

An event is `tracks_per_event` consecutive tracks of the split, carried as one
batch. Every method carries the same events, after one event of warm-up that is
not timed. The time is the median over the events; the error is the radial
error at the end of the region against the reference, over every timed track.
The code paths are those of the reports: `whole_track` of the network, and
`one_sweep` and `forward_euler_path` of newton_sweep.py. RK4 uses the numpy
equation of motion, as the step-length study did.

A method whose run is not given is left out. To time, from `RK_Pinn_module`:

    PYTHONNOUSERSITE=1 PYTHONPATH=src /data/bfys/gscriven/conda/envs/TE/bin/python \\
        -m rkpinn.evaluation.time_per_event --store /data/bfys/gscriven/rkpinn_store \\
        --tracks 12a8d35c3165 --split test --number-of-steps 64 --number-of-stages 2 \\
        --chain-run 25dfa17ef168 --path-run <key> --round last \\
        --tracks-per-event 100 --events 14

Added 2026-11-04 for the experiment does_a_one_shot_path_network_remove_the_chain.
Gates: tests/test_time_and_sweeps.py.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from rkpinn import registry
from rkpinn.evaluation import conventions
from rkpinn.evaluation.newton_sweep import forward_euler_path, one_sweep
from rkpinn.integrators.gauss_legendre_tableau import gauss_legendre_tableau
from rkpinn.predicted_track.track_layout import TrackLayout
from rkpinn.run_record.configuration import read_configuration
from rkpinn.run_record.store import Store
from rkpinn.track_data.load_tracks import load_tracks
from rkpinn.training.checkpoints import RunFolder


def network_of_a_run(store: Store, key: str, round_number):
    """The network of a run with the weights of one round ("last" or a number)."""
    from rkpinn.training.round_trainer import build_run
    folder = RunFolder(store, key)
    if round_number == "last":
        round_number = folder.state()["rounds_done"]
    run = build_run(read_configuration(folder.path("configuration.yaml")), store,
                    folder.constants())
    run.network.load_state_dict(folder.weights_of_round(int(round_number)))
    run.network.eval()
    return run.network, run.layout


def rk4_whole_track(equation, start_states, layout):
    """The final states, shape (n, 5), of classic RK4 over the layout."""
    S, h = np.array(start_states, dtype=np.float64), layout.step_length_mm
    for z in layout.start_planes_mm:
        k1 = equation.rates(S, z)
        k2 = equation.rates(S + 0.5 * h * k1, z + 0.5 * h)
        k3 = equation.rates(S + 0.5 * h * k2, z + 0.5 * h)
        k4 = equation.rates(S + h * k3, z + h)
        S = S + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return S


def methods(*, numpy_equation, differentiable_equation, layout, chain=None, path=None):
    """{name: (function of the start states of an event giving its final
    states (n, 5), field lookups per track)}."""
    N, q = layout.number_of_steps, layout.number_of_stages

    def euler_then(sweeps):
        def carry(states):
            track = forward_euler_path(differentiable_equation, states, layout)
            for _ in range(sweeps):
                track = one_sweep(track, differentiable_equation).track
            return np.asarray(track.final_state())
        return carry

    found = {}
    if chain is not None:
        cost = chain.cost_per_step()
        found["chain network"] = (
            lambda states: np.asarray(chain.whole_track(states, layout).final_state()),
            N * (cost["field_lookups_per_step"]
                 + cost["field_lookups_of_the_output_scales_per_step"]))
    if path is not None:
        # the per-track scale of the straight-line form samples the field over
        # the whole region (path_network.py)
        own = path.field_lookups_per_track()
        found["path network"] = (
            lambda states: np.asarray(path.whole_track(states, layout).final_state()), own)
        found["path network + 1 sweep"] = (
            lambda states: np.asarray(one_sweep(path.whole_track(states, layout),
                                                differentiable_equation).track.final_state()),
            own + N * q)
    found["Euler + 1 sweep"] = (euler_then(1), N + N * q)
    found["Euler + 2 sweeps"] = (euler_then(2), N + 2 * N * q)
    found["RK4"] = (lambda states: rk4_whole_track(numpy_equation, states, layout), 4 * N)
    return found


def time_per_event(store: Store, *, tracks_key, split, number_of_steps, number_of_stages,
                   chain_run, path_run, round_number, tracks_per_event, events,
                   report=print) -> list:
    """One row per method: milliseconds per event and per track (median over
    the events), field lookups per track, and the radial error at the end of
    the region over the timed tracks."""
    torch.set_num_threads(1)
    tracks = load_tracks(store, tracks_key)
    of_split = tracks.split(split)
    if (events + 1) * tracks_per_event > of_split.number_of_tracks:
        raise ValueError("%d events of %d tracks and one of warm-up need more than the %d "
                         "tracks of the split" % (events, tracks_per_event,
                                                  of_split.number_of_tracks))
    field_map = registry.component("field_map", tracks.description["settings"]["field_map"])()
    numpy_equation = registry.component("equation_of_motion", "lhcb")(field_map)
    differentiable = registry.component("equation_of_motion", "lhcb_differentiable")(field_map)
    layout = TrackLayout(tracks.first_plane_mm, tracks.last_plane_mm, number_of_steps,
                         gauss_legendre_tableau(number_of_stages))
    chain = path = None
    for key, role in ((chain_run, "chain"), (path_run, "path")):
        if key is None:
            continue
        network, its_layout = network_of_a_run(store, key, round_number)
        if (its_layout.number_of_steps, its_layout.number_of_stages) != (
                number_of_steps, number_of_stages):
            raise ValueError("the run %s has %d steps and %d stages, not %d and %d"
                             % (key, its_layout.number_of_steps, its_layout.number_of_stages,
                                number_of_steps, number_of_stages))
        if role == "chain":
            chain = network
        else:
            path = network
    rows = []
    for name, (carry, lookups) in methods(
            numpy_equation=numpy_equation, differentiable_equation=differentiable,
            layout=layout, chain=chain, path=path).items():
        with torch.no_grad():
            carry(of_split.start_state[:tracks_per_event])                  # warm-up
            seconds, finals = [], []
            for e in range(1, events + 1):
                states = of_split.start_state[e * tracks_per_event:(e + 1) * tracks_per_event]
                started = time.perf_counter()
                finals.append(carry(states))
                seconds.append(time.perf_counter() - started)
        reference = of_split.reference_end_state[tracks_per_event:(events + 1) * tracks_per_event]
        radial = conventions.errors(np.concatenate(finals), reference)["radial"]
        ms = 1e3 * float(np.median(seconds))
        rows.append({"method": name, "milliseconds_per_event": ms,
                     "milliseconds_per_track": ms / tracks_per_event,
                     "field_lookups_per_track": lookups,
                     "tracks_per_event": tracks_per_event, "events": events, "threads": 1,
                     "radial_median_micrometres": conventions.median(radial),
                     "radial_percentile_95_micrometres": conventions.quantile(radial, 0.95)})
        report("%-24s %9.2f ms per event  %7.4f ms per track  median %.4g um"
               % (name, ms, ms / tracks_per_event, rows[-1]["radial_median_micrometres"]))
    return rows


def main(arguments=None):
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    parser = argparse.ArgumentParser(description="Time the carrying of one event.")
    parser.add_argument("--store", required=True)
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--number-of-steps", type=int, required=True)
    parser.add_argument("--number-of-stages", type=int, required=True)
    parser.add_argument("--chain-run", required=True, help="a run key, or none")
    parser.add_argument("--path-run", required=True, help="a run key, or none")
    parser.add_argument("--round", required=True, help="a round, or last")
    parser.add_argument("--tracks-per-event", type=int, required=True)
    parser.add_argument("--events", type=int, required=True)
    a = parser.parse_args(arguments)
    none = lambda key: None if key == "none" else key
    return time_per_event(
        Store(a.store), tracks_key=a.tracks, split=a.split,
        number_of_steps=a.number_of_steps, number_of_stages=a.number_of_stages,
        chain_run=none(a.chain_run), path_run=none(a.path_run), round_number=a.round,
        tracks_per_event=a.tracks_per_event, events=a.events,
        report=lambda text: print(text, flush=True))


if __name__ == "__main__":
    main()
