"""The errors of every series of the figure, per track, and their statistics.

    step_length_errors.py networks     the trained networks of the store and the frozen networks
    step_length_errors.py references   the exact scheme and classic fourth-order Runge-Kutta

Every error is radial, sqrt(dx^2 + dy^2) in micrometres, on the test split,
against the sixth-order reference at 0.1 mm stored with the tracks.

    endpoint     the whole chain from the first plane (2648.2 mm) to the last (7826.0 mm),
                 over the 1,452 test tracks
    single step  one step from the reference state on every start plane, against the
                 reference one step later; networks and Runge-Kutta over all 1,452 test
                 tracks, the exact scheme over the first 200 (a solve per step is slow)

The statistics are written to results/*.csv; the per-track errors are kept outside
the repository, in CACHE, so that a series is not computed twice. Reads the store
and the archive; writes nothing to either.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
STORE = "/data/bfys/gscriven/rkpinn_store"
CACHE = "/data/bfys/gscriven/rkpinn_analysis_cache/does_the_error_fall_with_the_step_length"
RESULTS = os.path.join(HERE, "results")
TRACKS_KEY = "12a8d35c3165"
TRACKS_FOR_THE_EXACT_SINGLE_STEP = 200
BANDS = {"all momenta": (0.0, np.inf), "3 to 8 GeV": (3.0, 8.0)}
QUANTILES = (0.16, 0.50, 0.84)

from rkpinn import registry                                              # noqa: E402
from rkpinn.integrators.gauss_legendre_tableau import gauss_legendre_tableau  # noqa: E402
from rkpinn.predicted_track.predicted_track import steps_of_the_exact_scheme  # noqa: E402
from rkpinn.run_record.configuration import read_configuration           # noqa: E402
from rkpinn.run_record.store import Store                                # noqa: E402
from rkpinn.track_data.exact_states import load_exact_states             # noqa: E402
from rkpinn.track_data.load_tracks import load_tracks                    # noqa: E402
from rkpinn.training.checkpoints import RunFolder                        # noqa: E402
from rkpinn.training.round_trainer import build_run                      # noqa: E402

# the trained networks of the package: (series, run key)
NETWORKS = [
    ("straight line, per-track scale", "a66c06472d3d"),    # 8 / 2
    ("straight line, per-track scale", "6c4a3b022a2f"),    # 16 / 2
    ("straight line, per-track scale", "29cbba959300"),    # 32 / 2
    ("straight line, per-track scale", "25dfa17ef168"),    # 64 / 2
    ("straight line, per-track scale", "15242aa17c65"),    # 128 / 2
    ("straight line, per-track scale", "eeb809e9254d"),    # 256 / 2
    ("straight line, per-track scale", "4543836de1ce"),    # 64 / 4
    ("straight line, per-track scale", "9aed5ce39ff8"),    # 64 / 8
    ("straight line, per-track scale", "06a45b4dd0d3"),    # 64 / 16
    ("direct states", "49e5e2450eaa"),                     # 64 / 2
    ("direct states", "75200e48408c"),                     # 128 / 8
    ("direct states", "f38ec3e29819"),                     # 256 / 16
]
# the frozen networks of the second mini-paper, pooled loss, read with their stages summed
ARCHIVE = PROJECT + "/Archive/single_network_chain_discrete_approach"
FROZEN = [(64, 2, "N064_q02"), (128, 8, "N128_q08"), (256, 16, "N256_q16")]
REFERENCE_STEPS = [2, 4, 8, 16, 32, 64, 128, 256]
REFERENCE_STAGES = [2, 4, 8, 16]

store = Store(STORE)
tracks = load_tracks(store, TRACKS_KEY)
test = tracks.test
reference = test.reference_states_on_planes               # (n, 257, 5)
momentum = test.momentum_gev
LENGTH_MM = tracks.last_plane_mm - tracks.first_plane_mm


def radial_um(predicted, wanted):
    d = (np.asarray(predicted)[..., :2] - np.asarray(wanted)[..., :2]) * 1e3
    return np.hypot(d[..., 0], d[..., 1])


def reference_on_the_planes_of(number_of_steps):
    on = np.append(tracks.planes_a_step_starts_on(number_of_steps),
                   tracks.number_of_steps_between_planes)
    return reference[:, on]                                # (n, N + 1, 5)


def statistics(endpoint, single_step):
    """endpoint (n,), single_step (n_tracks_used, N): rows per momentum band."""
    rows = []
    for band, (low, high) in BANDS.items():
        here = (momentum >= low) & (momentum < high)
        e = endpoint[here]
        s = single_step[here[:len(single_step)]].reshape(-1)
        row = dict(momentum_band=band, number_of_tracks=int(here.sum()),
                   tracks_of_the_single_step=int(here[:len(single_step)].sum()))
        for name, values in (("endpoint", e), ("single_step", s)):
            for qu, v in zip(QUANTILES, np.quantile(values, QUANTILES)):
                row["%s_radial_p%02d_micrometres" % (name, round(qu * 100))] = float(v)
        rows.append(row)
    return rows


def cached(name, compute):
    path = os.path.join(CACHE, name + ".npz")
    if os.path.isfile(path):
        with np.load(path) as loaded:
            return loaded["endpoint"], loaded["single_step"]
    endpoint, single_step = compute()
    os.makedirs(CACHE, exist_ok=True)
    np.savez(path, endpoint=endpoint, single_step=single_step)
    return endpoint, single_step


def write_rows(file, rows):
    os.makedirs(RESULTS, exist_ok=True)
    columns = list(rows[0])
    with open(os.path.join(RESULTS, file), "w", newline="") as handle:
        writer = csv.DictWriter(handle, columns)
        writer.writeheader()
        writer.writerows(rows)


# ------------------------------------------------------------------ networks
def errors_of_a_run(key):
    folder = RunFolder(store, key)
    state = folder.state()
    round_number = state["rounds_done"]
    configuration = read_configuration(folder.path("configuration.yaml"))
    run = build_run(configuration, store, folder.constants())
    run.network.load_state_dict(folder.weights_of_round(round_number))
    N, layout = run.layout.number_of_steps, run.layout

    def compute():
        on_planes = reference_on_the_planes_of(N)
        whole = run.network.whole_track(test.start_state, layout)
        endpoint = radial_um(np.asarray(whole.final_state()), reference[:, -1])
        single = np.empty((len(on_planes), N))
        with torch.no_grad():
            for k in range(N):
                planes = torch.full((len(on_planes),), float(layout.start_planes_mm[k]),
                                    dtype=torch.float64)
                track = run.network.predict(torch.as_tensor(on_planes[:, k]), planes)
                single[:, k] = radial_um(track.end_state(0).numpy(), on_planes[:, k + 1])
        return endpoint, single

    endpoint, single = cached("network_%s_round_%04d" % (key, round_number), compute)
    summary = None
    reports = folder.path("reports")
    if os.path.isdir(reports):
        for name in sorted(os.listdir(reports)):
            if name.startswith("round_%04d_at_" % round_number):
                with open(os.path.join(reports, name, "summary.json")) as handle:
                    summary = json.load(handle)
    check = {}
    if summary is not None:
        check = dict(
            endpoint_median_of_the_standard_report_micrometres=summary["endpoint_radial_median_micrometres"],
            single_step_median_of_the_standard_report_micrometres=summary["single_step_radial_median_micrometres"])
    described = dict(run_key=key, number_of_steps=N, number_of_stages=layout.number_of_stages,
                     step_length_mm=layout.step_length_mm, round=round_number,
                     state=state["state"], commit_of_the_run=folder.provenance()["commit"][:8])
    return described, endpoint, single, check


def errors_of_a_frozen_network(N, q, cell):
    """The frozen ChainNetwork with its end state summed from its stages, as in the
    audit of 2026-10-19 (project library, shared_scorer.py)."""
    e1 = ARCHIVE + "/Block_E_single_network_chain/E1_Network_grid"
    sys.path.insert(0, e1)
    sys.path.insert(0, ARCHIVE)
    import use_shared  # noqa: F401
    from _shared.reference import make_field
    from _shared.model import LHCbRates
    import chain_network

    field = make_field("up")
    rates = LHCbRates(field)
    model = chain_network.load_network(e1 + "/results/" + cell, field)
    tableau = gauss_legendre_tableau(q)
    b, c = torch.as_tensor(tableau.weights), torch.as_tensor(tableau.nodes)
    dz = float(model.dz)
    starts = tracks.start_planes_mm(N)

    def one_step(states, z):
        St = torch.as_tensor(states)
        zt = torch.full((len(St),), float(z), dtype=torch.float64)
        emitted = model(St, zt)
        F = rates(emitted[:, :q], St[:, 4:5], zt[:, None] + c[None, :] * dz)
        end = St[:, :4] + dz * torch.einsum("j,njd->nd", b, F)
        return np.concatenate([end.numpy(), states[:, 4:5]], axis=1)

    def compute():
        on_planes = reference_on_the_planes_of(N)
        with torch.no_grad():
            S = np.array(test.start_state, dtype=np.float64)
            for k in range(N):
                S = one_step(S, starts[k])
            endpoint = radial_um(S, reference[:, -1])
            single = np.stack([radial_um(one_step(on_planes[:, k], starts[k]), on_planes[:, k + 1])
                               for k in range(N)], axis=1)
        return endpoint, single

    endpoint, single = cached("frozen_%s_stages_summed" % cell, compute)
    described = dict(run_key="frozen " + cell, number_of_steps=N, number_of_stages=q,
                     step_length_mm=dz, round="last restart", state="at the restart cap",
                     commit_of_the_run="archive")
    return described, endpoint, single, {}


def networks():
    rows = []
    for series, key in NETWORKS:
        if not os.path.isfile(os.path.join(STORE, "runs", key, "state.json")):
            print("%s: not in the store yet" % key)
            continue
        if not RunFolder(store, key).state().get("rounds_done"):
            print("%s: no round written yet" % key)
            continue
        described, endpoint, single, check = errors_of_a_run(key)
        for row in statistics(endpoint, single):
            rows.append(dict(series=series, **described, **row, **check))
        print("%s: done" % key)
    for N, q, cell in FROZEN:
        described, endpoint, single, check = errors_of_a_frozen_network(N, q, cell)
        for row in statistics(endpoint, single):
            rows.append(dict(series="frozen weights, stages summed", **described, **row))
        print("frozen %s: done" % cell)
    for row in rows:
        for name in ("endpoint_median_of_the_standard_report_micrometres",
                     "single_step_median_of_the_standard_report_micrometres"):
            row.setdefault(name, "")
    write_rows("networks.csv", rows)


# ---------------------------------------------------------------- references
_equation = None


def _equation_of_motion():
    global _equation
    if _equation is None:
        field_map = registry.component("field_map", "v8r1_up")()
        _equation = registry.component("equation_of_motion", "lhcb")(field_map)
    return _equation


def _exact_single_steps(arguments):
    N, q, k = arguments
    on_planes = reference_on_the_planes_of(N)[:TRACKS_FOR_THE_EXACT_SINGLE_STEP]
    step = steps_of_the_exact_scheme(_equation_of_motion(), on_planes[:, k],
                                     float(tracks.start_planes_mm(N)[k]), LENGTH_MM / N,
                                     gauss_legendre_tableau(q))
    return radial_um(step.end_state(0), on_planes[:, k + 1])


def rk4_step(equation, S, z, h):
    k1 = equation.rates(S, z)
    k2 = equation.rates(S + 0.5 * h * k1, z + 0.5 * h)
    k3 = equation.rates(S + 0.5 * h * k2, z + 0.5 * h)
    k4 = equation.rates(S + h * k3, z + h)
    return S + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def references(workers):
    rows = []
    equation = _equation_of_motion()
    for N in REFERENCE_STEPS:
        h, starts = LENGTH_MM / N, tracks.start_planes_mm(N)

        def compute_rk4():
            on_planes = reference_on_the_planes_of(N)
            S = np.array(test.start_state, dtype=np.float64)
            for k in range(N):
                S = rk4_step(equation, S, starts[k], h)
            single = np.stack([radial_um(rk4_step(equation, on_planes[:, k], starts[k], h),
                                         on_planes[:, k + 1]) for k in range(N)], axis=1)
            return radial_um(S, reference[:, -1]), single

        endpoint, single = cached("rk4_%03d_steps" % N, compute_rk4)
        for row in statistics(endpoint, single):
            rows.append(dict(scheme="classic fourth-order Runge-Kutta", number_of_steps=N,
                             number_of_stages=4, step_length_mm=h, **row))
        for q in REFERENCE_STAGES:
            def compute_exact():
                states = load_exact_states(store, TRACKS_KEY, "test", N, q)[0]
                endpoint = radial_um(states[:, -1], reference[:, -1])
                with Pool(workers) as pool:
                    columns = pool.map(_exact_single_steps, [(N, q, k) for k in range(N)])
                return endpoint, np.stack(columns, axis=1)

            started = time.time()
            endpoint, single = cached("exact_%03d_steps_%02d_stages" % (N, q), compute_exact)
            for row in statistics(endpoint, single):
                rows.append(dict(scheme="exact Gauss-Legendre scheme", number_of_steps=N,
                                 number_of_stages=q, step_length_mm=h, **row))
            print("exact %d / %d: %.0f s" % (N, q, time.time() - started))
    write_rows("references.csv", rows)


if __name__ == "__main__":
    torch.set_num_threads(1)
    if sys.argv[1] == "networks":
        networks()
    elif sys.argv[1] == "references":
        references(int(sys.argv[2]) if len(sys.argv) > 2 else 8)
    else:
        raise SystemExit(__doc__)
