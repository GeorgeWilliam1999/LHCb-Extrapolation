"""The two analyses of tranche 1 of the improvement plan (2026-10-28) that need no farm time.

    tranche1_analyses.py rounds      a. the test endpoint error at the last round, at the best
                                        round by validation, and for the average of the outputs
                                        of the last five snapshots
    tranche1_analyses.py steps       b. the correlation of the single-step errors along each test
                                        track, and the endpoint error against sqrt(64) times the
                                        single-step error
    tranche1_analyses.py symmetry    b. the mirror symmetry of the field map in y

Every position error is in micrometres, every slope error is dimensionless, on the 1,452 test
tracks, against the sixth-order reference at 0.1 mm stored with the tracks. The endpoint is the
last plane (7826.0 mm), after the whole chain from the first plane (2648.2 mm).

Reads the store and the field map; writes only to results/ beside this file.
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
STORE = "/data/bfys/gscriven/rkpinn_store"
RESULTS = os.path.join(HERE, "results")
TRACKS_KEY = "12a8d35c3165"
SNAPSHOTS_AVERAGED = 5

from rkpinn.equation_of_motion.field_map import load_field_map         # noqa: E402
from rkpinn.run_record.configuration import read_configuration         # noqa: E402
from rkpinn.run_record.report_of_a_run import best_round               # noqa: E402
from rkpinn.run_record.store import Store                              # noqa: E402
from rkpinn.track_data.load_tracks import load_tracks                  # noqa: E402
from rkpinn.training.checkpoints import RunFolder                      # noqa: E402
from rkpinn.training.round_trainer import build_run                    # noqa: E402

RUNS_OF_ANALYSIS_A = [("straight line, per-track scale", "25dfa17ef168"),
                      ("straight line, one global scale", "9cef9c160b7d"),
                      ("direct states", "49e5e2450eaa")]
RUNS_OF_ANALYSIS_B = [("straight line, per-track scale", "25dfa17ef168"),
                      ("direct states", "49e5e2450eaa")]

store = Store(STORE)
tracks = load_tracks(store, TRACKS_KEY)
test = tracks.test
reference = test.reference_states_on_planes               # (n, 257, 5)


def radial_um(predicted, wanted):
    d = (np.asarray(predicted)[..., :2] - np.asarray(wanted)[..., :2]) * 1e3
    return np.hypot(d[..., 0], d[..., 1])


def write_rows(file, rows):
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, file), "w", newline="") as handle:
        writer = csv.DictWriter(handle, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("written: results/%s" % file)


def run_of(key):
    folder = RunFolder(store, key)
    configuration = read_configuration(folder.path("configuration.yaml"))
    run = build_run(configuration, store, folder.constants())
    return folder, configuration, run


def final_states_of_round(run, folder, round_number):
    run.network.load_state_dict(folder.weights_of_round(round_number))
    with torch.no_grad():
        whole = run.network.whole_track(test.start_state, run.layout)
    return np.asarray(whole.final_state())


def endpoint_report_of_round(folder, round_number):
    """The endpoint radial median of the standard report of that round, when one exists."""
    reports = folder.path("reports")
    if os.path.isdir(reports):
        for name in sorted(os.listdir(reports)):
            if name.startswith("round_%04d_at_" % round_number) and "refit" not in name:
                with open(os.path.join(reports, name, "summary.json")) as handle:
                    return json.load(handle)["endpoint_radial_median_micrometres"]
    return ""


# ------------------------------------------------------------------ analysis a
def rounds():
    rows = []
    for series, key in RUNS_OF_ANALYSIS_A:
        folder, configuration, run = run_of(key)
        last = folder.state()["rounds_done"]
        quantity = configuration["training"]["stopping_rule"]["quantity"]
        best = best_round(folder, quantity)
        validation = {int(r["round"]): float(r[quantity]) for r in folder.rounds()}
        averaged = list(range(last - SNAPSHOTS_AVERAGED + 1, last + 1))
        finals = {r: final_states_of_round(run, folder, r) for r in sorted(set(averaged + [best]))}
        choices = [("last round", [last]), ("best round by validation", [best]),
                   ("average of the outputs of the last %d snapshots" % SNAPSHOTS_AVERAGED,
                    averaged)]
        for choice, used in choices:
            final = np.mean([finals[r][:, :4] for r in used], axis=0)
            error = radial_um(final, reference[:, -1])
            rows.append(dict(
                series=series, run_key=key, choice=choice,
                rounds_used=" ".join(str(r) for r in used), rounds_done=last,
                validation_quantity_of_the_rounds_used_micrometres=" ".join(
                    "%.4g" % validation[r] for r in used),
                test_endpoint_radial_median_micrometres=float(np.median(error)),
                test_endpoint_radial_p84_micrometres=float(np.quantile(error, 0.84)),
                standard_report_of_the_round_micrometres=(
                    endpoint_report_of_round(folder, used[0]) if len(used) == 1 else "")))
            print(rows[-1])
    write_rows("a_best_round_and_snapshot_average.csv", rows)


# ------------------------------------------------------------------ analysis b
def lag_one_autocorrelation(errors):
    """errors (n, S): the lag-1 autocorrelation along the steps, one value per track."""
    centred = errors - errors.mean(axis=1, keepdims=True)
    above = (centred[:, :-1] * centred[:, 1:]).sum(axis=1)
    below = (centred ** 2).sum(axis=1)
    return above / below


def steps():
    rows = []
    for series, key in RUNS_OF_ANALYSIS_B:
        folder, configuration, run = run_of(key)
        last = folder.state()["rounds_done"]
        layout = run.layout
        N = layout.number_of_steps
        on = np.append(tracks.planes_a_step_starts_on(N), tracks.number_of_steps_between_planes)
        on_planes = reference[:, on]                            # (n, N + 1, 5)
        run.network.load_state_dict(folder.weights_of_round(last))
        signed = np.empty((len(on_planes), N, 4))
        with torch.no_grad():
            for k in range(N):
                planes = torch.full((len(on_planes),), float(layout.start_planes_mm[k]),
                                    dtype=torch.float64)
                end = run.network.predict(torch.as_tensor(on_planes[:, k]), planes).end_state(0)
                signed[:, k] = end.numpy()[:, :4] - on_planes[:, k + 1, :4]
            whole = run.network.whole_track(test.start_state, layout)
        signed[..., :2] *= 1e3                                  # positions in micrometres
        chained = np.asarray(whole.final_state())[:, :4] - reference[:, -1, :4]
        chained[:, :2] *= 1e3
        single_radial = np.hypot(signed[..., 0], signed[..., 1])     # (n, N)
        endpoint_radial = np.hypot(chained[:, 0], chained[:, 1])
        rms_single = np.sqrt((single_radial ** 2).mean(axis=1))
        # the endpoint error a sum of the single-step errors would give, each carried to
        # the last plane along a straight line: e_x + e_tx (z_last - z_end of the step)
        lever_mm = layout.last_plane_mm - (np.asarray(layout.start_planes_mm) + layout.step_length_mm)
        carried_x = (signed[..., 0] + signed[..., 2] * lever_mm * 1e3).sum(axis=1)
        carried_y = (signed[..., 1] + signed[..., 3] * lever_mm * 1e3).sum(axis=1)
        row = dict(series=series, run_key=key, round=last, number_of_steps=N,
                   number_of_tracks=len(on_planes))
        for index, name in enumerate(("x", "y", "slope_in_x", "slope_in_y")):
            r1 = lag_one_autocorrelation(signed[..., index])
            row["lag_1_autocorrelation_of_%s_median_over_tracks" % name] = float(np.median(r1))
            row["lag_1_autocorrelation_of_%s_p16" % name] = float(np.quantile(r1, 0.16))
            row["lag_1_autocorrelation_of_%s_p84" % name] = float(np.quantile(r1, 0.84))
        row.update(
            single_step_radial_median_micrometres=float(np.median(single_radial)),
            endpoint_radial_median_micrometres=float(np.median(endpoint_radial)),
            ratio_of_medians_endpoint_over_sqrt_steps_times_single_step=float(
                np.median(endpoint_radial) / (np.sqrt(N) * np.median(single_radial))),
            ratio_per_track_endpoint_over_sqrt_steps_times_rms_single_step_median=float(
                np.median(endpoint_radial / (np.sqrt(N) * rms_single))),
            ratio_per_track_endpoint_over_steps_times_rms_single_step_median=float(
                np.median(endpoint_radial / (N * rms_single))),
            carried_sum_radial_median_micrometres=float(np.median(np.hypot(carried_x, carried_y))),
            correlation_of_carried_sum_and_chained_endpoint_in_x=float(
                np.corrcoef(carried_x, chained[:, 0])[0, 1]),
            correlation_of_carried_sum_and_chained_endpoint_in_y=float(
                np.corrcoef(carried_y, chained[:, 1])[0, 1]))
        print(row)
        rows.append(row)
    write_rows("b_correlation_of_step_errors.csv", rows)


def symmetry():
    field = load_field_map("v8r1_up")
    nx, ny, nz = (int(n) for n in field.number_of_points)
    cell_y = 1.0 / field.inverse_cell_size[1]
    highest_y = field.lowest_corner[1] + (ny - 1) * cell_y
    grid_is_mirrored = bool(np.isclose(field.lowest_corner[1], -highest_y))
    if not grid_is_mirrored:
        raise SystemExit("the grid in y is not symmetric about 0: %r to %r"
                         % (field.lowest_corner[1], highest_y))
    s = field.scale_to_tesla
    grids = (field.field_x * s, field.field_y * s, field.field_z * s)       # (nz, ny, nx)
    cells = 1.0 / field.inverse_cell_size
    x_of_grid, z_of_grid = (field.lowest_corner[i] + np.arange(n) * cells[i]
                            for i, n in ((0, nx), (2, nz)))
    y_of_grid = field.lowest_corner[1] + np.arange(ny) * cell_y
    # the points the test tracks pass: their reference states on all 257 planes
    planes = np.append(tracks.start_planes_mm(tracks.number_of_steps_between_planes),
                       tracks.last_plane_mm)
    x, y = reference[..., 0].reshape(-1), reference[..., 1].reshape(-1)
    z = np.broadcast_to(planes[None, :], reference.shape[:2]).reshape(-1)
    on_tracks, mirrored_on_tracks = field(x, y, z), field(x, -y, z)
    # the box of the grid that holds those points
    box = np.ix_((z_of_grid >= planes[0] - cells[2]) & (z_of_grid <= planes[-1] + cells[2]),
                 np.abs(y_of_grid) <= np.abs(y).max() + cells[1],
                 (x_of_grid >= x.min() - cells[0]) & (x_of_grid <= x.max() + cells[0]))
    regions = [
        ("every point of the grid", [g for g in grids], [g[:, ::-1, :] for g in grids]),
        ("grid points in the box the test tracks occupy", [g[box] for g in grids],
         [g[:, ::-1, :][box] for g in grids]),
        ("the test tracks, reference states on 257 planes", list(on_tracks),
         list(mirrored_on_tracks)),
    ]
    rows = []
    for region, values, mirrored in regions:
        peak = float(np.abs(values[1]).max())
        for index, name, sign in ((1, "By(x,-y,z) - By(x,y,z)", +1),
                                  (0, "Bx(x,-y,z) + Bx(x,y,z)", -1),
                                  (2, "Bz(x,-y,z) + Bz(x,y,z)", -1)):
            difference = np.abs(mirrored[index] - sign * values[index]).reshape(-1)
            rows.append(dict(
                region=region, relation=name, points=int(difference.size),
                peak_abs_By_in_the_region_tesla=peak,
                max_abs_difference_tesla=float(difference.max()),
                max_abs_difference_over_peak_By=float(difference.max() / peak),
                p99_abs_difference_over_peak_By=float(np.quantile(difference, 0.99) / peak),
                median_abs_difference_over_peak_By=float(np.median(difference) / peak)))
            print(rows[-1])
    write_rows("b_field_map_mirror_symmetry.csv", rows)


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("THREADS", "1")))
    {"rounds": rounds, "steps": steps, "symmetry": symmetry}[sys.argv[1]]()
