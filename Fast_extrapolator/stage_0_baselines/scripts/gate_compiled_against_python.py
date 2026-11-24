"""Gate: every compiled method gives the final states of its Python version.

The tracks are the 1,000 test tracks that are flat in momentum, on the map of
the method. Three comparisons are made and written to one table.

  1. The gate. The compiled program built with -ffp-contract=off (no fused
     multiply-add, so the same operations as Python, one by one) against the
     Python version, in double precision.

         Python version                                    compiled method
         rk4_whole_track of the present package            rk4:N
         fastex.lhcb_integrators (copies of Rec, Allen)    rec_cash_karp,
                                                           allen_adaptive_nystrom,
                                                           allen_fast_nystrom
         whole_track of the trained network                whole_crossing:RUN, chain:RUN

     A method passes when the largest difference of x and y is below 1e-9 mm
     and that of the slopes below 1e-12, and when the counts (field reads,
     accepted steps, failed steps) are the same on every track.

  2. The timed build (fused multiply-add, the default of the compiler) against
     the build of comparison 1, in double precision. Fused arithmetic changes
     the last digits. A fixed-step method must stay below the same limits. An
     adaptive method can take other steps on a few tracks, because its step
     control reacts to the last digits; the table gives how many tracks and how
     far they move.

  3. Single precision against double precision in the timed build. No limit;
     the table gives the median and the largest difference in micrometres.

  4. For a network, the fast tanh (from the vector exponential) against the
     tanh of the C library, in the timed build and double precision.

    gate_compiled_against_python.py <fx built with FX_CONTRACT=off> <fx of the timed build>
        writes results/gate_compiled_against_python.csv
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "python"))

from fastex import FOLDER_OF_EXPORTS, STORE_OF_THE_PRESENT_RUNS, TRACK_SETS  # noqa: E402
from fastex import lhcb_integrators  # noqa: E402
from fastex.export import network_and_layout  # noqa: E402

LIMIT_POSITION_MM = 1e-9
LIMIT_SLOPE = 1e-12

FIXED = [("rk4:16", "cubic"), ("rk4:64", "cubic"), ("rk4:16", "linear"), ("rk4:64", "linear")]
COPIES = [("rec_cash_karp", "linear"), ("rec_cash_karp", "cubic"),
          ("allen_adaptive_nystrom", "linear"), ("allen_adaptive_nystrom", "cubic"),
          ("allen_fast_nystrom", "linear"), ("allen_fast_nystrom", "cubic")]
NETWORKS = ["dd9acf59ac88", "29510f525130", "400182ba3233", "2382b5d314a5", "a8a4083032fd",
            "2e8f364b79e2", "84306b16e99d", "ffc653fadc1e", "6c4a3b022a2f", "29cbba959300",
            "a66c06472d3d", "15242aa17c65", "6af35e016a58", "a0f26d05a973"]


def read_meta(path):
    return dict(line.rstrip("\n").split(" ", 1) for line in open(path))


def compiled(fx, method, of_map, states_file, z0, z1, work, precision="double"):
    out = os.path.join(work, "out.f64")
    counts = os.path.join(work, "counts.i64")
    reader = "cubic" if of_map == "cubic" else "linear"
    subprocess.run([fx, "run", "method=" + method, "precision=" + precision, "map=v8r1_up",
                    "reader=" + reader, "states=" + states_file, "z0=%r" % z0, "z1=%r" % z1,
                    "out=" + out, "counts=" + counts], check=True)
    return (np.fromfile(out, dtype=np.float64).reshape(-1, 5),
            np.fromfile(counts, dtype=np.int64).reshape(-1, 3))


def main(fx_strict, fx_timed):
    from rkpinn import registry
    from rkpinn.evaluation.time_per_event import rk4_whole_track
    from rkpinn.integrators.gauss_legendre_tableau import gauss_legendre_tableau
    from rkpinn.predicted_track.track_layout import TrackLayout
    from rkpinn.run_record.store import Store

    work = os.path.join(FOLDER_OF_EXPORTS, "gate")
    os.makedirs(work, exist_ok=True)
    tracks, fields, equations, files = {}, {}, {}, {}
    for of_map in TRACK_SETS:
        folder = os.path.join(FOLDER_OF_EXPORTS, "tracks_" + of_map)
        meta = read_meta(os.path.join(folder, "meta"))
        start = np.fromfile(os.path.join(folder, "start_state.f64")).reshape(-1, 5)
        sample = np.fromfile(os.path.join(folder, "flat_sample.i64"), dtype=np.int64)
        tracks[of_map] = dict(start=start[sample], z0=float(meta["first_plane_mm"]),
                              z1=float(meta["last_plane_mm"]))
        files[of_map] = os.path.join(work, "states_%s.f64" % of_map)
        tracks[of_map]["start"].tofile(files[of_map])
        fields[of_map] = registry.component("field_map", meta["field_map"])()
        equations[of_map] = registry.component("equation_of_motion", "lhcb")(fields[of_map])

    rows = []

    def compare(method, of_map, python_finals, python_counts=None, adaptive=False):
        t = tracks[of_map]
        args = (method, of_map, files[of_map], t["z0"], t["z1"], work)
        strict, strict_counts = compiled(fx_strict, *args)
        timed, timed_counts = compiled(fx_timed, *args)
        single, _ = compiled(fx_timed, *args, precision="float")
        same = np.ones(len(strict), dtype=bool)
        if python_counts is not None:
            same = (strict_counts == python_counts).all(axis=1)
        d = np.abs(strict - python_finals)
        passed = bool(d[:, :2].max() < LIMIT_POSITION_MM and d[:, 2:4].max() < LIMIT_SLOPE
                      and same.all())
        same_timed = (timed_counts == strict_counts).all(axis=1)
        dt = np.abs(timed - strict)
        fused_ok = bool(adaptive or (dt[:, :2].max() < LIMIT_POSITION_MM
                                     and dt[:, 2:4].max() < LIMIT_SLOPE))
        radial_single = np.hypot(single[:, 0] - timed[:, 0], single[:, 1] - timed[:, 1]) * 1e3
        fast_tanh = float("nan")
        if method.split(":")[0] in ("chain", "whole_crossing"):
            other, _ = compiled(fx_timed, method + ":fast_tanh=1", *args[1:])
            fast_tanh = float(np.abs(other - timed)[:, :2].max())
        rows.append(dict(
            method=method, map=of_map, tracks=len(strict),
            field_reads_mean=float(strict_counts[:, 0].mean()),
            gate_largest_difference_of_position_mm=d[:, :2].max(),
            gate_largest_difference_of_slope=d[:, 2:4].max(),
            gate_tracks_with_other_counts=int((~same).sum()), gate_passed=passed,
            fused_tracks_with_other_counts=int((~same_timed).sum()),
            fused_largest_difference_same_counts_mm=(dt[same_timed][:, :2].max()
                                                     if same_timed.any() else float("nan")),
            fused_largest_difference_all_tracks_mm=dt[:, :2].max(),
            fused_within_limits=fused_ok,
            single_precision_median_um=float(np.median(radial_single)),
            single_precision_largest_um=float(radial_single.max()),
            fast_tanh_largest_difference_mm=fast_tanh))
        r = rows[-1]
        print("%-30s %-6s gate %.1e mm, %d other counts, %s | fused %.1e mm (%d other counts, "
              "all %.1e mm) | single %.3g um median, %.3g um largest | fast tanh %.1e mm"
              % (method, of_map, r["gate_largest_difference_of_position_mm"],
                 r["gate_tracks_with_other_counts"], "pass" if passed else "FAIL",
                 r["fused_largest_difference_same_counts_mm"], r["fused_tracks_with_other_counts"],
                 r["fused_largest_difference_all_tracks_mm"], r["single_precision_median_um"],
                 r["single_precision_largest_um"], r["fast_tanh_largest_difference_mm"]),
              flush=True)

    for method, of_map in FIXED:
        t = tracks[of_map]
        steps = int(method.split(":")[1])
        layout = TrackLayout(t["z0"], t["z1"], steps, gauss_legendre_tableau(2))
        compare(method, of_map, rk4_whole_track(equations[of_map], t["start"], layout))

    for method, of_map in COPIES:
        t = tracks[of_map]
        finals, counts = lhcb_integrators.carry_tracks(method, t["start"], t["z0"], t["z1"],
                                                       fields[of_map])
        compare(method, of_map, finals, counts, adaptive=method != "allen_fast_nystrom")

    store = Store(STORE_OF_THE_PRESENT_RUNS)
    for key in NETWORKS:
        network, layout, configuration, _ = network_and_layout(store, key)
        of_map = "cubic" if configuration["equation_of_motion"]["field_map"].endswith("_cubic") \
            else "linear"
        method = ("whole_crossing:" if network.kind == "whole_crossing_network" else "chain:") + key
        finals = np.asarray(network.whole_track(tracks[of_map]["start"], layout).final_state())
        compare(method, of_map, finals)

    with open(os.path.join(HERE, "results", "gate_compiled_against_python.csv"), "w",
              newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    failed = [r for r in rows if not (r["gate_passed"] and r["fused_within_limits"])]
    print("%d of %d methods pass" % (len(rows) - len(failed), len(rows)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
