"""The polynomial baseline: a start plus a polynomial correction, as a method.

It is the baseline that a network of stage 2 must beat, and it needs no
training on the farm. The correction is fitted on the training tracks
(fastex/correction.py), checked on the validation tracks, written for the
compiled program, and then scored and timed there on the test tracks like
every other method.

    start                         reference   degree of the correction
    the straight line             cubic       7, 9
    Nystrom, 4, 8, 12 steps       cubic       7      (the integrator reads the cubic map)
    the straight line             linear      7
    Nystrom, 6, 8, 12 steps       linear      7      (the integrator reads the map as LHCb does)

    polynomial_baseline.py fit          the corrections in the store, results/polynomial_baseline_fit.csv
    polynomial_baseline.py gate         results/polynomial_baseline_gate.csv (compiled against numpy)
    polynomial_baseline.py accuracy     results/polynomial_baseline_accuracy.csv (test tracks)
    polynomial_baseline.py time CORE    results/polynomial_baseline_timing.csv
    polynomial_baseline.py table        results/polynomial_baseline.csv
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "python"))

from fastex import FOLDER_OF_EXPORTS, bench, correction  # noqa: E402
from fastex.compiled import run, summary  # noqa: E402

Z0, Z1 = 2648.2, 7826.0
RESULTS = os.path.join(HERE, "results")
READER_OF = {"cubic": "cubic", "linear": "rec"}
CASES = ([("cubic", 0, 7), ("cubic", 0, 9)] + [("cubic", steps, 7) for steps in (4, 8, 12)]
         + [("linear", 0, 7)] + [("linear", steps, 7) for steps in (6, 8, 12)])


def name_of(reference, steps, degree):
    return "%s_%s_polynomial_%d" % (reference, "nystrom_%d" % steps if steps else "straight_line", degree)


def label_of(steps, degree):
    start = "Nystrom, %d steps" % steps if steps else "Straight line"
    return "%s + polynomial of degree %d" % (start, degree)


def split(reference, name):
    folder = os.path.join(FOLDER_OF_EXPORTS, "tracks_" + reference, "splits", name)
    return (np.fromfile(os.path.join(folder, "start_state.f64")).reshape(-1, 5),
            np.fromfile(os.path.join(folder, "reference_end_state.f64")).reshape(-1, 5),
            np.fromfile(os.path.join(folder, "momentum_gev.f64")))


def start_of(states, reference, steps, precision="double"):
    """The start on the last plane, (n, 4): the straight line, or the Nystrom
    integrator with the reader of the reference."""
    if steps == 0:
        length = Z1 - Z0
        return np.stack([states[:, 0] + length * states[:, 2], states[:, 1] + length * states[:, 3],
                         states[:, 2], states[:, 3]], 1)
    return run("nystrom:%d" % steps, states, Z0, Z1, reader=READER_OF[reference],
               precision=precision)[0][:, :4]


def fit():
    rows = []
    for reference, steps, degree in CASES:
        t0 = time.time()
        train_s, train_ref, _ = split(reference, "training")
        val_s, val_ref, val_p = split(reference, "validation")
        # the start is computed in single precision, as the compiled method computes it
        target = train_ref[:, :4] - start_of(train_s, reference, steps, "float")
        coefficients, scale = correction.fit(train_s, target, degree - 1)
        name = name_of(reference, steps, degree)
        correction.export(name, coefficients, scale, degree - 1,
                          start="nystrom" if steps else "straight_line", steps=steps,
                          reference=reference, reader=READER_OF[reference], first_plane_mm=Z0,
                          last_plane_mm=Z1, more=[("total_degree", degree),
                                                  ("training_tracks", len(train_s))])
        start_val = start_of(val_s, reference, steps, "float")
        corrected = start_val + correction.correction(val_s, coefficients, scale, degree - 1)
        row = dict(name=name, reference=reference, steps=steps, degree=degree,
                   terms=len(coefficients), largest_coefficient=float(np.abs(coefficients).max()))
        row.update({"start_" + k: v for k, v in summary(start_val, val_ref[:, :4], val_p).items()
                    if k in ("radial_median_um", "radial_95_um", "worst_band_median_um")})
        row.update({"validation_" + k: v for k, v in summary(corrected, val_ref[:, :4], val_p).items()
                    if k.startswith("radial") or k.startswith("worst")})
        rows.append(row)
        print("%-34s validation: start %.3g / %.3g um -> %.3g / %.3g / %.3g um (median / 95 %% / worst band); "
              "largest coefficient %.3g  (%.0f s)"
              % (name, row["start_radial_median_um"], row["start_radial_95_um"],
                 row["validation_radial_median_um"], row["validation_radial_95_um"],
                 row["validation_worst_band_median_um"], row["largest_coefficient"], time.time() - t0),
              flush=True)
        bench.write(os.path.join(RESULTS, "polynomial_baseline_fit.csv"), rows)


def methods():
    rows = []
    for reference, steps, degree in CASES:
        rows.append(dict(label=label_of(steps, degree), family="start plus polynomial",
                         method="corrected:" + name_of(reference, steps, degree), precision="float",
                         reader=READER_OF[reference], tracks=reference, reference=reference,
                         steps=steps, degree=degree))
        if steps and degree == 7:
            rows.append(dict(label="Nystrom, %d steps, alone" % steps, family="start alone",
                             method="nystrom:%d" % steps, precision="float",
                             reader=READER_OF[reference], tracks=reference, reference=reference,
                             steps=steps, degree=0))
    # the LHCb methods, timed in the same run
    rows += [dict(label="HLT2 adaptive Nystrom (LHCb reference)", family="LHCb", method="rec_step",
                  precision="double", reader="rec", tracks="linear", reference="linear"),
             dict(label="HLT2 adaptive Cash-Karp", family="LHCb", method="rec_cash_karp",
                  precision="double", reader="rec", tracks="linear", reference="linear"),
             dict(label="HLT1 parametrised step", family="LHCb", method="allen_utt",
                  precision="float", reader="allen", tracks="linear_at_2665", reference="linear"),
             dict(label="HLT2 adaptive Nystrom, as LHCb runs it, against the cubic reference",
                  family="LHCb", method="rec_step", precision="double", reader="rec",
                  tracks="cubic", reference="cubic")]
    return rows


def gate():
    """The compiled correction in double precision against numpy, on 1,000 test tracks."""
    rows = []
    for reference, steps, degree in CASES:
        name = name_of(reference, steps, degree)
        start, _, _, _, _ = bench.tracks_of(reference)
        states = start[:1000]
        meta, coefficients, scale = correction.read(name)
        in_numpy = start_of(states, reference, steps, "double") + correction.correction(
            states, coefficients, scale, int(meta["degree"]))
        compiled, _ = run("corrected:" + name, states, Z0, Z1, reader=READER_OF[reference],
                          precision="double")
        single, _ = run("corrected:" + name, states, Z0, Z1, reader=READER_OF[reference],
                        precision="float")
        difference = float(np.abs(compiled[:, :2] - in_numpy[:, :2]).max())
        of_single = np.hypot(single[:, 0] - compiled[:, 0], single[:, 1] - compiled[:, 1]) * 1e3
        rows.append(dict(name=name, largest_difference_from_numpy_mm=difference,
                         passes=difference < 1e-8,
                         single_precision_median_change_um=float(np.median(of_single)),
                         single_precision_largest_change_um=float(of_single.max())))
        print("%-34s compiled against numpy: %.2g mm  %s | single precision: %.3g um median, %.3g um largest"
              % (name, difference, "pass" if difference < 1e-8 else "FAIL",
                 rows[-1]["single_precision_median_change_um"],
                 rows[-1]["single_precision_largest_change_um"]), flush=True)
    bench.write(os.path.join(RESULTS, "polynomial_baseline_gate.csv"), rows)
    print("%d of %d pass" % (sum(r["passes"] for r in rows), len(rows)))


def table():
    both = bench.join(os.path.join(RESULTS, "polynomial_baseline_accuracy.csv"),
                      os.path.join(RESULTS, "polynomial_baseline_timing.csv"))
    both.to_csv(os.path.join(RESULTS, "polynomial_baseline.csv"), index=False)
    print(both[["label", "reference", "field_reads_mean", "ns_lowest", "radial_median_um",
                "radial_95_um", "worst_band_median_um"]].to_string())


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "fit":
        fit()
    elif command == "gate":
        gate()
    elif command == "accuracy":
        bench.accuracy(methods(), os.path.join(RESULTS, "polynomial_baseline_accuracy.csv"))
    elif command == "time":
        bench.time(methods(), sys.argv[2], os.path.join(RESULTS, "polynomial_baseline_timing.csv"))
    elif command == "table":
        table()
    else:
        raise SystemExit("fit, gate, accuracy, time CORE or table")
