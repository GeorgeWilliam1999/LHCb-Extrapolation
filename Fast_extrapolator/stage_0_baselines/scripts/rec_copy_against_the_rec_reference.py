"""Check of the copies of the two HLT2 integrators against the LHCb software.

The Rec release holds a test of its extrapolators (Tr/TrackExtrapolators,
tests/options/test_extrapolators.py) and the reference output of that test
(tests/refs/test_extrapolators.yaml). The test algorithm ExtrapolatorTester
carries a fixed grid of 1,331 states from z = 3000 mm to z = 7000 mm:

    q/p from -0.0004 to +0.0004 per MeV (2.5 GeV), 11 values
    tx  from -0.3 to +0.3, 11 values;  ty from -0.25 to +0.25, 11 values
    x = tx * 3000, y = ty * 3000

and prints the final (x, y, tx, ty) of every extrapolator with 3 decimals for
the positions (1 um) and 4 for the slopes.

Two tools are checked, each against the numbers that the real tool printed:

    rec_cash_karp   TrackRungeKuttaExtrapolator with its default properties. The
                    printed numbers are those of the extrapolator "Reference" in
                    the reference output of the Rec release.
    rec_step        TrackSTEPExtrapolator with its default properties (the
                    adaptive Runge-Kutta-Nystrom integrator). The Rec test does
                    not have it, so the printed numbers come from our own run of
                    the test with that tool
                    (lhcb_software/rec_extrapolator_test/output/print_one_event_step.log).

The test runs with a magnet-down simulation tag; for simulation the conditions
database names the field map field.v5r0 (Conditions/LHCb/HardwareProperties/
Magnet.yml, first interval), and the log of our run shows that the software
opened field.v5r0.down.bin.

This script carries the same grid through the compiled copy, with the field
read of LHCb (reader `rec`), on each map we have, and compares with the
printed numbers. The map that the test used shows itself: on it the copy must
print the same numbers.

    rec_copy_against_the_rec_reference.py <path of fx>
        writes results/rec_copy_against_the_rec_reference.csv
"""
from __future__ import annotations

import csv
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "python"))

from fastex import FOLDER_OF_EXPORTS  # noqa: E402

PRINTED = {
    "rec_cash_karp": ("/cvmfs/lhcb.cern.ch/lib/lhcb/REC/REC_v41r3/Tr/TrackExtrapolators/tests/refs/"
                      "test_extrapolators.yaml"),
    "rec_step": os.path.join(os.path.dirname(HERE), "lhcb_software", "rec_extrapolator_test",
                             "output", "print_one_event_step.log"),
}
C_LIGHT = 299.792458                  # Gaudi::Units::c_light in mm/ns
Z1, Z2 = 3000.0, 7000.0
MAPS = ["v5r0_down", "v5r0_up", "v8r1_down", "v8r1_up"]


def grid_of_the_test() -> np.ndarray:
    """The 1,331 states of ExtrapolatorTester, in its order, with q/p in the
    unit of our states (q/p per MeV times c_light)."""
    qopmax, qopmin, txmax, txmin, tymax, tymin, nbins = 0.0004, -0.0004, 0.3, -0.3, 0.25, -0.25, 11
    dqop, dtx, dty = (qopmax - qopmin) / (nbins - 1), (txmax - txmin) / (nbins - 1), \
        (tymax - tymin) / (nbins - 1)
    states = []
    for iqop in range(nbins):
        for itx in range(nbins):
            for ity in range(nbins):
                qop, tx, ty = qopmin + iqop * dqop, txmin + itx * dtx, tymin + ity * dty
                states.append([tx * Z1, ty * Z1, tx, ty, qop * C_LIGHT])
    return np.array(states)


def reference_of_the_test(path):
    """The printed final states of the extrapolator "Reference": (1331, 4),
    not a number where the test printed none."""
    number = r"\(\s*(-?[\d.]+),\s*(-?[\d.]+),\s*(-?[\d.]+),\s*(-?[\d.]+)\s*\)"
    printed, row = [], -1
    for line in open(path):
        if "ExtrapolatorTester" not in line:
            continue
        if "Propagating" in line:
            row += 1
            printed.append([np.nan] * 4)
        elif re.search(r"\bReference ->", line):
            printed[row] = [float(v) for v in re.search(number, line).groups()]
    return np.array(printed)


def main(fx):
    work = os.path.join(FOLDER_OF_EXPORTS, "gate")
    os.makedirs(work, exist_ok=True)
    states = grid_of_the_test()
    states_file = os.path.join(work, "grid_of_the_rec_test.f64")
    states.tofile(states_file)
    rows = []
    for method, path in PRINTED.items():
        reference = reference_of_the_test(path)
        assert len(states) == len(reference) == 1331
        printed = np.isfinite(reference).all(axis=1)
        for name in MAPS:
            out = os.path.join(work, "grid_out.f64")
            counts = os.path.join(work, "grid_counts.i64")
            subprocess.run([fx, "run", "method=" + method, "precision=double", "map=" + name,
                            "reader=rec", "states=" + states_file, "z0=%r" % Z1, "z1=%r" % Z2,
                            "out=" + out, "counts=" + counts], check=True)
            finals = np.fromfile(out).reshape(-1, 5)[:, :4]
            reads = np.fromfile(counts, dtype=np.int64).reshape(-1, 3)[:, 0]
            # the copy, printed as the test prints: 3 decimals for x and y, 4 for the slopes
            d_position = np.abs(np.round(finals[:, :2], 3) - reference[:, :2])[printed]
            d_slope = np.abs(np.round(finals[:, 2:], 4) - reference[:, 2:])[printed]
            raw = np.abs(finals[:, :2] - reference[:, :2])[printed].max(axis=1)
            same = (d_position.max(axis=1) < 0.5e-3) & (d_slope.max(axis=1) < 0.5e-4)
            within = (d_position.max(axis=1) < 1.5e-3) & (d_slope.max(axis=1) < 1.5e-4)
            rows.append(dict(
                method=method, printed_by=path, map=name, states=len(states),
                states_printed_by_the_test=int(printed.sum()),
                states_with_the_same_printed_numbers=int(same.sum()),
                states_within_one_last_digit=int(within.sum()),
                median_difference_of_position_um=float(np.median(raw) * 1e3),
                percentile_99_of_difference_um=float(np.quantile(raw, 0.99) * 1e3),
                largest_difference_of_position_mm=float(raw.max()),
                field_reads_mean=float(reads.mean())))
            r = rows[-1]
            print("%-14s %-10s same printed numbers %4d of %4d, largest difference %.4g mm, "
                  "field reads %.1f" % (method, name, r["states_with_the_same_printed_numbers"],
                                         r["states_printed_by_the_test"],
                                         r["largest_difference_of_position_mm"],
                                         r["field_reads_mean"]), flush=True)
    with open(os.path.join(HERE, "results", "rec_copy_against_the_rec_reference.csv"), "w",
              newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main(sys.argv[1])
