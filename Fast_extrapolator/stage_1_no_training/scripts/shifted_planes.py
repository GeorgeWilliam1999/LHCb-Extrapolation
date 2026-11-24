"""Does a fixed-step integrator keep its accuracy when the planes move?

On the linear map the error of a fixed-step integrator depends on where its
steps fall on the grid of the map (step_length_scan.py). The numbers of steps
that are good for the crossing of this experiment (2648.2 mm to 7826 mm) are
scored here on crossings with other planes: the first plane, the last plane or
both are moved by 5 mm to 50 mm. The adaptive Nystrom integrator of LHCb is
scored beside them, at its default tolerance.

The test tracks are carried from 2648.2 mm to the new first plane with RK4 in
steps of about 0.2 mm; the reference on the new last plane is RK4 with 4,096
steps, both in double precision on the linear map.

    shifted_planes.py      results/shifted_planes.csv
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "python"))

from fastex import bench  # noqa: E402
from fastex.compiled import run, summary  # noqa: E402

Z0, Z1 = 2648.2, 7826.0
SHIFTS = [(0, 0), (0, 5), (0, 10), (0, 25), (0, 50), (5, 0), (10, 0), (25, 0), (50, 0), (-25, 0),
          (25, 25), (50, 50), (-25, -25), (16.8, 0), (51.8, -26.0)]
METHODS = [("nystrom:%d" % n, "Nystrom, %d equal steps" % n, "float", "rec") for n in (19, 20, 31, 32, 33, 36, 37, 47, 62, 69)]
METHODS += [("rec_step", "HLT2 adaptive Nystrom, tolerance 5 um", "double", "rec"),
            ("rec_cash_karp", "HLT2 adaptive Cash-Karp, tolerance 1 um", "double", "rec")]


def main():
    start, _, momentum, _, _ = bench.tracks_of("linear")
    rows = []
    for d0, d1 in SHIFTS:
        z0, z1 = Z0 + d0, Z1 + d1
        if d0 == 0:
            first = start
        else:
            a, b = (Z0, z0) if d0 > 0 else (Z0, z0)
            first, _ = run("rk4:256", start, a, b, reader="linear", precision="double")
        reference, _ = run("rk4:4096", first, z0, z1, reader="linear", precision="double")
        for method, label, precision, reader in METHODS:
            finals, counts = run(method, first, z0, z1, reader=reader, precision=precision)
            row = dict(first_plane_mm=z0, last_plane_mm=z1, shift_of_the_first_plane_mm=d0,
                       shift_of_the_last_plane_mm=d1, label=label, method=method,
                       field_reads_mean=float(counts[:, 0].mean()))
            row.update(summary(finals, reference, momentum))
            rows.append(row)
        print("planes %.1f mm to %.1f mm: %s" % (z0, z1, "  ".join(
            "%s %.2f" % (r["method"].replace("nystrom:", "N"), r["radial_median_um"])
            for r in rows[-len(METHODS):])), flush=True)
    bench.write(os.path.join(HERE, "results", "shifted_planes.csv"), rows)


if __name__ == "__main__":
    main()
