"""The error of a fixed-step integrator against its step length.

On the linear map the error does not fall smoothly with the number of steps:
32 equal steps are far better than 24 or 48. The linear map has a kink at
every plane of its grid (the planes are 100 mm apart, and every track crosses
them at the same z). The scan shows how the error depends on where the steps
fall on that grid.

Two scans, on the linear and on the cubic map, with the Nystrom integrator
(RK4 gives the same errors):

    equal steps      N = 8 .. 96 equal steps from 2648.2 mm to 7826 mm
    one step length  steps of length h from the first plane, h = 100 .. 400 mm;
                     the last step is the rest of the way

Each scan is made on the 2,000 training tracks and on the 13,447 test tracks,
so that a step length can be chosen on tracks that the scores do not use.

    step_length_scan.py      results/step_length_scan.csv
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "python"))

from fastex import FOLDER_OF_EXPORTS, bench  # noqa: E402

Z0, Z1 = 2648.2, 7826.0
CELL = 100.0
FAST = {"linear": ("float", "rec"), "cubic": ("float", "cubic")}
FOLDER_OF_PLANS = os.path.join(FOLDER_OF_EXPORTS, "plans", "step_length_scan")


def planes_of_step_length(h):
    """Planes h apart from the first plane; the last step is the rest. A rest
    that is shorter than a tenth of h is added to the step before it."""
    n = int(np.floor((Z1 - Z0) / h + 1e-9))
    planes = Z0 + h * np.arange(n + 1)
    if Z1 - planes[-1] > 0.1 * h:
        planes = np.append(planes, Z1)
    else:
        planes[-1] = Z1
    return planes


def rows_of(of_map, tracks):
    rows = []
    precision, reader = FAST[of_map]
    for steps in range(8, 97):
        rows.append(dict(label="Nystrom, %d equal steps" % steps, family="equal steps",
                         method="nystrom:%d" % steps, precision=precision, reader=reader,
                         tracks=tracks, map_of_the_field=of_map, steps=steps,
                         step_length_mm=(Z1 - Z0) / steps, field_reads=2 * steps + 1))
    for h in np.arange(100.0, 400.5, 2.0):
        planes = planes_of_step_length(h)
        file = os.path.join(FOLDER_OF_PLANS, "h%g.f64" % h)
        planes.tofile(file)
        rows.append(dict(label="Nystrom, steps of %g mm" % h, family="one step length",
                         method="nystrom_plan:" + file, precision=precision, reader=reader,
                         tracks=tracks, map_of_the_field=of_map, steps=len(planes) - 1,
                         step_length_mm=h, field_reads=2 * (len(planes) - 1) + 1))
    return rows


def main():
    os.makedirs(FOLDER_OF_PLANS, exist_ok=True)
    rows = []
    for of_map in ("linear", "cubic"):
        for tracks in (of_map + "_training", of_map):
            for m in rows_of(of_map, tracks):
                r = bench.score(m, report=lambda text: None)
                r["sample"] = "training" if tracks.endswith("_training") else "test"
                r["step_over_cell"] = r["step_length_mm"] / CELL
                rows.append(r)
            print(of_map, tracks, "done", flush=True)
    bench.write(os.path.join(HERE, "results", "step_length_scan.csv"), rows)


if __name__ == "__main__":
    main()
