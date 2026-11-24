"""The polynomial probe of what_must_a_correction_learn.py at degrees 9 and 11,
for the start with no field read (the straight line) on the cubic reference.

    higher_degrees_for_the_straight_line.py      results/polynomial_probe_straight_line.csv
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastex import bench  # noqa: E402
from fastex.compiled import summary  # noqa: E402
from what_must_a_correction_learn import design, exponents, robust_fit, split, straight_line  # noqa: E402


def main():
    train_s, train_ref, _ = split("cubic", "training")
    val_s, val_ref, val_p = split("cubic", "validation")
    scale = np.quantile(np.abs(train_s), 0.995, axis=0)
    target = train_ref[:, :4] - straight_line(train_s)
    start_val = straight_line(val_s)
    rows = []
    for degree in (5, 7, 9, 11):
        t0 = time.time()
        powers = exponents(degree)
        q, r = np.linalg.qr(design(train_s / scale, powers))
        q_val = np.linalg.solve(r.T, design(val_s / scale, powers).T).T
        coefficients = robust_fit(q, target)
        row = dict(reference="cubic", start="the straight line", field_reads=0, degree=degree,
                   monomials=len(powers), sample="validation")
        row.update({k: v for k, v in summary(start_val + q_val @ coefficients, val_ref[:, :4], val_p).items()
                    if k.startswith("radial") or k.startswith("worst")})
        rows.append(row)
        print("degree %2d, %4d monomials: median %.3g um, 95 %% %.3g um, worst band %.3g um, bands %s  (%.0f s)"
              % (degree, len(powers), row["radial_median_um"], row["radial_95_um"],
                 row["worst_band_median_um"],
                 " ".join("%.3g" % row[k] for k in row if k.startswith("radial_median_um_")),
                 time.time() - t0), flush=True)
        bench.write(os.path.join(HERE, "results", "polynomial_probe_straight_line.csv"), rows)


if __name__ == "__main__":
    main()
