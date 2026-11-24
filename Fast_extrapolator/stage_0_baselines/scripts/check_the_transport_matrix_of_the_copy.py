"""A check of the transport matrix in the copy of the HLT2 integrator.

The copy (cpp/lhcb/rec_cash_karp.h, option jacobian=1) computes the 4 x 3
block of the transport matrix as Rec does: the derivatives of the final
(x, y, tx, ty) with respect to the start (tx, ty, q/p). The Rec test prints no
transport matrix, so the block cannot be compared with the LHCb software.
Here it is compared with central differences of a fine integration (RK4 with
512 steps) on the 1,000 flat tracks.

The two are not the same quantity: Rec takes no derivative of the field with
respect to the position, and differences of the true motion do. The check
therefore finds a wrong sign, index or factor, not a small error.

    check_the_transport_matrix_of_the_copy.py     writes results/transport_matrix_of_the_copy.csv
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "python"))

from fastex import FOLDER_OF_EXPORTS  # noqa: E402
from fastex.compiled import SCRATCH, program, run  # noqa: E402

Z0, Z1 = 2648.2, 7826.0
NAMES_OUT, NAMES_IN = ("x", "y", "tx", "ty"), ("tx0", "ty0", "qop0")


def main():
    folder = os.path.join(FOLDER_OF_EXPORTS, "tracks_linear")
    start = np.fromfile(os.path.join(folder, "start_state.f64")).reshape(-1, 5)
    start = start[np.fromfile(os.path.join(folder, "flat_sample.i64"), dtype=np.int64)]
    os.makedirs(SCRATCH, exist_ok=True)
    states = os.path.join(SCRATCH, "tm_states.f64")
    start.tofile(states)
    out, jac = os.path.join(SCRATCH, "tm_out.f64"), os.path.join(SCRATCH, "tm_jac.f64")
    subprocess.run([program(), "run", "method=rec_cash_karp:jacobian=1", "precision=double",
                    "map=v8r1_up", "reader=rec", "states=" + states, "z0=%r" % Z0, "z1=%r" % Z1,
                    "out=" + out, "jacobian=" + jac], check=True)
    copy = np.fromfile(jac).reshape(-1, 4, 3)
    differences = np.empty_like(copy)
    for column, index in enumerate((2, 3, 4)):
        h = 1e-6 if index != 4 else None
        plus, minus = start.copy(), start.copy()
        step = np.full(len(start), 1e-6) if index != 4 else 1e-5 * np.abs(start[:, 4])
        plus[:, index] += step
        minus[:, index] -= step
        up, _ = run("rk4:512", plus, Z0, Z1, map_name="v8r1_up", reader="linear")
        down, _ = run("rk4:512", minus, Z0, Z1, map_name="v8r1_up", reader="linear")
        differences[:, :, column] = (up[:, :4] - down[:, :4]) / (2 * step[:, None])
    rows = []
    for r in range(4):
        for c in range(3):
            a, b = copy[:, r, c], differences[:, r, c]
            relative = np.abs(a - b) / np.maximum(np.abs(b), 1e-300)
            rows.append(dict(element="d %s / d %s" % (NAMES_OUT[r], NAMES_IN[c]),
                             median_of_copy=float(np.median(a)),
                             median_of_differences=float(np.median(b)),
                             median_relative_difference=float(np.median(relative)),
                             percentile_95_relative_difference=float(np.quantile(relative, 0.95)),
                             same_sign_fraction=float((np.sign(a) == np.sign(b)).mean())))
            print("%-18s copy %12.5g  differences %12.5g  relative difference: median %.3g, 95 %% %.3g, "
                  "same sign %.3f" % (rows[-1]["element"], rows[-1]["median_of_copy"],
                                      rows[-1]["median_of_differences"],
                                      rows[-1]["median_relative_difference"],
                                      rows[-1]["percentile_95_relative_difference"],
                                      rows[-1]["same_sign_fraction"]))
    with open(os.path.join(HERE, "results", "transport_matrix_of_the_copy.csv"), "w",
              newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
