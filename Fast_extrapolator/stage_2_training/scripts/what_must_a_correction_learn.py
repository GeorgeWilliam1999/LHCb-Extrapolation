"""What must a correction learn? No network is trained here.

A method of stage 2 has the form

    end state = start(start state) + correction(start state)

The start is a cheap method with no network: the straight line, or a
fixed-step integrator with a few steps. The correction is what a network must
learn. This script measures that correction on the training and validation
tracks, with no network:

  1. its size: the error of each start against the reference;
  2. how it grows with q/p: the exponent k of error ~ |q/p|^k;
  3. how much of it a smooth function gives. A polynomial of the start state
     is fitted to the correction on the 108,915 training tracks and scored on
     the 13,684 validation tracks. The polynomial is

         correction_c = sum over monomials m of  a_cm  m(x, y, tx, ty, q/p)

     with every monomial of total degree D or less that holds q/p at least
     once (with q/p = 0 a track is a straight line and every start is exact).
     The five variables are divided by a fixed scale each.

The fit is robust. A plain least-squares fit is ruled by the 1 % of the tracks
whose correction is 100 times the typical one (tracks of low momentum that
leave the middle of the map), and it makes the typical track worse. So the fit
minimises the Huber loss of each component: quadratic below 2 um (1e-6 for a
slope) and linear above, by least squares with weights that are renewed 12
times.

Two references: the cubic reading of the map (the reference of the
experiment), and the linear reading (the field of LHCb) for the starts that
read the map linearly.

The polynomial is a probe of the target, not a model of the experiment: it
says how smooth the correction is, and so what a small network can hope to
reach. The test tracks are not used.

    what_must_a_correction_learn.py      results/what_a_correction_must_learn.csv
                                         results/polynomial_probe.csv
"""
from __future__ import annotations

import itertools
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "python"))

from fastex import FOLDER_OF_EXPORTS, bench  # noqa: E402
from fastex.compiled import run, summary  # noqa: E402

Z0, Z1 = 2648.2, 7826.0
DEGREES = (3, 5, 7)
STEPS = (4, 6, 8, 12)
RENEWALS_OF_THE_WEIGHTS = 12
HUBER = np.array([2e-3, 2e-3, 1e-6, 1e-6])      # mm, mm, slope, slope


def split(of_map, name):
    folder = os.path.join(FOLDER_OF_EXPORTS, "tracks_" + of_map, "splits", name)
    return (np.fromfile(os.path.join(folder, "start_state.f64")).reshape(-1, 5),
            np.fromfile(os.path.join(folder, "reference_end_state.f64")).reshape(-1, 5),
            np.fromfile(os.path.join(folder, "momentum_gev.f64")))


def straight_line(s):
    length = Z1 - Z0
    return np.stack([s[:, 0] + length * s[:, 2], s[:, 1] + length * s[:, 3], s[:, 2], s[:, 3]], 1)


def integrator(steps, reader):
    return lambda s: run("nystrom:%d" % steps, s, Z0, Z1, reader=reader, precision="float")[0][:, :4]


def cases():
    """reference, name of the start, field reads, cost of a read in ns, start."""
    rows = [("cubic", "the straight line", 0, straight_line)]
    for steps in STEPS:
        rows.append(("cubic", "Nystrom, %d equal steps, linear reading" % steps, 2 * steps + 1,
                     integrator(steps, "rec")))
    for steps in STEPS:
        rows.append(("cubic", "Nystrom, %d equal steps, cubic reading" % steps, 2 * steps + 1,
                     integrator(steps, "cubic")))
    for steps in STEPS:
        rows.append(("linear", "Nystrom, %d equal steps, linear reading" % steps, 2 * steps + 1,
                     integrator(steps, "rec")))
    return rows


def exponents(degree):
    """The monomials: exponents (a_x, a_y, a_tx, a_ty, a_qop) with a total of
    `degree` or less and a_qop at least 1."""
    return np.array([e for e in itertools.product(range(degree + 1), repeat=5)
                     if sum(e) <= degree and e[4] >= 1])


def design(u, powers):
    """The monomials of the scaled variables u (n, 5), shape (n, monomials)."""
    highest = int(powers.max())
    table = np.ones((highest + 1,) + u.shape)
    for k in range(1, highest + 1):
        table[k] = table[k - 1] * u
    out = np.ones((len(u), len(powers)))
    for j in range(5):
        out *= table[powers[:, j], :, j].T
    return out


def robust_fit(q, target):
    """The coefficients (in the orthonormal basis q) that minimise the Huber
    loss of each component of the target, by renewed weights."""
    coefficients = np.zeros((q.shape[1], 4))
    for c in range(4):
        weight = np.ones(len(q))
        for _ in range(RENEWALS_OF_THE_WEIGHTS):
            qw = q * weight[:, None]
            gram = q.T @ qw + 1e-12 * np.eye(q.shape[1])
            coefficients[:, c] = np.linalg.solve(gram, qw.T @ target[:, c])
            residual = np.abs(target[:, c] - q @ coefficients[:, c])
            weight = np.minimum(1.0, HUBER[c] / np.maximum(residual, 1e-300))
    return coefficients


def main():
    size_rows, probe_rows = [], []
    bases = {}
    for reference in ("cubic", "linear"):
        train_s, _, _ = split(reference, "training")
        val_s, _, _ = split(reference, "validation")
        # one fixed scale a variable: the value that 99.5 % of the training tracks are below
        scale = np.quantile(np.abs(train_s), 0.995, axis=0)
        for degree in DEGREES:
            powers = exponents(degree)
            q, r = np.linalg.qr(design(train_s / scale, powers))
            bases[reference, degree] = (powers, q, np.linalg.solve(r.T, design(val_s / scale, powers).T).T)
            print("%s reference, degree %d: %d monomials" % (reference, degree, len(powers)), flush=True)

    for reference, name, reads, start in cases():
        t0 = time.time()
        train_s, train_ref, _ = split(reference, "training")
        val_s, val_ref, val_p = split(reference, "validation")
        target_train = train_ref[:, :4] - start(train_s)
        start_val = start(val_s)
        target_val = val_ref[:, :4] - start_val
        row = dict(reference=reference, start=name, field_reads=reads)
        row.update({k: v for k, v in summary(start_val, val_ref[:, :4], val_p).items()
                    if k.startswith("radial") or k.startswith("worst")})
        radial = np.hypot(target_val[:, 0], target_val[:, 1])
        keep = radial > 0
        row["exponent_of_qop"] = float(np.polyfit(np.log(np.abs(val_s[keep, 4])),
                                                  np.log(radial[keep]), 1)[0])
        size_rows.append(row)
        line = []
        for degree in DEGREES:
            powers, q_train, q_val = bases[reference, degree]
            coefficients = robust_fit(q_train, target_train)
            residual = target_train - q_train @ coefficients
            radial = np.hypot(residual[:, 0], residual[:, 1]) * 1e3
            probe_rows.append(dict(reference=reference, start=name, field_reads=reads, degree=degree,
                                   monomials=len(powers), sample="training",
                                   radial_median_um=float(np.median(radial)),
                                   radial_95_um=float(np.quantile(radial, 0.95))))
            corrected = start_val + q_val @ coefficients
            r = dict(reference=reference, start=name, field_reads=reads, degree=degree,
                     monomials=len(powers), sample="validation")
            r.update({k: v for k, v in summary(corrected, val_ref[:, :4], val_p).items()
                      if k.startswith("radial") or k.startswith("worst")
                      or k.endswith("absolute_median")})
            probe_rows.append(r)
            line.append("D%d %.3g/%.3g/%.3g" % (degree, r["radial_median_um"], r["radial_95_um"],
                                                r["worst_band_median_um"]))
        print("%-6s %-42s start %7.3g/%7.3g/%7.3g um | with the polynomial: %s   (%.0f s)"
              % (reference, name, row["radial_median_um"], row["radial_95_um"],
                 row["worst_band_median_um"], "   ".join(line), time.time() - t0), flush=True)
        bench.write(os.path.join(HERE, "results", "what_a_correction_must_learn.csv"), size_rows)
        bench.write(os.path.join(HERE, "results", "polynomial_probe.csv"), probe_rows)


if __name__ == "__main__":
    main()
