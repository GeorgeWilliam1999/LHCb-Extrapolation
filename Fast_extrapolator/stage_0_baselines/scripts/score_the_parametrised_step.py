"""The error of the parametrised step of HLT1 on our tracks.

The step (Allen v9r3, copied in cpp/lhcb/allen_utt.h) goes from z = 2665 mm
to z = 7826 mm with no field read. Our tracks start on z = 2648.2 mm, so each
start state is first carried to 2665 mm (RK4, 64 steps over 16.8 mm). The
reference on 7826 mm is RK4 with 4,096 steps from 2665 mm, in double precision.
The 13,447 test tracks of the linear track set are used.

What is not known from the code is the field map that the tabulated
polynomials were fitted to. So the reference is made on two maps, the map of
data taking in 2025 (v8r1) and the map of the simulation (v5r0), both magnet
up, and the step is scored against each. Four variants of the step are scored:

    polynomial   extrapUTT alone, the tabulated polynomials with the q/p of the state
    full         ExtrapolateUTT: q/p changed first, then terms in q/p added (UTTFParams)
    each with polarity +1 and -1

For comparison on the same planes and tracks: the HLT2 integrator (copy,
field read of LHCb) and the fixed-step integrator of the HLT1 T tracks.

It also checks our stored reference: RK4 with 4,096 steps from 2648.2 mm to
7826 mm against the reference end states of the track set.

    score_the_parametrised_step.py      writes results/parametrised_step.csv
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "python"))

from fastex import FOLDER_OF_EXPORTS  # noqa: E402
from fastex.compiled import radial_um, run, summary  # noqa: E402

Z_OURS, Z_INI, Z_FIN = 2648.2, 2665.0, 7826.0


def main():
    folder = os.path.join(FOLDER_OF_EXPORTS, "tracks_linear")
    start = np.fromfile(os.path.join(folder, "start_state.f64")).reshape(-1, 5)
    stored = np.fromfile(os.path.join(folder, "reference_end_state.f64")).reshape(-1, 5)
    momentum = np.fromfile(os.path.join(folder, "momentum_gev.f64"))
    rows = []

    check, _ = run("rk4:4096", start, Z_OURS, Z_FIN, map_name="v8r1_up", reader="linear")
    r = radial_um(check, stored)
    print("RK4 with 4,096 steps against the stored reference (v8r1, linear): median %.3g um, "
          "largest %.3g um" % (np.median(r), r.max()), flush=True)
    rows.append(dict(method="RK4, 4,096 steps, 2648.2 to 7826 mm, against the stored reference",
                     reference_map="v8r1_up", tracks=len(start), **summary(check, stored, momentum)))

    for map_name in ("v8r1_up", "v5r0_up"):
        at_ini, _ = run("rk4:64", start, Z_OURS, Z_INI, map_name=map_name, reader="linear")
        reference, _ = run("rk4:4096", at_ini, Z_INI, Z_FIN, map_name=map_name, reader="linear")
        methods = [("HLT1 parametrised step, polynomial, polarity %+d" % s,
                    "allen_utt:polarity=%d" % s, "linear", "float") for s in (1, -1)]
        methods += [("HLT1 parametrised step, full, polarity %+d" % s,
                     "allen_utt:full=1:polarity=%d" % s, "linear", "float") for s in (1, -1)]
        methods += [("HLT2 integrator (Cash-Karp copy)", "rec_cash_karp", "rec", "double"),
                    ("HLT1 T-track integrator (fixed Nystrom, 500 mm)", "allen_fast_nystrom",
                     "allen", "float"),
                    ("Straight line", "rk4:1", None, "double")]
        for label, method, reader, precision in methods:
            if reader is None:      # the straight line, as the scale of the bending
                finals = at_ini.copy()
                finals[:, 0] += finals[:, 2] * (Z_FIN - Z_INI)
                finals[:, 1] += finals[:, 3] * (Z_FIN - Z_INI)
                reads = np.zeros(len(finals))
            else:
                finals, counts = run(method, at_ini, Z_INI, Z_FIN, map_name=map_name,
                                     reader=reader, precision=precision)
                reads = counts[:, 0]
            row = dict(method=label, reference_map=map_name, tracks=len(start),
                       field_reads_mean=float(reads.mean()), **summary(finals, reference, momentum))
            rows.append(row)
            print("%-8s %-52s median %10.4g um  95 %% %10.4g um  worst band %10.4g um  x bias %9.3g um"
                  % (map_name, label, row["radial_median_um"], row["radial_95_um"],
                     row["worst_band_median_um"], row["x_signed_median_um"]), flush=True)

    names = []
    for row in rows:
        names += [k for k in row if k not in names]
    with open(os.path.join(HERE, "results", "parametrised_step.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
