"""Calls of the compiled program (cpp/bench/fx.cpp) from Python.

    run(...)     the final states of one method for an array of start states
    errors(...)  the radial error in micrometres and its summary

The program is built per node (cpp/build/<node>/fx), because the default
target is the processor of the node that compiles.
"""
from __future__ import annotations

import os
import socket
import subprocess
import tempfile

import numpy as np

from fastex import STORE_OF_THIS_EXPERIMENT

FOLDER_OF_THE_PROGRAM = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "cpp", "build")
SCRATCH = os.path.join(STORE_OF_THIS_EXPERIMENT, "scratch")

BANDS_GEV = [(0.0, 3.0), (3.0, 8.0), (8.0, 20.0), (20.0, 50.0), (50.0, float("inf"))]


def program(build=None) -> str:
    """The path of fx for this node; `build` names another build folder."""
    path = os.path.join(FOLDER_OF_THE_PROGRAM, build or socket.gethostname().split(".")[0], "fx")
    if not os.path.exists(path):
        raise FileNotFoundError("no compiled program at %s; build it on this node first" % path)
    return path


def run(method, states, z0, z1, *, map_name="v8r1_up", reader="linear", precision="double",
        fx=None):
    """Returns the final states (n, 5) and the counts (n, 3): field reads,
    accepted steps, failed steps."""
    os.makedirs(SCRATCH, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=SCRATCH) as work:
        states_file = os.path.join(work, "states.f64")
        np.ascontiguousarray(states, dtype=np.float64).tofile(states_file)
        out, counts = os.path.join(work, "out.f64"), os.path.join(work, "counts.i64")
        subprocess.run([fx or program(), "run", "method=" + method, "precision=" + precision,
                        "map=" + map_name, "reader=" + reader, "states=" + states_file,
                        "z0=%r" % float(z0), "z1=%r" % float(z1), "out=" + out,
                        "counts=" + counts], check=True)
        return (np.fromfile(out, dtype=np.float64).reshape(-1, 5),
                np.fromfile(counts, dtype=np.int64).reshape(-1, 3))


def radial_um(finals, reference) -> np.ndarray:
    """The distance in the plane between two sets of final states, in micrometres."""
    return np.hypot(finals[:, 0] - reference[:, 0], finals[:, 1] - reference[:, 1]) * 1e3


def summary(finals, reference, momentum_gev=None) -> dict:
    """Median, 68 %, 95 %, 99 % and largest radial error in micrometres, the
    fraction above 100 um, the signed medians of the four components, and with
    the momenta the median in each band."""
    r = radial_um(finals, reference)
    d = finals[:, :4] - reference[:, :4]
    row = dict(radial_median_um=float(np.median(r)), radial_68_um=float(np.quantile(r, 0.68)),
               radial_95_um=float(np.quantile(r, 0.95)), radial_99_um=float(np.quantile(r, 0.99)),
               radial_largest_um=float(r.max()),
               fraction_above_100_um=float((r > 100.0).mean()),
               x_signed_median_um=float(np.median(d[:, 0]) * 1e3),
               y_signed_median_um=float(np.median(d[:, 1]) * 1e3),
               tx_signed_median=float(np.median(d[:, 2])),
               ty_signed_median=float(np.median(d[:, 3])),
               tx_absolute_median=float(np.median(np.abs(d[:, 2]))),
               ty_absolute_median=float(np.median(np.abs(d[:, 3]))))
    if momentum_gev is not None:
        p = np.asarray(momentum_gev)
        for low, high in BANDS_GEV:
            inside = (p >= low) & (p < high)
            name = "radial_median_um_%s_gev" % ("%g_%g" % (low, high) if np.isfinite(high)
                                                else "above_%g" % low)
            row[name] = float(np.median(r[inside])) if inside.any() else float("nan")
        row["worst_band_median_um"] = float(np.nanmax(
            [row[k] for k in row if k.startswith("radial_median_um_") and k.endswith("_gev")]))
    return row
