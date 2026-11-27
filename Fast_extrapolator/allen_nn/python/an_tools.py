"""Calls of the standalone program allen_nn/build/<node>/an, and the track sets."""
from __future__ import annotations

import os
import socket
import subprocess
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.path.dirname(HERE)
EXPORTS = "/data/bfys/gscriven/fast_extrapolator_store/exports"
SCRATCH = "/data/bfys/gscriven/fast_extrapolator_store/scratch"
WEIGHTS = os.path.join(FOLDER, "weights")
Z0, Z1 = 2648.2, 7826.0
C_LIGHT = 299.792458


def program():
    path = os.path.join(FOLDER, "build", socket.gethostname().split(".")[0], "an")
    if not os.path.exists(path):
        raise FileNotFoundError("build the program first: allen_nn/build.sh (%s)" % path)
    return path


def run(method, states, z_in, z_out, precision="float", map_name="v8r1_up"):
    """Final states (n, 5), the 4 x 5 matrices (n, 4, 5) and the counts (n, 4).
    z_in and z_out are numbers or arrays (one plane a track)."""
    os.makedirs(SCRATCH, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=SCRATCH) as work:
        f = lambda name: os.path.join(work, name)  # noqa: E731
        np.ascontiguousarray(states, dtype=np.float64).tofile(f("s.f64"))
        command = [program(), "run", "method=" + method, "precision=" + precision, "map=" + map_name,
                   "states=" + f("s.f64"), "out=" + f("o.f64"), "jac=" + f("j.f64"), "counts=" + f("c.i64")]
        if np.ndim(z_in) == 0 and np.ndim(z_out) == 0:
            command += ["z_in=%r" % float(z_in), "z_out=%r" % float(z_out)]
        else:
            n = len(states)
            np.broadcast_to(np.asarray(z_in, dtype=np.float64), (n,)).copy().tofile(f("zi.f64"))
            np.broadcast_to(np.asarray(z_out, dtype=np.float64), (n,)).copy().tofile(f("zo.f64"))
            command += ["z_in_file=" + f("zi.f64"), "z_out_file=" + f("zo.f64")]
        subprocess.run(command, check=True)
        return (np.fromfile(f("o.f64")).reshape(-1, 5), np.fromfile(f("j.f64")).reshape(-1, 4, 5),
                np.fromfile(f("c.i64"), dtype=np.int64).reshape(-1, 4))


def test_tracks(reference="linear"):
    folder = os.path.join(EXPORTS, "tracks_" + reference)
    s = np.fromfile(os.path.join(folder, "start_state.f64")).reshape(-1, 5)
    e = np.fromfile(os.path.join(folder, "reference_end_state.f64")).reshape(-1, 5)
    p = np.fromfile(os.path.join(folder, "momentum_gev.f64"))
    return s, e, p
