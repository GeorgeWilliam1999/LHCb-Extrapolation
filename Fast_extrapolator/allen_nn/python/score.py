"""P3: the offline score of the magnet-crossing tool on the linear-map test tracks.

Methods (float unless said): the tool (K Nystrom steps + network, six networks), the
ARKN of Allen (host copy), plain K Nystrom steps, the HLT1 parametrised step of ParKF
(on its own planes, 2665 -> 7826 mm), and the Rec STEP integrator (double).

    1. endpoint error against the reference of the track set (13,447 test tracks,
       trilinear map v8r1 up, the reference of the present package): |error| in x, y
       (um) and tx, ty (urad): median, 95 %, 99.9 %, largest; by momentum band.
    2. transport matrix: the shift of the end state for a one-sigma change of the
       start (50 um in x or y, 0.3 mrad in tx or ty, 1 % of q/p), predicted with the
       matrix, against central differences of a dense reference (RK4, 4000 steps,
       double, trilinear map).
    3. way A: start on a plane of the last UT layer (2600-2700 mm) and end on a plane
       of the first SciFi layer (7826-7900 mm), the reference carried there by RK4.
    4. time per call, one core.

    python allen_nn/python/score.py           writes allen_nn/results/*.csv and allen_nn/figures/*.png
"""
from __future__ import annotations

import csv
import os
import socket
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import an_tools as an  # noqa: E402

RESULTS = os.path.join(an.FOLDER, "results")
FIGURES = os.path.join(an.FOLDER, "figures")
NETWORKS = ["linear_nystrom_6_2x32", "linear_nystrom_6_2x64", "linear_nystrom_8_2x32", "linear_nystrom_8_2x64",
            "linear_nystrom_12_2x32", "linear_nystrom_12_2x64"]
BANDS = [("all", 0, np.inf), ("p<3", 0, 3), ("3-8", 3, 8), ("8-20", 8, 20), (">20", 20, np.inf)]
SIGMA = np.array([0.05, 0.05, 3e-4, 3e-4, np.nan])  # q/p: 1 % of the track's own
REFERENCE = "reference:4000"


def methods():
    out = [("tool K=%s %s" % (n.split("_")[2], n.split("_")[3]), "tool:" + os.path.join(an.WEIGHTS, n + ".json"), "float")
           for n in NETWORKS]
    out += [("tool K=12 2x64, no fallback", "tool:" + os.path.join(an.WEIGHTS, "linear_nystrom_12_2x64.json")
             + ":fallback=1e30", "float")]
    out += [("tool K=12 2x64, network in the matrix", "tool:" + os.path.join(an.WEIGHTS, "linear_nystrom_12_2x64.json")
             + ":netjac=1", "float")]
    out += [("ARKN (Allen)", "arkn", "float")]
    out += [("Nystrom %d steps" % k, "nystrom:%d" % k, "float") for k in (6, 8, 12, 32)]
    out += [("Rec STEP (double)", "rec_step", "double")]
    return out


def quantiles(v):
    v = np.abs(v)
    return [np.median(v), np.percentile(v, 95), np.percentile(v, 99.9), v.max()]


def endpoint_rows(label, end, ref, p, extra):
    rows = []
    err = np.column_stack([(end[:, 0] - ref[:, 0]) * 1e3, (end[:, 1] - ref[:, 1]) * 1e3,
                           (end[:, 2] - ref[:, 2]) * 1e6, (end[:, 3] - ref[:, 3]) * 1e6])
    radial = np.hypot(err[:, 0], err[:, 1])
    for band, lo, hi in BANDS:
        m = (p >= lo) & (p < hi)
        row = dict(method=label, band=band, tracks=int(m.sum()))
        for name, v in zip(("x_um", "y_um", "tx_urad", "ty_urad", "radial_um"), list(err.T) + [radial]):
            q = quantiles(v[m])
            for qn, qv in zip(("median", "p95", "p99.9", "max"), q):
                row["%s_%s" % (name, qn)] = qv
        row.update(extra)
        rows.append(row)
    return rows


def finite_differences(states, z_in, z_out, map_name="v8r1_up"):
    """d_true[n, j, 4]: the central difference of the reference for the change j of the start."""
    out = np.zeros((len(states), 5, 4))
    for j in range(5):
        delta = np.zeros_like(states)
        delta[:, j] = SIGMA[j] if j < 4 else 0.01 * states[:, 4]
        plus = an.run(REFERENCE, states + delta, z_in, z_out, precision="double", map_name=map_name)[0]
        minus = an.run(REFERENCE, states - delta, z_in, z_out, precision="double", map_name=map_name)[0]
        out[:, j] = 0.5 * (plus[:, :4] - minus[:, :4])
    return out


def transport_rows(label, J, states, d_true, p):
    rows = []
    names = ["x0 50um", "y0 50um", "tx0 0.3mrad", "ty0 0.3mrad", "q/p 1%"]
    for j in range(5):
        step = SIGMA[j] if j < 4 else 0.01 * states[:, 4]
        predicted = J[:, :, j] * (step[:, None] if j == 4 else step)
        e = predicted - d_true[:, j]
        pos = np.hypot(e[:, 0], e[:, 1]) * 1e3
        slo = np.hypot(e[:, 2], e[:, 3]) * 1e6
        shift = np.hypot(d_true[:, j, 0], d_true[:, j, 1]) * 1e3
        for band, lo, hi in BANDS:
            m = (p >= lo) & (p < hi)
            row = dict(method=label, change=names[j], band=band, tracks=int(m.sum()),
                       true_shift_median_um=np.median(shift[m]))
            for qn, qv in zip(("median", "p95", "p99.9", "max"), quantiles(pos[m])):
                row["position_error_um_" + qn] = qv
            for qn, qv in zip(("median", "p95", "p99.9", "max"), quantiles(slo[m])):
                row["slope_error_urad_" + qn] = qv
            rows.append(row)
    return rows


def write(name, rows):
    os.makedirs(RESULTS, exist_ok=True)
    keys = list(rows[0])
    with open(os.path.join(RESULTS, name), "w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (("%.6g" % v) if isinstance(v, float) else v) for k, v in r.items()})


def main():
    s, ref, p = an.test_tracks()
    folder = os.path.join(an.EXPORTS, "tracks_linear")
    s2665 = np.fromfile(os.path.join(folder, "start_state_at_2665.f64")).reshape(-1, 5)
    ref2665 = np.fromfile(os.path.join(folder, "reference_end_state_from_2665.f64")).reshape(-1, 5)

    # the dense reference itself, against the reference of the track set
    dense = an.run(REFERENCE, s, an.Z0, an.Z1, precision="double")[0]
    d = np.hypot(*(dense[:, :2] - ref[:, :2]).T) * 1e3
    print("dense reference (%s) vs the track-set reference: median %.3g um, max %.3g um" % (REFERENCE, np.median(d), d.max()))
    end_rows, tr_rows, cost = [], [], []
    d_true = finite_differences(s, an.Z0, an.Z1)
    d_true_2665 = finite_differences(s2665, 2665.0, 7826.0)
    for label, spec, precision in methods():
        end, J, counts = an.run(spec, s, an.Z0, an.Z1, precision=precision)
        extra = dict(field_reads_mean=float(counts[:, 0].mean()), fell_back=int((counts[:, 3] == 1).sum()),
                     failed=int((counts[:, 3] == -1).sum()))
        end_rows += endpoint_rows(label, end, ref, p, extra)
        tr_rows += transport_rows(label, J, s, d_true, p)
        cost.append((label, spec, extra["field_reads_mean"]))
        print("%-22s radial median %.2f um, 95 %% %.1f um, reads %.1f, fell back %d" % (
            label, end_rows[-5]["radial_um_median"], end_rows[-5]["radial_um_p95"], extra["field_reads_mean"],
            extra["fell_back"]), flush=True)
    end, J, counts = an.run("parkf", s2665, 2665.0, 7826.0)
    end_rows += endpoint_rows("ParKF step (2665->7826)", end, ref2665, p, dict(field_reads_mean=0.0, fell_back=0, failed=0))
    tr_rows += transport_rows("ParKF step (2665->7826)", J, s2665, d_true_2665, p)
    write("endpoint_errors.csv", end_rows)
    write("transport_errors.csv", tr_rows)

    # way A
    rng = np.random.default_rng(20261127)
    z_in = rng.uniform(2600.0, 2700.0, len(s))
    z_out = rng.uniform(7826.0, 7900.0, len(s))
    start = an.run(REFERENCE.replace("4000", "200"), s, an.Z0, z_in, precision="double")[0]
    target = an.run(REFERENCE, start, z_in, z_out, precision="double")[0]
    rows = []
    for label, spec, precision in methods():
        if not spec.startswith(("tool", "arkn")):
            continue
        end, _, counts = an.run(spec, start, z_in, z_out, precision=precision)
        rows += endpoint_rows(label, end, target, p, dict(field_reads_mean=float(counts[:, 0].mean()),
                                                         fell_back=int((counts[:, 3] == 1).sum()), failed=0))
    write("way_a_endpoint_errors.csv", rows)

    # time, one core (core 40 of the node; the node is shared, so the lowest of 9 passes counts)
    import tempfile
    out = os.path.join(RESULTS, "timing_%s.csv" % socket.gethostname().split(".")[0])
    specs = [spec for _, spec, prec in methods() if prec == "float"] + \
            ["tool_state:" + os.path.join(an.WEIGHTS, n + ".json") for n in NETWORKS] + ["arkn_state"]
    with tempfile.TemporaryDirectory(dir=an.SCRATCH) as work:
        for name, states, z0, z1, these in (("main", s, an.Z0, an.Z1, specs), ("parkf", s2665, 2665.0, 7826.0, ["parkf"])):
            f = os.path.join(work, name + ".f64")
            np.ascontiguousarray(states).tofile(f)
            subprocess.run(["taskset", "-c", "40", an.program(), "time", "methods=" + ",".join(these), "states=" + f,
                            "z_in=%r" % z0, "z_out=%r" % z1, "out=" + os.path.join(work, name + ".csv"),
                            "repeats=9"], check=True)
        with open(out, "w") as handle:
            handle.write(open(os.path.join(work, "main.csv")).read())
            handle.write("".join(open(os.path.join(work, "parkf.csv")).readlines()[1:]))
    print("wrote", RESULTS)


if __name__ == "__main__":
    main()
