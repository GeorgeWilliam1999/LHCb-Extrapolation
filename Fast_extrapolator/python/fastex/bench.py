"""Scoring and timing a list of methods with the compiled program.

A method is a dict:

    label       the name in the tables
    family      a word to group the rows by
    method      the name for the compiled program (cpp/bench/fx.cpp)
    precision   double or float
    reader      linear, cubic (the present package), rec (the field read of
                LHCb), allen (the field read of Allen)
    tracks      linear, cubic            the test tracks of that track set,
                                         2648.2 mm to 7826 mm
                linear_at_2665           the linear tracks carried to 2665 mm, the
                                         first plane of the parametrised step of HLT1
                linear_flat, cubic_flat  the 1,000 tracks that are flat in momentum
                linear_training, cubic_training
                                         2,000 training tracks, for choices that
                                         must not see the test tracks
    map         the field map; v8r1_up when it is not given

    accuracy(methods)        final states in the compiled program; the errors
                             against the reference, over the tracks and over
                             the flat sample inside them
    time(methods, core, ...) every method carries its tracks one at a time on
                             one core (taskset); see `fx time`
"""
from __future__ import annotations

import csv
import os
import socket
import subprocess

import numpy as np

from fastex import FOLDER_OF_EXPORTS
from fastex.compiled import program, run, summary

Z_OURS, Z_UTT, Z_LAST = 2648.2, 2665.0, 7826.0


def planes(tracks: str):
    return (Z_UTT, Z_LAST) if tracks == "linear_at_2665" else (Z_OURS, Z_LAST)


def tracks_of(kind):
    """The start states, the reference end states, the momenta, the rows of the
    flat sample, and the file of the start states."""
    of_map = "cubic" if kind.startswith("cubic") else "linear"
    folder = os.path.join(FOLDER_OF_EXPORTS, "tracks_" + of_map)
    if kind.endswith("_training"):
        file = os.path.join(folder, "training_start_state.f64")
        start = np.fromfile(file).reshape(-1, 5)
        reference = np.fromfile(os.path.join(folder, "training_reference_end_state.f64")).reshape(-1, 5)
        momentum = np.fromfile(os.path.join(folder, "training_momentum_gev.f64"))
        return start, reference, momentum, np.arange(0), file
    momentum = np.fromfile(os.path.join(folder, "momentum_gev.f64"))
    flat = np.fromfile(os.path.join(folder, "flat_sample.i64"), dtype=np.int64)
    if kind == "linear_at_2665":
        file = os.path.join(folder, "start_state_at_2665.f64")
        reference_file = os.path.join(folder, "reference_end_state_from_2665.f64")
        if not os.path.exists(file):
            start = np.fromfile(os.path.join(folder, "start_state.f64")).reshape(-1, 5)
            at_ini, _ = run("rk4:64", start, Z_OURS, Z_UTT, map_name="v8r1_up", reader="linear")
            reference, _ = run("rk4:4096", at_ini, Z_UTT, Z_LAST, map_name="v8r1_up", reader="linear")
            at_ini.tofile(file)
            reference.tofile(reference_file)
        return (np.fromfile(file).reshape(-1, 5), np.fromfile(reference_file).reshape(-1, 5),
                momentum, flat, file)
    file = os.path.join(folder, "start_state.f64")
    start = np.fromfile(file).reshape(-1, 5)
    reference = np.fromfile(os.path.join(folder, "reference_end_state.f64")).reshape(-1, 5)
    if kind.endswith("_flat"):
        file = os.path.join(folder, "start_state_flat.f64")
        if not os.path.exists(file):
            start[flat].tofile(file)
        return start[flat], reference[flat], momentum[flat], np.arange(len(flat)), file
    return start, reference, momentum, flat, file


def multiply_adds(method):
    """The multiply-adds of one network call of a method, 0 for no network."""
    parts = method.split(":")
    if parts[0] not in ("chain", "whole_crossing"):
        return 0
    total = 0
    for line in open(os.path.join(FOLDER_OF_EXPORTS, "networks", parts[1], "meta")):
        key, value = line.rstrip("\n").split(" ", 1)
        if key.startswith("layer_"):
            n_out, n_in = (int(v) for v in value.split())
            total += n_out * n_in
    return total


def write(path, rows):
    names = []
    for row in rows:
        names += [k for k in row if k not in names]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def score(m, report=print):
    """One row: the counts and the errors of one method."""
    start, reference, momentum, flat, _ = tracks_of(m["tracks"])
    z0, z1 = planes(m["tracks"])
    finals, counts = run(m["method"], start, z0, z1, map_name=m.get("map", "v8r1_up"),
                         reader=m["reader"], precision=m["precision"])
    row = {k: v for k, v in m.items()}
    row.update(number_of_tracks=len(start), multiply_adds_a_call=multiply_adds(m["method"]),
               field_reads_mean=float(counts[:, 0].mean()),
               field_reads_largest=int(counts[:, 0].max()),
               steps_mean=float(counts[:, 1].mean()),
               failed_steps_mean=float(counts[:, 2].mean()))
    row.update(summary(finals, reference, momentum))
    if len(flat):
        row.update({"flat_" + k: v for k, v in
                    summary(finals[flat], reference[flat], momentum[flat]).items()})
    report("%-76s %-6s reads %7.1f  median %10.4g um  95 %% %10.4g um  worst band %10.4g um"
           % (m["label"], m["precision"], row["field_reads_mean"], row["radial_median_um"],
              row["radial_95_um"], row["worst_band_median_um"]))
    return row


def accuracy(methods, path, report=lambda text: print(text, flush=True)):
    rows = [dict(index=index, **score(m, report)) for index, m in enumerate(methods)]
    write(path, rows)
    return rows


def time(methods, core, out, *, repeats=15, budget=24, cold=200):
    """Writes the CSV of `fx time` to `out`; the label of row i is m<i>."""
    lines = []
    for index, m in enumerate(methods):
        z0, z1 = planes(m["tracks"])
        _, _, _, _, file = tracks_of(m["tracks"])
        lines.append("label=m%03d method=%s precision=%s map=%s reader=%s states=%s z0=%r z1=%r"
                     % (index, m["method"], m["precision"], m.get("map", "v8r1_up"), m["reader"],
                        file, z0, z1))
    list_file = out + ".list"
    with open(list_file, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    subprocess.run(["taskset", "-c", str(core), program(), "time", "list=" + list_file,
                    "out=" + out, "repeats=%d" % repeats, "budget=%g" % budget, "cold=%d" % cold],
                   check=True)
    print("timed on %s, core %s" % (socket.gethostname(), core))


def join(accuracy_csv, timing_csv):
    """The rows of the accuracy table with their times, as a pandas frame."""
    import pandas as pd
    a = pd.read_csv(accuracy_csv)
    t = pd.read_csv(timing_csv)
    t["index"] = t["label"].str[1:].astype(int)
    both = a.merge(t[["index", "ns_lowest", "ns_median", "ns_highest", "ns_shuffled_lowest",
                      "ns_shuffled_median", "cold_ns_median", "cold_ns_lowest", "host"]],
                   on="index")
    assert len(both) == len(a), (len(both), len(a))
    return both
