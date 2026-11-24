"""The error and the compiled time of every method of stage 0 (methods.py).

    accuracy     every method carries the 13,447 test tracks in the compiled
                 program; the errors against the reference are summarised over
                 the full test split (the real momentum spectrum) and over the
                 1,000 tracks that are flat in momentum (2 to 42 GeV).
                 Writes results/accuracy.csv.
    time CORE    every method carries the same tracks one at a time, on the
                 core CORE of this node (taskset). A pass is one loop over all
                 tracks; the passes of the methods take turns in a shuffled
                 order, 15 times. 200 calls a method are also timed one by one
                 with cold caches. Writes results/timing.csv.
    table        joins the two and writes results/methods.csv and
                 results/methods.md.

The clock of a call starts when the start state is given and stops when the
final state is returned. Nothing else is inside: no reading of files, no
conversion of units outside the method.
"""
from __future__ import annotations

import csv
import os
import socket
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastex import FOLDER_OF_EXPORTS  # noqa: E402
from fastex.compiled import program, run, summary  # noqa: E402
from methods import METHODS, Z_LAST, Z_OURS, Z_UTT, planes  # noqa: E402

RESULTS = os.path.join(HERE, "results")


def tracks_of(kind):
    """The start states, the reference end states, the momenta and the rows of
    the flat sample; the file of the start states."""
    of_map = "cubic" if kind == "cubic" else "linear"
    folder = os.path.join(FOLDER_OF_EXPORTS, "tracks_" + of_map)
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
    if kind == "linear_flat":
        file = os.path.join(folder, "start_state_flat.f64")
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


def write(name, rows):
    names = []
    for row in rows:
        names += [k for k in row if k not in names]
    with open(os.path.join(RESULTS, name), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def accuracy():
    rows = []
    for index, m in enumerate(METHODS):
        start, reference, momentum, flat, _ = tracks_of(m["tracks"])
        z0, z1 = planes(m["tracks"])
        finals, counts = run(m["method"], start, z0, z1, map_name="v8r1_up", reader=m["reader"],
                             precision=m["precision"])
        row = dict(index=index, label=m["label"], family=m["family"], method=m["method"],
                   precision=m["precision"], tanh=m.get("tanh", ""), reader=m["reader"],
                   tracks=m["tracks"],
                   number_of_tracks=len(start), network_calls=m.get("calls", 0),
                   multiply_adds_a_call=multiply_adds(m["method"]),
                   field_reads_mean=float(counts[:, 0].mean()),
                   field_reads_largest=int(counts[:, 0].max()),
                   steps_mean=float(counts[:, 1].mean()),
                   failed_steps_mean=float(counts[:, 2].mean()))
        row.update(summary(finals, reference, momentum))
        row.update({"flat_" + k: v for k, v in
                    summary(finals[flat], reference[flat], momentum[flat]).items()})
        rows.append(row)
        print("%-78s %-6s reads %7.1f  median %10.4g um  95 %% %10.4g um  worst band %10.4g um"
              % (m["label"], m["precision"], row["field_reads_mean"], row["radial_median_um"],
                 row["radial_95_um"], row["worst_band_median_um"]), flush=True)
    write("accuracy.csv", rows)


def time(core):
    lines = []
    for index, m in enumerate(METHODS):
        z0, z1 = planes(m["tracks"])
        _, _, _, _, file = tracks_of(m["tracks"])
        label = "m%03d" % index          # the row of methods.py; the table joins on it
        lines.append("label=%s method=%s precision=%s map=v8r1_up reader=%s states=%s z0=%r z1=%r"
                     % (label, m["method"], m["precision"], m["reader"], file, z0, z1))
    list_file = os.path.join(RESULTS, "timing_list.txt")
    with open(list_file, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    out = os.path.join(RESULTS, "timing.csv")
    subprocess.run(["taskset", "-c", str(core), program(), "time", "list=" + list_file,
                    "out=" + out, "repeats=15", "budget=24", "cold=200"], check=True)
    print("timed on %s, core %s" % (socket.gethostname(), core))


def table():
    import pandas as pd
    a = pd.read_csv(os.path.join(RESULTS, "accuracy.csv"))
    t = pd.read_csv(os.path.join(RESULTS, "timing.csv"))
    t["index"] = t["label"].str[1:].astype(int)
    both = a.merge(t[["index", "ns_lowest", "ns_median", "ns_highest", "ns_shuffled_lowest",
                      "ns_shuffled_median", "cold_ns_median", "cold_ns_lowest", "host"]], on="index")
    assert len(both) == len(a), (len(both), len(a))
    both.to_csv(os.path.join(RESULTS, "methods.csv"), index=False)
    lines = ["| Method | Precision | Field read | Network calls | Field reads | Time a track (ns), "
             "lowest | median | cold caches | Median error (um) | 95 % (um) | Worst band (um) | "
             "Flat sample: median (um) | 95 % (um) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for family in ("LHCb", "classical", "network"):
        for _, r in both[both["family"] == family].iterrows():
            precision = r["precision"] + (", fast tanh" if r["tanh"] == "fast" else "")
            lines.append("| %s | %s | %s | %d | %.1f | %.0f | %.0f | %.0f | %.3g | %.3g | %.3g | %.3g | %.3g |"
                         % (r["label"], precision, r["reader"], r["network_calls"],
                            r["field_reads_mean"], r["ns_lowest"], r["ns_median"], r["cold_ns_median"],
                            r["radial_median_um"], r["radial_95_um"], r["worst_band_median_um"],
                            r["flat_radial_median_um"], r["flat_radial_95_um"]))
    with open(os.path.join(RESULTS, "methods.md"), "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "accuracy":
        accuracy()
    elif command == "time":
        time(sys.argv[2])
    elif command == "table":
        table()
    else:
        raise SystemExit("accuracy, time CORE or table")
