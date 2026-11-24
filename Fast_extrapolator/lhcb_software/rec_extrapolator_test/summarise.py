"""The time of the Rec integrator in the LHCb software beside the time of our copy.

Reads the timing tables of the Gaudi jobs (output/time_rk_*.log,
output/time_step_*.log and output/time_linear_*.log, made with run.sh) and the
timing of the copies
(output/copy_on_the_grid_<node>.csv, made with `fx time`), all on the same node
and core and on the same 1,331 states of the Rec test.

In Gaudi the time of one call of the integrator is the time of the test
algorithm with the integrator minus its time with the straight-line
extrapolator, divided by 1,331. The Rec test asks for the transport matrix in
every call, so the copy with its transport matrix is the like-for-like row.

    summarise.py <node>      writes output/summary_<node>.csv
"""
from __future__ import annotations

import csv
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STATES = 1331


def microseconds_per_event(pattern):
    values = []
    for path in sorted(glob.glob(os.path.join(HERE, "output", pattern))):
        for line in open(path):
            if '"ExtrapolatorTester"' in line:
                values.append(float(line.split("|")[4]))
    return values


def main(node):
    linear = microseconds_per_event("time_linear_*.log")
    baseline = sum(linear) / len(linear)
    rows = []
    for tool, pattern, name in (("rk", "time_rk_*.log", "TrackRungeKuttaExtrapolator (adaptive Cash-Karp)"),
                                ("step", "time_step_*.log", "TrackSTEPExtrapolator (adaptive Nystrom)")):
        gaudi = [(a - baseline) / STATES * 1e3 for a in microseconds_per_event(pattern)]
        rows.append(dict(what="Rec v41r3 in Gaudi: %s with its transport matrix" % name,
                         jobs=len(gaudi), ns_a_state_lowest=min(gaudi), ns_a_state_highest=max(gaudi)))
    for row in csv.DictReader(open(os.path.join(HERE, "output", "copy_on_the_grid_%s.csv" % node))):
        rows.append(dict(what="our program: " + row["label"].replace("_", " "), jobs=1,
                         ns_a_state_lowest=float(row["ns_lowest"]),
                         ns_a_state_highest=float(row["ns_highest"])))
    with open(os.path.join(HERE, "output", "summary_%s.csv" % node), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print("%-88s %8.0f to %8.0f ns a state" % (row["what"], row["ns_a_state_lowest"],
                                                  row["ns_a_state_highest"]))


if __name__ == "__main__":
    main(sys.argv[1])
