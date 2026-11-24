"""How far apart are the two readings of the same field map?

LHCb reads its field map with linear interpolation between the points of the
grid. This project scores against the cubic reading of the same points. The
script carries the test tracks of the cubic track set with methods that read
the map linearly and scores them against the cubic reference. The first row is
an integration of the linear reading with 2,048 steps, so its error is the
difference of the two readings and nothing else.

    two_readings_of_the_map.py      results/two_readings_of_the_map.csv
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "python"))

from fastex import bench  # noqa: E402

METHODS = [
    ("The linear reading, integrated with 2,048 steps", "rk4:2048", "double", "linear"),
    ("HLT2 adaptive Nystrom, linear reading (as LHCb runs it)", "rec_step", "double", "rec"),
    ("HLT2 adaptive Cash-Karp, linear reading (as LHCb runs it)", "rec_cash_karp", "double", "rec"),
    ("HLT2 adaptive Nystrom on the cubic reading", "rec_step", "double", "cubic"),
    ("HLT2 adaptive Cash-Karp on the cubic reading", "rec_cash_karp", "double", "cubic"),
]

if __name__ == "__main__":
    bench.accuracy([dict(label=label, family="two readings", method=method, precision=precision,
                         reader=reader, tracks="cubic", map_of_the_field="cubic reference")
                    for label, method, precision, reader in METHODS],
                   os.path.join(HERE, "results", "two_readings_of_the_map.csv"))
