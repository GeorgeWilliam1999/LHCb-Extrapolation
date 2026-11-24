"""The test score of every run of this experiment that has a standard report.

    collect_scores.py      writes results/test_scores.csv

One row per configuration: its run key, state, round and the endpoint radial
median of the latest standard report on the test split (13,447 tracks).
Reads the store; writes nothing to it.
"""
import csv
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIGURATIONS = os.path.join(HERE, "..", "configurations")
STORE = "/data/bfys/gscriven/rkpinn_store/runs"
# the two seed-1 runs of the previous experiment, which this one does not run again
EARLIER = {"straight_line_per_track_scale__lbfgs__seed_1": "400182ba3233",
           "physics_layer__seed_1": "a8a4083032fd"}

sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "RK_Pinn_module", "src"))
from rkpinn.run_record.configuration import read_configuration, run_key  # noqa: E402

runs = dict(EARLIER)
for file in sorted(glob.glob(os.path.join(CONFIGURATIONS, "*.yaml"))):
    runs[os.path.basename(file)[:-5]] = run_key(read_configuration(file))
rows = []
for name, key in sorted(runs.items()):
    state = json.load(open(os.path.join(STORE, key, "state.json")))
    reports = sorted(glob.glob(os.path.join(STORE, key, "reports", "*", "summary.json")))
    summary = json.load(open(reports[-1])) if reports else {}
    rows.append(dict(configuration=name, run_key=key, state=state.get("state"),
                     rounds_done=state.get("rounds_done"),
                     report=os.path.basename(os.path.dirname(reports[-1])) if reports else "",
                     split=summary.get("split", ""),
                     test_endpoint_radial_median_micrometres=summary.get(
                         "endpoint_radial_median_micrometres", "")))
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
with open(os.path.join(HERE, "results", "test_scores.csv"), "w", newline="") as handle:
    writer = csv.DictWriter(handle, list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(len(rows), "rows,", sum(1 for r in rows if r["report"]), "with a report")
