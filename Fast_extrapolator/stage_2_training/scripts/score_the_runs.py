"""Score and time the trained runs of stage 2 in the compiled program, on the
test tracks, like every method of stages 0 and 1. A run is taken at its round
with the lowest validation median; a run with no round is left out.

    score_the_runs.py accuracy     results/runs_accuracy.csv
    score_the_runs.py time CORE    results/runs_timing.csv
    score_the_runs.py table        results/runs.csv
"""
from __future__ import annotations

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "python"))

from fastex import bench, train  # noqa: E402

RESULTS = os.path.join(HERE, "results")


def methods(export):
    rows = []
    for path in sorted(glob.glob(os.path.join(HERE, "configurations", "*.yaml"))):
        c = train.read_configuration(path)
        file_of_state = os.path.join(train.FOLDER_OF_RUNS, c["name"], "state.json")
        if not os.path.exists(file_of_state):
            continue
        state = json.load(open(file_of_state))
        if export:
            train.export(c["name"])
        best = min(state["validation_medians_um"])
        rows.append(dict(label=c["name"], family="start plus network",
                         method="corrected:%s:fast_tanh=1" % c["name"], precision="float",
                         reader=train.READER_OF[c["reference"]], tracks=c["reference"],
                         reference=c["reference"], start=c["start"], steps=c["steps"],
                         width=c["width"], depth=c["depth"], scale=c["scale"], loss=c["loss"],
                         rounds=state["rounds_done"], ended=state["ended"] or "running",
                         validation_median_um=best))
    return rows


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "accuracy":
        bench.accuracy(methods(True), os.path.join(RESULTS, "runs_accuracy.csv"))
    elif command == "time":
        # the list of the accuracy table, so that the two tables have the same rows
        import pandas as pd
        kept = pd.read_csv(os.path.join(RESULTS, "runs_accuracy.csv"))
        bench.time([dict(method=r["method"], precision=r["precision"], reader=r["reader"],
                         tracks=r["tracks"]) for _, r in kept.iterrows()], sys.argv[2],
                   os.path.join(RESULTS, "runs_timing.csv"))
    elif command == "table":
        both = bench.join(os.path.join(RESULTS, "runs_accuracy.csv"), os.path.join(RESULTS, "runs_timing.csv"))
        both.to_csv(os.path.join(RESULTS, "runs.csv"), index=False)
        print(both[["label", "rounds", "ended", "ns_lowest", "radial_median_um", "radial_95_um",
                    "worst_band_median_um"]].to_string())
    else:
        raise SystemExit("accuracy, time CORE or table")
