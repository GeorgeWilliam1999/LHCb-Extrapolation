#!/usr/bin/env python
"""Collect the grid's per-run json records into the two result tables.

Reads every `results/*.json` written by `_shared/train.py` and writes

  results/summary.csv          one row per run (the run's own numbers, plus the
                               parameter count of its architecture)
  results/by_architecture.csv  one row per (mode, width, depth): how many runs
                               converged, and the median / min / max over the
                               converged seeds of the test endpoint error, the
                               median validation endpoint error, and the median
                               restarts, wall time and final loss

Only converged runs enter the by-architecture statistics; the unconverged ones
are counted and kept in summary.csv but never pooled with them.  Runs still on
the farm simply have no json yet and are absent from both tables, so the
script is safe to run on a partially drained grid.

    PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python aggregate.py
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
SPLITS = ("train", "val", "test")
SCORES = ("endpoint_med_um", "endpoint_p95_um", "stage_med_um",
          "slope_med_mrad", "rho_mean", "rho_median", "straight_med_um", "n")


def n_params(width: int, depth: int, q: int = 8, n_in: int = 5) -> int:
    """Weights + biases of OneStepNetwork(q, width=width, depth=depth).

    `depth` Linear(->width)+Tanh blocks, then one Linear(width, 4*(q+1)).
    """
    n = 0
    prev = n_in
    for _ in range(depth):
        n += prev * width + width
        prev = width
    n += prev * 4 * (q + 1) + 4 * (q + 1)
    return n


def load_runs() -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(os.path.join(RESULTS, "*.json"))):
        if path.endswith("_meta.json"):
            continue
        with open(path) as f:
            rec = json.load(f)
        if "width" not in rec or "test" not in rec:
            continue                      # not a train.py record
        row = {k: rec[k] for k in ("tag", "mode", "seed", "q", "width", "depth",
                                   "restarts", "final_loss", "converged",
                                   "wall_s", "outer_cap")}
        row["n_params"] = n_params(rec["width"], rec["depth"], rec["q"])
        for split in SPLITS:
            for key in SCORES:
                row["%s_%s" % (split, key)] = rec[split][key]
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["mode", "width", "depth", "seed"])
    return df


def by_architecture(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (mode, width, depth), g in df.groupby(["mode", "width", "depth"]):
        conv = g[g["converged"]]
        allt = g["test_endpoint_med_um"].to_numpy(float)
        rec = {"mode": mode, "width": width, "depth": depth,
               "n_params": int(g["n_params"].iloc[0]),
               "n_runs": int(len(g)), "n_converged": int(len(conv)),
               # The converged-only statistics below are the headline, but a
               # architecture whose seeds mostly failed the confirmation would
               # report a median over a biased subsample, so the same three
               # numbers over every finished run are carried alongside.
               "test_endpoint_med_um_median_allruns": float(np.median(allt)),
               "test_endpoint_med_um_min_allruns": float(allt.min()),
               "test_endpoint_med_um_max_allruns": float(allt.max())}
        if len(conv):
            t = conv["test_endpoint_med_um"].to_numpy(float)
            rec.update({
                "test_endpoint_med_um_median": float(np.median(t)),
                "test_endpoint_med_um_min": float(t.min()),
                "test_endpoint_med_um_max": float(t.max()),
                "test_endpoint_spread_ratio": float(t.max() / t.min()),
                "val_endpoint_med_um_median":
                    float(np.median(conv["val_endpoint_med_um"])),
                "test_endpoint_p95_um_median":
                    float(np.median(conv["test_endpoint_p95_um"])),
                "restarts_median": float(np.median(conv["restarts"])),
                "wall_s_median": float(np.median(conv["wall_s"])),
                "final_loss_median": float(np.median(conv["final_loss"])),
            })
        else:
            for k in ("test_endpoint_med_um_median", "test_endpoint_med_um_min",
                      "test_endpoint_med_um_max", "test_endpoint_spread_ratio",
                      "val_endpoint_med_um_median", "test_endpoint_p95_um_median",
                      "restarts_median", "wall_s_median", "final_loss_median"):
                rec[k] = float("nan")
        out.append(rec)
    arch = pd.DataFrame(out)
    if len(arch):
        arch = arch.sort_values(["mode", "n_params", "depth"])
    return arch


def main():
    df = load_runs()
    if not len(df):
        print("no run records in %s yet" % RESULTS)
        return
    df.to_csv(os.path.join(RESULTS, "summary.csv"), index=False)
    arch = by_architecture(df)
    arch.to_csv(os.path.join(RESULTS, "by_architecture.csv"), index=False)

    print("%d run records; %d converged, %d not"
          % (len(df), int(df["converged"].sum()), int((~df["converged"]).sum())))
    cols = ["mode", "width", "depth", "n_params", "n_runs", "n_converged",
            "test_endpoint_med_um_median", "test_endpoint_med_um_min",
            "test_endpoint_med_um_max", "test_endpoint_med_um_median_allruns",
            "val_endpoint_med_um_median", "restarts_median", "wall_s_median",
            "final_loss_median"]
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(arch[cols].to_string(index=False, float_format=lambda v: "%.4g" % v))
    print("\nwrote results/summary.csv and results/by_architecture.csv")


if __name__ == "__main__":
    main()
