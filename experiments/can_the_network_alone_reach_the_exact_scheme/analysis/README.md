# Analysis

**Last updated:** 2026-11-12

## What this directory is

The checks made before the runs, and later the results of the runs. The scripts read the store and write nothing to it.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file | built |
| `estimate_against_measured.py` | on existing runs: the label-free estimate of the carried-defect loss, summed per track, against the measured endpoint error against the exact chain | built |
| `results/estimate_against_measured.csv` | its table, per run and momentum band | built |
| `smoke.py` | builds each configuration with its checks before training, times one loss and gradient on one thread and takes 30 iterations in memory | built; all seven configurations pass (2026-11-12) |
