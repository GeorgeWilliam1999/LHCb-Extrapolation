# Analysis

**Last updated:** 2026-10-28

## What this directory is

The numbers and figures of the two straight-line runs, beside the direct-states run of the same cell and loss, and a short note that answers the experiment's question. The script copies them from the standard reports in the store and writes nothing to the store.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file | built |
| `results.md` | the answer to the question: each criterion against its number | built |
| `results_and_figures.py` | reads the standard report of the last round of each run and its `rounds.csv`; writes the tables and the figures | built |
| `results/` | `runs.csv` (one row per run), `endpoint_by_momentum_band.csv`, `error_along_the_track.csv`, `validation_per_round.csv`, `references.csv` (the numbers the criteria are read against, with their source) | built |
| `figures/` | `validation_per_round.png`, `endpoint_by_momentum_band.png`, `error_along_the_track.png` | built |

## The contract

Every error is on the 1,452 test tracks, against the sixth-order reference, in micrometres, from the standard report of the last round of the run. Nothing is recomputed. A run is drawn hollow if it did not plateau on the validation error.

| Run | Key | Report |
|---|---|---|
| straight line, per-track scale | `25dfa17ef168` | `rkpinn_store/runs/25dfa17ef168/reports/round_0044_at_bc467594` |
| straight line, one global scale | `9cef9c160b7d` | `rkpinn_store/runs/9cef9c160b7d/reports/round_0022_at_bc467594` |
| direct states, read beside them | `49e5e2450eaa` | `rkpinn_store/runs/49e5e2450eaa/reports/round_0059_at_c44a599c` |

The frozen network's numbers (2.18 and 0.081 µm) are in `results/references.csv`, copied from `../README.md`.

## How to run

From this folder:

```bash
PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python results_and_figures.py
```

It needs only the standard library and matplotlib, and reads only the store.

## How to add to it

1. Add a script, with its row above.
2. Change the date.
