# Analysis

**Last updated:** 2026-10-28

## What this directory is

The numbers and figures of the nine runs, with the 256-step 10 to 50 GeV run read at its old cap (round 80) and at its plateau after the extension (round 111), and a short note that answers the experiment's question. The script copies them from the standard reports in the store and writes nothing to the store.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file | built |
| `results.md` | the answer to the question: each criterion against its number | built |
| `results_and_figures.py` | reads the standard report of the last round of each run and its `rounds.csv`; writes the tables and the figures | built |
| `results/` | `runs.csv` (one row per run and report), `endpoint_by_momentum_band.csv`, `error_along_the_track.csv`, `validation_per_round.csv`, `references.csv` (the numbers the criteria are read against, with their source) | built |
| `figures/` | `validation_per_round.png`, `endpoint_by_momentum_band.png`, `error_along_the_track.png` | built |

## The contract

Every error is on the 1,452 test tracks, against the sixth-order reference, in micrometres, from the standard report of the last round of the run. Nothing is recomputed. A run is drawn hollow if it did not plateau on the validation error.

The runs and their reports are listed in `RUNS` at the top of the script and, with the report each number came from, in `results/runs.csv`. The paper's numbers the criteria are read against are in `results/references.csv`, copied from `../README.md`. The column `endpoint_radial_median_in_the_window_of_the_loss_micrometres` is filled for the cost-weighted runs only.

## How to run

From this folder:

```bash
PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python results_and_figures.py
```

It needs only the standard library and matplotlib, and reads only the store.

## How to add to it

1. Add a script, with its row above.
2. Change the date.
