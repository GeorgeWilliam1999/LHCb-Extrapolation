# Analysis

**Last updated:** 2026-10-28

## What this directory is

The numbers and figures of the pilot, and a short note that answers its question. The script copies them from the standard reports of the two runs in the store and writes nothing to the store.

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
| A, end state summed | `bddea42ea1ef` | `rkpinn_store/runs/bddea42ea1ef/reports/round_0041_at_184cf50e` |
| B, end state predicted | `4cc888dcc710` | `rkpinn_store/runs/4cc888dcc710/reports/round_0045_at_184cf50e` |

## How to run

From this folder:

```bash
PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python results_and_figures.py
```

It needs only the standard library and matplotlib, and reads only the store.

## How to add to it

1. Add a script, with its row above.
2. Change the date.
