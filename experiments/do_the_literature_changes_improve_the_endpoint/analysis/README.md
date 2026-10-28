# Analysis

**Last updated:** 2026-10-28

## What this directory is

The numbers and the figure of the literature tests, and a short note that answers the question as far as the stopped tests allow. The six arms stopped after 3 to 6 rounds and have no standard report, so each is compared with its baseline at the same round, on the validation error that every round records. The script reads the store and writes nothing to it.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file | built |
| `results.md` | the answer, change by change, with what is not known | built |
| `results_and_figures.py` | reads `rounds.csv` of every arm and baseline, the standard reports of the baselines, and the tranche-1 table of change 2; writes the tables and the figure | built |
| `results/` | `arms_against_baselines.csv` (one row per arm), `validation_per_round.csv`, `baselines_on_the_test_split.csv`, `best_round_and_snapshot_average.csv` | built |
| `figures/` | `validation_per_round.png`, one panel per output form | built |

## The contract

| Change | Form | Arm | Key |
|---|---|---|---|
| 1 | direct states | fixed draw, own predictions after round 1 | `23f3bc73c61b` |
| 1 | direct states | fixed draw, reference states of round 1 | `c594bdddd98f` |
| 4 | direct states | self-scaled Broyden | `f4c86e05eafe` |
| 1 | straight line, one global scale | fixed draw, own predictions after round 1 | `1212e5c801c2` |
| 1 | straight line, one global scale | fixed draw, reference states of round 1 | `68a9f0f91e9e` |
| 4 | straight line, one global scale | self-scaled Broyden | `9a3df286d5fb` |

The baselines are `49e5e2450eaa` (direct states) and `9cef9c160b7d` (straight line, one global scale). The validation quantity is the one of the stopping rule: the median over the validation tracks of the larger of the endpoint errors in x and y, in micrometres. Validation numbers are not test numbers and are not compared with them.

Change 2 (best round) is not a training run. Its numbers are copied from `../../../analysis/tranche1_2026-10-28/results/a_best_round_and_snapshot_average.csv`, which scored the baselines on the test split; that folder must exist when the script runs. Change 3 (the refit of the last layer) was not run: no report with a refit is in the store.

The arm `f4c86e05eafe` was resumed on the farm on 2026-10-28 at 11:58 (its `state.json`). The tables hold what the store had when the script ran; run it again to update them.

## How to run

From this folder:

```bash
PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python results_and_figures.py
```

## How to add to it

1. Add a script, with its row above.
2. Change the date.
