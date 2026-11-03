# Analysis

**Last updated:** 2026-11-03

## What this directory is

The scripts, tables and figures of Parts A and B. They read the store, the field map and the cache of the per-step bias study. They write only to `results/`, `figures/`, `logs/` and to the scratch cache `/data/bfys/gscriven/scratch/kalman_2026-10-28/cache/` (not used by the runs below: every array needed was already in the bias-study cache).

## Contents

| Name | What it does | Writes |
|---|---|---|
| `results.md` | the results note: every number, the decision point | — |
| `common.py` | copy of the bias study's `common.py`: the runs, the tracks, the chain, the scores; reads that study's cache | — |
| `stage_equations.py` | copy of the bias study's file: the 2-stage stage equations, the Jacobian of f, the exact step | — |
| `part_b.py check` | the summed end state of the script against the network's end state | `logs/part_b_check.txt` |
| `part_b.py run` | Part B: plain, B1, B2 and the two controls; single-step errors; one low-momentum track | `results/b_endpoint.csv`, `results/b_single_step.csv`, `results/b_low_momentum_track.npz` |
| `toy_fit_with_q.py fit` | Part A: the A.7 toy fit with no Q, Q per step, Q at the end; the exact extrapolator; the reference | `results/a_toy_fit_with_q.csv`, `results/a_q_per_band.csv` |
| `timing.py run` | the time of each part of a step (CPU, torch); `THREADS` sets the threads | `results/timing_threads_<n>.csv` |
| `figures.py` | the figures, from the tables | `figures/b_endpoint_by_band.png`, `figures/b_low_momentum_track.png` |
| `run_one_step.sh` | runs one step from this folder and writes its log | `logs/` |
| `part_c_smoke.sh <store>` | Part C smoke run: C1a and C2 on the frozen per-track straight-line network, a small draw, 3 rounds, and the standard report of each; into a scratch store | the scratch store; `logs/part_c_smoke.txt` |
| `part_c_smoke_configurations/` | the two configurations of the smoke run (each a package default with 4,000 states, 5 restarts of 100 iterations, 3 rounds) | — |
| `part_c_smoke.md` | the numbers of the smoke run | — |

## How to run

From this folder, on a login node that is not loaded (2026-10-28: `stbc-i3`). About 5 minutes for `part_b.py run` and 5 minutes for the fit.

```bash
bash run_one_step.sh part_b.py check
bash run_one_step.sh part_b.py run
bash run_one_step.sh toy_fit_with_q.py fit
THREADS=1 bash run_one_step.sh timing.py run
PYTHONPATH=../../../RK_Pinn_module/src /data/bfys/gscriven/conda/envs/TE/bin/python figures.py
```

Package code at commit `bb38f113` (the last commit touching `RK_Pinn_module/src`), with uncommitted edits of another thread in `training/` only (a new training protocol, a new loss-constant hook); the networks, the field map, the integrators and the scoring are those of the commit. The bias-study cache must hold `e_<key>_validation_round_<last>.npy`, `e_exact_gl2_test.npy`, `jacobians_of_the_steps_test.npy` and `a7_ingredients.npz`.
