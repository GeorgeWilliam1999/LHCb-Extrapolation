# Tranche 1: the two analyses that need no farm time

**Last updated:** 2026-10-28

## What this directory is

Items 1a and 1b of tranche 1 of the improvement plan of 2026-10-28 (Claude Docs, "RK-PINN: proposed changes to the loss, training and architecture"), approved by George on 2026-10-28. They read existing runs of the store and the field map; they train nothing and write nothing to the store.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file: what was computed and the numbers | built |
| `tranche1_analyses.py` | the script: `rounds` (item 1a), `steps` and `symmetry` (item 1b) | built |
| `results/a_best_round_and_snapshot_average.csv` | item 1a | built |
| `results/b_correlation_of_step_errors.csv` | item 1b, the errors along the track | built |
| `results/b_field_map_mirror_symmetry.csv` | item 1b, the mirror symmetry of the field map | built |
| `log_rounds.txt`, `log_steps.txt`, `log_symmetry.txt` | what the script printed | built |

To run, from this folder: `PYTHONNOUSERSITE=1 PYTHONPATH=../../RK_Pinn_module/src THREADS=4 /data/bfys/gscriven/conda/envs/TE/bin/python tranche1_analyses.py <rounds|steps|symmetry>`. Code at commit `d028404d` of the package (unchanged since `3ec05291`).

All errors are on the 1,452 test tracks, against the sixth-order reference. The endpoint is the last plane, 7826.0 mm. Positions are in micrometres.

## 1a. Best round by validation, and the average of the last 5 snapshots

"Best round" is the round with the lowest validation error in the quantity of the stopping rule (`--round best` of the standard report). "Average" is the mean of the predicted end states (x, y, tx, ty) of the networks of the last 5 rounds, each chained on its own. The last-round numbers equal the standard reports of those rounds to the last digit, which checks the script.

| Run | Form | Last round | Best round by validation | Average of the last 5 snapshots |
|---|---|---|---|---|
| `25dfa17ef168` | straight line, per-track scale | 1.83 (round 44) | 1.69 (round 33) | 1.41 (rounds 40 to 44) |
| `9cef9c160b7d` | straight line, one global scale | 2.04 (round 22) | 1.91 (round 19) | 1.37 (rounds 18 to 22) |
| `49e5e2450eaa` | direct states | 5.71 (round 59) | 5.71 (round 59, the last) | 5.24 (rounds 55 to 59) |

Test endpoint radial median, micrometres.

## 1b. Correlation of the single-step errors along a track

For each test track, the network takes one step from the reference state on each of the 64 start planes; the signed error is the network's end state minus the reference one step later. The lag-1 autocorrelation is taken along the 64 steps of each track, then the median over tracks (16th to 84th percentile in brackets). Last round of each run.

| | `25dfa17ef168` straight line, per-track | `49e5e2450eaa` direct states |
|---|---|---|
| lag-1 autocorrelation, x | 0.85 (0.74 to 0.92) | 0.87 (0.83 to 0.93) |
| lag-1 autocorrelation, y | 0.90 (0.77 to 0.96) | 0.89 (0.81 to 0.95) |
| lag-1 autocorrelation, tx | -0.37 (-0.39 to -0.35) | -0.36 (-0.38 to -0.30) |
| lag-1 autocorrelation, ty | -0.15 (-0.32 to 0.10) | 0.12 (-0.11 to 0.44) |
| single-step radial error, median | 0.072 | 0.486 |
| endpoint radial error, median | 1.83 | 5.71 |
| endpoint median / (sqrt(64) x single-step median) | 3.17 | 1.47 |
| per track: endpoint / (sqrt(64) x rms of its single-step errors), median | 2.51 | 1.21 |
| per track: endpoint / (64 x rms of its single-step errors), median | 0.31 | 0.15 |
| endpoint from the sum of the single-step errors, each carried to the last plane on a straight line, median | 1.84 | 5.70 |
| correlation of that sum with the chained endpoint error, x / y | 0.976 / 0.9995 | 0.998 / 0.999 |

The sum "carried on a straight line" is, per track, the sum over steps k of e_x,k + e_tx,k (z_last - z_end of step k), and the same in y.

## 1b. Mirror symmetry of the field map in y

The map `v8r1_up` is on a grid of 81 x 81 x 146 points, 100 mm apart, y from -4000 to 4000 mm. For a field symmetric under y -> -y: By(x,-y,z) = By(x,y,z), Bx(x,-y,z) = -Bx(x,y,z), Bz(x,-y,z) = -Bz(x,y,z). The table gives the departure, as a fraction of the largest |By| in the region.

| Region | Points | Peak |By| | By: max / p99 / median | Bx: max / p99 / median | Bz: max / p99 / median |
|---|---|---|---|---|---|
| every grid point | 957,906 | 4.94 T | 0.455 / 0.126 / 1.7e-4 | 0.588 / 0.131 / 2.2e-4 | 0.483 / 0.118 / 4.5e-4 |
| grid points in the box the test tracks occupy | 115,830 | 4.94 T | 0.353 / 0.134 / 1.2e-3 | 0.312 / 0.123 / 8.7e-4 | 0.394 / 0.133 / 1.3e-3 |
| the test tracks, reference states on 257 planes | 373,164 | 1.50 T | 0.126 / 6.5e-3 / 1.6e-4 | 0.171 / 4.1e-3 / 5.7e-4 | 0.240 / 9.5e-3 / 1.7e-3 |

The peak of 4.94 T is at x = -1600, y = 1100, z = 4300 mm, outside the region the tracks cross. The grid is not mirror-symmetric to better than about 13 per cent at its 99th percentile.

## Findings

None yet. They are discussed with George before anything is written up.
