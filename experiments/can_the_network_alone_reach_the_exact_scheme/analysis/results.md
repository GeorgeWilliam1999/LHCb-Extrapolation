# Results: can the network alone reach the exact scheme

**Last updated:** 2026-11-14 · **State:** all seven runs ended and scored; for discussion with George. Not a write-up.

Test split, 1,452 tracks, radial error at the end of the region against the sixth-order reference, from the standard report of each run at its last round. All seven runs plateaued by the validation rule; none reached its cap.

## The runs

| Run | Rounds | All momenta, median / p95 (µm) | Mean (µm) | Below 3 GeV | 3 to 8 GeV | 8 to 20 GeV | 20 to 50 GeV | Above 50 GeV |
|---|---|---|---|---|---|---|---|---|
| baseline `25dfa17ef168` (the run every file is copied from) | 44 | 1.83 / 12.7 | 4.93 | 4.94 | 2.10 | 1.44 | 1.21 | 1.89 |
| control: whole tracks, pooled | 46 | 2.80 / 36.0 | 11.7 | 9.81 | 3.41 | 1.64 | 2.26 | 4.07 |
| A: carried term, share 0.5 | 41 | 1.77 / 35.7 | 9.95 | 10.0 | 2.57 | 1.02 | 1.10 | 2.51 |
| A: carried term, share 0.9 | 46 | 2.07 / 29.0 | 8.98 | 7.59 | 2.49 | 1.36 | 1.51 | 2.01 |
| B: physics layer, 64 steps | 22 | 0.146 / 1.17 | 0.943 | 0.652 | 0.249 | 0.0851 | 0.0353 | 0.0115 |
| B: physics layer, 64 steps, width 32 | 22 | 0.147 / 1.19 | 0.944 | 0.665 | 0.249 | 0.0852 | 0.0353 | 0.0115 |
| B: physics layer, 32 steps | 22 | 0.782 / 4.98 | 4.67 | 3.02 | 1.34 | 0.461 | 0.188 | 0.0783 |
| B: physics layer, 16 steps | 22 | 1.68 / 50.2 | 31.8 | 9.98 | 3.37 | 0.846 | 0.391 | 0.0892 |
| exact two-stage scheme, 64 / 32 / 16 steps | | 0.147 / 0.787 / 1.68 | | 0.66 (64 steps) | 0.25 (64 steps) | 0.085 | 0.035 | 0.012 |

The band columns are medians in µm. The physics-layer run at 64 steps is 0.0006 µm from the exact scheme in the median (0.024 µm at the 95th percentile).

## Time per track

`evaluation/time_per_event.py`, one thread, stbc-i1, 13 events of 100 test tracks, 2026-11-14.

| Method | Steps | ms per track | Median of the timed tracks (µm) |
|---|---|---|---|
| baseline chain network | 64 | 2.98 | 1.83 |
| physics layer | 64 | 3.96 | 0.145 |
| physics layer, width 32 | 64 | 3.85 | 0.146 |
| physics layer | 32 | 2.08 | 0.816 |
| physics layer | 16 | 1.05 | 1.75 |
| Euler stages + 2 sweeps, no network | 64 | 2.41 to 2.52 | 0.145 |
| Euler stages + 2 sweeps, no network | 32 | 1.42 | 0.819 |
| Euler stages + 2 sweeps, no network | 16 | 0.88 | 1.74 |
| RK4 | 64 | 0.98 | 1.11 |
| RK4 | 32 | 0.49 | 0.318 |
| RK4 | 16 | 0.25 | 4.06 |

## Against the criteria

| # | Criterion | Outcome |
|---|---|---|
| 1 | the control is the baseline (1.46 to 2.57 µm) | **fails**: 2.80 µm, and its p95 is 36.0 µm against 12.7 µm. Training on whole tracks is worse than the baseline's draw. A is read against the control only |
| 2 | A is below the control by more than 0.37 µm | **passes**: 1.77 µm (share 0.5) against 2.80 µm |
| 3 | A below 0.59 µm (Broyden level) or 0.162 µm (exact scheme) | **fails**: 1.77 µm. A is not below the baseline (1.83 µm) by more than the seed range, and its p95 and mean are worse than the baseline's |
| 4 | coherence ratio of the better A run | being computed |
| 5 | B within 10 % of the exact scheme of its step count | **passes at all three**: 0.146 (limit 0.162), 0.782 (limit 0.866), 1.68 (limit 1.85) µm |
| 6 | a body of width 32 is enough | **passes**: 0.147 µm |
| 7 | time per track, reported | see the table above |
| 8 | convergence by validation | all seven plateaued |

## What the numbers say

1. **The physics layer makes the network equal to the exact scheme.** At 64, 32 and 16 steps, and with a body 4 times narrower, in every momentum band. The run ends at round 22, the earliest round the rule allows.
2. **On one CPU thread the layer network is not faster than a classical method of the same accuracy.** At 64 steps it takes 3.96 ms per track; Euler stages with two sweeps, with no network, reach the same 0.145 µm in 2.4 ms; RK4 at 32 steps reaches 0.32 µm in 0.49 ms. The narrow body saves 3 %: the time is in the field lookups and the per-track scale of the output (16 field samples a step), not in the body.
3. **The carried-defect term helps against its control, and not against the baseline.** The whole-track protocol itself costs accuracy: a round holds 500 tracks on all their planes, where the baseline's draw holds 500 different tracks on each plane. The tails show it (p95 29 to 36 µm against 12.7 µm).
4. **The test error of the three whole-track runs moves by about 1 µm from round to round** in validation (1.6 to 3.2 µm in the last 12 rounds). The last round is not the best one: by validation the best rounds are 1.80, 1.68 and 1.62 µm.

## Not done

- The coherence ratio of the A runs (criterion 4).
- A with more tracks a round, or with the baseline's draw for the pooled part and whole tracks for the carried part only.
- A timing on a GPU in batches, where the cost of a network call and of a field lookup differ from one CPU thread.
- No second seed of any run.
