# Can the network alone reach the exact scheme

**Last updated:** 2026-11-12 · **State:** built and checked; **not submitted** (a farm submission needs George's word)

## What this directory is

Strand 2 of the plan "RK-PINN: plan for the next phase" (Claude Docs, 2026-11-12). George ruled on 2026-11-12 that the goal is a full network replacement, with physics-informed techniques to raise its accuracy, and that a physics layer inside the network counts as "the network is the process". Two techniques are tested, each as a new component of the package with the old behaviour unchanged.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file: the question, the two techniques, the runs, the criteria | built |
| `configurations/` | one file per run, seven in all | built |
| `farm/` | what puts the runs on the farm and keeps them going | built, not submitted |
| `analysis/` | the check of the label-free estimate on existing runs, the smoke check of the configurations, and later the results | started |

## The question

The best network chain has an endpoint error of 1.83 µm with L-BFGS (0.59 µm with the Broyden optimiser). The exact two-stage scheme at the same 64 steps has 0.147 µm. Can the network reach the exact scheme with no solver after it?

## Why these two techniques

The experiment `how_does_the_stage_residual_become_the_endpoint_error` measured the chain from the loss to the endpoint. Two facts follow from it.

1. **The endpoint error can be estimated without labels.** For each step, the end state summed from the network's stages minus the end state summed from the stages after one fixed-point sweep is the defect of the step. Carried on a straight line to the last plane and summed over the steps of a track, it is an estimate of (network endpoint − exact-chain endpoint). `analysis/results/estimate_against_measured.csv` has the check on eight existing runs:

   | Run | Steps | Measured against the exact chain (µm) | Estimate (µm) | Measured minus estimate (µm) | Correlation, x / y |
   |---|---|---|---|---|---|
   | straight line, per-track | 8 | 13.21 | 13.22 | 0.079 | 0.943 / 1.000 |
   | straight line, per-track | 16 | 7.20 | 7.21 | 0.036 | 0.992 / 0.999 |
   | straight line, per-track | 32 | 3.80 | 3.79 | 0.019 | 0.977 / 0.999 |
   | straight line, per-track | 64 | 1.862 | 1.857 | 0.010 | 0.954 / 1.000 |
   | straight line, per-track | 128 | 1.119 | 1.115 | 0.005 | 0.992 / 0.994 |
   | straight line, per-track | 256 | 0.635 | 0.637 | 0.003 | 0.989 / 0.998 |
   | direct states | 64 | 5.73 | 5.72 | 0.033 | 0.999 / 0.999 |
   | Broyden, straight line | 64 | 0.531 | 0.531 | 0.003 | 0.999 / 0.990 |

   Medians of the radial value over the 1,452 test tracks. The estimate leaves 0.5 % of the error. So a loss can train on the endpoint error of each track with no label.
2. **The defects of a track add up.** They keep one sign over a fixed length of track. The pooled loss does not see the sign.

## The two techniques

| Technique | Component | What it does | Field lookups per step at use |
|---|---|---|---|
| A. Carried-defect loss | loss `pooled_with_a_carried_defect_term`, protocol `rounds_on_whole_tracks_of_own_predictions` | adds to the pooled loss the squared estimate above, per track. The network and its cost at use do not change | 2, as now |
| B. Physics layer | network `stage_network_with_a_physics_layer`, `sweeps_inside: 1` | the last layer of the network is one application of the stage map, Y ← S0 + dz A f(Y); the body is trained through it | 4 |

## The runs

All: 2 stages, pooled loss over the stages, end state summed, straight line plus a correction scaled per track, L-BFGS, seed 1, cap 80 rounds, track set `12a8d35c3165`. Each file is the configuration of run `25dfa17ef168` (the baseline, 1.83 µm) changed only in what its row names.

| File | Change | Key |
|---|---|---|
| `whole_tracks_pooled.yaml` | the protocol only: whole tracks (control of A) | `6da7cb64cc48` |
| `whole_tracks_carried_share_0.5.yaml` | A, the term at a share of 0.5 of the loss of round one | `744ddcd5c8b7` |
| `whole_tracks_carried_share_0.9.yaml` | A, the term at a share of 0.9 | `ff8f8c2f67ba` |
| `physics_layer_64_steps.yaml` | B, 64 steps | `857f4642d383` |
| `physics_layer_32_steps.yaml` | B, 32 steps of 161.8 mm | `003029e43f24` |
| `physics_layer_16_steps.yaml` | B, 16 steps of 323.6 mm | `6af35e016a58` |
| `physics_layer_64_steps_width_32.yaml` | B, 64 steps, a body of width 32 in place of 128 | `a0f26d05a973` |

## The criteria, written before the runs are submitted

All errors are `endpoint_radial_median_micrometres` of the standard report on the 1,452 test tracks, at the round where the run plateaued or reached its cap. Every momentum band is reported; the 3 to 8 GeV band is named in the results beside the whole sample.

Numbers the criteria use: the baseline 1.83 µm; the range of three L-BFGS seeds of the baseline, 0.37 µm (1.83, 2.20, 2.10); the exact two-stage scheme, 0.147 µm at 64 steps, 0.787 µm at 32, 1.68 µm at 16.

| # | Criterion |
|---|---|
| 1 | The control is the baseline: `whole_tracks_pooled` is between 1.46 and 2.57 µm (the three seeds, widened by their range). If not, the protocol changes the result, and A is read against the control only |
| 2 | A passes when the better of its two runs is below the control by more than 0.37 µm |
| 3 | A reaches the Broyden level when that run is below 0.59 µm; it reaches the exact scheme when it is below 0.162 µm (the exact scheme plus 10 %) |
| 4 | The coherence ratio of the better A run (script `decompose.py` of the theory experiment) is reported beside that of the control. A removes the coherence if the ratio falls below 1.2 (random signs give 0.67; the baseline has 2.4) |
| 5 | B passes at a step count when its run is within 10 % of the exact scheme of that step count: below 0.162 µm at 64 steps, 0.866 µm at 32, 1.85 µm at 16 |
| 6 | A body of width 32 is enough when `physics_layer_64_steps_width_32` also passes criterion 5 |
| 7 | The time per track of every B run that passes, and of RK4 and the exact scheme at the nearest accuracy, is measured with `evaluation/time_per_event.py` and reported. No pass mark |
| 8 | Convergence is read from the validation error. A run at its cap without a plateau is reported as such and is not extended without George's word |

What the outcomes mean:

| Outcome | What follows |
|---|---|
| A passes criterion 3 at the exact-scheme level | the network alone, with 2 field lookups a step, is a full replacement at 64 steps; the loss becomes the proposal for the default |
| A passes criterion 2 only | the term helps; its share and the Broyden optimiser are the next settings to vary |
| A fails | the body cannot represent the defect-free stages at this size; B is the route |
| B passes at 16 or 32 steps | a network with the layer is a full replacement at 4 to 8 times fewer steps than now; the timing of criterion 7 says what that is worth |

## What is not known before the runs

- Whether the body, at width 128, can make the carried sum small on tracks it has not seen. The estimate is exact to 0.5 %; the capacity is the open point.
- Whether training through the layer is stable. The loss of a layer run starts 4 orders of magnitude lower than that of the baseline (6.6e-9 against 4.9e-5 on round-one states), because the sweep contracts the stage error.
- The time of a round on the farm. On one thread of the login node, one loss and gradient takes 0.46 s against 0.42 s for the baseline, so about 10 % more than the baseline's 42 minutes a round.

## To run

See `farm/README.md`. Nothing is submitted until George says so in the thread.

## Findings

None yet.
