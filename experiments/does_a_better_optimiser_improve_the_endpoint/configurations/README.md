# Configurations

**Last updated:** 2026-10-28

## What this directory is

One configuration file per run of the experiment, ten in all. The name of a file is the output form, the optimiser and the seed.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file | built |
| `direct_states__lbfgs__seed_2.yaml` | direct states, L-BFGS, seed 2; run `b6bd9570e6f2` | built |
| `direct_states__lbfgs__seed_3.yaml` | direct states, L-BFGS, seed 3; run `5cae49832580` | built |
| `direct_states__self_scaled_broyden__seed_1.yaml` | direct states, self-scaled Broyden, seed 1; run `f4c86e05eafe`, the 23 Oct trial, resumed | built |
| `direct_states__self_scaled_broyden__seed_2.yaml` | direct states, self-scaled Broyden, seed 2; run `c3a1aa54c6fe` | built |
| `direct_states__self_scaled_broyden__seed_3.yaml` | direct states, self-scaled Broyden, seed 3; run `c19552a6831c` | built |
| `straight_line_per_track_scale__lbfgs__seed_2.yaml` | straight line with the per-track scale, L-BFGS, seed 2; run `dedaab236b3d` | built |
| `straight_line_per_track_scale__lbfgs__seed_3.yaml` | straight line with the per-track scale, L-BFGS, seed 3; run `b68802fcce0e` | built |
| `straight_line_per_track_scale__self_scaled_broyden__seed_1.yaml` | straight line with the per-track scale, self-scaled Broyden, seed 1; run `166cfc8cdfd6` | built |
| `straight_line_per_track_scale__self_scaled_broyden__seed_2.yaml` | straight line with the per-track scale, self-scaled Broyden, seed 2; run `3400f9c572ef` | built |
| `straight_line_per_track_scale__self_scaled_broyden__seed_3.yaml` | straight line with the per-track scale, self-scaled Broyden, seed 3; run `c8b210923994` | built |

## The contract

A file copies the seed-1 L-BFGS configuration of its form and changes only the seed, or the seed and the optimiser block. The direct-states source is `../../redo_the_papers_nine_runs_in_the_summed_form/configurations/64_steps_2_stages_pooled.yaml` (run `49e5e2450eaa`); the straight-line source is `../../does_the_straight_line_form_restore_the_old_accuracy/configurations/64_steps_2_stages_pooled_per_track_scale.yaml` (run `25dfa17ef168`). A seed-1 L-BFGS copy made the same way gives those two keys again, which was checked on 2026-10-28.

## How to add to it

1. Copy a file and change one block.
2. Print its key with `python -m rkpinn.run_record.configuration <file>` and add its row here.
3. Add its name, cpus and memory to `../farm/runs.txt`.
