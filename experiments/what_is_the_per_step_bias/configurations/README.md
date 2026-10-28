# Configurations

**Last updated:** 2026-10-28

## What this directory is

One configuration file per run of Part B, four in all: items B.3 and B.4 in both forms (George, 2026-10-28). The name of a file is the form and the item. The folder is called `configurations/`, as in every other experiment.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file | built |
| `direct_states__b3_residual_draws.yaml` | direct states, B.3 residual draws; run `5cb2aeb6778b` | built |
| `direct_states__b4_bias_term.yaml` | direct states, B.4 bias term; run `6aba11cc7475` | built |
| `straight_line_per_track_scale__b3_residual_draws.yaml` | straight line with the per-track scale, B.3 residual draws; run `8a0f3b8f422f` | built |
| `straight_line_per_track_scale__b4_bias_term.yaml` | straight line with the per-track scale, B.4 bias term; run `f0f219df805e` | built |

## The contract

A file copies the configuration of its form and changes one block. Direct states: `../../redo_the_papers_nine_runs_in_the_summed_form/configurations/64_steps_2_stages_pooled.yaml` (run `49e5e2450eaa`). Straight line: `../../does_the_straight_line_form_restore_the_old_accuracy/configurations/64_steps_2_stages_pooled_per_track_scale.yaml` (run `25dfa17ef168`). Both: 64 steps, 2 stages, pooled loss over the stages, end state summed, seed 1, 32,000 states a round, 25 restarts of 200 L-BFGS iterations, the validation plateau rule, a cap of 80 rounds.

| Item | Block changed | Settings |
|---|---|---|
| B.3 | `training.protocol` → `rounds_on_own_predictions_with_residual_draws` | `share_drawn_by_residual` 0.5, `candidates_per_residual_state` 4, `weight_of_a_step_across_a_z_face` 2.0 |
| B.4 | `loss` → `pooled_with_a_bias_term` | `share_of_the_bias_term_at_round_one` 0.1, `number_of_bands_of_start_planes` 8, `momentum_band_edges_gev` [3, 8, 20, 50] |

The values of B.4 make 8 × 5 × 2 = 80 bins, about 400 states a bin in a round. The z-face weight of B.3 is a choice of this experiment: 81% of the steps of 80.9 mm cross a z face of the 100 mm grid, so a weight of 2 favours them without making the draw a draw of faces alone.

## How to add to it

1. Copy a file and change one block.
2. Print its key with `python -m rkpinn.run_record.configuration <file>` and add its row here.
3. Add its name, cpus and memory to `../farm/runs.txt`.
