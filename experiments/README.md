# Experiments

**Last updated:** 2026-11-12

## What this directory is

The experiments made with the package `rkpinn`, one folder each. An experiment is a set of configuration files; the package is in `../RK_Pinn_module/` and holds no experiment. The rules are in `../RK_Pinn_module/WORKFLOW.md`, section 8.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file, with the index of the results | built |
| `RUNS.md` | one row per run in the store: its settings, its state, its latest score and the experiments that name it | built (written by the script below) |
| `index_of_the_runs.py` | writes `RUNS.md` from the store; reads the store only | built |
| `does_a_one_shot_path_network_remove_the_chain/` | one supervised network call emits the whole 64-step path of a track; one parallel Newton sweep corrects it; against the chain on accuracy and cost per event | built; P1, P2 and the control scored, P3 and P4 not submitted |
| `what_is_the_per_step_bias/` | the per-step bias of the trained networks: its size and sign along the track, its correlation, and corrections for it (Part B: residual draws and a bias term in the loss) | built |
| `how_does_the_stage_residual_become_the_endpoint_error/` | Strand 1 of the plan of 2026-11-12: the chain from the trained stage residual to the endpoint error, step by step, on existing runs; no training | built |
| `can_the_network_alone_reach_the_exact_scheme/` | Strand 2 of the plan: a loss on the label-free estimate of each track's endpoint error, and a physics layer inside the network; seven runs | built, not submitted |
| `does_a_kalman_like_gain_remove_the_bias/` | a Kalman-like gain from the free stage residual: a model-error covariance in a toy fit (Part A), the analytic gain (Part B), a learned gain and an error-conditioned network (Part C) | built; second tranche of Part C plateaued, not scored |
| `does_a_summed_end_state_train/` | the pilot: one network of 64 steps and 2 stages, trained with its end state summed from the stages and with its end state predicted | built |
| `redo_the_papers_nine_runs_in_the_summed_form/` | the paper's three cells under the pooled and the two cost-weighted losses, trained again by the package in the summed form, seed 1 | built |
| `does_the_straight_line_form_restore_the_old_accuracy/` | the 64-step pooled run again with the output form of the frozen networks, a straight line plus a scaled correction, with the per-track scale and with one global scale | built |
| `do_the_literature_changes_improve_the_endpoint/` | short local tests of four changes the literature suggests (one fixed draw, the best round by validation, a refit of the last layer, the self-scaled Broyden optimiser), in both output forms | built |
| `does_the_error_fall_with_the_step_length/` | the per-track straight-line run trained again at 8, 16, 32, 128 and 256 steps of 2 stages, and at 64 steps of 4, 8 and 16 stages, for the figure of the step length against the single-step error and the error at the end of the region | built |
| `does_a_better_optimiser_improve_the_endpoint/` | the 64-step pooled run with the self-scaled Broyden optimiser in place of L-BFGS, seeds 1 to 3, in both output forms, and two more L-BFGS seeds of each form for the seed spread | built |

## The results, one line each

Test endpoint error: radial, median over the 1,452 test tracks, against the sixth-order reference, at the end of the magnet region. 64 steps and 2 stages unless the line says otherwise. The exact two-stage scheme at 64 steps gives 0.147 µm. Each experiment's README has the full numbers and the criteria.

| Experiment | Question | Answer | Main number |
|---|---|---|---|
| `does_a_summed_end_state_train/` | Does a network train when its end state is summed from its stages? | Yes. Summing trains; predicting the end state fails. | 8.97 µm summed against 623 µm predicted |
| `redo_the_papers_nine_runs_in_the_summed_form/` | What do the paper's nine runs give in the summed form? | The pooled runs beat the paper; the cost-weighted runs do not reach the pooled ones. | pooled 5.71 / 6.60 / 4.81 µm at 64/2, 128/8, 256/16 |
| `does_the_straight_line_form_restore_the_old_accuracy/` | Does the old output form (straight line plus scaled correction) restore the old accuracy? | Yes, with a per-track scale. | 1.83 µm (per-track scale), 2.04 µm (global scale) |
| `do_the_literature_changes_improve_the_endpoint/` | Do four changes from the literature help? | Interrupted after 3 to 6 rounds; only short-run validation numbers. | see its `analysis/results.md` |
| `does_the_error_fall_with_the_step_length/` | Does the error fall with the step length? | Yes for the network: single step as dz², end of region about as dz. The exact scheme and RK4 are not monotone (field-map kinks). | 31.0 µm at 8 steps to 0.657 µm at 256 steps |
| `does_a_better_optimiser_improve_the_endpoint/` | Does the self-scaled Broyden optimiser beat L-BFGS by more than the seed spread? | Yes, in both output forms. | 0.59 µm against 2.10 µm (straight line), 1.55 µm against 7.80 µm (direct states), medians of 3 seeds |
| `what_is_the_per_step_bias/` | What is the per-step bias, and can a correction remove it? | The step errors in x and y are strongly correlated along the track (lag-1 0.85 to 0.90), so they add up; one fixed-point sweep per step removes them. | see its `analysis/RESULTS.md` |
| `does_a_kalman_like_gain_remove_the_bias/` | Does a Kalman-like gain from the stage residual remove the bias? | The analytic gain is one Newton step and reaches the exact scheme. The learned correction that works learns one fixed-point sweep and adds nothing to it. | 0.147 µm (analytic), 0.145 to 0.148 µm (learned, fixed scale) |
| `how_does_the_stage_residual_become_the_endpoint_error/` | How does the trained residual become the endpoint error? | The stage error is the residual; the step defect is linear in it; the defects of a track keep one sign over a fixed length, about 11 segments a track. | endpoint ∝ dz^0.88 measured, dz^0.93 from the theory |
| `does_a_one_shot_path_network_remove_the_chain/` | Can one network give the whole path in one call? | Not alone; one Newton sweep from it reaches the exact scheme. | 196.9 µm alone, 0.1455 µm after one sweep, 0.83 ms per track |

## The contract

A folder is named by the question it asks. It has a row in the Experiments table of the Notion project page from the day it is made. Its criteria for success are written in its README before its runs are submitted.

Runs, weights and reports are not kept here. They are in the store, `/data/bfys/gscriven/rkpinn_store`, under the key of each run.

## How to add to it

1. Write the question in one sentence and agree the design with George.
2. Make the folder, with a README in this format, and add its row above and in the Notion table.
3. Write the configurations, one file per run.
4. Follow the README of the experiment.
