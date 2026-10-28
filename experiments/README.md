# Experiments

**Last updated:** 2026-10-28

## What this directory is

The experiments made with the package `rkpinn`, one folder each. An experiment is a set of configuration files; the package is in `../RK_Pinn_module/` and holds no experiment. The rules are in `../RK_Pinn_module/WORKFLOW.md`, section 8.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file | built |
| `does_a_summed_end_state_train/` | the pilot: one network of 64 steps and 2 stages, trained with its end state summed from the stages and with its end state predicted | built |
| `redo_the_papers_nine_runs_in_the_summed_form/` | the paper's three cells under the pooled and the two cost-weighted losses, trained again by the package in the summed form, seed 1 | built |
| `does_the_straight_line_form_restore_the_old_accuracy/` | the 64-step pooled run again with the output form of the frozen networks, a straight line plus a scaled correction, with the per-track scale and with one global scale | built |
| `do_the_literature_changes_improve_the_endpoint/` | short local tests of four changes the literature suggests (one fixed draw, the best round by validation, a refit of the last layer, the self-scaled Broyden optimiser), in both output forms | built |
| `does_the_error_fall_with_the_step_length/` | the per-track straight-line run trained again at 8, 16, 32, 128 and 256 steps of 2 stages, and at 64 steps of 4, 8 and 16 stages, for the figure of the step length against the single-step error and the error at the end of the region | built |
| `does_a_better_optimiser_improve_the_endpoint/` | the 64-step pooled run with the self-scaled Broyden optimiser in place of L-BFGS, seeds 1 to 3, in both output forms, and two more L-BFGS seeds of each form for the seed spread | built |

## The contract

A folder is named by the question it asks. It has a row in the Experiments table of the Notion project page from the day it is made. Its criteria for success are written in its README before its runs are submitted.

Runs, weights and reports are not kept here. They are in the store, `/data/bfys/gscriven/rkpinn_store`, under the key of each run.

## How to add to it

1. Write the question in one sentence and agree the design with George.
2. Make the folder, with a README in this format, and add its row above and in the Notion table.
3. Write the configurations, one file per run.
4. Follow the README of the experiment.
