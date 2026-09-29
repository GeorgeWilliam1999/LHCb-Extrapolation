# Experiments

**Last updated:** 2026-09-30

## What this directory is

The experiments made with the package `rkpinn`, one folder each. An experiment is a set of configuration files; the package is in `../RK_Pinn_module/` and holds no experiment. The rules are in `../RK_Pinn_module/WORKFLOW.md`, section 8.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file | built |
| `does_a_summed_end_state_train/` | the pilot: one network of 64 steps and 2 stages, trained with its end state summed from the stages and with its end state predicted | built |

## The contract

A folder is named by the question it asks. It has a row in the Experiments table of the Notion project page from the day it is made. Its criteria for success are written in its README before its runs are submitted.

Runs, weights and reports are not kept here. They are in the store, `/data/bfys/gscriven/rkpinn_store`, under the key of each run.

## How to add to it

1. Write the question in one sentence and agree the design with George.
2. Make the folder, with a README in this format, and add its row above and in the Notion table.
3. Write the configurations, one file per run.
4. Follow the README of the experiment.
