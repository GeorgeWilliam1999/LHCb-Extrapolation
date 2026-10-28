# Does the straight-line form restore the old accuracy

**Last updated:** 2026-10-28

## What this directory is

The third experiment of the package: the 64-step, 2-stage pooled run of the nine-runs experiment, trained again with the output form of the frozen networks, a straight line plus a scaled correction, in the two ways of scaling the correction. Correction 2 of the plan of 2026-10-20 (project library, `correction_plan_final.md`). Prepared on 2026-10-22; not submitted until George says so.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file: the question, the runs, the criteria, how to run | built |
| `configurations/` | one configuration file per run, two in all | built |
| `farm/` | what puts the runs on the farm and keeps them going | built |
| `analysis/` | the results tables, the figures and `results.md`, the answer to the question; read from the standard reports in the store | built |

## The question

Does a stage network that emits a correction to the straight line, with its end state summed from the stages and trained from scratch by the package, reach the accuracy of the frozen network with its stages summed, 2.2 micrometres at the end of the region?

## Why it is asked

The audit of 2026-10-19 showed that the package's gain over the paper comes from the summed end state alone: the frozen 64-step pooled network, chained with its own stages summed and no retraining, gives 2.18 micrometres at the end of the region, where the package run `49e5e2450eaa` gives 5.71. At the stages the package network is worse than the frozen one by a factor of six. Eighteen differences between the two were checked; seventeen are cleared, and the one that remains is the output form. The package emits the states themselves, scaled by the spread of the first round's states, about 445 mm in x; a micrometre is two parts in a million of what the network emits. The frozen network emits the straight line plus a correction scaled by the bend of the track over one step. Isolated on the login node with the same trainer, loss, states, seed and body, the straight line lowers the loss 190 to 390 times and the single-step error 12 to 15 times.

Whether the per-track scale of the frozen code adds anything over one global scale is not known: on the login node the two were within 20 per cent of each other, with the global scale slightly ahead. Both are run.

## The runs

Both copy the 64-step pooled configuration of the nine-runs experiment, run key `49e5e2450eaa`, and change only the output form and the scale of the outputs. Seed 1, cap 80 rounds, as there.

| Run | Output form | Scale of the correction | Key |
|---|---|---|---|
| per-track scale | straight line plus a scaled correction | per track: kappa |q/p| times the field integral along the straight line, as in the frozen code | see `farm/runs.txt` and the key printed by `rkpinn.run_record.configuration` |
| global scale | the same | one number per component: the median of the per-track scale over the first round's states | the same |

What is read beside them, with no new training: run `49e5e2450eaa` (the same cell and loss with direct states, 5.71 micrometres) and the frozen 64-step pooled network with its stages summed (2.18 micrometres, 0.081 micrometres a step).

## The criteria, written before the runs were submitted

All errors are on the 1,452 test tracks, against the sixth-order reference, from the standard report at the last round.

| # | Criterion | Number it is read against |
|---|---|---|
| 1 | Each run reaches the frozen network with its stages summed at the end of the region: `endpoint_radial_median_micrometres` at or below 2.2 | 2.18 micrometres, the frozen 64-step pooled weights chained with the end state summed |
| 2 | Each run reaches the frozen network at the stages: `single_step_radial_median_micrometres` from reference states at or below 0.08 | 0.081 micrometres, the same weights, one step from the reference states |
| 3 | Convergence is read from the validation error; a run that reaches its cap without a plateau is reported as such | |

A single-step error is judged against the frozen network's stages summed, never against its predicted end state (0.333 micrometres): that is the mistake of the pilot's criterion 1.

What the outcomes mean:

| Outcome | What follows |
|---|---|
| 1 and 2 hold for both runs | the output form was the whole cause; the one that is better sets the default, and the nine runs are redone in that form (correction 3 of the plan) |
| 1 and 2 hold for one run | that scale is the default; the other is reported with its numbers |
| 1 fails for both | the form is not the whole cause; the list of eighteen differences is reopened at the training length and at the states of the later rounds, which now come from a summed chain |

## What is not known before the runs

- The time of a round. The frozen runs and the nine runs give about 90 s a restart at 64 steps, so about 40 minutes a round and up to 55 hours for 80 rounds; the farm's time limit restarts the job and the run resumes.
- Whether the per-track scale matters over a full run. On the login node it did not.

## To run

See `farm/README.md`. The standard report of a run, when it has ended, from `RK_Pinn_module`:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src /data/bfys/gscriven/conda/envs/TE/bin/python \
    -m rkpinn.run_record.report_of_a_run --store /data/bfys/gscriven/rkpinn_store \
    --run <key> --round last --split test --tracks-for-the-stage-errors 200 \
    --step-length-of-the-carrying-mm 1.0 --fraction-for-the-derivative 1e-3 \
    --states-for-the-loss-shares 8000 --bins-of-the-histograms 60
```

## What was measured when the runs were submitted

| | |
|---|---|
| Commit the runs train at | `bc467594`, from the copy `/data/bfys/gscriven/rkpinn_code/bc467594` |
| Checks on the login node before submission, on the real configurations and the real store, nothing written | both runs built; loss at the exact stages 5.8e-34; loss before training 4.93e-5 (per-track scale) and 5.36e-5 (global scale), against 0.966 for the direct-states run `49e5e2450eaa`; ten iterations of one restart took 7 s on four threads |
| Global scale measured on the first round's states, seed 1 | 0.0957 mm in x, 0.000845 mm in y, 2.37e-3 in tx, 2.09e-5 in ty |
| Submitted | 2026-10-22 11:38, clusters 5985540 (per-track scale, `25dfa17ef168`) and 5985541 (global scale, `9cef9c160b7d`), by the keeper, on George's word in the thread |
| Logs of the farm and of the keeper | `/data/bfys/gscriven/rkpinn_farm_logs/does_the_straight_line_form_restore_the_old_accuracy` |

## Findings

The numbers against the criteria are in `analysis/results.md`. They are discussed with George before anything is written up.
