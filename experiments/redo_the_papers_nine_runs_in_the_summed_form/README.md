# Redo the paper's nine runs in the summed form

**Last updated:** 2026-10-28

## What this directory is

The second experiment of the package: the nine runs of the second mini-paper, three cells under three losses, trained again by the package in the summed form, with total independence from the frozen code, its results and the pilot. Agreed with George on 2026-10-12.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file: the question, the runs, the criteria, how to run | built |
| `configurations/` | one configuration file per run, nine in all | built |
| `farm/` | what puts the runs on the farm and keeps them going | built |
| `analysis/` | the results tables, the figures and `results.md`, the answer to the question; read from the standard reports in the store | built |

## The question

What do the paper's three cells and three losses give when the network emits its stages and sums its end state from them, trained from scratch by the package?

## Why it is asked

The pilot showed that the summed form trains, and that at 64 steps and 2 stages with the pooled loss it reaches 8.97 micrometres at the endpoint where the frozen network of the same size reached 145.7. The paper's other findings, that the cost-weighted loss buys a factor of 2.6 to 4.4 in its window at a price below 3 GeV, and that shorter steps do not lower the endpoint error, were measured on the old form. This experiment measures them on the new form, with nothing carried over.

## The runs

All nine copy the LHCb default, `RK_Pinn_module/src/rkpinn/defaults/lhcb_64_steps_2_stages.yaml`, and change only the cell, the loss, the seed and the cap. Seed 1 for every run, so that no run is the pilot's run A again, and a cap of 80 rounds, because two of the paper's six runs needed 63 and 74 rounds to plateau.

| Cell | Step | Pooled | Cost-weighted, 10 to 50 GeV | Cost-weighted, 3 to 8 GeV |
|---|---|---|---|---|
| 64 steps, 2 stages | 80.9 mm | `49e5e2450eaa` | `da3a1d00b33d` | `f977c7b61049` |
| 128 steps, 8 stages | 40.5 mm | `75200e48408c` | `237dc4101ad5` | `521415f2c534` |
| 256 steps, 16 stages | 20.2 mm | `f38ec3e29819` | `aec34bbd4934` | `9c3c5d86784b` |

## What total independence means here

1. No frozen weights, states or scores are used. The package's parity gates read the frozen code; a run never does.
2. No run is the pilot's: every run has seed 1, so the 64-step pooled run is a fresh start of the default, not run A.
3. Every run is its own job with its own key and snapshots.
4. The runs are in the summed form, the form the package was built for. They are the paper's cells and losses redone, not a replication of the paper's numbers, which were made with a predicted end state written as a correction to a straight line. That form is not in the package (George, 2026-09-28).

## The criteria, written before the runs were submitted

All errors are on the 1,452 test tracks, against the sixth-order reference, from the standard report at the last round.

| # | Criterion | Number it is read against |
|---|---|---|
| 1 | Each pooled run beats the paper's pooled run of the same cell at the endpoint radial median | 145.7, 122.2 and 147.7 micrometres at 64, 128 and 256 steps |
| 2 | Each cost-weighted run with the 10 to 50 GeV window beats the paper's cost-weighted run of the same cell at the endpoint radial median | 88.8, 104.6 and 92.2 micrometres |
| 3 | The 64-step pooled run at seed 1 reproduces the pilot's run A: its endpoint radial median is within a factor of two of 8.97 micrometres | 8.97 micrometres, seed 0 |
| 4 | Convergence is read from the validation error; a run that reaches its cap without a plateau is reported as such | |

Read beside them, not criteria: the ceilings of the exact scheme, 0.147, 0.0115 and 0.00105 micrometres at the three cells; the paper's in-window and below-3-GeV medians, to see whether the cost-weighted trade of the paper (the window better, the soft tracks worse) survives the change of form; the 3 to 8 GeV window runs have no frozen counterpart in the paper and are read against the pooled run of the same cell.

What the outcomes mean:

| Outcome | What follows |
|---|---|
| 1 to 3 hold | the summed form holds across the paper's cells and losses; the first real experiment is designed on the default |
| 3 fails | the default does not reproduce; the seed matters, and that is reported to George before anything else |
| 1 holds, 2 fails | the cost-weighted loss does not carry over to the summed form; the anatomy of output 6 is read first |
| a cell fails where 64 steps holds | the summed form has a step-length dependence; reported with the numbers |

## What is not known before the runs

- The time of a round at 128 and 256 steps on the farm. From the pilot's 92 s a restart at 64 steps and the frozen runs' ratios, about 140 and 260 s a restart, so about 60 and 110 minutes a round.
- Whether 80 rounds are enough for a plateau at the larger cells.

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
| Commit the runs train at | `5eb3b068`, from the copy `/data/bfys/gscriven/rkpinn_code/5eb3b068` |
| Smoke jobs, two configurations made small (width 16, 640 states, 2 restarts of 5 iterations, 2 rounds), on a scratch store | the 64-step cost-weighted 10 to 50 GeV run and the 256-step 16-stage pooled run; both passed on worker nodes, exit 0, clusters 5945464 and 5945465; the same two run on the login node gave the same losses to every printed digit |
| Field integral of the cost-weighted loss, 1,024 samples | 3752.054296250222 tesla mm, equal to the frozen study's value to the last digit |
| Reference bend of the cost-weighted smoke run | 891.5 mm on 640 states of seed 1; the frozen study measured 861.6 mm on 32,000 states of seed 0; the real runs measure it on 32,000 states of seed 1 |
| Submitted | 2026-10-12 20:25, clusters 5945466 to 5945474, by the keeper, in the order of `farm/runs.txt` |
| Logs of the farm and of the keeper | `/data/bfys/gscriven/rkpinn_farm_logs/redo_the_papers_nine_runs_in_the_summed_form` |
| The farm at submission | about 3,200 jobs of others idle; the nine jobs started idle |

## Findings

The numbers against the criteria are in `analysis/results.md`. They are discussed with George before anything is written up.
