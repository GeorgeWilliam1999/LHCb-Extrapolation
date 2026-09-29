# Does a network whose end state is summed from its stages train?

**Last updated:** 2026-09-30 · **State:** planned · **Agreed with George:** 2026-09-30

## What this directory is

The pilot of the package `rkpinn`: the first networks it trains. It is phase 6 of `../../RK_Pinn_module/PACKAGE_PLAN.md`, section 9, and the measurement of the risk in its section 11.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file: the question, the runs, the criteria, the findings | built |
| `configurations/` | one configuration file per run | built |
| `farm/` | what puts the runs on the farm and keeps them going | built |
| `analysis.ipynb` | loads the standard reports of the runs; computes nothing | planned |
| `extra_analysis/` | anything beyond the standard report | planned |

## The question

Can one network of 64 steps and 2 stages be trained when it emits the stage states themselves and its end state is summed from them, and how does it compare with the same network when the end state is predicted?

## Why it is asked

Every frozen network wrote its outputs as a correction to a straight line and predicted its end state. The package does neither by default: the network emits the states, and the end of a step is collected and summed from the stages (George, 2026-09-28). At 64 steps a state changes by a median of 2.6 mm in x over a stage, against a spread of 446 mm. A micrometre is two parts in a million of what the network emits. Whether an optimiser reaches that is not known. It is measured here before any farm time is spent on an experiment.

## The runs

Both: 64 steps of 80.9 mm, 2 stages, the pooled loss, a body of 2 layers of 128 units with tanh, 32,000 states a round, 25 restarts a round of 200 iterations, seed 0, the tracks `12a8d35c3165`.

| Run | End state | Terms of the loss | Outputs of the network | Configuration | Key |
|---|---|---|---|---|---|
| A | summed from the stages | the stages | 8 | `64_steps_2_stages_summed_end_state.yaml` | `bddea42ea1ef` |
| B | predicted | the stages and the end state | 12 | `64_steps_2_stages_predicted_end_state.yaml` | `4cc888dcc710` |

B is the form of the frozen runs without the straight line. It is there so that a failure of A can be told from a fault of the new trainer: if B trains and A does not, the summed end state is the cause.

The two runs differ in two settings, which cannot be changed one at a time; see `configurations/README.md`.

A run ends when its validation error has plateaued or at 60 rounds. The rule: the median of the last 10 rounds no more than 5 % below the median of the 10 before, at each of the last 3 rounds, on the median over the validation tracks of the larger of the endpoint errors in x and y.

## The criteria, written before the runs were submitted

All errors are on the 1,452 test tracks, against the sixth-order reference.

| # | Criterion | Number it is read against |
|---|---|---|
| 1 | The pilot criterion of the plan: the single-step radial error of run A, at the median, is within a factor of ten of that of the frozen network of the same size | frozen, pooled loss, 64 steps 2 stages: 0.333 µm. So run A passes below 3.33 µm |
| 2 | The same for run B | the same |
| 3 | Convergence is read from the validation error, and a run that reached its cap without a plateau is reported as such | |

Read beside them, not criteria:

| Number | Value |
|---|---|
| Endpoint radial error of the frozen network, pooled loss, at the median | 145.7 µm |
| Endpoint radial error of the exact scheme of 64 steps and 2 stages: the ceiling | 0.147 µm |

What the outcomes mean:

| Outcome | What follows |
|---|---|
| A and B pass | the summed form trains; the first experiment can be designed on it |
| B passes, A fails | the summed end state is the cause. Reported to George with the numbers. No workaround is added without his ruling |
| A passes, B fails | unexpected; the trainer and the predicted form are looked at before anything else |
| Both fail | the direct output is the likely cause, since both emit the states directly. Reported to George with the numbers |

## What is not known before the runs

- How long a round takes on the farm. One timing on the login node gives 0.48 s for the loss and its gradient on 32,000 states, so about 2 minutes a restart and about 50 minutes a round. That is an estimate from one timing.
- Whether 60 rounds are enough for a plateau. The frozen runs of this size took 40 to 50 rounds.

## To run

See `farm/README.md`. The standard report of a run, when it has ended, from `RK_Pinn_module`:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src /data/bfys/gscriven/conda/envs/TE/bin/python \
    -m rkpinn.run_record.report_of_a_run --store /data/bfys/gscriven/rkpinn_store \
    --run <key> --round last --split test --tracks-for-the-stage-errors 200 \
    --step-length-of-the-carrying-mm 1.0 --fraction-for-the-derivative 1e-3 \
    --states-for-the-loss-shares 8000 --bins-of-the-histograms 60
```

## Findings

None yet. They are discussed with George before anything is written up.
