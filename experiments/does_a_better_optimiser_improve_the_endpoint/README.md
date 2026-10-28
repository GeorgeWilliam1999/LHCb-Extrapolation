# Does a better optimiser improve the endpoint

**Last updated:** 2026-10-28

## What this directory is

The 64-step, 2-stage pooled network trained with the self-scaled Broyden optimiser in place of L-BFGS, three seeds, in both output forms, with two more L-BFGS seeds in each form for the seed spread. Tranche 1 of the improvement plan of 2026-10-28 (Claude Docs, "RK-PINN: proposed changes to the loss, training and architecture", items 1c and 1d), approved by George on 2026-10-28 in both output forms, ten farm runs.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file: the question, the arms, the criterion, how to run | built |
| `configurations/` | one configuration file per run, ten in all | built |
| `farm/` | what puts the runs on the farm and keeps them going | built |

## The question

Does the self-scaled Broyden optimiser (arXiv:2405.04230, with the dense inverse Hessian), run to the plateau rule, give a lower endpoint error on the test split than L-BFGS, by more than the spread between seeds?

## The arms

Every run is 64 steps of 80.903 mm, 2 stages, pooled loss over the stages, end state summed from the stages, the standard track set `12a8d35c3165`, 32,000 states a round, 25 restarts a round of 200 iterations, the validation plateau rule, a cap of 80 rounds. Each configuration copies the seed-1 L-BFGS run of its form and changes only the seed, or the seed and the optimiser block.

| Output form | Optimiser | Seed 1 | Seed 2 | Seed 3 |
|---|---|---|---|---|
| direct states | L-BFGS | `49e5e2450eaa` (exists, not rerun) | `b6bd9570e6f2` | `5cae49832580` |
| straight line, per-track scale | L-BFGS | `25dfa17ef168` (exists, not rerun) | `dedaab236b3d` | `b68802fcce0e` |
| direct states | self-scaled Broyden | `f4c86e05eafe` (resumed from round 3) | `c3a1aa54c6fe` | `c19552a6831c` |
| straight line, per-track scale | self-scaled Broyden | `166cfc8cdfd6` | `3400f9c572ef` | `c8b210923994` |

The self-scaled Broyden block is that of the trials of 2026-10-23 (`../do_the_literature_changes_improve_the_endpoint/`): 200 iterations a restart, gradient tolerance 1e-13, change tolerance 1e-16, scale renewed when the loss falls to 0.1 of its value, the matrix products on 4 threads.

Two facts about the seed-1 Broyden runs:

- **Direct states.** The configuration is the 23 Oct trial's to the byte, so its key is the trial's, `f4c86e05eafe`. The trial stopped after 3 rounds on the login node; the farm resumes it from round 3 to the plateau rule. Its first 3 rounds were trained at commit `3ec05291` on `stbc-i2`.
- **Straight line.** The 23 Oct trial `9a3df286d5fb` used one global scale. The plan names the per-track scale for the straight line, and the L-BFGS baselines `25dfa17ef168` use it, so the Broyden arm uses it too. The seed-1 run is new, `166cfc8cdfd6`; `9a3df286d5fb` is left as it is.

## The criterion, written before the runs were submitted

All errors are `endpoint_radial_median_micrometres` of the standard report on the 1,452 test tracks, at the round where the run plateaued or reached its cap.

| # | Criterion |
|---|---|
| 1 | In a form, the self-scaled Broyden arm passes when the median over its 3 seeds is below the median over the 3 L-BFGS seeds by more than the seed spread. The seed spread is the larger of the two arms' ranges (largest minus smallest of the 3 seeds). |
| 2 | Targets, read beside criterion 1: straight line below 1.08 µm, the classic fourth-order Runge-Kutta at 64 steps; direct states below 2.18 µm, the frozen 64-step pooled weights with the stages summed. |
| 3 | Convergence is read from the validation error. A run that reaches its cap without a plateau is reported as such. |

What the outcomes mean:

| Outcome | What follows |
|---|---|
| Criterion 1 holds in a form | the self-scaled Broyden optimiser is the default for that form in tranche 2 |
| Criterion 1 fails in both forms | the optimiser is not what limits the endpoint; tranche 2 starts from the L-BFGS runs |
| Criterion 1 holds and the target of criterion 2 is met | the network reaches the reference scheme at the endpoint |

## What is not known before the runs

- The time of a round on the farm. On the login node, with 4 threads for the matrix, a Broyden round took 1.9 to 3.2 hours (restarts of about 250 s against about 100 s for L-BFGS). At the plateaus of the L-BFGS runs (22 to 59 rounds) that is about 2 to 8 days; the cap of 80 rounds would be up to about 10 days.
- The memory of a Broyden job. The dense inverse Hessian of about 18,000 parameters is about 2.7 GB in double precision. The jobs ask for 16 GB; the keeper doubles the memory of a job held for its memory.

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
| Commit the runs train at | `53d158fb`, from the copy `/data/bfys/gscriven/rkpinn_code/53d158fb`; the package is unchanged since `3ec05291` |
| Gates | on 2026-10-28, on `stbc-i1`: 488 passed, 2 failed. Both failures are in what a report of a network without stages, or of an old run, lists as left out: it now also lists output 11, the resolution per momentum band, and the two gates still expect the older list. They touch no component these runs use. The files of those gates and of the report were last changed at `e42b73d2`. On `stbc-i2` the farm-stop gate also failed, because its 300 s wait ran out before round 1 on the slow node; on `stbc-i1` it passed. Logs: `gates_2026-10-28*.log` in the logs folder |
| Submitted | 2026-10-28 11:57 by the keeper, started on `stbc-i1`; clusters 6015202 to 6015211 in the order of `farm/runs.txt` |
| First restarts on the farm | L-BFGS 73 to 85 s; the resumed self-scaled Broyden run `f4c86e05eafe` 224 s (round 4, restart 1) |
| Logs of the farm and of the keeper | `/data/bfys/gscriven/rkpinn_farm_logs/does_a_better_optimiser_improve_the_endpoint` |

| Cluster | Configuration | Run |
|---|---|---|
| 6015202 | `direct_states__lbfgs__seed_2.yaml` | `b6bd9570e6f2` |
| 6015203 | `direct_states__lbfgs__seed_3.yaml` | `5cae49832580` |
| 6015204 | `straight_line_per_track_scale__lbfgs__seed_2.yaml` | `dedaab236b3d` |
| 6015205 | `straight_line_per_track_scale__lbfgs__seed_3.yaml` | `b68802fcce0e` |
| 6015206 | `direct_states__self_scaled_broyden__seed_1.yaml` | `f4c86e05eafe` |
| 6015207 | `direct_states__self_scaled_broyden__seed_2.yaml` | `c3a1aa54c6fe` |
| 6015208 | `direct_states__self_scaled_broyden__seed_3.yaml` | `c19552a6831c` |
| 6015209 | `straight_line_per_track_scale__self_scaled_broyden__seed_1.yaml` | `166cfc8cdfd6` |
| 6015210 | `straight_line_per_track_scale__self_scaled_broyden__seed_2.yaml` | `3400f9c572ef` |
| 6015211 | `straight_line_per_track_scale__self_scaled_broyden__seed_3.yaml` | `c8b210923994` |

## Findings

None yet. They are discussed with George before anything is written up.
