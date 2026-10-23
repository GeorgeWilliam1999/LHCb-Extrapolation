# Does the error fall with the step length

**Last updated:** 2026-10-23

## What this directory is

The fourth experiment of the package: the per-track straight-line run of 64 steps and 2 stages (`25dfa17ef168`), trained again at other numbers of steps and of stages, so that one figure can show the step length against the single-step error and against the error at the end of the region. Planned with George in the thread of 2026-10-23 (project library, `step_size_figure_plan.md`); he agreed the plan and chose the per-track scale for every run.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file: the question, the runs, the criteria, how to run | built |
| `configurations/` | one configuration file per run, eight in all | built |
| `farm/` | what puts the runs on the farm and keeps them going | built |
| `analysis/` | the reference curves and the figure: reads the store, writes CSV and PNG | built |

## The question

When one network of the same form, loss and training is trained at each step length, does its error fall as the step length falls? And at one step length, does the number of stages matter?

## Why it is asked

A trained network works at its own step length only: the step length is a fixed constant of the network (`RK_Pinn_module/src/rkpinn/networks/stage_network.py`), not an input. The runs of the store change the steps and the stages together (64/2, 128/8, 256/16), so they cannot separate the two.

The exact scheme, the Gauss–Legendre method of the same steps and stages solved exactly, is the best a perfect network can do. At 64 steps or more it is 10 to 1,000 times below every network, so the error of a network is set by its training. The figure shows how far each network is from its own exact scheme.

## The runs

All copy `../does_the_straight_line_form_restore_the_old_accuracy/configurations/64_steps_2_stages_pooled_per_track_scale.yaml` and change only the number of steps and of stages. Pooled loss over the stages, end state summed from the stages, per-track scale, seed 1, cap 80 rounds.

| Series | Steps | Stages | Step length (mm) | Configuration |
|---|---|---|---|---|
| step length | 8 | 2 | 647.2 | `8_steps_2_stages_pooled_per_track_scale.yaml` |
| step length | 16 | 2 | 323.6 | `16_steps_2_stages_pooled_per_track_scale.yaml` |
| step length | 32 | 2 | 161.8 | `32_steps_2_stages_pooled_per_track_scale.yaml` |
| step length | 64 | 2 | 80.9 | the existing run `25dfa17ef168`, not trained again |
| step length | 128 | 2 | 40.5 | `128_steps_2_stages_pooled_per_track_scale.yaml` |
| step length | 256 | 2 | 20.2 | `256_steps_2_stages_pooled_per_track_scale.yaml` |
| stages | 64 | 4 | 80.9 | `64_steps_4_stages_pooled_per_track_scale.yaml` |
| stages | 64 | 8 | 80.9 | `64_steps_8_stages_pooled_per_track_scale.yaml` |
| stages | 64 | 16 | 80.9 | `64_steps_16_stages_pooled_per_track_scale.yaml` |

The keys are in `farm/runs.txt` order and are printed by `rkpinn.run_record.configuration`; they are recorded below when the runs are submitted.

What is read beside them, with no new training: the exact scheme at every steps and stages; classic fourth-order Runge–Kutta at every step length; the three direct-states cells of the nine-runs experiment; the frozen weights with their stages summed.

## The criteria, written before the runs were submitted

All errors are radial, sqrt(dx² + dy²), on the 1,452 test tracks, against the sixth-order reference at 0.1 mm, from the standard report at the last round.

| # | Criterion |
|---|---|
| 1 | The figure has two panels against the step length: `single_step_radial_median_micrometres` and `endpoint_radial_median_micrometres`, log-log, with the 16th to 84th percentile as the bar |
| 2 | Each network point is read only when its run has plateaued on the validation error; a run at its cap without a plateau is drawn hollow and named as such |
| 3 | The step-length series answers the question if the endpoint error falls monotonically from 8 to 256 steps; if not, the figure says where it does not |
| 4 | The stages series answers whether stages matter if the 64-step points at 2, 4, 8 and 16 stages differ by more than their spread between neighbouring rounds |

## What is not known before the runs

- The time of a round. A restart trains on the same number of states at every step length, so it should cost about the same; the validation chain costs in proportion to the steps. At 64 steps and 2 stages a round is about 40 minutes.
- Why the exact scheme does not converge cleanly at 2 stages (128 steps worse than 64). The field map is interpolated between grid points and is not smooth; this is a guess, not checked.

## To run

See `farm/README.md`. The figure, from this folder: see `analysis/README.md`.

## What was measured when the runs were submitted

| | |
|---|---|
| Commit the runs train at | `73f900ac`, from the copy `/data/bfys/gscriven/rkpinn_code/73f900ac` |
| The configuration of `25dfa17ef168` at that commit | gives the key `25dfa17ef168`, so the schema did not change since `bc467594` |
| Submitted | 2026-10-23 14:58 to 15:00, by the keeper, on George's word in the thread |
| Logs of the farm and of the keeper | `/data/bfys/gscriven/rkpinn_farm_logs/does_the_error_fall_with_the_step_length` |

Checks on the login node before submission, on the real configurations and the real store, nothing written. Ten iterations of one restart on one thread of a loaded login node (about 65 processes on 28 cores); a restart has 200.

| Steps / stages | Key | Cluster | Loss at the exact stages | Loss before training | Ten iterations (s) | Validation chain (s) |
|---|---|---|---|---|---|---|
| 8 / 2 | `a66c06472d3d` | 5996427 | 9.9e-34 | 3.59e-3 | 21.3 | 0.2 |
| 16 / 2 | `6c4a3b022a2f` | 5996428 | 5.7e-33 | 8.21e-4 | 22.8 | 2.7 |
| 32 / 2 | `29cbba959300` | 5996429 | 2.8e-35 | 1.98e-4 | 16.9 | 1.1 |
| 128 / 2 | `15242aa17c65` | 5996430 | 0 | 1.22e-5 | 15.9 | 3.7 |
| 256 / 2 | `eeb809e9254d` | 5996431 | 0 | 3.04e-6 | 40.4 | 16.1 |
| 64 / 4 | `4543836de1ce` | 5996432 | 2.5e-34 | 6.79e-5 | 18.4 | 2.4 |
| 64 / 8 | `9aed5ce39ff8` | 5996433 | 1.7e-33 | 6.85e-5 | 17.3 | 1.6 |
| 64 / 16 | `06a45b4dd0d3` | 5996434 | 4.4e-34 | 6.54e-5 | 55.4 | 9.2 |

The 16-stage run is about three times slower a restart than the others; at its cap of 80 rounds it can take several times as long as the 64 / 2 run.

The exact scheme for 4, 8, 16 and 32 steps at 2, 4, 8 and 16 stages was written to the store on 2026-10-23 from the same copy (0 solves not converged).

## Findings

None yet. They are discussed with George before anything is written up.
