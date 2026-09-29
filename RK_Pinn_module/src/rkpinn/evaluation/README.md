# Evaluation

**Last updated:** 2026-09-30

## What this directory is

The standard for results. One set of conventions and one report, the same for every run and every comparator.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file | built |
| `__init__.py` | marks the directory as part of the package; it holds no code | built |
| `conventions.py` | positions in micrometres, slopes with no unit, the momentum bands, the statistics, the name of a network; defined here and nowhere else | built |
| `endpoint_error.py` | the error on the last plane per component and per momentum band, the five statistics, the radial error, and the quantiles of the signed error. Outputs 1, 2 and 3 | built |
| `error_along_the_track.py` | the error plane by plane (output 5); the error of one application (output 4); stage errors held at every step and carried to the end of the track (output 6) | built |
| `three_references.py` | the prediction against the reference integrator, against the exact scheme, and against the true state. Output 7 | built |
| `convergence.py` | the validation error per round, whether the stopping rule holds now, and the first round it held. Output 8 | built |
| `standard_report.py` | the ten standard outputs of `PACKAGE_PLAN.md`, section 5, as tables and figures; where the loss puts its weight (output 9); writes them | built |

## Where each file was ported from

| File | Ported from | Gate |
|---|---|---|
| `conventions.py` | `Self_chained_paper/scripts/common.py` | `tests/test_evaluation_reproduces_the_paper.py` |
| `endpoint_error.py` | `Self_chained_paper/scripts/numbers.py` (`chain_block`, `sec_headline`) | the same |
| `error_along_the_track.py`, outputs 4 and 5 | `numbers.py` (`sec_along_z`) | the same |
| `error_along_the_track.py`, output 6 | new | `tests/test_stage_errors_held_and_carried.py` |
| `three_references.py` | `numbers.py` (`sec_against_true`); the comparison with the exact scheme is added | `tests/test_evaluation_reproduces_the_paper.py` |
| `convergence.py` | `common.py` (`first_plateau_round`, `validation_headline`) | the same |
| `standard_report.py`, output 9 | `numbers.py` (`sec_preflight`) | the same |

## What was measured when it was built

Run on the stored states of the six old runs the second mini-paper compares, the evaluation gives the paper's numbers.

| Output | Against the paper | Agreement |
|---|---|---|
| 1, 2, 3 | endpoint error per component and band, five statistics; radial error per band | every digit, all six runs |
| 4 | single-step error per step | every digit, the two runs of 64 steps |
| 5 | error along the track, every plane | every digit, all six runs |
| 7 | the three references, per band | every digit, all six runs |
| 8 | the headline of the validation error, the verdict, the first round the rule held | every digit, all six runs |
| 9 | the shares of the loss, pooled and cost-weighted | to a relative 1e-12 |
| 6, 10 | no number in the paper | output 6 has its own gate |

Found while building output 6: the error of x and y at the end of a step is first order in the stage errors exactly, because the rates of x and y are the slopes, which are linear in the state. What is beyond first order is in the slopes only, and it falls with the square of the stage error.

## The contract

The evaluation reads states and targets. It never imports a network. Figures and tables name a network by its number of steps, number of stages and step length, and say whether an error is an endpoint error or a single-step error.

An output whose input is not given is left out of a report, and the summary of the report says so.

The command that makes the report of a run is `../run_record/report_of_a_run.py`. It is there, not here, because it builds the network of the run.

The validation error of the training is in `../training/validation_error.py`. It is one of the magnitudes of `conventions.py`, the larger of the errors in x and y, computed the same way.

## How to add to it

1. An analysis used by one experiment stays in that experiment's folder.
2. An analysis wanted for every run is added here as one file, and its output is added to the standard report.
3. Add its row to the Contents table above and change the date.
4. Update the card of the standard evaluation and its entry in the Notion master index.
