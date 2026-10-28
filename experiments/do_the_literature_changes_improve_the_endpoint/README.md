# Do the literature's changes to the training improve the endpoint

**Last updated:** 2026-10-28

## What this directory is

Short local tests of four changes to the training that the literature suggests, each on the 64-step, 2-stage pooled network in both output forms: direct states, and the straight line plus a scaled correction. Asked by George on 2026-10-23 ("please try the suggested improvements out on the networks, both the direct sums and the deviation from a straight line").

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file: the question, the changes, the tests, the results | built |
| `configurations/` | one configuration file per run: two forms times three training changes | built |
| `analysis/` | the results tables, the figures and `results.md`, the answer to the question; read from the store; the arms stopped after 3 to 6 rounds | built |

## The question

Does any of the four changes lower the endpoint error on the test split (`endpoint_radial_median_micrometres` of the standard report) against the run without it, for either output form?

## The changes, each a setting with the old behaviour unchanged

| # | Change | Where in the package | How it is chosen |
|---|---|---|---|
| 1 | one fixed set of training states, reused every round, in place of a fresh draw of 32,000 | `training/training_protocols.py`, protocol `rounds_on_one_fixed_draw` | `training.protocol.name`; `states_after_round_one` says whether later rounds take the network's own predictions on the same tracks and planes, or the reference states of round 1 again |
| 2 | the round of the lowest validation error, in place of the last round | `run_record/report_of_a_run.py` | `--round best` |
| 3 | a Levenberg-Marquardt (Gauss-Newton) refit of the last linear layer on the stage residual, applied to a trained snapshot | `training/last_layer_refit.py` | a separate command; the report scores it with `--refit <name>` |
| 4 | the self-scaled Broyden quasi-Newton method (arXiv:2405.04230) with the dense inverse Hessian, in place of L-BFGS | `training/optimiser.py`, optimiser `self_scaled_broyden_restarts` | `training.optimiser.name` |

`rounds_on_own_predictions`, `lbfgs_restarts` and `--round last` are unchanged; every run made before keeps its key. Gates: `RK_Pinn_module/tests/test_literature_changes.py`.

## The baselines

| Form | Run | Rounds | Test endpoint, last round |
|---|---|---|---|
| direct states | `49e5e2450eaa` | 59, plateaued | 5.71 µm |
| straight line, one global scale | `9cef9c160b7d` | 22, plateaued | 2.04 µm |

Every arm copies its baseline's configuration and changes one block. Seed 1, as the baselines. Round 1 of the fixed-draw arms is identical to round 1 of the baseline.

## Results

The tests stopped after 3 to 6 rounds. What the store shows, each arm against its baseline at the same round, is in `analysis/results.md`. It is discussed with George before anything is written up.
