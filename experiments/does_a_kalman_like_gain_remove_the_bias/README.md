# Does a Kalman-like gain remove the bias

**Last updated:** 2026-11-03 · **State:** Parts A and B run; Part C (a learned gain) built, smoke-run, eight farm runs submitted 2026-11-03

## What this directory is

Parts A, B and C of the plan "Kalman filter aided network". Parts A and B are offline: no training, no farm job, nothing written to the store. Part C trains a learned gain; its components are in the package (2026-11-03). It reads three trained runs, the track set `12a8d35c3165`, the field map `v8r1_up` and the cache of the per-step bias study (`../what_is_the_per_step_bias/`), which it does not change.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file | built |
| `analysis/` | the scripts, the tables (`results/`), the figures (`figures/`) and `results.md`, the answer; Part C's smoke run (`part_c_smoke.sh`, `part_c_smoke.md`) | built |
| `configurations/` | Part C: one configuration per farm run, eight in all | built |
| `farm/` | Part C: what puts the runs on the farm and keeps them going; not submitted | built |
| `WRITEUP_DRAFT.md` | the draft of the write-up: theory, output forms, methodology, results of Parts A and B, places for Part C | built |

## The question

The summed form already computes the rates F at the network's stages. The stage residual r is therefore free. Does a linearised, Kalman-like gain K, built from r and the Jacobian of the field at the stages, remove the per-step bias of the network without new field lookups (Part B)? Does a model-error covariance Q for the network step make a toy track fit honest (Part A)?

## The runs

| Form | Key | Round | Test endpoint (um) |
|---|---|---|---|
| straight line, per-track scale | `25dfa17ef168` | 44 | 1.83 |
| straight line, one global scale | `9cef9c160b7d` | 22 | 2.04 |
| direct states | `49e5e2450eaa` | 59 | 5.71 |

All: 64 steps of 80.903 mm, 2 stages (Gauss–Legendre), pooled loss, end state summed, seed 1.

## The answer

See `analysis/results.md`. In short: with the gain every run gives the exact scheme (0.147 um) at 128 field lookups + 128 Jacobian evaluations per track. A Q in the toy fit does not change the momentum bias (≤ 1.1% of the resolution) and only narrows the wide pulls of direct states below 3 GeV.

## Part C: a learned gain (2026-11-03)

The question: can a small network learn the gain, so that inference needs no Jacobian and no solve?

| Variant | What | Inference per step | Loss |
|---|---|---|---|
| C1a | the frozen stage network; a gain network gives a 4 × 8 matrix W that multiplies r; end = summed − correction | 2 field lookups, 1 stage network call, 1 gain call | `distilled_analytic_gain`: the correction against K r from Jacobians at the stages (training only); no reference label |
| C1b | not built: see `WRITEUP_DRAFT.md`, section 4.3 | | |
| C2 | the frozen stage network is called; a corrector reads r (6 + 8 inputs) and gives an 8 × 8 matrix that multiplies r to correct the stages; end summed from the corrected stages | 4 field lookups, 1 stage network call, 1 corrector call | `pooled_on_both_calls` |

The components: `rkpinn/networks/stage_network_with_a_learned_gain.py`, `error_conditioned_stage_network.py`, `stage_network_of_a_run.py`, `rkpinn/losses/distilled_analytic_gain.py`, `pooled_on_both_calls.py`, `rkpinn/integrators/linearised_gain.py`. Gates: `RK_Pinn_module/tests/test_learned_gain_and_error_conditioned_network.py`.

Criteria, written before the farm runs:

| # | Criterion |
|---|---|
| 1 | C1a passes on a frozen network when its test endpoint radial median is within 2× of the analytic gain (0.147 um), i.e. below 0.30 um. |
| 2 | C2 passes on a frozen network when its test endpoint radial median is below that of the frozen network alone by more than 2×. |
| 3 | Either passes as an extrapolator when it beats classic RK4 at 64 steps (1.08 um) at no more field lookups than RK4 (256 a track). |
| 4 | Convergence is read from the validation error. A run at its cap without a plateau is reported as such. |

Smoke run: `analysis/part_c_smoke.md`. Farm runs: `configurations/README.md`, `farm/README.md`.
