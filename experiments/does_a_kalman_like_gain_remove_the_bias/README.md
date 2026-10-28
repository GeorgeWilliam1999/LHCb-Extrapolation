# Does a Kalman-like gain remove the bias

**Last updated:** 2026-10-28 · **State:** Parts A and B run; for discussion with George

## What this directory is

Parts A and B of the plan "Kalman filter aided network". Offline: no training, no farm job, nothing written to the store. It reads three trained runs, the track set `12a8d35c3165`, the field map `v8r1_up` and the cache of the per-step bias study (`../what_is_the_per_step_bias/`), which it does not change.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file | built |
| `analysis/` | the scripts, the tables (`results/`), the figures (`figures/`) and `results.md`, the answer | built |

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
