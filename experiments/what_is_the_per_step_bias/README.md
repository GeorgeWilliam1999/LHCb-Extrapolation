# What is the per-step bias

**Last updated:** 2026-10-28

## What this directory is

An analysis of three existing runs. It asks what the error of one network step is made of, whether it matters after a track fit, and whether a cheap correction at inference removes it. Parts A, B.1 and C read existing runs only: they train nothing and write nothing to the store. George approved them on 2026-10-28 (artifact "per-step bias study plan"). Part B (items B.3 and B.4, approved the same day at 15:19 UTC) trains four new runs on the farm; see the section "Part B" below.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file: the question, the runs, the items, the criteria | built |
| `analysis/` | the scripts, the tables (`results/`), the figures (`figures/`) and the results note | built |
| `configurations/` | the four configurations of Part B (B.3 and B.4, both forms) | built |
| `farm/` | what puts the Part B runs on the farm and keeps them going | built |

Everything of this experiment is in this folder. Two things are outside it, by the convention of `does_the_error_fall_with_the_step_length`: the cache of regenerable arrays (`/data/bfys/gscriven/rkpinn_analysis_cache/what_is_the_per_step_bias/`, 184 MB: single-step errors, Jacobians, fitted models) and, for Part B, the runs in the store and the farm logs in `/data/bfys/gscriven/rkpinn_farm_logs/what_is_the_per_step_bias/`. The study reads the store, the field map and the tranche-1 runs; it writes into no other experiment's folder. The tranche-1 analysis (`analysis/tranche1_2026-10-28/` at the project root) belongs to tranche 1 and is not part of this experiment; this study only cites its numbers.

## The question

One network step from the reference state has a signed error e. Along a track, e has a lag-1 autocorrelation of 0.85 to 0.90 (tranche-1 analysis, `analysis/tranche1_2026-10-28/` at the project root). So the errors of the steps add up instead of cancelling. Is e a bias that a smooth function of the state can predict and remove? Is it shared between forms, seeds and snapshots, or private to each network? Does it move the momentum of a track fit? Does a cheap correction at inference remove it?

## The runs

All are 64 steps of 80.903 mm, 2 stages, pooled loss over the stages, end state summed from the stages, seed 1, track set `12a8d35c3165`. Each is read at its last round.

| Form | Key | Last round | Test endpoint (um) |
|---|---|---|---|
| straight line, per-track scale | `25dfa17ef168` | 44 | 1.83 |
| straight line, one global scale | `9cef9c160b7d` | 22 | 2.04 |
| direct states | `49e5e2450eaa` | 59 | 5.71 |

The direct-states form and the straight-line form are two equal methods (George, 2026-10-28, reversing the ruling of 2026-10-22). This study changes no package setting.

## Definitions

- **Score.** The radial error sqrt(dx² + dy²) in micrometres on the 1,452 test tracks on the plane z = 7826 mm, against the sixth-order reference at 0.1 mm. The median is quoted.
- **Single-step error e.** The network's one-step state from the reference state on a start plane, minus the reference state on the next plane. Components x, y (um) and tx, ty (urad). All 64 steps between the 65 standard planes (every 4th of the 257 planes of the track set).
- **Momentum bands.** < 3, 3–8, 8–20, 20–50, > 50 GeV.
- **Bend direction.** For a step, the sign of the change of the reference tx over the step. For a whole track, sign(tx at 7826 mm − tx at 2648.2 mm), as in the resolution thread of 2026-10-28.

## The items

### Part A: what e is made of

| Item | What is computed | Pass criterion |
|---|---|---|
| A.1 | a smooth model of e on (x, y, tx, ty, q/p, z), fitted on the training tracks: gradient-boosted trees and a cubic polynomial; R² on the validation tracks per component per run | "removable" if R² ≥ 0.5 |
| A.2 | the binned mean of e against each of the six variables | — |
| A.3 | e per momentum band; the power n of e ∝ (q/p)^n | — |
| A.4 | e by charge, by sign of y, by quadrant, bending toward or away from the beam; the mean signed endpoint error along and across the bend per band | — |
| A.5 | each step's e carried to the last plane with the reference Jacobian (central differences on the sixth-order method); does the sum reproduce the endpoint error; the share of the endpoint error per z bin | — |
| A.6 | the steps that cross a z face of the field map's grid: mean \|e\| crossing and not, per run and for the exact 2-stage Gauss–Legendre scheme | — |
| A.7 | a toy track fit to UT and SciFi hits, with the network and with the reference between 2648.2 and 7826 mm: momentum bias dp/p and pull width per band | "bias matters" if \|bias\| ≥ 10% of the momentum resolution in any band |
| A.8 | the bias map of other seeds (tranche-1 runs of `does_a_better_optimiser_improve_the_endpoint`) | "shared" if correlation > 0.7 |
| A.9 | the correlation of bias maps between forms and between the last 5 snapshots of a run | "shared" if correlation > 0.7 |

### Part B.1: averaging

The mean of the last 5 snapshots' weights (one network call a step) against the mean of their outputs (five network calls a step). Test endpoint.

### Part C: corrections at inference only

New code in `analysis/`, not in the package.

| Item | What is applied at each step |
|---|---|
| C.1 | the end state minus the A.1 model of e (supervised: the model was fitted to reference errors; reported apart) |
| C.2 | 1 and 2 fixed-point sweeps of the 2-stage Gauss–Legendre stage equations from the network's stages; end state summed |
| C.3 | one Newton step of the stage equations, with the gradient of the trilinear map |
| C.4 | a small network trained without reference labels to predict the C.3 change of the end state from the state; applied as C.1 |
| C.5 | moved out of this experiment (George, 2026-10-28): the correction from measured SciFi residuals, and every question about a Kalman-filter fit, belong to a separate thread |

For every variant: the test endpoint radial median, per band; the lag-1 autocorrelation of the corrected single-step errors; the mean signed endpoint error ± standard error; field calls and network calls per track.

| Pass criterion | Value |
|---|---|
| direct states | ≤ 2.85 um |
| straight line | ≤ 0.92 um |
| no momentum band | more than 2× worse than the plain network |
| the speed claim is kept | ≤ 1.08 um at 128 field calls and 64 network calls |
| C.2 and C.3 | ≤ 0.54 um at ≤ 256 field calls |

Comparators: classic fourth-order Runge–Kutta at 64 steps, 1.08 um at 256 field calls; the exact 2-stage Gauss–Legendre scheme at 64 steps, 0.147 um. Each run is one seed, so differences under 2× are not claimed.

## Part B: B.3 and B.4 in both forms

Approved by George on 2026-10-28 (15:19 UTC), after the results of Parts A and C. Four farm runs. Two settings were added to the package as new components, so no earlier run changes its key; their gates are `RK_Pinn_module/tests/test_bias_study_changes.py`.

| Item | Component | What it does |
|---|---|---|
| B.3 | training protocol `rounds_on_own_predictions_with_residual_draws` | round 1 as now; each later round draws half of its 32,000 states uniformly, as now, and half from a pool of 4× as many by probability proportional to the pooled squared stage residual of the network as it is, × 2 for a step that crosses a z face of the field map |
| B.4 | loss `pooled_with_a_bias_term` | pooled loss + λ × mean over bins of the squared mean normalised residual; bins of 8 bands of start planes × 5 momentum bands × charge; λ fixed when the run is made so the term is 10% of the loss of round 1, recorded in the run's constants |

| Run | Form | Item | λ |
|---|---|---|---|
| `5cb2aeb6778b` | direct states | B.3 | — |
| `6aba11cc7475` | direct states | B.4 | 0.432 (P1 0.966, B1 0.248) |
| `8a0f3b8f422f` | straight line, per-track scale | B.3 | — |
| `f0f219df805e` | straight line, per-track scale | B.4 | 0.117 (P1 4.93e-5, B1 4.67e-5) |

Everything else is that of `49e5e2450eaa` and `25dfa17ef168`: 64 steps, 2 stages, pooled terms over the stages, summed end state, seed 1, cap 80 rounds, validation plateau rule.

Each run is scored by the standard report of the package (with output 11, `resolution_per_momentum_band`) and by `analysis/part_b_scores.py` (lag-1 autocorrelation of e, mean signed endpoint error ± SE, the bands, the criteria).

| Pass criterion | Value |
|---|---|
| direct states | ≤ 2.85 um |
| straight line | ≤ 0.92 um |
| no momentum band | more than 2× worse than the baseline run |

Each run is one seed, so a difference under 2× is not claimed.

## Findings

See `analysis/RESULTS.md`. They are discussed with George before anything is written up.
