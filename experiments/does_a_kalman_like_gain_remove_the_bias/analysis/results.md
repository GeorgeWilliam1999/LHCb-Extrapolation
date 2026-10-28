# Results: does a Kalman-like gain remove the bias

**Date:** 2026-10-28. **State:** for discussion with George. Nothing is written up.

## Scope

- Three runs, each at its last round: straight line per-track scale (`25dfa17ef168`, round 44), straight line one global scale (`9cef9c160b7d`, round 22), direct states (`49e5e2450eaa`, round 59). All are 64 steps of 80.903 mm, 2 stages, pooled loss, end state summed from the stages, seed 1.
- Test set: 1,452 tracks. Score: radial median at z = 7826 mm against the sixth-order reference at 0.1 mm. Bands: < 3, 3–8, 8–20, 20–50, > 50 GeV.
- No training. No farm job. Nothing in the store changed. The scripts ran on `stbc-i3` with 4 threads.
- Each run is one seed. A difference under 2× is not a claim.

## Summary

1. The gain removes the network error completely. With the correction, all three runs give 0.147 um at the endpoint. This is the exact 2-stage scheme (0.147 um). Every band equals the exact scheme.
2. The gain needs no new field lookup. It uses the Jacobian of the field at the same stage states. Cost per step: 2 field lookups, 2 Jacobian evaluations, 1 network call, one 8 × 8 solve. Per track: 128 lookups, 128 Jacobians, 64 network calls. The sweep and the Newton step of the bias study (C.2, C.3) needed 256 lookups for the same result.
3. After the gain, the single-step error equals that of the exact scheme: radial median 0.0007 um (before: 0.07 to 0.49 um). The lag-1 autocorrelation falls from 0.78–0.87 (x) and 0.87–0.90 (y) to −0.22 (x) and −0.18 (y), the values of the exact scheme.
4. Control at equal lookups (128 + 128 Jacobians, no network): the gain from straight-line stages gives 2.36 um (54 um below 3 GeV). So at this cost the network is worth 16× at the endpoint.
5. Control with one more lookup (192 + 128 Jacobians, no network): the gain from explicit Euler stages gives 0.146 um, the exact scheme. So the network saves 1 field lookup per step against a network-free start.
6. For the three runs and the Euler start, B2 (corrections carried to the end with the reference Jacobians) gives the same endpoint as B1 within 0.001 um (straight-line start: 2.40 against 2.36 um). So the correction is linear; it does not need to be fed into the next step. A plain sum without the carry is worse (0.15 to 0.20 um; up to 3 um below 3 GeV), because a slope correction moves the position downstream.
7. Part A: a model-error covariance Q does not change the momentum bias. The bias is at most 1.1% of the hits-only resolution with and without Q. Q only changes the pulls. Only direct states below 3 GeV has wide pulls (tx 1.37, q/p 1.12). The per-step Q narrows them to 1.26 and 1.10. A Q measured at the endpoint narrows them to 1.03 and 1.07.
8. The per-step Q is too small by 1.1× to 2.6× in sigma at the endpoint, because the step errors are correlated along the track and a per-step Q treats them as independent.

## Method

For one step from S0 on z0 of length dz, with stages Y_k and rates F_k = f(Y_k, z0 + c_k dz) that the summed end state already uses:

- residual r_k = Y_k − S0 − dz Σ_j a_kj F_j (the stage residual of `rkpinn/losses/stage_residual.py`);
- Jacobian J_k = ∂f/∂Y at Y_k, 4 × 4, q/p fixed, by torch autograd through the differentiable field map;
- gain K = dz (bᵀ ⊗ J)(I − dz A ⊗ J)⁻¹, an 8 × 8 solve per step and track;
- corrected end state = S0 + dz Σ_k b_k F_k − K r.

K r is the change that one Newton step of the stage equations makes to the end state, with f at the new stages replaced by its linearisation. So it needs no new lookup. The summed end state of the script agrees with the network's own end state to 5e-13 mm (`logs/part_b_check.txt`).

Variants:

- **B1.** Correct the end state of each step. Chain the corrected state into the next step.
- **B2.** Chain the uncorrected states. Compute the correction of each step along that chain. Carry each correction to 7826 mm with the reference Jacobians of the steps (A.5 of the bias study) and add the sum to the plain endpoint. A diagnosis, not a method. The row "summed, not carried" adds the corrections without the carry.
- **B3.** Not run. A second update needs r at the corrected stages, so f at the corrected stages: 2 more lookups per step. The linear model itself predicts r = 0 there. The brief says to skip it in that case.
- **Controls.** The same K r update from stages with no network. Straight line: Y_k = S0 + c_k dz (tx, ty, 0, 0); no lookup to build the stages. Explicit Euler: Y_k = S0 + c_k dz f(S0); 1 lookup to build the stages.

Counts are per step. A field lookup is f at one state. A Jacobian evaluation is ∂f/∂Y at one state; it reuses the lookup of f at that state (the gradient of the trilinear map comes from the same 8 cell corners). Multiply by 64 for the counts per track.

## Part B: endpoint (um)

Test endpoint radial median, per band, and mean signed endpoint error ± standard error.

| Stages from | Correction | Field / step | Jacobian / step | Network / step | Endpoint | < 3 | 3–8 | 8–20 | 20–50 | > 50 | Mean x | Mean y |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| straight line per-track | none | 2 | 0 | 1 | 1.83 | 4.94 | 2.10 | 1.44 | 1.21 | 1.89 | -0.03 ± 0.25 | +0.77 ± 0.46 |
|  | B1 gain, chained | 2 | 2 | 1 | 0.147 | 0.66 | 0.249 | 0.0852 | 0.0353 | 0.0115 | +0.01 ± 0.12 | +0.08 ± 0.15 |
|  | B2 carried to the end | 2 | 2 | 1 | 0.147 | 0.66 | 0.249 | 0.0852 | 0.0353 | 0.0115 | +0.01 ± 0.12 | +0.08 ± 0.15 |
|  | B2 summed, not carried | 2 | 2 | 1 | 0.149 | 0.856 | 0.274 | 0.0883 | 0.0369 | 0.0117 | +0.08 ± 0.16 | +0.04 ± 0.15 |
| straight line global | none | 2 | 0 | 1 | 2.04 | 7.42 | 3.21 | 1.31 | 1.30 | 0.654 | +0.02 ± 0.52 | +0.76 ± 0.32 |
|  | B1 gain, chained | 2 | 2 | 1 | 0.147 | 0.66 | 0.249 | 0.0852 | 0.0353 | 0.0115 | +0.01 ± 0.12 | +0.08 ± 0.15 |
|  | B2 carried to the end | 2 | 2 | 1 | 0.147 | 0.666 | 0.249 | 0.0852 | 0.0353 | 0.0115 | +0.01 ± 0.12 | +0.08 ± 0.15 |
|  | B2 summed, not carried | 2 | 2 | 1 | 0.16 | 1.19 | 0.3 | 0.0887 | 0.0336 | 0.0115 | -0.01 ± 0.15 | +0.25 ± 0.19 |
| direct states | none | 2 | 0 | 1 | 5.71 | 21.7 | 8.74 | 3.54 | 2.96 | 3.24 | +1.40 ± 1.46 | -0.39 ± 0.65 |
|  | B1 gain, chained | 2 | 2 | 1 | 0.147 | 0.666 | 0.249 | 0.0852 | 0.0353 | 0.0115 | +0.01 ± 0.12 | +0.08 ± 0.15 |
|  | B2 carried to the end | 2 | 2 | 1 | 0.147 | 0.666 | 0.248 | 0.0852 | 0.0353 | 0.0115 | +0.01 ± 0.12 | +0.08 ± 0.15 |
|  | B2 summed, not carried | 2 | 2 | 1 | 0.203 | 3.05 | 0.554 | 0.0915 | 0.0341 | 0.0112 | -0.23 ± 0.24 | -0.09 ± 0.21 |
| straight line (no network) | none | 2 | 0 | 0 | 7020 | 22369 | 10171 | 3690 | 1702 | 645 | +115.48 ± 293.61 | +1.62 ± 3.32 |
|  | B1 gain, chained | 2 | 2 | 0 | 2.36 | 53.9 | 6.88 | 0.307 | 0.0182 | 0.0111 | -0.11 ± 0.63 | +0.07 ± 0.15 |
|  | B2 carried to the end | 2 | 2 | 0 | 2.40 | 54.1 | 6.98 | 0.31 | 0.0178 | 0.0111 | -0.15 ± 0.63 | +0.03 ± 0.18 |
|  | B2 summed, not carried | 2 | 2 | 0 | 146 | 2532 | 415 | 27.9 | 3.26 | 0.33 | -31.97 ± 32.26 | +4.11 ± 6.99 |
| explicit Euler (no network) | none | 3 | 0 | 0 | 5.45 | 22.8 | 8.43 | 2.91 | 1.29 | 0.486 | +0.13 ± 0.31 | +0.17 ± 0.21 |
|  | B1 gain, chained | 3 | 2 | 0 | 0.146 | 0.658 | 0.248 | 0.0852 | 0.0353 | 0.0115 | +0.01 ± 0.12 | +0.08 ± 0.15 |
|  | B2 carried to the end | 3 | 2 | 0 | 0.146 | 0.658 | 0.248 | 0.0852 | 0.0353 | 0.0115 | +0.01 ± 0.12 | +0.08 ± 0.15 |
|  | B2 summed, not carried | 3 | 2 | 0 | 0.965 | 12.8 | 2.76 | 0.222 | 0.0453 | 0.0113 | -0.06 ± 0.18 | +0.21 ± 0.27 |

Comparators from the per-step bias study, same tracks: exact 2-stage Gauss–Legendre 0.147 um (0.66 / 0.25 / 0.085 / 0.035 / 0.012 per band); one fixed-point sweep from the network 0.145–0.147 um at 256 lookups (C.2); one Newton step 0.147 um at 256 lookups (C.3); classic RK4 1.08 um at 256 lookups. The floor is the exact scheme: 0.147 um. The best value at 256 lookups was 0.145 um.

Figure: `figures/b_endpoint_by_band.png`. Table: `results/b_endpoint.csv`.

## Part B: single steps

One step from the reference state on each start plane, all 64 steps of the 1,452 test tracks (92,928 steps). Signed error against the reference state on the next plane. Lag-1: median over tracks of the lag-1 autocorrelation along the track.

| Stages from | Correction | Radial median (um) | Radial rms (um) | rms tx (urad) | rms ty (urad) | Lag-1 x | Lag-1 y |
|---|---|---|---|---|---|---|---|
| exact 2-stage Gauss-Legendre | — | 0.000718 | 0.0219 | 0.295 | 0.247 | -0.22 | -0.18 |
| straight line per-track | none | 0.0721 | 0.893 | 0.3 | 0.249 | +0.85 | +0.90 |
| straight line per-track | gain | 0.000718 | 0.0219 | 0.295 | 0.247 | -0.22 | -0.18 |
| straight line global | none | 0.117 | 1.04 | 0.303 | 0.246 | +0.78 | +0.87 |
| straight line global | gain | 0.000718 | 0.0219 | 0.295 | 0.247 | -0.22 | -0.18 |
| direct states | none | 0.486 | 2.80 | 0.344 | 0.269 | +0.87 | +0.89 |
| direct states | gain | 0.000719 | 0.0219 | 0.295 | 0.247 | -0.22 | -0.18 |
| straight line (no network) | none | 95.2 | 169 | 10.3 | 2.61 | +0.94 | +0.98 |
| straight line (no network) | gain | 0.00132 | 0.0221 | 0.336 | 0.248 | -0.22 | -0.18 |
| explicit Euler (no network) | none | 0.985 | 2.12 | 0.31 | 0.256 | +0.99 | +0.95 |
| explicit Euler (no network) | gain | 0.000718 | 0.0219 | 0.295 | 0.247 | -0.22 | -0.18 |

- The slope error (tx, ty) hardly changes: 0.30 → 0.295 urad. It is the exact scheme's own error at the cell faces (A.6 of the bias study). No correction of the stage equations can remove it.
- The position error falls by 100× to 680× in the median and 40× to 130× in the rms.
- From the straight line the gain leaves a slope error of 0.34 urad (exact: 0.295). The linearisation is not exact when the stage slopes are far off. This is the 2.36 um at the endpoint.

Figure: `figures/b_low_momentum_track.png`: the signed single-step x error along the median test track below 3 GeV (p = 2.63 GeV), before and after. After the gain the curve lies on the exact scheme. Table: `results/b_single_step.csv`, `results/b_low_momentum_track.npz`.

## Part A: toy fit with a model-error covariance Q

The toy fit of A.7 of the bias study (copied, not edited). Fit of (x, y, tx, ty, q/p) on 2648.2 mm to:

- **UT:** 4 hits, z = 2327.5, 2372.5, 2597.5, 2642.5 mm, stereo 0, +5, −5, 0 degrees, **50 um** per hit;
- **SciFi:** 12 hits, stations at 7826, 8508, 9193 mm, layers 70 mm apart, stereo 0, +5, −5, 0 degrees, **75 um** per hit.

These follow the brief of the bias study and the LHCb Tracker Upgrade TDR (CERN-LHCC-2014-001); they were not re-checked against the document. No multiple scattering, no energy loss, no VELO. So the resolution is that of the hits alone. The models differ only from 2648.2 to 7826 mm. 200 noise draws per track. The fit is linear around the truth (A.7 showed that a full Gauss–Newton fit agrees: correlation 0.99993–0.99997).

Q settings:

- **no Q:** hit covariance only.
- **Q per step:** Q = covariance of the signed single-step errors (x, y, tx, ty) of the network on the 1,463 validation tracks, per momentum band. It is carried to 7826 mm as Σ_k T_k Q T_kᵀ with the reference Jacobians and added to the SciFi hit covariance. For a linear fit this is the same as a Kalman filter with process noise Q on every step.
- **Q measured at the end** (extra, not in the brief): the covariance of the chained endpoint error on 7826 mm of the validation tracks, per band. It keeps the correlation of the step errors along the track.
- **exact extrapolator:** the exact 2-stage Gauss–Legendre scheme, 64 steps, no Q. **reference:** the sixth-order method at 0.1 mm, no Q (pulls of a correct model).

Bias / resolution: mean dp/p from the model error alone, over the median dp/p resolution of the same fit. Pull: (fitted − true) / fit sigma, width over draws and tracks.

| Model | Q | Band | Resolution dp/p | Bias / resolution | Pull x | Pull y | Pull tx | Pull ty | Pull q/p |
|---|---|---|---|---|---|---|---|---|---|
| reference (sixth order, 0.1 mm) | no Q | all | 4.58e-04 | +0.0000 | 0.998 | 0.999 | 0.999 | 1.000 | 0.999 |
|  |  | < 3 GeV | 1.50e-04 | +0.0000 | 0.999 | 1.004 | 1.002 | 1.002 | 1.002 |
|  |  | 3–8 GeV | 3.20e-04 | +0.0000 | 0.997 | 1.000 | 0.997 | 0.999 | 0.997 |
|  |  | 8–20 GeV | 8.70e-04 | +0.0000 | 0.999 | 0.997 | 0.998 | 1.002 | 0.998 |
|  |  | 20–50 GeV | 1.89e-03 | +0.0000 | 1.003 | 0.997 | 1.006 | 0.997 | 1.005 |
|  |  | > 50 GeV | 4.98e-03 | +0.0000 | 0.999 | 0.991 | 1.000 | 0.991 | 1.000 |
| exact 2-stage Gauss-Legendre, 64 steps | no Q | all | 4.58e-04 | -0.0013 | 0.998 | 0.999 | 0.999 | 1.000 | 0.999 |
|  |  | < 3 GeV | 1.50e-04 | -0.0063 | 0.999 | 1.004 | 1.002 | 1.003 | 1.002 |
|  |  | 3–8 GeV | 3.20e-04 | -0.0021 | 0.997 | 1.000 | 0.997 | 0.999 | 0.997 |
|  |  | 8–20 GeV | 8.70e-04 | -0.0006 | 0.999 | 0.997 | 0.998 | 1.002 | 0.998 |
|  |  | 20–50 GeV | 1.89e-03 | -0.0001 | 1.003 | 0.997 | 1.006 | 0.997 | 1.005 |
|  |  | > 50 GeV | 4.98e-03 | -0.0002 | 0.999 | 0.991 | 1.000 | 0.991 | 1.000 |
| straight line per-track | no Q | all | 4.58e-04 | -0.0007 | 0.999 | 0.999 | 1.000 | 1.001 | 0.999 |
|  |  | < 3 GeV | 1.50e-04 | -0.0013 | 0.999 | 1.005 | 1.006 | 1.006 | 1.004 |
|  |  | 3–8 GeV | 3.20e-04 | -0.0010 | 0.997 | 1.000 | 0.999 | 1.000 | 0.997 |
|  |  | 8–20 GeV | 8.70e-04 | -0.0016 | 0.999 | 0.997 | 0.998 | 1.002 | 0.998 |
|  |  | 20–50 GeV | 1.89e-03 | -0.0022 | 1.003 | 0.997 | 1.006 | 0.997 | 1.005 |
|  |  | > 50 GeV | 4.98e-03 | +0.0071 | 0.999 | 0.990 | 1.001 | 0.991 | 1.000 |
| straight line per-track | Q per step | all | 4.58e-04 | -0.0007 | 0.999 | 0.999 | 0.999 | 1.000 | 0.998 |
|  |  | < 3 GeV | 1.51e-04 | -0.0013 | 0.999 | 1.004 | 0.998 | 1.005 | 0.998 |
|  |  | 3–8 GeV | 3.20e-04 | -0.0010 | 0.997 | 1.000 | 0.998 | 1.000 | 0.997 |
|  |  | 8–20 GeV | 8.71e-04 | -0.0016 | 0.999 | 0.997 | 0.998 | 1.002 | 0.998 |
|  |  | 20–50 GeV | 1.89e-03 | -0.0022 | 1.003 | 0.997 | 1.006 | 0.997 | 1.005 |
|  |  | > 50 GeV | 4.98e-03 | +0.0071 | 0.999 | 0.990 | 1.001 | 0.991 | 1.000 |
| straight line per-track | Q measured at the end | all | 4.58e-04 | -0.0007 | 0.998 | 0.999 | 0.996 | 1.000 | 0.997 |
|  |  | < 3 GeV | 1.52e-04 | -0.0013 | 0.997 | 1.004 | 0.980 | 1.002 | 0.986 |
|  |  | 3–8 GeV | 3.20e-04 | -0.0010 | 0.997 | 1.000 | 0.996 | 0.999 | 0.996 |
|  |  | 8–20 GeV | 8.71e-04 | -0.0016 | 0.999 | 0.997 | 0.998 | 1.002 | 0.998 |
|  |  | 20–50 GeV | 1.89e-03 | -0.0022 | 1.003 | 0.997 | 1.005 | 0.997 | 1.005 |
|  |  | > 50 GeV | 4.98e-03 | +0.0071 | 0.999 | 0.990 | 1.001 | 0.991 | 1.000 |
| straight line global | no Q | all | 4.58e-04 | -0.0004 | 0.999 | 0.999 | 1.006 | 1.000 | 1.001 |
|  |  | < 3 GeV | 1.50e-04 | -0.0099 | 1.003 | 1.005 | 1.042 | 1.004 | 1.012 |
|  |  | 3–8 GeV | 3.20e-04 | -0.0072 | 0.997 | 1.000 | 1.002 | 0.999 | 0.998 |
|  |  | 8–20 GeV | 8.70e-04 | +0.0015 | 0.999 | 0.997 | 0.998 | 1.002 | 0.998 |
|  |  | 20–50 GeV | 1.89e-03 | +0.0030 | 1.003 | 0.997 | 1.006 | 0.997 | 1.005 |
|  |  | > 50 GeV | 4.98e-03 | -0.0001 | 0.999 | 0.991 | 1.000 | 0.991 | 1.000 |
| straight line global | Q per step | all | 4.58e-04 | -0.0004 | 0.999 | 0.999 | 1.003 | 1.000 | 0.999 |
|  |  | < 3 GeV | 1.51e-04 | -0.0098 | 1.002 | 1.005 | 1.026 | 1.003 | 1.005 |
|  |  | 3–8 GeV | 3.20e-04 | -0.0072 | 0.997 | 1.000 | 1.000 | 0.999 | 0.997 |
|  |  | 8–20 GeV | 8.71e-04 | +0.0015 | 0.999 | 0.997 | 0.998 | 1.002 | 0.998 |
|  |  | 20–50 GeV | 1.89e-03 | +0.0030 | 1.003 | 0.997 | 1.006 | 0.997 | 1.005 |
|  |  | > 50 GeV | 4.98e-03 | -0.0001 | 0.999 | 0.991 | 1.000 | 0.991 | 1.000 |
| straight line global | Q measured at the end | all | 4.59e-04 | -0.0004 | 0.998 | 0.999 | 0.991 | 0.999 | 0.998 |
|  |  | < 3 GeV | 1.50e-04 | -0.0098 | 0.999 | 1.004 | 0.995 | 1.003 | 1.005 |
|  |  | 3–8 GeV | 3.21e-04 | -0.0072 | 0.995 | 0.999 | 0.982 | 0.997 | 0.994 |
|  |  | 8–20 GeV | 8.71e-04 | +0.0015 | 0.999 | 0.997 | 0.998 | 1.002 | 0.998 |
|  |  | 20–50 GeV | 1.89e-03 | +0.0030 | 1.003 | 0.997 | 1.006 | 0.997 | 1.005 |
|  |  | > 50 GeV | 4.98e-03 | -0.0001 | 0.999 | 0.991 | 1.000 | 0.991 | 1.000 |
| direct states | no Q | all | 4.58e-04 | +0.0084 | 1.006 | 1.001 | 1.080 | 1.001 | 1.023 |
|  |  | < 3 GeV | 1.50e-04 | -0.0036 | 1.034 | 1.011 | 1.365 | 1.008 | 1.123 |
|  |  | 3–8 GeV | 3.20e-04 | +0.0106 | 1.004 | 1.001 | 1.070 | 1.001 | 1.018 |
|  |  | 8–20 GeV | 8.70e-04 | +0.0051 | 0.999 | 0.997 | 1.001 | 1.002 | 0.999 |
|  |  | 20–50 GeV | 1.89e-03 | +0.0038 | 1.003 | 0.997 | 1.006 | 0.997 | 1.005 |
|  |  | > 50 GeV | 4.98e-03 | +0.0020 | 0.999 | 0.991 | 1.000 | 0.991 | 1.000 |
| direct states | Q per step | all | 4.61e-04 | +0.0083 | 1.004 | 1.000 | 1.051 | 1.001 | 1.016 |
|  |  | < 3 GeV | 1.53e-04 | -0.0034 | 1.027 | 1.010 | 1.259 | 1.006 | 1.096 |
|  |  | 3–8 GeV | 3.22e-04 | +0.0104 | 1.001 | 1.001 | 1.040 | 1.000 | 1.010 |
|  |  | 8–20 GeV | 8.71e-04 | +0.0051 | 0.999 | 0.997 | 1.000 | 1.002 | 0.998 |
|  |  | 20–50 GeV | 1.89e-03 | +0.0038 | 1.003 | 0.997 | 1.006 | 0.997 | 1.005 |
|  |  | > 50 GeV | 4.98e-03 | +0.0020 | 0.999 | 0.991 | 1.000 | 0.991 | 1.000 |
| direct states | Q measured at the end | all | 4.64e-04 | +0.0082 | 0.999 | 0.999 | 1.000 | 0.998 | 1.008 |
|  |  | < 3 GeV | 1.56e-04 | -0.0033 | 1.005 | 1.006 | 1.032 | 0.998 | 1.068 |
|  |  | 3–8 GeV | 3.25e-04 | +0.0103 | 0.997 | 1.000 | 0.992 | 0.997 | 1.001 |
|  |  | 8–20 GeV | 8.72e-04 | +0.0051 | 0.999 | 0.997 | 0.996 | 1.002 | 0.997 |
|  |  | 20–50 GeV | 1.89e-03 | +0.0038 | 1.003 | 0.997 | 1.005 | 0.997 | 1.005 |
|  |  | > 50 GeV | 4.98e-03 | +0.0020 | 0.999 | 0.991 | 1.000 | 0.991 | 1.000 |

- The momentum bias is the same with and without Q, to 2 digits. Q changes the weights, not the realised model error. Largest |bias / resolution|: 0.011 (direct states, 3–8 GeV). The criterion of the bias study (0.1) fails everywhere: the bias does not matter.
- The exact extrapolator gives |bias / resolution| ≤ 0.006. After the gain of Part B, a network endpoint equals the exact scheme (Part B, item 1), so it has this bias.
- The straight-line runs have pulls 0.98 to 1.04 in every setting. Q changes nothing there.
- Direct states below 3 GeV: pull tx 1.37 → 1.26 (Q per step) → 1.03 (Q at the end); pull q/p 1.12 → 1.10 → 1.07; pull x 1.03 → 1.03 → 1.01. The resolution cost: dp/p 1.50e-4 → 1.53e-4 (+2%) → 1.56e-4 (+4%).

Size of Q (validation, per band), and the per-step Q carried to 7826 mm against the measured spread of the endpoint error:

| Network | Band | per-step sigma x (um) | sigma y (um) | sigma tx (urad) | sigma ty (urad) | per-step Q carried to 7826: sigma x (um) | endpoint error sigma x (um) | ratio |
|---|---|---|---|---|---|---|---|---|
| straight line per-track | < 3 GeV | 1.67 | 1.54 | 0.726 | 0.665 | 22.4 | 26.4 | 1.2 |
| straight line per-track | 3–8 GeV | 0.59 | 1.04 | 0.353 | 0.266 | 10.4 | 11.3 | 1.1 |
| straight line per-track | 8–20 GeV | 0.188 | 0.242 | 0.0953 | 0.0641 | 2.68 | 3.11 | 1.2 |
| straight line per-track | 20–50 GeV | 0.242 | 0.0557 | 0.0265 | 0.00387 | 2.03 | 4.80 | 2.4 |
| straight line per-track | > 50 GeV | 0.223 | 0.0214 | 0.011 | 0.00018 | 1.80 | 4.67 | 2.6 |
| straight line global | < 3 GeV | 2.49 | 1.85 | 0.722 | 0.653 | 28.1 | 53.8 | 1.9 |
| straight line global | 3–8 GeV | 1.06 | 0.938 | 0.357 | 0.275 | 13.3 | 32.5 | 2.5 |
| straight line global | 8–20 GeV | 0.3 | 0.219 | 0.0953 | 0.0641 | 3.36 | 4.54 | 1.4 |
| straight line global | 20–50 GeV | 0.113 | 0.0574 | 0.0265 | 0.00386 | 1.12 | 2.12 | 1.9 |
| straight line global | > 50 GeV | 0.0727 | 0.0362 | 0.011 | 0.000153 | 0.645 | 0.764 | 1.2 |
| direct states | < 3 GeV | 6.08 | 2.61 | 0.952 | 0.684 | 62.0 | 146 | 2.4 |
| direct states | 3–8 GeV | 3.92 | 1.65 | 0.413 | 0.306 | 35.9 | 66.1 | 1.8 |
| direct states | 8–20 GeV | 0.881 | 0.336 | 0.0951 | 0.0648 | 7.48 | 14.3 | 1.9 |
| direct states | 20–50 GeV | 0.444 | 0.181 | 0.0265 | 0.0039 | 3.61 | 6.18 | 1.7 |
| direct states | > 50 GeV | 0.256 | 0.126 | 0.011 | 0.000189 | 2.06 | 2.25 | 1.1 |

- The per-step Q gives 1.1× to 2.6× too small a sigma at the endpoint. The step errors are correlated (lag-1 0.85 to 0.90), so they add faster than independent errors. This is why the per-step Q leaves the pulls of direct states wide.
- The slope part of Q (sigma tx about 0.7 urad below 3 GeV, the same for all three runs) is the exact scheme's own error. Carried over 5 m it gives most of the carried sigma x for the straight-line runs.

Tables: `results/a_toy_fit_with_q.csv`, `results/a_q_per_band.csv`.

## Decision point

| Question | Answer | Evidence |
|---|---|---|
| Does the gain remove the per-step bias? | Yes. Every run gives the exact scheme, 0.147 um, in every band. The lag-1 autocorrelation becomes that of the exact scheme. | Part B |
| Does it cost new field lookups? | No. It needs 2 Jacobians per step at the stages the network already gives, and an 8 × 8 solve. | Method |
| Does the network matter for it? | Yes at 128 lookups: 0.147 um against 2.36 um from the straight line. A network-free Euler start needs 1 more lookup per step (192) for 0.146 um. | Controls |
| Does a Q in the fit help? | Not for the bias (unchanged, ≤ 1.1% of the resolution). It narrows the wide pulls of direct states below 3 GeV, best with Q measured at the endpoint. | Part A |

Open for George: is a Jacobian evaluation and an 8 × 8 solve per step cheaper than one field lookup in the target code? If yes, the network with the gain beats the Euler start. If no, the Euler start with the gain gives the same accuracy without a network.

## What is not done

- B3: not run. It needs 2 new field lookups per step (see Method).
- The cost of a Jacobian evaluation and of the 8 × 8 solve is counted, not timed.
- The hit layout and resolutions of Part A were not re-checked against the TDR (as in A.7).
- The Part A fit of a corrected network (gain + fit) was not run separately; the exact-extrapolator row stands for it.
