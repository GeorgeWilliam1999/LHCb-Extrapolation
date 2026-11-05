# Results: what is the per-step bias

**Date:** 2026-10-28. **State:** for discussion with George. Nothing is written up.

## Scope

- Three runs: `25dfa17ef168` (straight line, per-track scale), `9cef9c160b7d` (straight line, one global scale), `49e5e2450eaa` (direct states). The two forms are equal methods.
- Each run is one seed. A difference under 2× is not a claim.
- Test set: 1,452 tracks. Score: radial median at z = 7826 mm against the sixth-order reference at 0.1 mm.
- e = network one step from the reference state, minus the reference state on the next plane. 64 steps. x, y in um. tx, ty in urad.
- The analysis ran on `stbc-i3`. No farm job ran. Nothing in the store changed.

## Checks of the code

| Quantity | This study | Known value |
|---|---|---|
| endpoint, `25dfa17ef168`, last round | 1.83 um | 1.83 um (standard report) |
| lag-1 autocorrelation of e_x / e_y, `25dfa17ef168` | 0.85 / 0.90 | 0.85 / 0.90 (tranche-1 analysis) |
| exact 2-stage Gauss–Legendre, 64 steps | 0.147 um | 0.147 um |
| classic RK4, 64 steps | 1.08 um | 1.08 um |
| largest residual of the exact single steps (Newton, this folder) | 2.3e-13 | tolerance 1e-12 |

## Summary

1. The slope part of e (tx, ty) is the exact scheme's own error. It comes from the cell faces of the field map. Every network shares it. Training cannot remove it.
2. The position part of e (x, y) is the network's own error. It does not come from the cell faces. It is private: forms do not share it, and the snapshots of a straight-line run do not share it.
3. The position error is a slow wave along the track. Its correlation length is about 5 steps. Its mean over the population is near zero. It is not a constant bias per track.
4. A smooth function of the state does not predict e well. The best validation R² is 0.43. The "removable" criterion (R² ≥ 0.5) fails for every component of every run.
5. The momentum bias of a toy fit is at most 1.1% of the hits-only momentum resolution. The criterion "bias matters" (≥ 10%) fails in every band. The bias does not matter after a fit.
6. One fixed-point sweep, or one Newton step, of the stage equations from the network's stages gives the exact scheme: 0.145 to 0.147 um for all three runs, at 256 field calls. This passes the C.2/C.3 criterion. It is 7× better than RK4 at the same number of field calls.
6b. Control (C.2b): without a network, one Newton step from straight-line stages also gives 0.133 um at 256 field calls. So with a Newton step the network adds nothing at 256 calls. With fixed-point sweeps only, the network saves one sweep: straight-line stages need 2 sweeps (384 calls) for 0.145 um, and give 1.67 um after 1 sweep (256 calls).
7. At 128 field calls, no correction passes. The speed claim (≤ 1.08 um at 128 field + 64 network calls) is not kept.
8. Weight averaging fails: 5.3 to 70 um. Output averaging helps by less than 2×.

## A.1 Smooth model of e

Fitted on the training tracks (740,288 steps). R² on the validation tracks (93,632 steps). The trees setting was chosen on the last fifth of the training tracks, held out. A first fit with one setting reached R² 0.8–0.9 on the training tracks and ≤ 0.4 on validation: it learned the tracks, not the state.

| Run | Component | rms of e | R² trees, training | R² trees, validation | R² cubic, validation | Removable (≥ 0.5) |
|---|---|---|---|---|---|---|
| straight line per-track | x | 0.72 | 0.87 | 0.40 | 0.01 | no |
| straight line per-track | y | 0.90 | 0.03 | -0.04 | -0.11 | no |
| straight line per-track | tx | 0.35 | 0.10 | 0.12 | -0.02 | no |
| straight line per-track | ty | 0.30 | 0.00 | 0.00 | 0.00 | no |
| straight line global | x | 1.14 | 0.03 | 0.02 | -0.18 | no |
| straight line global | y | 0.92 | 0.14 | 0.01 | 0.01 | no |
| straight line global | tx | 0.35 | 0.13 | 0.21 | 0.00 | no |
| straight line global | ty | 0.30 | 0.00 | 0.00 | -0.01 | no |
| direct states | x | 3.44 | 0.90 | 0.43 | 0.06 | no |
| direct states | y | 1.46 | 0.85 | 0.25 | -0.01 | no |
| direct states | tx | 0.44 | 0.69 | 0.34 | 0.03 | no |
| direct states | ty | 0.32 | 0.01 | 0.01 | 0.00 | no |

Table: `results/a1_smooth_model_r2.csv`.

## A.2 Binned mean of e against each variable

The binned means are 0.01 to 0.1 um in x and y. The rms of e is 0.7 to 3.4 um. So the mean of e in any slice of one variable is 1% to 10% of its spread. The largest structure: e_y of the straight-line run grows at |y| > 500 mm and |ty| > 0.1 (up to 0.37 um), and at low |q/p| it is near zero.

Figures: `figures/a2_binned_mean_<key>.png`. Table: `results/a2_binned_mean_of_e.csv`.

## A.3 e by momentum band

Mean e_x along the bend, ± standard error, in um. Positive: the step bends too much. rms of e_x and e_y in um.

| Band | per-track: mean along bend | per-track: rms x / y | global: mean along bend | global: rms x / y | direct: mean along bend | direct: rms x / y | exact: rms x / y |
|---|---|---|---|---|---|---|---|
| < 3 GeV | -0.024 ± 0.007 | 0.71 / 1.19 | +0.010 ± 0.013 | 1.36 / 0.98 | +0.042 ± 0.047 | 4.82 / 1.88 | 0.031 / 0.020 |
| 3–8 GeV | -0.007 ± 0.002 | 0.49 / 1.01 | +0.014 ± 0.005 | 0.98 / 0.85 | -0.048 ± 0.014 | 2.91 / 1.29 | 0.018 / 0.020 |
| 8–20 GeV | +0.003 ± 0.001 | 0.17 / 0.14 | -0.007 ± 0.001 | 0.19 / 0.10 | -0.022 ± 0.005 | 0.83 / 0.26 | 0.004 / 0.000 |
| 20–50 GeV | +0.008 ± 0.001 | 0.09 / 0.02 | -0.012 ± 0.001 | 0.08 / 0.04 | -0.015 ± 0.003 | 0.36 / 0.12 | 0.001 / 0.000 |
| > 50 GeV | -0.006 ± 0.004 | 0.19 / 0.02 | -0.000 ± 0.002 | 0.07 / 0.03 | -0.011 ± 0.006 | 0.27 / 0.09 | 0.001 / 0.000 |

The fraction of steps that bend too much is 0.41 to 0.53 in every band and run. The mean along the bend is 1% to 3% of the rms.

Power n of rms(e) ∝ |q/p|^n, fitted over 12 bins of |q/p|:

| Run | x | y | tx | ty |
|---|---|---|---|---|
| straight line per-track | 0.75 | 1.59 | 1.37 | 2.74 |
| straight line global | 1.21 | 1.48 | 1.38 | 2.75 |
| direct states | 1.12 | 1.20 | 1.44 | 2.76 |
| exact scheme | 1.37 | 2.82 | 1.36 | 2.75 |

The powers of tx and ty are the same for every network and the exact scheme. This agrees with item 1 of the summary. Tables: `results/a3_*.csv`.

## A.4 Sign structure

Single steps. The mean of e in each group is below 0.035 um for the straight-line runs and below 0.11 um for direct states. Every such mean is 4% or less of the rms of e. With the magnet up, the charge sets the bend direction, so the "charge" and "bend +x / -x" groups are the same steps. Table: `results/a4_sign_structure_of_steps.csv`.

Endpoint along and across the bend. "Along": d_x × sign(tx at 7826 − tx at 2648.2). "Across": d_y × sign(y at 7826). Means ± standard error, um.

| Run | Band | Tracks | Mean along bend | Median along bend | Fraction bending too much | Mean across (away from y = 0) | rms x / y |
|---|---|---|---|---|---|---|---|
| straight line per-track | all | 1452 | -0.13 ± 0.25 | +0.02 | 0.50 | +1.10 ± 0.46 | 9.4 / 17.4 |
| | < 3 GeV | 164 | -0.93 ± 1.18 | -1.25 | 0.39 | +3.76 ± 2.60 | 15.1 / 33.4 |
| | 3–8 GeV | 655 | -0.35 ± 0.44 | +0.12 | 0.54 | +1.35 ± 0.75 | 11.3 / 19.3 |
| | 8–20 GeV | 433 | +0.23 ± 0.17 | -0.17 | 0.44 | +0.09 ± 0.29 | 3.5 / 6.0 |
| | 20–50 GeV | 170 | +0.57 ± 0.14 | +0.40 | 0.63 | +0.32 ± 0.06 | 1.9 / 0.9 |
| | > 50 GeV | 30 | -0.35 ± 1.21 | +0.86 | 0.63 | +0.23 ± 0.11 | 6.5 / 0.6 |
| straight line global | all | 1452 | +0.21 ± 0.52 | -0.39 | 0.42 | -0.21 ± 0.32 | 19.8 / 12.2 |
| | < 3 GeV | 164 | +1.02 ± 3.88 | -0.88 | 0.45 | +0.83 ± 2.17 | 49.5 / 27.7 |
| | 3–8 GeV | 655 | +0.70 ± 0.61 | +0.45 | 0.56 | -0.43 ± 0.45 | 15.7 / 11.6 |
| | 8–20 GeV | 433 | -0.44 ± 0.14 | -0.58 | 0.30 | -0.14 ± 0.11 | 2.9 / 2.2 |
| | 20–50 GeV | 170 | -0.75 ± 0.06 | -0.91 | 0.19 | -0.51 ± 0.05 | 1.1 / 0.8 |
| | > 50 GeV | 30 | -0.01 ± 0.13 | +0.07 | 0.53 | -0.47 ± 0.07 | 0.7 / 0.6 |
| direct states | all | 1452 | -1.28 ± 1.46 | -0.75 | 0.44 | +1.39 ± 0.65 | 55.8 / 24.8 |
| | < 3 GeV | 164 | +6.31 ± 9.60 | +3.49 | 0.56 | +8.35 ± 3.39 | 122.8 / 44.1 |
| | 3–8 GeV | 655 | -3.24 ± 2.16 | -0.58 | 0.46 | -0.51 ± 1.13 | 55.2 / 29.0 |
| | 8–20 GeV | 433 | -1.37 ± 0.53 | -0.79 | 0.38 | +1.27 ± 0.36 | 11.1 / 7.5 |
| | 20–50 GeV | 170 | -0.94 ± 0.18 | -1.26 | 0.39 | +2.20 ± 0.18 | 2.6 / 3.2 |
| | > 50 GeV | 30 | -0.70 ± 0.57 | -1.51 | 0.43 | +2.07 ± 0.26 | 3.1 / 2.5 |

Table: `results/a4_endpoint_along_and_across_the_bend.csv`.

Agreement with the resolution thread of 2026-10-28 (`/data/bfys/gscriven/scratch/resolution_2026-10-28/`, `bend_signed.csv`, same bend definition). That thread finds the pooled straight-line runs at 64 steps balanced along the bend (28% to 64% bend too much, median -1.2 to +0.3 um). This study finds 42% to 50% overall and a median of -0.39 to +0.02 um for the two straight-line runs. The two agree. Above 8 GeV some band means are 3 to 12 standard errors from zero, but they are at most 0.75 um along the bend for the straight-line runs and 2.2 um across for direct states. Below 8 GeV the means along the bend are within 1.5 standard errors of zero.

## A.4b How a lag-1 autocorrelation of 0.85 fits a mean near zero

Test steps. "Along the bend" for x and tx. The effective standard error of a per-track mean uses n(1 − r)/(1 + r) steps, with r the track's lag-1 autocorrelation.

| Run | Component | Population mean | Population rms | Share of variance in per-track means | Tracks with mean > 3 eff. s.e. | Lag 1 | Lag 8 | Lag 16 | Lag 32 | Sign changes in 63 |
|---|---|---|---|---|---|---|---|---|---|---|
| straight line per-track | x along bend | -0.004 | 0.42 | 0.10 | 1.2% | 0.85 | 0.03 | -0.13 | -0.08 | 6 |
| straight line per-track | y | +0.012 | 0.79 | 0.14 | 2.7% | 0.90 | 0.13 | -0.08 | -0.10 | 5 |
| straight line global | x along bend | +0.004 | 0.81 | 0.09 | 0.1% | 0.78 | -0.02 | -0.04 | -0.10 | 10 |
| straight line global | y | +0.008 | 0.66 | 0.03 | 0.2% | 0.87 | -0.12 | -0.04 | -0.12 | 8 |
| direct states | x along bend | -0.025 | 2.58 | 0.12 | 0.0% | 0.87 | -0.28 | -0.19 | -0.08 | 6 |
| direct states | y | -0.005 | 1.08 | 0.12 | 0.6% | 0.89 | -0.12 | -0.20 | -0.01 | 5 |

The reading:

- The lag-1 autocorrelation is high, but the lag-8 autocorrelation is near zero. So e changes slowly from one step to the next, and it changes sign about 5 to 10 times along a track.
- The per-track mean holds only 3% to 14% of the variance. Almost no track has a mean that differs from zero.
- So e is a smooth wave along the track, not a constant offset per track. Over the population its sign is random, and its mean is near zero.
- Neighbouring steps add in the same direction, so the endpoint grows faster than √64 × the single-step error (ratio 2.5 to 3.2, tranche-1 analysis). The wave cancels over longer distances, so the endpoint has no direction.
- For direct states the wave has a period of about 2,000 mm and a constant phase in z (`figures/along_the_track.png`). This is a fit residual of the network in its z input.

Table: `results/a4b_per_track_bias_reconciliation.csv`.

## A.5 Each step's e carried to the end

The reference Jacobian of each step comes from central differences on the sixth-order method at 0.1 mm. T_k carries the end of step k to 7826 mm. The carried sum is Σ_k T_k e_k.

| Run | Correlation with the chained endpoint, x / y | Slope, x / y | Carried sum, median (um) | Chained, median (um) | Median of the difference (um) |
|---|---|---|---|---|---|
| straight line per-track | 0.99993 / 0.99999 | 0.998 / 1.003 | 1.831 | 1.831 | 6e-5 |
| straight line global | 0.99999 / 1.00000 | 1.000 / 1.000 | 2.039 | 2.039 | 7e-5 |
| direct states | 1.00000 / 1.00000 | 0.999 / 1.000 | 5.705 | 5.705 | 1e-3 |

The endpoint error is the linear sum of the single-step errors. Chaining adds nothing.

Share of the variance of the endpoint error per z bin of 8 steps (sum = 1; an even share is 0.125):

| z bin (mm) | per-track x | per-track y | global x | global y | direct x | direct y |
|---|---|---|---|---|---|---|
| 2648–3295 | 0.10 | 0.01 | 0.02 | 0.02 | 0.19 | 0.14 |
| 3295–3943 | 0.22 | 0.03 | 0.16 | 0.08 | 0.16 | 0.05 |
| 3943–4590 | 0.18 | 0.03 | 0.32 | 0.60 | 0.07 | 0.18 |
| 4590–5237 | 0.06 | 0.52 | 0.08 | 0.11 | -0.06 | 0.14 |
| 5237–5884 | 0.08 | 0.23 | 0.00 | -0.10 | 0.06 | 0.06 |
| 5884–6532 | 0.14 | 0.17 | 0.16 | 0.14 | 0.24 | 0.13 |
| 6532–7179 | 0.16 | -0.02 | 0.15 | 0.07 | 0.24 | 0.23 |
| 7179–7826 | 0.05 | 0.03 | 0.10 | 0.08 | 0.11 | 0.07 |

The y error of the straight-line runs comes mostly from one bin (0.52 in 4590–5237 mm; 0.60 in 3943–4590 mm). The x error spreads over the whole region. Figure: `figures/a5_share_by_z.png`. Tables: `results/a5_*.csv`.

## A.6 Steps that cross a z face of the field map

The grid has 100 mm cells. A step is 80.9 mm. So 81% of the steps cross one z face.

| Series | Steps | Fraction | Mean radial e (um) | Mean \|e_tx\| (urad) | Mean \|e_ty\| (urad) |
|---|---|---|---|---|---|
| exact scheme | cross a z face | 0.81 | 0.0058 | 0.115 | 0.018 |
| exact scheme | cross no z face | 0.19 | 0.0002 | 0.002 | 0.002 |
| exact scheme | cross no face at all | 0.16 | 0.00003 | 0.00001 | 0.00000 |
| straight line per-track | cross a z face | 0.81 | 0.177 | 0.116 | 0.019 |
| straight line per-track | cross no z face | 0.19 | 0.184 | 0.006 | 0.004 |
| straight line global | cross a z face | 0.81 | 0.276 | 0.117 | 0.020 |
| straight line global | cross no z face | 0.19 | 0.259 | 0.010 | 0.005 |
| direct states | cross a z face | 0.81 | 1.134 | 0.130 | 0.030 |
| direct states | cross no z face | 0.19 | 1.106 | 0.036 | 0.019 |

- For the exact scheme, a face crossing multiplies the error by 30 (position) and 50 (slope).
- For the networks, the position error is the same with and without a crossing. So the faces do not cause the position error.
- For the networks, the slope error follows the faces in the same way as the exact scheme, at the same size.

Table: `results/a6_cell_face_crossings.csv`.

## A.7 Toy track fit

Fit of (x, y, tx, ty, q/p) on 2648.2 mm to 4 UT hits (z 2327.5–2642.5 mm, 50 um) and 12 SciFi hits (z 7826–9403 mm, 75 um), stereo 0, +5, −5, 0 degrees. The layout and resolutions follow the brief and the LHCb Tracker Upgrade TDR (CERN-LHCC-2014-001); the values were not re-checked against the document. No scattering, no energy loss, no VELO. So the resolution is that of the hits alone. It is small, and the 10% test is strict. The network and the reference differ only from 2648.2 to 7826 mm. Both use the sixth-order method outside. 200 noise draws per track.

| Band | Resolution dp/p (median) | per-track: mean bias dp/p | per-track: bias / resolution | global: bias / resolution | direct: mean bias dp/p | direct: bias / resolution | direct: rms per-track bias / own resolution | direct: pull width |
|---|---|---|---|---|---|---|---|---|
| < 3 GeV | 1.5e-4 | -1.9e-7 | -0.001 | -0.010 | -5.4e-7 | -0.004 | 0.53 | 1.12 |
| 3–8 GeV | 3.2e-4 | -3.2e-7 | -0.001 | -0.007 | +3.4e-6 | +0.011 | 0.21 | 1.02 |
| 8–20 GeV | 8.7e-4 | -1.4e-6 | -0.002 | +0.002 | +4.4e-6 | +0.005 | 0.04 | 1.00 |
| 20–50 GeV | 1.9e-3 | -4.1e-6 | -0.002 | +0.003 | +7.2e-6 | +0.004 | 0.01 | 1.00 |
| > 50 GeV | 5.0e-3 | +3.5e-5 | +0.007 | -0.000 | +1.0e-5 | +0.002 | 0.01 | 1.00 |

- The largest |mean bias| / resolution is 0.011 (direct states, 3–8 GeV). The criterion is 0.1. "Bias matters" fails in every band of every run.
- The pull width of the reference fit is 1.00 in every band. For the straight-line runs the network fit gives 1.00 to 1.01.
- For direct states below 3 GeV, the per-track error is 0.53 of the resolution. It widens the pull to 1.12. This is a spread, not a bias.
- A full Gauss–Newton fit with both models evaluated exactly agrees with the linear answer: correlation 0.99993 to 0.99997.
- With multiple scattering the real resolution is larger (about 0.5% at 10 GeV), so every ratio above is an upper bound.

C.5 (a correction from the measured SciFi residuals) is moved out of this experiment to a separate thread (George, 2026-10-28), with every question about a Kalman-filter fit. Figure: `figures/a7_toy_fit.png`. Tables: `results/a7_*.csv`.

## A.8 Seeds

Updated 2026-11-05. The runs are those of `does_a_better_optimiser_improve_the_endpoint` (tranche 1), each at its own last finished round (`seeds.py last`), with the seed-1 runs of this note. Per form: L-BFGS seeds 1, 2, 3 and self-scaled Broyden seeds 1, 2, 3. Rounds: straight line per-track 44 / 36 / 35 (L-BFGS) and 22 / 22 / 22 (Broyden); direct states 59 / 42 / 56 (L-BFGS) and 37 / 56 / 33 (Broyden). All plateaued. 15 pairs of seeds per form.

On 2026-11-03 `c3a1aa54c6fe` (direct states, Broyden seed 2) was still training and was left out (10 pairs for direct states; `logs/seeds_last_c3a1aa54c6fe.txt`). It plateaued at round 56. With it in, the reading does not change.

Bias map as in A.9: mean e in 8 × 8 cells of signed q/p × z. "Shared" if the correlation is > 0.7. The table gives the median over the pairs, with the lowest and highest pair.

| Form | e | x | y | tx | ty |
|---|---|---|---|---|---|
| straight line per-track | e | -0.11 (-0.52 to 0.34) | -0.01 (-0.44 to 0.63) | 0.95 (0.89 to 0.99) | 0.59 (0.34 to 0.92) |
| straight line per-track | e minus the exact scheme's error | -0.10 (-0.52 to 0.34) | -0.01 (-0.44 to 0.63) | 0.02 (-0.43 to 0.65) | 0.09 (-0.37 to 0.68) |
| direct states | e | 0.22 (-0.19 to 0.70) | 0.35 (0.04 to 0.66) | 0.78 (0.64 to 0.98) | 0.51 (0.03 to 0.87) |
| direct states | e minus the exact scheme's error | 0.23 (-0.18 to 0.70) | 0.35 (0.03 to 0.66) | 0.12 (-0.32 to 0.61) | 0.27 (-0.56 to 0.66) |

Reading:

- The slope maps that look shared (tx in both forms; ty of the straight line between the L-BFGS seeds) are the exact scheme's own error. A.9 found the same for one seed. With that error removed, no component is shared in either form. Medians are -0.10 to 0.35. None of the 120 rows is above 0.7; the highest is 0.697 (direct states x, L-BFGS seeds 1–2).
- The network's own part of e is private to each seed. A correction learned from one network does not carry over to another seed.
- x and y do not carry the exact scheme's error (A.9: share of variance not exact = 1.00), so the two rows agree there.

Table: `results/a8_seed_correlations_last.csv`. Log: `logs/seeds_last.txt`. Package code at commit `27a054e1`. The round-1 smoke test of 2026-10-28 is in `results/a8_seed_correlations_common.csv`; round 1 is not a trained network and is not a result.

## A.9 Bias maps: forms and snapshots

Bias map: mean e in 8 bins of signed q/p × 8 bins of z (64 cells). "Shared" if the correlation is > 0.7.

| Comparison | x | y | tx | ty |
|---|---|---|---|---|
| straight per-track vs straight global | -0.22 | -0.12 | 0.96 | 0.70 |
| straight per-track vs direct | 0.41 | -0.32 | 0.74 | 0.44 |
| straight global vs direct | -0.51 | 0.12 | 0.60 | 0.09 |
| last 5 snapshots, straight per-track (mean of pairs) | 0.37 | 0.47 | 0.98 | 0.92 |
| last 5 snapshots, straight global | 0.42 | 0.16 | 0.95 | 0.76 |
| last 5 snapshots, direct | 0.89 | 0.60 | 0.90 | 0.75 |
| each run vs the exact scheme (per-track / global / direct) | 0.10 / -0.14 / -0.23 | 0.45 / 0.18 / -0.03 | 0.99 / 0.97 / 0.72 | 0.98 / 0.76 / 0.43 |

Share of the variance of e that is not the exact scheme's error (e minus the exact scheme's e):

| Run | x | y | tx | ty |
|---|---|---|---|---|
| straight line per-track | 1.00 | 1.00 | 0.02 | 0.01 |
| straight line global | 1.00 | 1.00 | 0.04 | 0.03 |
| direct states | 1.00 | 1.00 | 0.25 | 0.14 |

With the exact scheme's error removed, no slope map is shared between forms (-0.58 to 0.27), and the slope maps of the last 5 snapshots of the straight-line runs fall to 0.02–0.44. The x map of the direct-states snapshots stays shared (0.89): this network has a stable wave in z across its last rounds (A.4b). Table: `results/a9_bias_map_correlations.csv`.

## B.1 and Part C: corrections

Test endpoint radial median (um), per band, mean signed endpoint error ± standard error (um), lag-1 autocorrelation of the corrected single-step e, calls per track.

| Run | Variant | Field calls | Net calls | Endpoint | < 3 | 3–8 | 8–20 | 20–50 | > 50 | Mean x | Mean y | Lag-1 x / y |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| straight per-track | plain | 128 | 64 | 1.83 | 4.94 | 2.10 | 1.44 | 1.21 | 1.89 | -0.03 ± 0.25 | +0.77 ± 0.46 | 0.85 / 0.90 |
| | B.1 output averaging | 640 | 320 | 1.41 | 3.78 | 1.74 | 0.93 | 0.93 | 1.82 | -0.14 ± 0.22 | +0.68 ± 0.27 | 0.85 / 0.90 |
| | B.1 weight averaging | 128 | 64 | 5.25 | 21.65 | 9.02 | 2.91 | 2.31 | 3.90 | +1.20 ± 0.51 | +1.65 ± 0.30 | 0.93 / 0.92 |
| | C.1 trees (supervised) | 128 | 64 + 64 trees | 1.36 | 3.97 | 1.68 | 0.90 | 0.68 | 1.96 | +0.02 ± 0.24 | -0.65 ± 0.39 | 0.81 / 0.90 |
| | C.1 cubic (supervised) | 128 | 64 + 64 cubic | 6.46 | 9.10 | 6.78 | 5.99 | 5.10 | 6.20 | -0.07 ± 0.30 | -0.44 ± 0.45 | 0.86 / 0.91 |
| | C.2 one sweep | 256 | 64 | 0.147 | 0.66 | 0.25 | 0.085 | 0.035 | 0.011 | +0.02 ± 0.13 | +0.08 ± 0.15 | -0.22 / -0.17 |
| | C.2 two sweeps | 384 | 64 | 0.147 | 0.66 | 0.25 | 0.085 | 0.035 | 0.012 | +0.01 ± 0.12 | +0.08 ± 0.15 | -0.22 / -0.18 |
| | C.3 one Newton step | 256 | 64 | 0.147 | 0.66 | 0.25 | 0.085 | 0.035 | 0.012 | +0.01 ± 0.12 | +0.08 ± 0.15 | -0.22 / -0.18 |
| | C.4 distilled | 128 | 64 + 64 small | 2.28 | 6.33 | 3.08 | 1.41 | 1.81 | 2.35 | -0.18 ± 0.25 | -0.20 ± 0.29 | 0.85 / 0.92 |
| straight global | plain | 128 | 64 | 2.04 | 7.42 | 3.21 | 1.31 | 1.30 | 0.65 | +0.02 ± 0.52 | +0.76 ± 0.32 | 0.78 / 0.87 |
| | B.1 output averaging | 640 | 320 | 1.37 | 5.23 | 2.11 | 0.87 | 0.44 | 0.38 | +0.52 ± 0.43 | +0.65 ± 0.28 | 0.77 / 0.86 |
| | B.1 weight averaging | 128 | 64 | 43.5 | 96.8 | 59.4 | 27.9 | 28.7 | 30.5 | +4.15 ± 2.50 | -14.47 ± 0.55 | 0.98 / 0.96 |
| | C.1 trees (supervised) | 128 | 64 + 64 trees | 1.89 | 7.35 | 2.97 | 1.37 | 1.28 | 0.68 | +0.39 ± 0.54 | +0.52 ± 0.32 | 0.79 / 0.87 |
| | C.1 cubic (supervised) | 128 | 64 + 64 cubic | 3.92 | 16.13 | 6.35 | 2.28 | 1.33 | 1.30 | -0.05 ± 0.74 | +0.33 ± 0.31 | 0.83 / 0.89 |
| | C.2 one sweep | 256 | 64 | 0.147 | 0.66 | 0.25 | 0.085 | 0.035 | 0.012 | +0.01 ± 0.13 | +0.08 ± 0.15 | -0.22 / -0.17 |
| | C.2 two sweeps | 384 | 64 | 0.147 | 0.66 | 0.25 | 0.085 | 0.035 | 0.012 | +0.01 ± 0.12 | +0.08 ± 0.15 | -0.22 / -0.18 |
| | C.3 one Newton step | 256 | 64 | 0.147 | 0.66 | 0.25 | 0.085 | 0.035 | 0.012 | +0.01 ± 0.12 | +0.08 ± 0.15 | -0.22 / -0.18 |
| | C.4 distilled | 128 | 64 + 64 small | 2.31 | 7.27 | 3.55 | 1.47 | 1.21 | 1.01 | +0.02 ± 0.62 | +0.67 ± 0.35 | 0.80 / 0.88 |
| direct states | plain | 128 | 64 | 5.71 | 21.67 | 8.75 | 3.54 | 2.96 | 3.24 | +1.40 ± 1.46 | -0.39 ± 0.65 | 0.87 / 0.89 |
| | B.1 output averaging | 640 | 320 | 5.24 | 22.47 | 8.29 | 2.78 | 2.32 | 2.27 | +1.53 ± 1.25 | +0.56 ± 0.61 | 0.86 / 0.89 |
| | B.1 weight averaging | 128 | 64 | 69.5 | 76.8 | 75.5 | 69.8 | 68.0 | 69.0 | -64.6 ± 1.4 | -9.9 ± 0.7 | 0.91 / 0.93 |
| | C.1 trees (supervised) | 128 | 64 + 64 trees | 3.34 | 13.02 | 5.74 | 1.63 | 1.00 | 1.34 | +2.08 ± 1.23 | +0.25 ± 0.50 | 0.85 / 0.88 |
| | C.1 cubic (supervised) | 128 | 64 + 64 cubic | 8.72 | 29.52 | 12.86 | 5.59 | 4.52 | 5.30 | +1.66 ± 1.36 | +0.71 ± 0.62 | 0.87 / 0.89 |
| | C.2 one sweep | 256 | 64 | 0.145 | 0.67 | 0.25 | 0.087 | 0.035 | 0.011 | +0.02 ± 0.13 | +0.07 ± 0.15 | -0.22 / +0.04 |
| | C.2 two sweeps | 384 | 64 | 0.147 | 0.66 | 0.25 | 0.085 | 0.035 | 0.012 | +0.01 ± 0.12 | +0.08 ± 0.15 | -0.22 / -0.18 |
| | C.3 one Newton step | 256 | 64 | 0.147 | 0.66 | 0.25 | 0.085 | 0.035 | 0.012 | +0.01 ± 0.12 | +0.08 ± 0.15 | -0.22 / -0.18 |
| | C.4 distilled | 128 | 64 + 64 small | 6.60 | 15.96 | 10.05 | 4.90 | 2.95 | 1.73 | +2.11 ± 1.31 | +0.63 ± 0.59 | 0.84 / 0.89 |
| comparator | exact 2-stage Gauss–Legendre | — | 0 | 0.147 | 0.66 | 0.25 | 0.085 | 0.035 | 0.012 | +0.01 ± 0.12 | +0.08 ± 0.15 | -0.22 / -0.18 |
| comparator | classic RK4 | 256 | 0 | 1.08 | 3.08 | 1.66 | 0.62 | 0.28 | 0.11 | -0.22 ± 0.22 | -0.21 ± 0.14 | -0.15 / -0.13 |

"Field calls" counts f at one stage state. C.3 takes the field gradient with 128 of its 256 field calls. Tables: `results/c_variants_<key>.csv`, `results/c_comparators.csv`. Figure: `figures/c_variants.png`.

Pass criteria:

| Criterion | Result |
|---|---|
| direct states ≤ 2.85 um | passes with C.2 and C.3 (0.145–0.147); fails with B.1, C.1 (3.34), C.4 |
| straight line ≤ 0.92 um | passes with C.2 and C.3 (0.147); fails with B.1, C.1, C.4 |
| no band more than 2× worse | C.2 and C.3: every band is better, by 7× to 300×. C.1 cubic, C.4 and weight averaging break it |
| speed claim: ≤ 1.08 um at 128 field + 64 net calls | not kept. Best at that cost: C.1 trees, 1.36 um (straight per-track), supervised |
| C.2/C.3 ≤ 0.54 um at ≤ 256 field calls | passes for all three runs: 0.145–0.147 um at 256 |

What the corrections show:

- **C.2 and C.3.** The sweep sets each stage to S0 + dz A f(network stages). The new stage error is dz × (rate error), which is very small. One sweep is enough: the second sweep and the Newton step change the endpoint by less than 0.002 um. After the correction the endpoint equals the exact scheme for every run, whatever the form and whatever its plain error. The lag-1 autocorrelation falls from 0.85 to −0.22, the value of the exact scheme.
- **Cost.** 256 field calls and 64 network calls. Classic RK4 uses 256 field calls and no network for 1.08 um. So at equal field calls the corrected network is 7× more accurate than RK4, but it is not cheaper.
- **C.1.** The trees remove up to 40% of the error of direct states (5.71 → 3.34 um) and 26% for the straight per-track run. They are supervised. The cubic model makes every run worse.
- **C.4.** The small network cannot learn the Newton change from the state: validation R² of x and y 0.03 to 0.08, of tx and ty -0.25 to 0.08 (A.1 finds the same for e). It makes every run worse.
- **B.1.** Weight averaging breaks the network: the last 5 snapshots are not in one basin. Output averaging gives 1.1× to 1.5×, under the 2× bar, at 5× the cost.

## C.2b Control: the same sweeps from stages without a network

The same sweeps and Newton steps, from six starts. Field calls per track = 64 × (calls of the start + 2 per sweep or Newton step + 2 for the summed end). Straight line: Y_i = (x + tx c_i h, y + ty c_i h, tx, ty), no field call. Euler: Y_i = S0 + c_i h f(S0), 1 call a step. RK4: classic RK4 from z0 to each stage plane, 8 calls a step. A Newton step uses the field gradient with its 2 calls; a call with the gradient costs more than a call without it.

Test endpoint radial median (um) and per band:

| Start | Method | k | Field calls | Net calls | Endpoint | < 3 | 3–8 | 8–20 | 20–50 | > 50 |
|---|---|---|---|---|---|---|---|---|---|---|
| network, straight per-track | sweeps | 0 | 128 | 64 | 1.831 | 4.94 | 2.10 | 1.44 | 1.21 | 1.89 |
| | sweeps | 1 | 256 | 64 | 0.147 | 0.66 | 0.25 | 0.085 | 0.035 | 0.011 |
| | sweeps | 2 | 384 | 64 | 0.147 | 0.66 | 0.25 | 0.085 | 0.035 | 0.012 |
| | Newton | 1 | 256 | 64 | 0.147 | 0.66 | 0.25 | 0.085 | 0.035 | 0.012 |
| network, straight global | sweeps | 0 | 128 | 64 | 2.039 | 7.42 | 3.21 | 1.31 | 1.30 | 0.65 |
| | sweeps | 1 | 256 | 64 | 0.147 | 0.66 | 0.25 | 0.085 | 0.035 | 0.012 |
| | Newton | 1 | 256 | 64 | 0.147 | 0.66 | 0.25 | 0.085 | 0.035 | 0.012 |
| network, direct states | sweeps | 0 | 128 | 64 | 5.705 | 21.67 | 8.75 | 3.54 | 2.96 | 3.24 |
| | sweeps | 1 | 256 | 64 | 0.145 | 0.67 | 0.25 | 0.087 | 0.035 | 0.011 |
| | Newton | 1 | 256 | 64 | 0.147 | 0.66 | 0.25 | 0.085 | 0.035 | 0.012 |
| straight line (no field) | sweeps | 0 | 128 | 0 | 7020 | 22369 | 10171 | 3690 | 1702 | 645 |
| | sweeps | 1 | 256 | 0 | 1.665 | 37.96 | 4.95 | 0.27 | 0.020 | 0.010 |
| | sweeps | 2 | 384 | 0 | 0.145 | 0.63 | 0.25 | 0.085 | 0.035 | 0.012 |
| | Newton | 1 | 256 | 0 | 0.133 | 0.57 | 0.23 | 0.085 | 0.035 | 0.012 |
| | Newton | 2 | 384 | 0 | 0.147 | 0.66 | 0.25 | 0.085 | 0.035 | 0.012 |
| Euler predictor | sweeps | 0 | 192 | 0 | 5.450 | 22.83 | 8.43 | 2.91 | 1.29 | 0.49 |
| | sweeps | 1 | 320 | 0 | 0.148 | 0.68 | 0.26 | 0.087 | 0.035 | 0.011 |
| | Newton | 1 | 320 | 0 | 0.147 | 0.66 | 0.25 | 0.085 | 0.035 | 0.012 |
| RK4 predictor | sweeps | 0 | 640 | 0 | 0.145 | 0.63 | 0.24 | 0.085 | 0.035 | 0.011 |
| | sweeps | 1 | 768 | 0 | 0.147 | 0.66 | 0.25 | 0.085 | 0.035 | 0.012 |

Every start reaches 0.147 um with 3 and 5 sweeps and with 2 Newton steps (full rows in `results/c2_control_endpoint.csv`).

Stage error of one step from the reference state, against the exact stages: median over steps of the largest stage error, position (um) / slope (urad).

| Start | k = 0 | 1 sweep | 2 sweeps | 3 sweeps | 1 Newton step |
|---|---|---|---|---|---|
| network, straight per-track | 0.221 / 1.34 | 0.0458 / 2.9e-4 | 8.3e-6 / 1.0e-6 | 2.0e-8 / 2.8e-10 | 5.7e-11 / 2.1e-9 |
| network, straight global | 0.885 / 2.11 | 0.0692 / 4.6e-4 | 1.2e-5 / 1.5e-6 | 2.9e-8 / 3.6e-10 | 1.1e-10 / 5.1e-9 |
| network, direct states | 10.6 / 9.4 | 0.308 / 2.7e-3 | 7.7e-5 / 6.2e-6 | 1.5e-7 / 2.0e-9 | 2.8e-9 / 1.2e-7 |
| straight line (no field) | 59.1 / 1850 | 59.1 / 0.476 | 0.0111 / 3.3e-4 | 5.6e-6 / 1.7e-7 | 5.9e-5 / 3.5e-3 |
| Euler predictor | 59.1 / 22.6 | 0.528 / 4.5e-3 | 7.7e-5 / 2.7e-6 | 2.5e-8 / 6.7e-10 | 6.4e-9 / 4.7e-7 |
| RK4 predictor | 0.0487 / 0.0357 | 6.9e-4 / 5.3e-6 | 2.1e-7 / 1.1e-8 | 3.7e-10 / 8.7e-12 | 0 / 0 |

The reading:

- A sweep cuts the slope error of the stages by a factor 1e-3 to 1e-4. The position error follows one sweep later, because a sweep builds the stage positions from the old stage slopes. From the straight line the first sweep therefore leaves the position error at 59 um; the second sweep removes it.
- The endpoint needs the stage slopes right, not the stage positions: 1 sweep from the network, with stage positions still 0.05 to 0.3 um off, already gives 0.147 um.
- At 256 field calls with sweeps only, the network matters: 0.147 um against 1.67 um from the straight line (11×). A network-free start reaches 0.147 um at 320 calls (Euler + 1 sweep) or 384 calls (straight line + 2 sweeps).
- At 256 field calls with a Newton step, the network adds nothing: one Newton step from the straight line gives 0.133 um (not converged, so lucky by 10%; under 2×, not a claim). The Newton step needs the gradient of the field.
- The Euler start with no sweep gives 5.45 um at 192 field calls and no network. That is as good as the direct-states network alone (5.71 um at 128 field calls + 64 network calls).

Figure: `figures/c2_control.png` (left panel: a point at 1e-9 um is an error of zero to rounding). Tables: `results/c2_control_endpoint.csv`, `results/c2_control_stage_error.csv`. Script: `control_c2.py`.

## B.3 and B.4 runs

Added 2026-11-05; `5cb2aeb6778b` final the same day. Four runs on the farm at commit `27a054e1`, submitted 2026-11-03. Each is at its last finished round, scored on test (1,452 tracks): radial median of the endpoint at z = 7826 mm, in um. Baselines: `49e5e2450eaa` (direct states, round 59) and `25dfa17ef168` (straight line per-track, round 44). Target: direct states <= 2.85 um, straight line <= 0.92 um, and no momentum band more than 2× worse than the baseline.

| Run | Form, item | Round | State | Endpoint | Target | Worst band / baseline | Mean signed x ± SE | Mean signed y ± SE | Lag-1 x / y |
|---|---|---|---|---|---|---|---|---|---|
| `49e5e2450eaa` | direct states, baseline | 59 | plateaued | 5.71 | | | 1.40 ± 1.46 | -0.39 ± 0.65 | 0.87 / 0.89 |
| `5cb2aeb6778b` | direct states, B.3 residual draws | 70 | plateaued | 7.78 | fails | 1.86 | 0.57 ± 0.75 | 0.16 ± 0.46 | 0.89 / 0.94 |
| `6aba11cc7475` | direct states, B.4 bias term | 51 | plateaued | 8.36 | fails | 1.69 | 0.24 ± 1.43 | 0.70 ± 0.61 | 0.82 / 0.88 |
| `25dfa17ef168` | straight line, baseline | 44 | plateaued | 1.83 | | | -0.03 ± 0.25 | 0.77 ± 0.46 | 0.85 / 0.90 |
| `8a0f3b8f422f` | straight line, B.3 residual draws | 30 | plateaued | 3.33 | fails | 1.90 | 0.34 ± 0.23 | 0.48 ± 0.35 | 0.90 / 0.91 |
| `f0f219df805e` | straight line, B.4 bias term | 45 | plateaued | 1.70 | fails | 1.66 | 0.03 ± 0.21 | 0.37 ± 0.29 | 0.87 / 0.90 |

Endpoint per momentum band (um):

| Run | < 3 GeV | 3–8 | 8–20 | 20–50 | > 50 |
|---|---|---|---|---|---|
| `49e5e2450eaa` baseline | 21.67 | 8.74 | 3.54 | 2.96 | 3.24 |
| `5cb2aeb6778b` B.3 | 19.92 | 10.71 | 6.05 | 5.51 | 5.50 |
| `6aba11cc7475` B.4 | 24.14 | 12.06 | 5.69 | 5.01 | 3.53 |
| `25dfa17ef168` baseline | 4.94 | 2.10 | 1.44 | 1.21 | 1.89 |
| `8a0f3b8f422f` B.3 | 7.39 | 3.63 | 2.60 | 2.30 | 2.41 |
| `f0f219df805e` B.4 | 4.84 | 2.10 | 1.20 | 1.15 | 3.13 |

Reading:

- No run meets its target. Each run is one seed, so a difference under 2× is not a claim.
- B.4 straight line (1.70) is equal to its baseline (1.83). It is better below 50 GeV by 0–17% and worse above 50 GeV (3.13 against 1.89).
- B.3 straight line (3.33) is 1.8× worse than its baseline and stopped earliest (round 30). B.4 direct states (8.36) is 1.5× worse. B.3 direct states (7.78) is 1.4× worse. At round 66 it gave 6.41; its validation score moves by up to 1.6× from round to round (2.89 to 4.66 um, rounds 63–70).
- The lag-1 autocorrelation stays at 0.82 to 0.93 in every run. Neither item makes the step errors less correlated along the track.
- Every mean signed endpoint error is within 2 SE of zero, as for the baselines.

Table: `results/b_part_b_scores.csv`. Log: `logs/part_b_scores.txt`. The standard reports of the four runs (`report_of_a_run --split test --round last`) are in the store, in `runs/<key>/reports/round_<n>_at_27a054e1/`; logs `logs/report_of_a_run_<key>.txt`. Run on `stbc-i2` (the `stbc-i3` login failed on 2026-11-05), package at `27a054e1`.

## Decision point

| Question | Answer | Evidence |
|---|---|---|
| Is the bias private or shared? | The position part is private. The slope part is shared, but it is the exact scheme's own error at the cell faces, not a network bias. | A.6, A.9 |
| Does it matter after a fit? | No. Mean bias ≤ 1.1% of the hits-only resolution in every band. | A.7 |
| Is it removable by a smooth model? | No. Best validation R² 0.43. | A.1, C.4 |
| Which Part B item does it point to? | The plan maps a private bias to B.1. B.1 weight averaging fails here. Output averaging and a seed ensemble give under 2× at M× the cost. The shared slope part is the scheme's own error, so B.3 and B.5 (aimed at cell faces) cannot improve on the exact scheme. No shared bias elsewhere, so B.4 has no target. | A.6, A.9, B.1 |
| What removes the error? | One fixed-point sweep of the stage equations at inference (C.2): every run gives the exact scheme, 0.147 um, at 256 field calls. | C.2, C.3 |
| Does the network matter for that? | With sweeps only, yes: it saves one sweep (0.147 um at 256 calls against 1.67 um from the straight line; a network-free start needs 320–384 calls). With one Newton step, no: the straight line gives 0.133 um at 256 calls. | C.2b |

By the plan's own rule, A.7 says stop and write up. The open choice is George's: accept 256 field calls with one sweep (exact-scheme accuracy for either form), or keep 128 field calls and accept the plain error.

## What is not done

- A.8 (2026-11-03, with `c3a1aa54c6fe` 2026-11-05): the network's own bias is private to each seed. The slope maps that look shared are the exact scheme's error.
- C.5 and the Kalman-filter question are moved out of this experiment to a separate thread (George, 2026-10-28).
- The UT and SciFi layout and resolutions in A.7 were not re-checked against the TDR.
