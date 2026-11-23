# Results: where and how A1 fails

**Last updated:** 2026-11-23 · **State:** computed; for discussion with George. Not a write-up.

## What was done

- A1 (`3ccd44d61cd8`) is at round 37. A1 was still training; round 37 was its latest saved round at 23:00 on 23 Nov. A2 (`0b41ecdd2505`) is at its best round by validation, round 36.
- Parts 1 to 3 use all 13,447 cubic test tracks of `5558be31abeb`. Parts 4 to 6 use 2,000 random test tracks (seed 0). Part 4 also uses the 4,096 first-round training states.
- The reference is the sixth-order reference on the 65 planes. The exact scheme is the 64-step, 2-stage solution in the store.
- `compute.py` makes the arrays in `/data/bfys/gscriven/rkpinn_analysis_cache/failure_of_A1/`. `analyse.py` makes `figures/` and `tables/`.

Checks:
- The exact stage solve reaches a scaled residual below 1e-12 on every step.
- The local defect in x is h times the mean stage error in tx, to 1e-4 µm.
- The linear carry of the local defects gives the network-minus-exact endpoint with a correlation of 0.996.

## The endpoint

Endpoint radial error against the reference.

| Tracks | A1 median (µm) | A1 p95 (µm) | A2 median (µm) | A2 p95 (µm) | Exact median (µm) |
|---|---|---|---|---|---|
| all 13,447 | 509.1 | 3022 | 565.9 | 3231 | 0.0183 |
| old 1,452 | 495.0 | 2875 | 522.3 | 3163 | 0.0178 |

## 1. Error along the track

Median absolute error at every eighth plane. Figures: `figures/1_error_along_the_track.png` and `figures/1b_signed_error_along_the_track.png`.

| Plane | z (mm) | median \|B\| (T) | A1 \|dx\| (µm) | A1 \|dy\| (µm) | A1 \|dtx\| (µrad) | A1 \|dty\| (µrad) |
|---|---|---|---|---|---|---|
| 8 | 3295 | 0.52 | 8.8 | 8.6 | 0.022 | 0.020 |
| 16 | 3943 | 0.89 | 21.4 | 41.3 | 0.091 | 0.103 |
| 24 | 4590 | 1.05 | 35.6 | 99.6 | 0.20 | 0.26 |
| 32 | 5237 | 1.01 | 54.9 | 167 | 0.69 | 0.45 |
| 40 | 5884 | 0.88 | 81.6 | 234 | 1.29 | 0.61 |
| 48 | 6532 | 0.70 | 111 | 298 | 2.07 | 0.63 |
| 56 | 7179 | 0.49 | 146 | 352 | 2.70 | 0.50 |
| 64 | 7826 | 0.30 | 175 | 382 | 2.81 | 0.45 |

- The error is a smooth drift. It has no kink and it does not start from one point. It grows from the first step and grows fastest in the strong field, from plane 16 to plane 48.
- The signed median is near zero at every plane, for every component. The error is a spread from track to track, not a bias.
- The position error is large and the slope error is small. At the last plane, \|dy\| is 382 µm but \|dty\| is only 0.45 µrad.
- A2 is the same shape and 10 to 30 % larger.

## 2. Who fails

Figure: `figures/2_who_fails.png`. Table: `tables/2_endpoint_by_group.md`.

| Group | Tracks | A1 median (µm) | A1 p90 (µm) | A2 median (µm) | Exact median (µm) |
|---|---|---|---|---|---|
| p < 3 GeV | 1582 | 1495 | 3899 | 1723 | 0.086 |
| p 3-8 GeV | 6025 | 686 | 2073 | 814 | 0.036 |
| p 8-20 GeV | 3974 | 274 | 914 | 309 | 0.0085 |
| p 20-50 GeV | 1573 | 247 | 603 | 305 | 0.0024 |
| p > 50 GeV | 293 | 893 | 4359 | 896 | 0.00087 |
| charge + | 6806 | 492 | 1952 | 578 | 0.018 |
| charge − | 6641 | 532 | 2157 | 554 | 0.018 |
| \|tx\| start, lowest quarter | 3362 | 244 | 1333 | 273 | 0.0078 |
| \|tx\| start, highest quarter | 3362 | 1097 | 3220 | 1237 | 0.043 |
| \|ty\| start, highest quarter | 3362 | 1148 | 3986 | 1349 | 0.062 |
| \|x\| start, highest quarter | 3362 | 1033 | 3060 | 1169 | 0.044 |
| \|y\| start, highest quarter | 3362 | 1067 | 3695 | 1243 | 0.063 |

- From 2 to 20 GeV the error falls about as 1/p. From 20 to 50 GeV it is flat at about 250 µm, with dips near 15 and 28 GeV.
- Above 50 GeV the error rises again, to 1 to 4 mm at 100 GeV. These nearly straight tracks fail. They are 2.2 % of the test tracks.
- Both charges fail equally.
- Large start angles and large start positions fail most: the error in the highest quarter is 4 to 5 times the error in the lowest quarter.

The bend-strength test. The fit is dx_end = a·(q/p) + b.

| Run | Component | a (µm GeV) | R² | median \|d\| (µm) | median \|d − fit\| (µm) |
|---|---|---|---|---|---|
| A1 | x | −698 | 0.0006 | 174.8 | 179.1 |
| A1 | y | −1001 | 0.0016 | 382.4 | 381.3 |
| A2 | x | −487 | 0.0003 | 222.7 | 222.4 |
| A2 | y | 710 | 0.0007 | 409.6 | 414.2 |

- A bend-scale error explains less than 0.2 % of the endpoint error. A fit to the bend of each reference track gives the same result (R² 0.0007). The error is not a wrong field strength or a wrong q/p.

## 3. Position against slope

Figure: `figures/3_position_against_slope.png`.

| Run | Component | Correlation of d_end with ∫ dt dz | Share of \|d_end\| from ∫ dt dz |
|---|---|---|---|
| A1 | x | 0.80 | 0.095 |
| A1 | y | 0.21 | 0.014 |
| A2 | x | 0.78 | 0.084 |
| A2 | y | 0.23 | 0.012 |

- The plane slope error, carried over the track, gives only 10 % of dx_end and 1 % of dy_end.
- The position error comes from the stage slopes. In the summed chain, x_{n+1} = x_n + h Σ b_i tx(S_ni): the slope of the emitted stages moves the position directly. The local defect in x is h times the mean stage error in tx (checked to 1e-4 µm).
- Per step, A1 has a median stage error of 47.8 µrad in tx and 69.3 µrad in ty. The two stages of one step have nearly the same error (correlation 0.96 in tx, 0.995 in ty), so the errors add in the sum. This gives a local defect of 3.5 µm in x and 5.6 µm in y per step.
- The same stage error changes the plane slope by only 0.028 µrad per step, because the slope rate depends on the slope only through the small term κ (q/p) B. So the plane slopes are nearly right, but the plane positions are not.

## 4. Where the training lands

Figure: `figures/4_where_the_training_lands.png`. Table: `tables/4_shares_by_quarter.md`.

Share of each quantity in each quarter of the 64 steps (A1, training states):

| Quantity | Steps 0-15 | 16-31 | 32-47 | 48-63 |
|---|---|---|---|---|
| loss L_n | 0.019 | 0.183 | 0.309 | 0.489 |
| \|∂loss/∂outputs_n\|² | 0.024 | 0.370 | 0.296 | 0.311 |
| \|∇θ L_n\| | 0.141 | 0.243 | 0.167 | 0.449 |
| carried local defect² (endpoint) | 0.009 | 0.228 | 0.388 | 0.375 |

Loss per step, L_n:

| Step n | A1 | A2 | P1 (linear map, own 1,452 tracks) |
|---|---|---|---|
| 0 | 6.2e-7 | 7.8e-7 | 2.3e-7 |
| 16 | 6.0e-6 | 4.7e-6 | 1.1e-6 |
| 32 | 2.4e-5 | 2.8e-5 | 4.0e-6 |
| 48 | 2.4e-5 | 2.6e-5 | 4.3e-6 |
| 63 | 7.9e-5 | 1.0e-4 | 2.7e-5 |
| mean | 2.1e-5 | 2.4e-5 | 5.3e-6 |

- The loss is in the second half of the track: 80 % in steps 32 to 63. The last step alone has 6 %.
- The endpoint error is made in steps 16 to 63, with most in steps 32 to 47 (39 %). Steps 0 to 15 make 1 % of it.
- The loss and the endpoint agree in the first half. In the last quarter the loss gives more weight (49 %) than the endpoint needs (37 %). The match is close; the weighting of the steps is not the main fault.
- In A1, the gradient through the plane states moves weight to steps 16 to 31: 37 % of the output gradient, against 18 % of the loss. In A2 the output gradient follows the loss exactly. This does not help A1 much: A1 is 10 to 20 % better than A2.
- The gradient of each step's loss points in a different direction. The cosine with the total gradient is 0.03 to 0.35. The steps compete for the same parameters.
- The carried defects of one track have the same sign over many steps. The median coherence ratio in x is 4.6 (8 for 64 equal defects, 0.67 for random signs). P1 had 2.4.
- A1 and A2 have a loss 4 to 5 times the loss of P1, at every step.

## 5. Does a small residual mean a small error?

Figure: `figures/5_residual_against_error.png`.

| Run | Component | median \|r\| | median \|ε\| | median \|ε\|/\|r\| | p99 \|ε\|/\|r\| |
|---|---|---|---|---|---|
| A1 | x (µm) | 72.2 | 72.3 | 1.00 | 2.3 |
| A1 | y (µm) | 9.8 | 11.2 | 1.02 | 19 |
| A1 | tx (µrad) | 47.8 | 47.8 | 1.00 | 1.03 |
| A1 | ty (µrad) | 69.3 | 69.3 | 1.00 | 1.005 |

- Yes. The stage residual r equals the stage error ε at every step: the median ratio is 1.00. No residual floor hides the error.
- Per track, the mean loss predicts the endpoint error: Spearman 0.91 (A1) and 0.90 (A2).
- So the loss tells the truth. It is only too large: the stage errors are 50 to 70 µrad in slope.

## 6. One Newton sweep

2,000 test tracks; the exact scheme on the same tracks has a median of 0.0178 µm and a p95 of 0.130 µm.

| Run | Sweeps | Median against the reference (µm) | p95 (µm) | Median against the exact scheme (µm) |
|---|---|---|---|---|
| A1 | 0 | 514.8 | 2974 | 514.8 |
| A1 | 1 | 0.0299 | 0.795 | 0.0096 |
| A2 | 0 | 563.9 | 3283 | 563.9 |
| A2 | 1 | 0.0308 | 0.938 | 0.0102 |

- One sweep takes A1 to 1.7 times the exact scheme in the median. The p95 is 6 times the exact p95.

## Summary

What A1 gets:
- The plane slopes. At the last plane the slope error is 2.8 µrad in tx and 0.45 µrad in ty (median).
- The first 16 steps (8 to 21 µm in x at plane 16).
- Tracks of 8 to 50 GeV with small start angles (about 250 µm).
- A start for the Newton sweep: one sweep gives 0.030 µm.

What A1 misses:
- The positions. The stage slopes are wrong by 50 to 70 µrad. The two stages of a step have the same error, and the summed chain adds h times this error to the position at every step. The defects keep their sign over many steps, so they add up to 175 µm in x and 382 µm in y.
- Low-momentum tracks (1.5 mm below 3 GeV), large-angle tracks, and the nearly straight tracks above 50 GeV (0.9 mm median, 4.4 mm p90).
- The strong-field region, steps 16 to 47, where 62 % of the endpoint error is made.

The most likely cause:
- The network is not precise enough in its stage outputs. It emits 512 numbers per track. The stage error in slope is about 7e-4 of the spread of the slopes (48 µrad against 0.071). This is the precision that a 2-layer, 256-wide tanh network reaches with L-BFGS. To reach 197 µm the stage error must fall by about 2.6 times. To reach the exact scheme it must fall by about 3e4 times.
- The loss is not the problem. It measures the stage error without a floor, and it puts its weight about where the endpoint error is made.
- The stage error is a smooth error of the whole output curve. It is common to both stages and keeps its sign over many steps. This is the error of a function fit, not of the stage equations.
- The failure above 50 GeV is not explained. These tracks are 2 % of the test tracks and sit at q/p near zero, where the network must fit a very small bend. The cause must be checked; one candidate is the per-track scale of the outputs.

## Not done

- No run at another width, depth or number of training states, to test the precision limit.
- The failure above 50 GeV is not traced to a cause.
- Parts 4 to 6 use 2,000 tracks, not all 13,447.
