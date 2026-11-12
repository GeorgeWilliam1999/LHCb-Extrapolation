# Results: how does the stage residual become the endpoint error

**Last updated:** 2026-11-12 · **State:** all 11 runs computed; for discussion with George. Not a write-up.

Test split, 1,452 tracks, each run at its last round, along the network's own chain. The exact carried defects use the first 200 tracks. Numbers are in `results/per_run.csv` and `results/per_band.csv`; the figure is `figures/step_length.png`.

## Checks of the computation

- The exact stage solve reaches a scaled residual below 1e-12 on every step of every run.
- The exact carried defects sum to (network endpoint − exact-chain endpoint) exactly: the difference is 0.0 mm on the smoke run.
- The endpoint error of every run equals its standard report (1.83, 2.04, 5.71, 0.59, 8.97 and 623 µm), and the exact chain gives 0.147 µm at 64 steps.

## The chain, run by run (64 steps of 2 stages)

Medians. Stage error and defect are in x and y.

| Run | Endpoint against the reference (µm) | Endpoint against the exact chain (µm) | Stage error (µm) | Local defect (µm) | Coherence ratio, x / y |
|---|---|---|---|---|---|
| straight line, per-track scale | 1.83 | 1.86 | 0.239 | 0.0725 | 2.37 / 2.49 |
| straight line, global scale | 2.04 | 2.00 | 0.952 | 0.116 | 1.38 / 1.91 |
| direct states | 5.71 | 5.73 | 15.2 | 0.486 | 0.86 / 2.49 |
| Broyden, straight line | 0.59 | 0.531 | 0.0588 | 0.0226 | 1.77 / 1.58 |
| pilot, summed | 8.97 | 8.99 | 19.8 | 0.516 | 1.32 / 1.62 |
| pilot, predicted | 623 | 623 | 16.8 | 11.8 | 1.55 / 2.14 |

The coherence ratio is |Σ c| / √(Σ c²) over the steps of a track, c the exact carried defect. Its value is √N = 8 for 64 equal defects of one sign, and 0.67 in the median for random signs.

## Step length (straight line, per-track scale, 2 stages)

| Steps | Step length (mm) | Stage error (µm) | Stage error / step length (1e-6) | Local defect (µm) | Endpoint against the exact chain (µm) | Coherence ratio, x / y | Coherence / √N |
|---|---|---|---|---|---|---|---|
| 8 | 647.2 | 4.37 | 6.75 | 3.98 | 13.2 | 0.84 / 0.82 | 0.30 |
| 16 | 323.6 | 1.82 | 5.61 | 1.11 | 7.20 | 1.05 / 1.29 | 0.26 |
| 32 | 161.8 | 0.613 | 3.79 | 0.294 | 3.80 | 1.28 / 1.86 | 0.23 |
| 64 | 80.9 | 0.239 | 2.96 | 0.0725 | 1.86 | 2.37 / 2.49 | 0.30 |
| 128 | 40.5 | 0.113 | 2.80 | 0.0201 | 1.12 | 3.51 / 3.68 | 0.31 |
| 256 | 20.2 | 0.0447 | 2.21 | 0.00507 | 0.635 | 5.86 / 5.84 | 0.37 |

Fitted exponents against the step length: stage error 1.33, local defect 1.93, endpoint against the exact chain 0.88.

## Against the criteria

| # | Claim | Outcome |
|---|---|---|
| 1 | ε = (I − dz A⊗J)⁻¹ r to first order | **holds**, far inside the limit: median relative difference 3e-10 to 8e-8 in every run. The stage equations are linear at the size of the stage errors |
| 2 | ε ≈ r | **holds from 32 steps on** (6.4 %, 3.2 %, 1.6 %, 0.8 % at 32, 64, 128, 256 steps; 1.5 to 3.2 % in the other 64-step runs). **Fails at 8 and 16 steps** (24.9 % and 12.9 %): there dz A J is not small |
| 3 | δ = dz Σ b J ε for a summed end state | **holds**: median relative difference 1e-8 to 5e-7 |
| 4 | the linear carry reproduces the carried defect | **holds**: correlation 0.984 to 0.999 in x and 0.998 to 1.000 in y; the sums differ by 0.3 to 2.0 % |
| 5 | the defects are coherent (ratio above √N / 2) | **fails**: 0.86 to 2.5 at 64 steps against the limit 4. The defects are partly coherent: 0.23 to 0.37 of √N at every step length |
| 6 | a predicted end state is not first order in its stage error | **holds**: δ is 31.9 times dz J ε for the predicted run, 1.00 for the summed runs. With the same stage error (16.8 against 19.8 µm), its defect is 23 times larger (11.8 against 0.516 µm) |
| 7 | fixed relative precision | **half holds**: the defect falls with exponent 1.93 (inside 1.7 to 2.3). The stage error over the step length is not constant: it falls by a factor 3.1 from 8 to 256 steps (exponent 1.33, not 1) |

## What the numbers say

1. **The loss measures the stage error.** From 32 steps on, r and ε agree to a few per cent. The map from ε to the local defect is linear and exact. So nothing is hidden between the loss and the defect of one step; the summed form has no other error source.
2. **Summing gains by the factor dz J.** In the direct-states run the stage error is 15.2 µm and the defect is 0.486 µm, a factor 31. In the predicted run the defect is the size of the stage error. This is the cause of 8.97 against 623 µm.
3. **The defects are coherent over a fixed length of track, not over a fixed number of steps.** The coherence ratio is about 0.3 √N at every step length from 8 to 256 steps. This is what a sum of about 1 / 0.3² ≈ 11 independent segments per track gives, whatever the step count. It agrees with the per-step bias study (lag-1 autocorrelation 0.85 to 0.90, lag-8 near zero, 5 to 10 sign changes per track).
4. **This explains the step-length law.** The endpoint is about (N / √11) defects. With the defect ∝ dz^1.93 and N ∝ 1/dz, the endpoint ∝ dz^0.93. The measured exponent is 0.88.
5. **What limits the network.** The endpoint is the product of three factors: the stage error ε, the factor dz J, and the number of steps over √(independent segments). Training lowers the first. A smaller step lowers the second but raises the third. The third is the one no run has attacked: if the defect changed sign from step to step, the 64-step endpoint would fall by about 0.3 √64 / 0.67 ≈ 3.5.

## Not done

- No cubic field map yet: all numbers are on the linear map.
- The coherence is measured in x and y only, not per momentum band beyond `per_band.csv`.
- No test on van der Pol.
