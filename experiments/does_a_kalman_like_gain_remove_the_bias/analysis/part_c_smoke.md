# Part C smoke run

**Date:** 2026-11-03 · **Code:** commit `2ccd1916` · **Machine:** `stbc-i2`, 1 torch thread · **Store:** a scratch store with links to the track set `12a8d35c3165` and the run `25dfa17ef168`; nothing was written to the project store.

## What was run

`part_c_smoke.sh`: C1a and C2, each on the frozen per-track straight-line network (`25dfa17ef168`, round 44), then the standard report on the test split.

| Setting | Smoke run | Farm runs (planned) |
|---|---|---|
| states per round | 4,000 | 32,000 |
| restarts per round × iterations | 5 × 100 | 25 × 200 |
| rounds | 3 (cap) | to the plateau, cap 60 |
| small network | 2 tanh layers of 64 | the same |

| Variant | Run key | Loss | Small network parameters |
|---|---|---|---|
| C1a, learned gain | `61aba2432fcc` | `distilled_analytic_gain` | 7,200 (of 25,640 with the frozen network) |
| C2, error-conditioned | `8a64dfa936fa` | `pooled_on_both_calls` | 9,280 (of 27,720) |

## The loss falls

| Variant | Round | Source of states | Loss at start | Loss at end | Validation (um, larger of x and y, median) |
|---|---|---|---|---|---|
| C1a | 1 | reference | 1.25e-12 | 1.60e-13 | 1.38 |
| C1a | 2 | own predictions | 8.14e-12 | 2.92e-13 | 1.40 |
| C1a | 3 | own predictions | 2.88e-12 | 2.48e-13 | 1.39 |
| C2 | 1 | reference | 4.41e-09 | 2.25e-09 | 1.27 |
| C2 | 2 | own predictions | 1.84e-07 | 1.20e-08 | 1.31 |
| C2 | 3 | own predictions | 2.39e-08 | 3.67e-09 | 1.26 |

The frozen network alone has 1.71 um on the same validation quantity. The checks before training passed: the loss at the exact scheme's stages is 2.3e-37 (C1a) and 3.6e-34 (C2).

The loss rises at the start of round 2. Round 2 trains on the states the network itself predicts along the training tracks, which are further from the reference than the states of round 1.

## The report runs

Test split, 1,452 tracks, endpoint radial median (um):

| Model | All | < 3 | 3–8 | 8–20 | 20–50 | > 50 | Single-step x, median of absolute (um) |
|---|---|---|---|---|---|---|---|
| frozen network alone | 1.83 | 4.94 | 2.10 | 1.44 | 1.21 | 1.89 | 0.0515 |
| C1a after 3 smoke rounds | 1.45 | 3.69 | 1.54 | 1.19 | 1.12 | 1.83 | 0.0297 |
| C2 after 3 smoke rounds | 1.31 | 4.12 | 1.30 | 1.11 | 1.05 | 1.80 | 0.0278 |
| analytic gain (Part B) | 0.147 | 0.66 | 0.249 | 0.0852 | 0.0353 | 0.0115 | — |

Counts per step, from the report's `cost.csv`:

| Variant | Field lookups | Field samples of the per-track scale | Jacobians | Solves | Stage network calls | Small network calls | Jacobians in training |
|---|---|---|---|---|---|---|---|
| C1a | 2 | 16 | 0 | 0 | 1 | 1 | 2 |
| C2 | 4 | 16 | 0 | 0 | 1 | 1 | 0 |

Time to carry one test track: 1,395 us (C1a) and 1,448 us (C2), on a loaded login node.

## Time per restart

| Variant | Seconds per restart of 100 iterations on 4,000 states |
|---|---|
| C1a | 3 to 7 |
| C2 | 4 to 7 |

## What this shows and what it does not

- Both variants train: the loss falls in every round, by 4× to 15× within a round.
- Both reports hold the standard tables and the counts.
- Three rounds on 4,000 states are far from a plateau. In round 1 the loss fell 7.8×, so the error of the correction against K r fell about 2.8×. The analytic gain needs the correction to match K r closely to reach 0.147 um. So these numbers say nothing about what the farm runs reach.
