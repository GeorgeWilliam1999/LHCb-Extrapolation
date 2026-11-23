# Can a label-free path network reach the exact scheme?

**Last updated:** 2026-11-20 · **State:** stage 0 done on branch `label_free_path`; no run submitted · **Agreed with George:** 2026-11-20 (the plan "Plan: a label-free path network")

## What this directory is

An experiment of the package `rkpinn`. One network call emits every Gauss-Legendre stage state of the crossing. The plane states are summed from those stages. The loss is the stage residual of every step, with the chaining inside the loss. No label enters the loss.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file: the question, the design, the runs, the criteria, the stage 0 numbers | built |
| `configurations/` | one configuration file per run | built |
| `farm/` | what puts the runs on the farm and keeps them going | built; not submitted |
| `analysis/` | tables, figures and the results note, made after the runs end | planned |

## The question

Can one network call emit every Gauss-Legendre stage of the crossing, trained on the stage residual alone, and reach the exact scheme without a Newton sweep?

## Why it is asked

The path network of `../does_a_one_shot_path_network_remove_the_chain/` learns from exact-scheme labels. It is supervised. This experiment removes the labels.

A network with a loss of zero solves G(U) = 0, the system that the Newton sweep solves. Its endpoint error is then the error of the exact scheme.

## The design

The network emits the stages S_ni of the steps n = 0 to 63 and the stages i = 1, 2. The loss builds the plane states from the stages and scores the stage equations of every step:

```
s_{n+1} = s_n + h sum_i b_i f(z_ni, S_ni)
R_ni    = S_ni - s_n - h sum_j a_ij f(z_nj, S_nj)
L_n     = mean over the tracks, i and d of (R_ni,d / sigma_d)^2
loss    = sum_n w_n L_n / sum_n w_n
```

where s_0 is the start state on the first plane, h the step length (80.9 mm), a and b the stage matrix and the weights of the tableau, z_ni the plane of stage i of step n, f the rates of the equation of motion, and sigma_d the spread of component d over the first round's states.

| Block | Setting | Why |
|---|---|---|
| Network | `label_free_path_network`: 2 tanh layers of width 256, 5 inputs, 512 outputs (128 stages x 4) | the path network without its plane-state outputs |
| Output form | straight line over the whole region, scale per track | the form that was best on the chain |
| Field map | `v8r1_up_cubic` | George's ruling of 2026-11-20 |
| Tracks | `5558be31abeb`: 108,915 / 13,684 / 13,447 train / validation / test | the cubic track set; the 1,452 old test tracks are a subset and are scored too |
| Loss | `stage_residual_of_a_path` | the equations above |
| Training states | start states of the training tracks, 4,096 drawn each round | see the timing below; the plan said 16,384 |
| Optimiser | L-BFGS with restarts, 200 iterations, 5 restarts a round, cap 40 rounds | as in the supervised path runs |
| Stopping | validation plateau: window 10, tolerance 5 %, held 3 | the standard rule |

The weights w_n:

| Weights | Rule |
|---|---|
| uniform | w_n = 1 |
| sliding window | w_n = 1 / (1 + exp((n - c) / tau)), tau = 2 steps. c = 4 (1 + the number of leading blocks of 4 steps whose mean L_k is below a threshold) |
| causal (Wang, Sankaran and Perdikaris, arXiv 2203.07404) | w_n = exp(-eps sum_{k<n} L_k). eps is set at round one so that the last step has a given weight |

The weights carry no gradient. They are held fixed for one restart of L-BFGS, so that each restart minimises a fixed objective.

## The runs

| Run | Stage | Weights | Gradient through s_n | Configuration | Key |
|---|---|---|---|---|---|
| A1 | A | uniform | yes | `label_free_path_uniform.yaml` | `3ccd44d61cd8` |
| A2 | A | uniform | no (detached) | `label_free_path_uniform_detached.yaml` | `0b41ecdd2505` |
| B1 | B | sliding window | yes | `label_free_path_sliding_window.yaml` | `4cb5f4821e9f` |
| B2 | B | causal, last weight 0.1 | yes | `label_free_path_causal_last_weight_0_1.yaml` | `332072fbffa9` |
| B3 | B | causal, last weight 0.001 | yes | `label_free_path_causal_last_weight_0_001.yaml` | `07a3299ece5e` |
| C1 | C | uniform + drift term `endpoint_sum` | yes | `label_free_path_drift_endpoint_sum.yaml` | `450fff13e800` |
| C2 | C | uniform + drift term `block_sums` (4, 8, 16, 32, 64 steps) | yes | `label_free_path_drift_block_sums.yaml` | `26c5a53de1bc` |
| C3 | C | uniform + drift term `newton_endpoint` (J detached) | yes | `label_free_path_drift_newton_endpoint.yaml` | `02d0d81b399b` |

Stage B runs only if criterion 1 passes. Stage C (2026-11-23, approved by George the same day) adds to A1's loss a term that sees how the step errors add along the track; λ is set so the term is half the loss of the first round's states. Why: `analysis/failure_of_A1/results.md`; the terms measured first: `analysis/loss_terms_check/`. One seed per run. Differences under 2x are not claimed.

The threshold of B1 (`settled_below`, now 1e-10) is provisional. It is set from the loss scale of stage A before B1 is submitted. Its key then changes.

## The criteria, written before the runs

All errors are the endpoint radial median on the 13,447 cubic-map test tracks against the sixth-order reference. The 1,452-track subset is reported beside it.

| # | Criterion | What it shows |
|---|---|---|
| 1 | The best stage A run alone is below 197 µm | the label-free loss trains at least as well as the labels did |
| 2 | The best run alone is at or below 1.83 µm | one call matches 64 chained calls |
| 3 | The best run alone is within 2x of the exact scheme on the cubic map | the network alone reaches the scheme; no sweep is needed |
| 4 | The best run after one sweep is at the exact scheme | the label-free start is as good a sweep start as the supervised one |
| 5 | A stage B run beats A1 by more than 2x | the curriculum helps |

## Stage 0: the numbers before training

Measured on stbc-i2, 2026-11-20, one or two threads, no training.

### The network-free references on the cubic map

Endpoint radial median against the sixth-order reference.

| Method | All 13,447 test tracks (µm) | A random 1,452 of them (µm) | The old 1,452 test tracks (µm) | On the linear map, old 1,452 (µm) |
|---|---|---|---|---|
| exact scheme, 64 steps, 2 stages | not computed | 0.01763 | 0.01784 | 0.147 |
| RK4, 64 steps | 0.02435 | — | 0.02421 | 1.11 |

The exact scheme costs 1.7 s a track on one thread (2,415 s for 1,452 tracks), so the full 13,447 tracks need about 6.3 h. That number is left to the scoring of the runs. The random 1,452 tracks (seed 0) stand in for it.

The old 1,452 test tracks of `12a8d35c3165` are all in the cubic test split, found by (event, particle in event). Their start states agree to 8e-6 mm and 1e-6 in slope.

On the cubic map both references are about 8 (exact scheme) and 46 (RK4) times smaller than on the linear map. Criterion 3 therefore asks for about 0.035 µm.

### The supervised path network P1 under this loss

P1 is run `25934ed22604`: linear map, track set `12a8d35c3165`, round 40, the 1,452 test tracks. Its stages are scored with the plane states summed from them, not with its own emitted plane states. Uniform weights.

| Step n | L_n |
|---|---|
| 0 | 2.29e-7 |
| 8 | 4.75e-7 |
| 16 | 1.08e-6 |
| 24 | 2.34e-6 |
| 32 | 4.00e-6 |
| 40 | 6.11e-6 |
| 48 | 4.35e-6 |
| 56 | 1.24e-5 |
| 63 | 2.73e-5 |
| mean over the 64 steps | 5.31e-6 |

### The cost of a run

One loss and gradient of A1 on one thread:

| States a round | Seconds a loss and gradient | Peak memory (GB) | Hours a restart (200 iterations) | Hours a run (40 rounds x 5 restarts) |
|---|---|---|---|---|
| 4,096 | 2.5 to 3.3 | 7.9 | 0.14 to 0.19 | 29 to 38 |
| 8,192 | 5.4 to 9.3 | 13.4 | 0.31 to 0.54 | 62 to 107 |
| 16,384 | 12.2 to 15.3 | 24.4 | 0.70 to 0.88 | 141 to 177 |

An iteration costs 1.04 evaluations (P1's restarts). The validation carry of 13,684 tracks takes 6 s a round.

The run at 16,384 states does not fit in 48 h. The configurations use 4,096 states a round.

### The checks before training

| Run | Loss at the exact scheme | Loss on the first round's states |
|---|---|---|
| A1 | 2.0e-33 | 1.364 |
| A2 | 2.0e-33 | 1.364 |
| B1 | 2.0e-33 | 0.193 |
| B2 | 2.0e-33 | 0.852 (eps 0.0273) |
| B3 | 2.0e-33 | 0.490 (eps 0.0818) |

## What is added to the package

| File | What |
|---|---|
| `networks/label_free_path_network.py` | the network |
| `losses/stage_residual_of_a_path.py` | the loss and its three weights |
| `losses/stage_residual_of_a_path_with_a_drift_term.py` | stage C: the loss plus one of three drift terms (2026-11-23) |
| `tests/test_drift_term.py` | its gates |
| `training/round_trainer.py` | calls `start_a_restart()` before every restart, on a loss that has it |
| `tests/test_label_free_path.py` | the gates |

## Findings

None yet.
