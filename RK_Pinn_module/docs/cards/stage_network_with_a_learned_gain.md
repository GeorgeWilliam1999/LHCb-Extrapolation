# Card: the stage network with a learned gain

**Date:** 2026-11-03 · **Source:** `src/rkpinn/networks/stage_network_with_a_learned_gain.py`, `src/rkpinn/losses/distilled_analytic_gain.py`, `src/rkpinn/integrators/linearised_gain.py` · **Gates:** `tests/test_learned_gain_and_error_conditioned_network.py`

## What it is

A trained stage network gives the q stage states of one step. A small network then reads the stage residual of those stages and corrects the end state of the step. The correction needs no new field lookup and no Jacobian at inference.

## The step

A step starts from the state S0 = (x, y, tx, ty, q/p) on the plane z0 and has the length dz. The Gauss–Legendre tableau has the nodes c, the stage matrix A and the weights b. The equation of motion gives the rates f(Y, z) of a state Y with four components.

1. The stage network gives the stages Y_1 … Y_q.
2. The rates at the stages are F_k = f(Y_k, z0 + c_k dz). These are q field lookups.
3. The summed end state is E = S0 + dz Σ_k b_k F_k.
4. The stage residual is r_j = Y_j − S0 − dz Σ_k A_jk F_k. It is free: F is already there.
5. The gain network G reads S0 divided by its scale, z0 mapped onto [−1, 1], and r divided by its root mean square s_r. It gives a matrix W of 4 × 4q numbers.
6. The correction is D_d = s_out,d Σ_m W_dm r_m / s_r,m, where s_out is the scale of the correction.
7. The end state is S_end = E − D.

A residual of zero gives a correction of zero, whatever the weights.

## The analytic gain it learns

The stage equations are G_j(Y) = Y_j − S0 − dz Σ_k A_jk f(Y_k) = 0. One Newton step moves the stages by dY = −M⁻¹ r, where M = I − dz (A ⊗ J) and J_k = ∂f/∂Y at Y_k (4 × 4, q/p fixed). If f at the new stages is replaced by F_k + J_k dY_k, the summed end state moves by dz Σ_k b_k J_k dY_k = −K r, with

    K = dz (bᵀ ⊗ J) M⁻¹,   a 4 × 4q matrix.

So W = diag(1/s_out) K diag(s_r) reproduces the analytic gain. K depends on the stages through J, and the stages are a function of S0 and z0. So G can represent K.

## The loss

    loss = mean over n, d of ((D[n, d] − (K r)[n, d]) / s[d])²

s[d] is the spread of component d over the first round's states. K r comes from the equation of motion and the stages alone. The loss needs q Jacobians a step in training. It needs no reference integrator and no label of a target.

## The Kalman analogy

| Kalman filter | Here |
|---|---|
| predicted state | the stages and the summed end state of the stage network |
| observation model | the stage equations G(Y) = 0 |
| innovation | the stage residual r |
| gain | K, or the learned W |
| update | S_end = E − K r |
| noise covariances | none: the gain is the linearised Newton step |

## Cost of one step

| | Inference | Training |
|---|---|---|
| field lookups | q | q |
| field samples of a per-track scale | 16 if the stage network has one | the same |
| Jacobians | 0 | q |
| linear solves | 0 | 1 (4q × 4q) |
| network calls | 1 stage network + 1 gain network | the same |

## Structural choices

- The correction is a matrix applied to r, not a free vector. This is the form of the KalmanNet update (arXiv:2107.10043). KalmanNet uses a recurrent network and is trained on labels; here the network is a feed-forward body and the label is K r.
- r enters G twice: as the vector W multiplies and as an input of G.
- The last layer of G starts at zero. A run then starts as the stage network alone.
- r and F are detached before they enter G.
