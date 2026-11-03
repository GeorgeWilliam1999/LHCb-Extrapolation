# Card: the error-conditioned stage network

**Date:** 2026-11-03 · **Source:** `src/rkpinn/networks/error_conditioned_stage_network.py`, `src/rkpinn/losses/pooled_on_both_calls.py` · **Gates:** `tests/test_learned_gain_and_error_conditioned_network.py`

## What it is

A trained stage network is called once. A second network, the corrector, reads the stage residual of that call and corrects the stages. The end state is summed from the corrected stages.

## The step

The symbols are those of the card of the stage network with a learned gain.

1. First call: the stages Y_k, their rates F_k (q lookups) and the residual r.
2. The corrector C reads S0 divided by its scale, z0 mapped onto [−1, 1], and r / s_r: 6 + 4q inputs. It gives a matrix W of 4q × 4q numbers.
3. The correction is dY_(k,d) = s_out,d Σ_m W_(k,d),m r_m / s_r,m.
4. The second stages are Y' = Y + dY. Y and r are detached.
5. The rates at the second stages are F'_k = f(Y'_k, z0 + c_k dz): q more lookups.
6. The end state is S_end = S0 + dz Σ_k b_k F'_k.

With W = −diag(1/s_out) M⁻¹ diag(s_r) the correction is one Newton step of the stage equations.

## The loss

    loss = mean over the 2q stage terms of (r' / s)²

r' is the stage residual of the second stages and of the first stages. It is the mean of two pooled losses. With a frozen stage network the second half is a constant.

## Cost of one step

| | Inference |
|---|---|
| field lookups | 2q |
| Jacobians, solves | 0 |
| network calls | 1 stage network + 1 corrector |

## How it differs from the error-conditioned neural solvers (arXiv:2606.27354)

| The paper | Here |
|---|---|
| trained on labels; no residual loss | trained on the residual only |
| corrector gives a free field, times a fixed step size, applied 5 times | corrector gives a matrix applied to r, once |
| corrector also reads the current solution | reads the state and the plane |
| predictor and corrector trained jointly | the stage network is frozen by default; joint training is a setting |
| the input of each correction step detached | the same |
