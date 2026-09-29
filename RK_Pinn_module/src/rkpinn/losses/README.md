# Loss functions

**Last updated:** 2026-09-29

## What this directory is

The stage residual, and the losses built on it. All label-free losses are the mean of the squared residual over a divisor; they differ only in the divisor.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file | built |
| `__init__.py` | marks the directory as part of the package; it holds no code | built |
| `stage_residual.py` | each stage state, minus the step length times the stage matrix applied to the stage rates, minus the input state; and the same for a predicted end state | built |
| `unweighted.py` | divisor 1: the baseline with no weighting | built |
| `pooled.py` | divisor per component: the spread of that component over the first round's states. Equation 19 of the second mini-paper | built |
| `cost_weighted.py` | divisor per track, stage plane and component: lever arm, track bend, momentum window, clamp. Equation 20 | built |
| `supervised_endpoint.py` | mean squared difference between a predicted end state and a target end state, over a divisor per component | built |

## Settings of a loss

Every setting is an argument. None has a default.

| Setting | Values | Ruled |
|---|---|---|
| `terms` | `stages`: the q stages. `stages_and_end_state`: the q stages and the end state, as equations 19 and 20 are written | both kept: it depends on the experiment (George, 2026-09-28) |
| cost-weighted: `momentum_window_gev`, `roll_off`, `floor`, `clamp`, `samples_of_the_field_integral` | numbers | the window of the frozen runs was 10 to 50 GeV, then 3 to 8 GeV |
| cost-weighted: `lever_arm_is_on`, `track_bend_is_on`, `momentum_window_is_on` | yes or no; a factor that is off is 1 | |
| cost-weighted: `reference_bend_mm` | `median_of_first_round_states`, or a number | |

How `terms` meets the way the end state is formed:

| End state | Terms | |
|---|---|---|
| predicted | `stages_and_end_state` | the loss of the paper |
| predicted | `stages` | refused: the end state would be an output that nothing trains |
| summed | `stages` | the loss of the summed form |
| summed | `stages_and_end_state` | allowed. The term of the end state is zero to rounding, so it only divides the mean by q + 1 instead of q |

## Where each file was ported from

| File | Ported from | Gate |
|---|---|---|
| `stage_residual.py`, `pooled.py` | `reconstruction_residuals` and `physics_loss` in `single_network_chain_discrete_approach/_shared/model.py` | identical to the last bit, value and gradient |
| `cost_weighted.py` | `Block_F_reweighted_loss/F0_Weighting/weighted_loss.py` and `Block_G_low_momentum_window/G0_Weighting/windowed_loss.py` | weights identical to the last bit; loss equal to a relative 1e-12 |
| `unweighted.py`, `supervised_endpoint.py` | new | their own gates |

The cost-weighted loss is not identical to the frozen one to the last bit, because the frozen code divides the residual by the scale of the inputs and multiplies it back, which rounds. The package multiplies the residual by the weight directly.

Not ported: the ablations of the frozen study, which replaced a factor by its average. Here a factor that is switched off is 1.

## The contract

A loss gives `constants(first_round_states)` and `value(predicted_track, target, constants)`. It never imports a network. It takes every constant as an argument, and the run records the constants it used.

`value` reads every constant from the `constants` it is given, and from nowhere else.

## Gates every loss must pass

| Gate | What it proves |
|---|---|
| the exact collocation solution gives machine zero | the loss has the right minimum |
| the torch weights equal an independent numpy implementation | the weight is what the formula says |
| with every weight switched off it equals the unweighted loss to the last bit | the weighting changes nothing else |
| the constants recorded equal the constants used | no setting lives in two places |
| a loss of zero implies the summed end state equals the exact scheme's | the collect-and-sum step is wired correctly |

All are in `tests/test_losses.py` and `tests/test_networks.py`.

## How to add to it

1. Write the loss as one file in this directory, named in plain English.
2. Register it under its name with `register` from `../registry.py`, and add the file to `COMPONENT_MODULES` there.
3. Write its gates in `tests/`.
4. Add its row to the Contents table above and change the date.
5. Write or update its card in `docs/cards/`.
6. Add or update its entry in the master index of the Notion project page.
7. Run all gates.
