# Loss functions

**Last updated:** 2026-09-28

## What this directory is

The stage residual, and the losses built on it. All label-free losses are the mean of the squared residual over a divisor; they differ only in the divisor.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file | built |
| `stage_residual.py` | each stage state, minus the step length times the stage matrix applied to the stage rates, minus the input state | planned |
| `unweighted.py` | divisor 1: the baseline with no weighting | planned |
| `pooled.py` | divisor per component: the spread of that component over the first round's states. Equation 19 of the second mini-paper | planned |
| `cost_weighted.py` | divisor per track, stage plane and component: lever arm, track bend, momentum window, clamp. Equation 20 | planned |
| `supervised_endpoint.py` | mean squared difference between a predicted end state and a target end state | planned |

## The contract

A loss gives `constants(first_round_states)` and `value(predicted_track, target)`. It never imports a network. It takes every constant as an argument, and the run records the constants it used.

## Gates every loss must pass

| Gate | What it proves |
|---|---|
| the exact collocation solution gives machine zero | the loss has the right minimum |
| the torch weights equal an independent numpy implementation | the weight is what the formula says |
| with every weight switched off it equals the unweighted loss to the last bit | the weighting changes nothing else |
| the constants recorded equal the constants used | no setting lives in two places |

## How to add to it

1. Write the loss as one file in this directory, named in plain English.
2. Register it under its name in `../registry.py`.
3. Write its gates in `tests/`.
4. Add its row to the Contents table above and change the date.
5. Write or update its card in `docs/cards/`.
6. Add or update its entry in the master index of the Notion project page.
7. Run all gates.
