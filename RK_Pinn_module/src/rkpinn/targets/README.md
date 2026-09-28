# Targets

**Last updated:** 2026-09-28

## What this directory is

What a prediction is trained on or compared with. A target is data; it contains no network and no loss.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file | built |
| `no_target.py` | label-free training: the loss needs only the equation of motion | planned |
| `reference_end_state.py` | the RK6 state at the first SciFi plane; the label of the supervised twin | planned |
| `reference_states_on_planes.py` | the RK6 state on every plane of the layout; used by the evaluation | planned |
| `exact_stage_states.py` | the exact scheme's stage states, started from the same input; used to hold the stage errors | planned |
| `true_state.py` | the Geant4-true state at the particle's own first SciFi plane; the floor no field-only method can beat | planned |

## The contract

A target gives `needs_labels` and `for_states(track_data, split)`. It is aligned with the track data by particle, in the same order.

## How to add to it

1. Write the target as one file in this directory, named in plain English.
2. Register it under its name in `../registry.py`.
3. Write its gates in `tests/`.
4. Add its row to the Contents table above and change the date.
5. Write or update its card in `docs/cards/`.
6. Add or update its entry in the master index of the Notion project page.
7. Run all gates.
