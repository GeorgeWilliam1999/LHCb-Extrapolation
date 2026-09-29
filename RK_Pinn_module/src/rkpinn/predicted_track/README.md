# The predicted track

**Last updated:** 2026-09-29

## What this directory is

The seam of the package. Every network, the exact scheme and the reference integrator fill this structure. Every loss and every evaluation reads it. Nothing here knows how the states were produced.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file | built |
| `__init__.py` | marks the directory as part of the package; it holds no code | built |
| `track_layout.py` | the planes of a track: first and last plane, number of steps, number of stages, and the stage planes of each step (`TrackLayout`) | built |
| `predicted_track.py` | for each step: its input state, its stage states and its end state; and the final state of the track (`PredictedTrack`). Also the three functions by which the exact scheme and the reference integrator fill it | built |

## What a predicted track holds

| Array | Shape | Content |
|---|---|---|
| `start_planes_mm` | (rows, steps) | z of the plane each step starts on |
| `input_states` | (rows, steps, 5) | the state each step starts from |
| `stage_states` | (rows, steps, stages, 4), or none | the state on every stage plane |
| `end_states` | (rows, steps, 4) | the state at the end of each step |
| `stage_rates` | (rows, steps, stages, 4), or none | the rates at the stage states, kept when whoever filled the track had to compute them |

It also holds the step length, the tableau, the layout, and `end_state_was`, which is `summed_from_the_stages` or `predicted`.

It is used in two ways. A whole track has a layout, and its steps follow one another. Separate steps have no layout: each row is one step on its own start plane, as drawn for a round of training.

The arrays are torch tensors while training and numpy arrays otherwise.

## The contract

A predicted track may cover one step or many. A step may have no stages, as for the whole-crossing network. Charge over momentum is carried unchanged.

A track refuses arrays whose shapes do not agree, and an end state summed from stages it does not have.

## How to add to it

1. A change here changes every component. Raise it with George first.
2. Gate the change against every network, loss and evaluation.
3. Update the Contents table above and change the date.
