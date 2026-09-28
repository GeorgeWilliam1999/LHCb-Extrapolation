# The predicted track

**Last updated:** 2026-09-28

## What this directory is

The seam of the package. Every network, the exact scheme and the reference integrator fill this structure. Every loss and every evaluation reads it. Nothing here knows how the states were produced.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file | built |
| `track_layout.py` | the planes of a track: first and last plane, number of steps, number of stages, and the stage planes of each step | planned |
| `predicted_track.py` | for each step: its input state, its stage states and its end state; and the final state of the track | planned |

## The contract

A predicted track may cover one step or many. A step may have no stages, as for the whole-crossing network. Charge over momentum is carried unchanged.

## How to add to it

1. A change here changes every component. Raise it with George first.
2. Gate the change against every network, loss and evaluation.
3. Update the Contents table above and change the date.
