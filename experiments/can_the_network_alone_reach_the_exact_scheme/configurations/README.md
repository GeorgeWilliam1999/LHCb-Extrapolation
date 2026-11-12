# Configurations

**Last updated:** 2026-11-12

## What this directory is

One file per run of the experiment. A run takes every setting from its file. The key of a run is computed from its file.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file | built |
| `whole_tracks_pooled.yaml` | the baseline with the whole-track protocol: the control of the carried-defect loss | built |
| `whole_tracks_carried_share_0.5.yaml` | the carried-defect term at a share of 0.5 of the loss of round one | built |
| `whole_tracks_carried_share_0.9.yaml` | the carried-defect term at a share of 0.9 | built |
| `physics_layer_64_steps.yaml` | the physics layer, one sweep inside, 64 steps of 80.9 mm | built |
| `physics_layer_32_steps.yaml` | the physics layer, 32 steps of 161.8 mm | built |
| `physics_layer_16_steps.yaml` | the physics layer, 16 steps of 323.6 mm | built |
| `physics_layer_64_steps_width_32.yaml` | the physics layer, 64 steps, a body of width 32 | built |

## The contract

Each file is `../../does_the_straight_line_form_restore_the_old_accuracy/configurations/64_steps_2_stages_pooled_per_track_scale.yaml` (run key `25dfa17ef168`), changed only in what its first line names: the protocol, the loss, or the network kind with its `sweeps_inside`, number of steps and width.

## How to add to it

1. Copy a file, change what the new run asks about, and say so in its first line.
2. Add its row above, its name to `../farm/runs.txt` and its key to `../README.md`.
