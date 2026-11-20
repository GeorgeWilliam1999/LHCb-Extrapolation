# Configurations

**Last updated:** 2026-11-20 · **State:** written; all five pass the configuration check and the checks before training (2026-11-20, stbc-i2)

## What this directory is

One file per run of the experiment. A run takes every setting from its file. The key of a run is computed from its file.

All five runs share: the cubic field map `v8r1_up_cubic`, the track set `5558be31abeb`, 64 steps and 2 stages, the network `label_free_path_network` (width 256, depth 2, tanh, straight line plus a correction scaled per track), no target, the protocol `start_states_of_the_tracks` with 4,096 states a round, L-BFGS restarts as in `path_256_lbfgs.yaml` of the one-shot experiment, 5 restarts a round, the validation plateau rule, a cap of 40 rounds, seed 1.

## Contents

| Name | Run | Loss block | Key | State |
|---|---|---|---|---|
| `README.md` | | this file | | built |
| `label_free_path_uniform.yaml` | A1 | uniform weights, gradient through the plane states | `3ccd44d61cd8` | built |
| `label_free_path_uniform_detached.yaml` | A2 | uniform weights, plane states detached | `0b41ecdd2505` | built |
| `label_free_path_sliding_window.yaml` | B1 | sliding window: block 4 steps, width 2 steps, `settled_below` 1e-10 (provisional, TODO) | `4cb5f4821e9f` | built; threshold not final |
| `label_free_path_causal_last_weight_0_1.yaml` | B2 | causal, last-step weight 0.1 at round one | `332072fbffa9` | built |
| `label_free_path_causal_last_weight_0_001.yaml` | B3 | causal, last-step weight 0.001 at round one | `07a3299ece5e` | built |

## The contract

A controlled comparison changes one block. Every B run differs from A1 in the loss block only. A2 differs from A1 in `gradient_through_plane_states` only.

## How to add to it

1. Copy a file, change one block, and name the file by what the run is.
2. Print its key with `python -m rkpinn.run_record.configuration <file>`.
3. Add its row above, and its line to `../farm/runs.txt` when it is to be submitted.
