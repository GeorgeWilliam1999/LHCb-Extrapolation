# Configurations

**Last updated:** 2026-11-04 · **State:** agreed; checked against the package on branch `one_shot_path`

## What this directory is

One file per run of the experiment. A run takes every setting from its file; the key of a run is computed from its file. Every file is read back by the package's configuration check (2026-11-04).

Names fixed on 2026-11-04 against the stbc checkout:

| Written | In the package |
|---|---|
| `output_form: straight_line_per_track` | `output_form: straight_line_plus_scaled_correction` with `scale_of_outputs: per_track` |
| (no setting) | `network.extra_outputs`: 0, or 4 in P3 (a log-sigma per state and component) |
| P4: `line_search`, `history` | not settings of `self_scaled_broyden_restarts`; its block is that of `does_a_better_optimiser_improve_the_endpoint` (`threads_for_the_matrix: 4`) |
| E1: `steps: {64, 2}` | `steps: {1, 0}`: the check refuses a whole-crossing network on any other layout |
| E1: straight-line form | `direct_states` scaled by `spread_of_target_end_states`: the whole-crossing network binds its form with no field map, and the straight-line form refuses that. Open for George's ruling |

Round size and cap ruled on 2026-11-04: P1, P2, P4 and E1 have `restarts_per_round` 5 and `rounds_at_most` 40. Run keys: P1 `25934ed22604`, P2 `45093f0c80d6`, P4 `e11f63bdc8be`, E1 `dd9acf59ac88`; P3 `053ed209bdb9`, unchanged. The smoke run is `047de8df7668`. P3 and P4 are not in `../farm/runs.txt`; see `../README.md`, The runs.

## Contents

| Name | Run | Differs from P1 in | State |
|---|---|---|---|
| `README.md` | | this file | planned |
| `path_256_lbfgs.yaml` | P1 | — | built |
| `path_512_lbfgs.yaml` | P2 | `network.width` | built |
| `path_256_gaussian_lbfgs.yaml` | P3 | `loss.uncertainty`, `network.extra_outputs` | built; needs a warm start from P1; not in the first submission |
| `path_256_broyden.yaml` | P4 | `training.optimiser` | built; not in the first submission: the dense matrix of the optimiser would be 561 GB |
| `endpoint_256_lbfgs.yaml` | E1 | `network.kind`, `network.output_form`, `steps`, `loss`, `target` (the control) | built |
| `smoke_path_256_lbfgs.yaml` | smoke run of P1 (stage 2) | `training.restarts_per_round` 2, `training.rounds_at_most` 2; not a run of the experiment | built |

## How to add to it

1. Copy a file, change one block, and name the file by what the run is.
2. Add its row above and its line to `../farm/runs.txt`.
