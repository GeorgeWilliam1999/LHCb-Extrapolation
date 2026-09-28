# Track data

**Last updated:** 2026-09-28

## What this directory is

Building the tracks from the official sample, loading them, drawing the states a round trains on, and writing the exact scheme's states of a track set.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file | built |
| `__init__.py` | marks the directory as part of the package; it holds no code | built |
| `build_tracks.py` | each particle's true state on the Upstream Tracker carried across the crossing with the reference integrator, stored on every plane, with the true state at its own plane of the fibre tracker; the cuts and their counts; written to the store under a key computed from the content | built |
| `load_tracks.py` | loads a track set from the store by its key, with its training, validation and test splits, and checks that the content gives the key | built |
| `draw_training_states.py` | draws the states of one round, the same number on every start plane | built |
| `exact_states.py` | the exact scheme chained across the crossing from every start state of a split, solved once on one machine and written to the store; and its loader | built |

## Where each file was ported from

| File | Ported from | Gate |
|---|---|---|
| `build_tracks.py` | `single_network_chain_discrete_approach/Block_E_single_network_chain/E0_Track_dataset/build_tracks.py`, and `_shared/prepare.py` (`magnet_leg_rows`, `_inside_map`, `_particle_key`), `_shared/reference.py` (`load_training`) | `tests/test_tracks_match_the_frozen_file.py` |
| `draw_training_states.py` | `E1_Network_grid/train_network.py` (`draw_states`) | `tests/test_draw_of_training_states_matches_the_frozen_code.py` |
| `load_tracks.py` | new | `tests/test_store_keys_and_manifest.py` |
| `exact_states.py` | new; the solver is `../integrators/exact_collocation.py` | `tests/test_exact_states_in_the_store.py` |

## What differs from the frozen file, and what does not

The particles, their order and every number are those of the frozen file. The names differ:

| Frozen name | Name in the package |
|---|---|
| `train`, `val`, `test` | `training`, `validation`, `test` |
| `S0` | `start_state` |
| `truth` | `reference_states_on_planes` |
| `truth_zpost` | `reference_state_on_own_fibre_plane` |
| `S_pre`, `S_post` | `true_state_on_own_upstream_plane`, `true_state_on_own_fibre_plane` |
| `z_pre`, `z_post` | `own_upstream_plane_mm`, `own_fibre_plane_mm` |
| `P`, `ETA`, `PID` | `momentum_gev`, `pseudorapidity`, `particle_type` |
| `EVT`, `MCKEY` | `event`, `particle_in_event` |

Left out on purpose:

- The momentum band of each particle. The frozen file stored the bands 1, 2, 5, 10, 25, 200 GeV. The package's bands are those of the paper and are defined once, in `evaluation/conventions.py`, from `momentum_gev`.
- The comparison of the true state with the reference state, which the frozen builder wrote into its record. It is an analysis and belongs to the evaluation.
- The figure of the crossing.
- The cap on the number of training tracks.

## The contract

A track set is named by a key computed from its content. It is split by particle. It records the field map's hash, the sample and the cuts.

Every setting of a track set is an argument without a default. The settings of the project's track set are in the docstring of `build_tracks.py`.

The exact states of a track set are kept under that track set in the store. Each file records the machine and the processor it was solved on.

## How to add to it

1. Write the builder or loader as one file in this directory, named in plain English.
2. Write its gates in `tests/`.
3. Add its row to the Contents table above and change the date.
4. Write or update its card in `docs/cards/`.
5. Add or update its entry in the master index of the Notion project page.
6. Run all gates.
