# Track data

**Last updated:** 2026-09-28

## What this directory is

Building the tracks from the official sample, loading them, and drawing the states a round trains on.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file | built |
| `build_tracks.py` | each particle's true last-UT state carried across the crossing with RK6, stored on every plane of the layout, with the true state at its own first SciFi plane | planned |
| `load_tracks.py` | loads a track set from the store by its key, with its training, validation and test splits | planned |
| `draw_training_states.py` | draws the states of one round, the same number per plane | planned |

## The contract

A track set is named by a key computed from its content. It is split by particle. It records the field map's hash, the sample and the cuts.

## How to add to it

1. Write the builder or loader as one file in this directory, named in plain English.
2. Write its gates in `tests/`.
3. Add its row to the Contents table above and change the date.
4. Write or update its card in `docs/cards/`.
5. Add or update its entry in the master index of the Notion project page.
6. Run all gates.
