# The run record

**Last updated:** 2026-09-28

## What this directory is

What a run is: its configuration, its key, its provenance, and the manifest that lists every run.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file | built |
| `configuration.py` | the schema of a configuration file, its validation, and the run key computed from it | planned |
| `manifest.py` | the list of track sets and runs in the store, and the metrics recomputed from snapshots | planned |

## The contract

A run refers to its tracks by key, never by path. It records the commit it was trained at. Metrics are recomputed from snapshots, never read from a run's own scores.

## How to add to it

1. Write the field of the configuration as one file in this directory, named in plain English.
2. Write its gates in `tests/`.
3. Add its row to the Contents table above and change the date.
4. Write or update its card in `docs/cards/`.
5. Add or update its entry in the master index of the Notion project page.
6. Run all gates.
