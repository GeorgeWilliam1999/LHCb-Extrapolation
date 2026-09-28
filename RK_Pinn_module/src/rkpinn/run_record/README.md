# The run record

**Last updated:** 2026-09-28

## What this directory is

What is recorded: the store and its keys, where a thing came from, the manifest that lists what the store holds, and, later, the configuration of a run.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file | built |
| `__init__.py` | marks the directory as part of the package; it holds no code | built |
| `store.py` | the folders and files of the store; the key computed from content; refuses to overwrite; the location is an argument or the environment variable `RKPINN_STORE` | built |
| `provenance.py` | the commit, the package version, the machine and its processor; refuses to write from uncommitted changes unless asked, and then records that the result cannot be traced | built |
| `manifest.py` | the list of track sets and the list of exact states in the store. The list of runs and the metrics recomputed from snapshots are added with the trainer and the evaluation | built |
| `configuration.py` | the schema of a configuration file, its validation, and the run key computed from it | planned |

## The layout of the store

```
<store>/
├── README.md
├── tracks/<key>/
│   ├── tracks.npz
│   ├── description.json
│   └── exact_scheme/<split>/<N>_steps_<q>_stages.npz  and its .json
├── runs/<key>/                 written by the trainer, phase 4
└── manifest/
    ├── tracks.csv
    └── exact_states.csv
```

The store of the project is `/data/bfys/gscriven/rkpinn_store`. It is outside git.

## The contract

A run refers to its tracks by key, never by path. It records the commit it was trained at. Metrics are recomputed from snapshots, never read from a run's own scores.

A key is the first 12 characters of the sha256 hash of the content it names. Nothing in the store is overwritten. No absolute path is written into a record.

## How to add to it

1. Write the field of the configuration as one file in this directory, named in plain English.
2. Write its gates in `tests/`.
3. Add its row to the Contents table above and change the date.
4. Write or update its card in `docs/cards/`.
5. Add or update its entry in the master index of the Notion project page.
6. Run all gates.
