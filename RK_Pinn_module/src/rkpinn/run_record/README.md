# The run record

**Last updated:** 2026-09-30

## What this directory is

What is recorded: the store and its keys, where a thing came from, the manifest that lists what the store holds, and, later, the configuration of a run.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file | built |
| `__init__.py` | marks the directory as part of the package; it holds no code | built |
| `store.py` | the folders and files of the store; the key computed from content; refuses to overwrite; the location is an argument or the environment variable `RKPINN_STORE` | built |
| `provenance.py` | the commit, the package version, the machine and its processor; refuses to write from uncommitted changes unless asked, and then records that the result cannot be traced | built |
| `manifest.py` | the lists of track sets, of exact states and of runs in the store, and the metrics: the endpoint errors of a snapshot, computed again from its weights | built |
| `report_of_a_run.py` | the standard report of a run, from one command: builds the network of the run, makes the predictions, hands states and targets to the evaluation, writes the tables and figures into the run | built |
| `configuration.py` | the schema of a configuration file, its validation, the run key computed from it, and the difference of two configurations | built |

## The layout of the store

```
<store>/
├── README.md
├── tracks/<key>/
│   ├── tracks.npz
│   ├── description.json
│   └── exact_scheme/<split>/<N>_steps_<q>_stages.npz  and its .json
├── runs/<key>/                 see ../training/checkpoints.py
│   ├── configuration.yaml, provenance.json, constants.json, state.json
│   ├── restarts.csv, rounds.csv
│   ├── snapshots/round_0001/   network.pt and scores.json
│   ├── reports/round_0001_at_<commit>/   the tables and figures of the standard report
│   └── LOCK
└── manifest/
    ├── tracks.csv
    ├── exact_states.csv
    ├── runs.csv
    └── metrics.csv
```

The store of the project is `/data/bfys/gscriven/rkpinn_store`. It is outside git.

## The contract

A run refers to its tracks by key, never by path. It records the commit it was trained at. Metrics are recomputed from snapshots, never read from a run's own scores.

A key is the first 12 characters of the sha256 hash of what it names: the arrays of a track set, or the configuration of a run. The cap on the rounds is left out of the key of a run, so that raising it extends the run.

A configuration with a setting that is not recognised, a setting that is missing, or a name the registry does not know, is refused. No setting has a default. Nothing in the store is overwritten. No absolute path is written into a record.

## How to add to it

1. Write the field of the configuration as one file in this directory, named in plain English.
2. Write its gates in `tests/`.
3. Add its row to the Contents table above and change the date.
4. Write or update its card in `docs/cards/`.
5. Add or update its entry in the master index of the Notion project page.
6. Run all gates.
