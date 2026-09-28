# Training

**Last updated:** 2026-09-28

## What this directory is

The trainer and everything it needs. The trainer names no loss, no network and no target: it takes them from the configuration.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file | built |
| `round_trainer.py` | trains in rounds; writes a snapshot after every round; resumes from the last snapshot | planned |
| `training_protocols.py` | what is drawn each round. One now: the first round on reference states, later rounds on the network's own predictions | planned |
| `optimiser.py` | L-BFGS restarts with the loss rescaled at the start of a round and on every tenfold fall | planned |
| `stopping_rule.py` | the validation plateau: the median of the last ten rounds no more than 5 % below the ten before, at each of the last three rounds | planned |
| `checkpoints.py` | snapshots that are never overwritten, and a lock so that one job writes to a run at a time | planned |

## The contract

Convergence is judged on the validation error, never on the loss. The test split is never used for stopping.

## How to add to it

1. Write the protocol, optimiser or stopping rule as one file in this directory, named in plain English.
2. Register it under its name in `../registry.py`.
3. Write its gates in `tests/`.
4. Add its row to the Contents table above and change the date.
5. Write or update its card in `docs/cards/`.
6. Add or update its entry in the master index of the Notion project page.
7. Run all gates.
