# Configurations

**Last updated:** 2026-11-03 · **State:** planned, not agreed

## What this directory is

One file per run of the experiment. A run takes every setting from its file; the key of a run is computed from its file. The components `path_network`, `reference_path` and `supervised_path` are planned, not built: the package refuses these files until they exist (see `../README.md`, "What is added to the package"). The name of the straight-line output form and of the Broyden optimiser must be checked against the stbc checkout before the smoke run.

## Contents

| Name | Run | Differs from P1 in | State |
|---|---|---|---|
| `README.md` | | this file | planned |
| `path_256_lbfgs.yaml` | P1 | — | planned |
| `path_512_lbfgs.yaml` | P2 | `network.width` | planned |
| `path_256_gaussian_lbfgs.yaml` | P3 | `loss.uncertainty` | planned |
| `path_256_broyden.yaml` | P4 | `training.optimiser` | planned |
| `endpoint_256_lbfgs.yaml` | E1 | `network.kind`, `loss`, `target` (the control) | planned |

## How to add to it

1. Copy a file, change one block, and name the file by what the run is.
2. Add its row above and its line to `../farm/runs.txt`.
