# rkpinn: the package

**Last updated:** 2026-09-28

## What this directory is

The package trains Runge–Kutta physics-informed networks that are chained to themselves to give a full track, and scores them in one standard way. It holds no experiment.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file | built |
| `__init__.py` | the package version | planned |
| `registry.py` | turns a name in a configuration file into a component | planned |
| `equation_of_motion/` | the differential equations and the field map | built |
| `integrators/` | the Gauss–Legendre tableau, the exact scheme, the reference integrator | built |
| `predicted_track/` | the one structure every network fills and every loss and evaluation reads | built |
| `networks/` | the networks, how their outputs become states, and how steps are chained | built |
| `targets/` | what a prediction is trained on or compared with | built |
| `losses/` | the stage residual and the losses built on it | built |
| `track_data/` | building, loading and drawing from the tracks | built |
| `training/` | the trainer, its protocols, the optimiser and the stopping rule | built |
| `evaluation/` | the conventions and the standard report | built |
| `run_record/` | the configuration of a run, its key, and the manifest | built |

A directory marked built exists and has its own README. Whether the files inside it are built is stated in that README.

## The contract

A directory depends only on the directories above it in the dependency chart of `PACKAGE_PLAN.md`, section 4. Losses and evaluation never import a network.

## How to add to it

1. Decide which directory the component belongs to. If none fits, raise it with George before making a new one.
2. Follow that directory's own README.
3. If a new directory is made, give it a README in this format and add its row above.
