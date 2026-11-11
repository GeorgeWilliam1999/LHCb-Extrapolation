# LHCb Extrapolation

**Last updated:** 2026-11-12

Runge–Kutta physics-informed networks (RK-PINNs) for charged-track extrapolation through the LHCb magnet. A small network gives the stage states of an implicit Gauss–Legendre Runge–Kutta step. The step sums its end state from the stages. The network is applied to itself, step after step, from the last UT plane (z = 2648 mm) to the first SciFi plane (z = 7826 mm).

## Where to start

| To read | Go to |
|---|---|
| What every experiment asked and found, one line each | [`experiments/README.md`](experiments/README.md) |
| Every trained run, its settings and its score | [`experiments/RUNS.md`](experiments/RUNS.md) |
| How the package behaves and how work on it proceeds | [`RK_Pinn_module/WORKFLOW.md`](RK_Pinn_module/WORKFLOW.md) |
| The package itself | [`RK_Pinn_module/README.md`](RK_Pinn_module/README.md) |

## Contents

| Folder | What it is | State |
|---|---|---|
| `RK_Pinn_module/` | the `rkpinn` package: field map, equation of motion, tableaux, exact scheme, networks, losses, trainer, standard evaluation, and the gates (`tests/`) | live |
| `experiments/` | one folder per experiment made with the package, each named by its question; configurations, farm files and analysis, never weights | live |
| `analysis/` | two read-only analyses of existing runs: the late-training oscillations (2026-10-22) and tranche 1 of the improvement plan (2026-10-28) | done |
| `Data_generation_exploration/` | the simulated events (Gauss, Boole, Moore), the official samples, the field map and the training tracks the package reads | done; the package's track set is built from it |
| `Talks/` | the progress talk of 2026-09-07 | done |
| `Archive/` | the studies before the package: the multi-network and single-network chains (Blocks 0 to F), the cost-function study and the two mini-papers. Read-only; moved here on 2026-10-13 | frozen |

## Data

Runs, weights, reports and exact states are not in git. They are in the project store, `/data/bfys/gscriven/rkpinn_store`, under the key of each run. Bulky files of the archive (.sim, .digi, .root, .npz, .pt, farm logs) are also git-ignored; the scripts, seeds and metadata here regenerate them.

## Conventions

- Errors are in micrometres, slopes are dimensionless, momenta in GeV. The momentum bands are below 3, 3 to 8, 8 to 20, 20 to 50 and above 50 GeV.
- The reference is a sixth-order Runge–Kutta integration at 0.1 mm steps on the same field map.
- Every directory has a README that lists what is in it.
