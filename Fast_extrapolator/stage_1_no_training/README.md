# Stage 1: what is possible with no training

Questions of this stage, before any network is trained:

1. The frontier. For a given compiled time, what is the smallest error that a
   classical integrator reaches? A network must beat this, and a tuned
   classical integrator can be the answer.
2. The floor. For a layout of N steps and q Gauss-Legendre stages, what is the
   error of the exact scheme? A chained network with a summed end state on that
   layout cannot be better.

The results are in `RESULTS.md` (written from the tables) and in the notebook
`stage_1_no_training.ipynb`.

## Set-up

| Choice | Value |
|---|---|
| Tracks | The 13,447 test tracks of the two track sets (linear map, cubic map). Choices (the exponent of a step plan) are made on 2,000 training tracks |
| Crossing | 2648.2 mm to 7826 mm; `shifted_planes.py` moves the planes |
| Field map | v8r1, magnet up, read linearly (as LHCb reads it) or with cubic B-splines (the standard of the project) |
| Error | The distance on the last plane between a method and the reference of the track set, in um |
| Time | As in stage 0: one track a call, one thread, one pinned core of stbc-i3, the lowest of up to 15 passes |
| Fixed-step methods | Single precision; on the linear map with the field read of LHCb |
| LHCb integrators | The copies of stage 0, in the precision of their originals |

## Scripts and their outputs

| Script | Output |
|---|---|
| `scripts/step_plans.py` | the step plans (in the store) and `results/cost_of_a_step_along_z.csv` |
| `scripts/classical_frontier.py choose_plans` | `results/plans_on_the_training_tracks.csv`, `results/chosen_plans.csv` |
| `scripts/classical_frontier.py accuracy` | `results/classical_accuracy.csv` |
| `scripts/classical_frontier.py floor` | `results/floor_of_the_layouts.csv` |
| `scripts/classical_frontier.py time CORE` | `results/classical_timing.csv` |
| `scripts/classical_frontier.py table` | `results/classical_frontier.csv` |
| `scripts/step_length_scan.py` | `results/step_length_scan.csv` |
| `scripts/shifted_planes.py` | `results/shifted_planes.csv` |
| `scripts/two_readings_of_the_map.py` | `results/two_readings_of_the_map.csv` |
| `scripts/analysis.py` | the four figures, `results/cheapest_methods.csv`, `results/what_a_chained_network_costs.csv`, `RESULTS.md` |
| `scripts/make_notebook.py` | `stage_1_no_training.ipynb` |

## Limits

- The times are from a shared login node, as in stage 0.
- A fixed-step integrator is scored on one crossing. `shifted_planes.py` shows
  what happens on 14 other crossings; it is not a study of all planes that a
  track fit uses.
- The tracks have no material and no energy loss.
- No transport matrix is computed for the fixed-step integrators.
- The cost of a chained network in `RESULTS.md` (section 8) is a lower limit
  from the unit costs of stage 0. No such network is built or trained here.
