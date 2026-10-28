# Results: does the straight-line form restore the old accuracy?

**Last updated:** 2026-10-28 · **Trust:** provisional, not yet discussed with George

## The answer

Yes, at the endpoint, for both scales. With the per-track scale it also holds at the stages. The per-track run reaches 1.83 µm at the endpoint and 0.072 µm in a single step, both below the frozen network with its stages summed (2.18 and 0.081 µm). The global-scale run reaches 2.04 µm at the endpoint but 0.117 µm in a single step, so it misses criterion 2. This is the outcome "1 and 2 hold for one run": by the README, the per-track scale is the default. The output form explains the gap to the frozen network: the direct-states run of the same cell, loss and seed gives 5.71 µm.

## The criteria

All errors are radial medians on the 1,452 test tracks, against the sixth-order reference, from the standard report of the last round. Source: `results/runs.csv`, `results/references.csv`.

| # | Criterion | Per-track scale (`25dfa17ef168`) | One global scale (`9cef9c160b7d`) |
|---|---|---|---|
| 1 | endpoint at or below 2.2 µm | 1.83, **holds** | 2.04, **holds** |
| 2 | single step at or below 0.08 µm | 0.072, **holds** | 0.117, **fails** |
| 3 | convergence read from the validation error | plateaued at round 44 of 80 | plateaued at round 22 of 80 |

Read beside them:

| Endpoint radial median (µm) | Per-track scale | One global scale | Direct states (`49e5e2450eaa`) | Frozen, stages summed |
|---|---|---|---|---|
| all momenta | 1.83 | 2.04 | 5.71 | 2.18 |
| below 3 GeV | 4.9 | 7.4 | 21.7 | |
| 3 to 8 GeV | 2.1 | 3.2 | 8.7 | |
| 8 to 20 GeV | 1.4 | 1.3 | 3.5 | |
| 20 to 50 GeV | 1.2 | 1.3 | 3.0 | |
| above 50 GeV | 1.9 | 0.7 | 3.2 | |
| 95th percentile, all momenta | 12.7 | 20.5 | 67.8 | |
| hours of training | 37 | 18 | 34 | |

The per-track scale is better below 8 GeV and in the tail. The global scale is better above 50 GeV, where there are only 30 test tracks.

## What the figures show

- `figures/endpoint_by_momentum_band.png`: both straight-line runs are below the direct-states run in every band, by 1.7 to 4.6 times; the frozen network is the cross at "all momenta".
- `figures/validation_per_round.png`: the straight-line runs start at 5.6 µm (per-track) and 3.8 µm (global) in round 1, where the direct-states run starts at 20.9 µm.
- `figures/error_along_the_track.png`: the radial error of the straight-line runs grows along the whole chain; that of the direct-states run peaks at 5.9 µm three steps before the end.

## Related analysis, beyond the criteria

Item 1a of the tranche-1 analysis (`../../../analysis/tranche1_2026-10-28/README.md`) scored these runs at other rounds. The round with the lowest validation error gives 1.69 µm (per-track, round 33) and 1.91 µm (global, round 19). The mean of the end states of the last five snapshots gives 1.41 and 1.37 µm. Those numbers are read in the literature experiment, `../../do_the_literature_changes_improve_the_endpoint/analysis/`.

The step-length experiment uses the per-track run as its 64-step, 2-stage point.
