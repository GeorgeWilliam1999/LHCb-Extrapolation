# Results: does a network whose end state is summed from its stages train?

**Last updated:** 2026-10-28 · **Trust:** provisional, not yet discussed with George

## The answer

Yes. Run A, whose end state is summed from its stages, trains: its single-step error is 0.516 micrometres, inside the pass line of 3.33, and its endpoint error is 8.97 micrometres, against 145.7 for the frozen network of the same size. Run B, whose end state is predicted, fails: its single-step error is 11.8 micrometres and its endpoint error 623 micrometres. This is the outcome "A passes, B fails", which the README calls unexpected and which points at the predicted form or at the trainer's handling of it, not at the summed end state.

## The criteria

All errors are radial medians on the 1,452 test tracks, against the sixth-order reference, at the last round. Source: `results/runs.csv`.

| # | Criterion | Run A, summed (`bddea42ea1ef`) | Run B, predicted (`4cc888dcc710`) |
|---|---|---|---|
| 1, 2 | single-step error at or below 3.33 µm (ten times the frozen 0.333) | 0.516, **passes** | 11.77, **fails** |
| 3 | convergence read from the validation error | plateaued at round 41 of 60 | plateaued at round 45 of 60 |

Read beside them:

| Number | Run A | Run B | Frozen network | Exact scheme (the ceiling) |
|---|---|---|---|---|
| Endpoint radial median (µm) | 8.97 | 622.6 | 145.7 | 0.147 |
| Endpoint, below 3 GeV (µm) | 28.8 | 1766 | | |
| Endpoint, above 50 GeV (µm) | 4.7 | 195 | | |
| Hours of training | 24 | 23 | | |

## Figures

- `figures/validation_per_round.png`: run B stays one hundred times above run A from the first round to the last.
- `figures/endpoint_by_momentum_band.png`: run A is below the frozen network in every band; run B is above it in every band.
- `figures/error_along_the_track.png`: the error grows along the chain for both runs.

## Caveat

A note written after the pilot (`../../does_the_straight_line_form_restore_the_old_accuracy/README.md`) says criterion 1 should have compared a single step with the frozen network's stages summed (0.081 µm), not with its predicted end state (0.333 µm). Against 0.081 the pass line would be 0.81 µm, and run A still passes.
