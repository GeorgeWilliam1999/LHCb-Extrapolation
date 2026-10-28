# Results: the paper's nine runs redone in the summed form

**Last updated:** 2026-10-28 · **Trust:** provisional, not yet discussed with George

## The answer

The summed form holds across the paper's three cells: every pooled run and every cost-weighted 10 to 50 GeV run is better at the endpoint than the paper's run of the same cell and loss: the pooled runs by 19 to 31 times, the cost-weighted runs by 1.5 to 4.8 times, and the 64-step pooled run at seed 1 reproduces the pilot. Criteria 1 to 4 all hold.

Two findings of the paper do not carry over:

1. **The cost-weighted loss no longer pays in its window.** In the summed form the pooled run of each cell is lower than the cost-weighted runs in every momentum band, inside their windows included; the one tie is above 50 GeV at 256 steps (2.48 against 2.53 µm, 30 tracks). The 10 to 50 GeV window medians of the cost-weighted runs are 16.2, 12.8 and 6.4 µm, where the pooled runs give 3.0 to 4.9 µm in the 8 to 20 and 20 to 50 GeV bands. The 3 to 8 GeV runs give 78.2, 57.6 and 35.3 µm in their window, against 8.7, 8.6 and 6.0 µm for the pooled runs.
2. **Shorter steps still do not lower the pooled endpoint cleanly**: 5.71, 6.60 and 4.81 µm at 64, 128 and 256 steps, while the exact scheme's ceiling falls from 0.147 to 0.0115 to 0.00105 µm. The error is set by training, not by the step. The step-length experiment measures this on its own.

## The criteria

All errors are radial medians on the 1,452 test tracks, against the sixth-order reference, from the standard report of the last round. Source: `results/runs.csv`, `results/references.csv`.

| # | Criterion | 64 steps, 2 stages | 128 steps, 8 stages | 256 steps, 16 stages |
|---|---|---|---|---|
| 1 | pooled beats the paper's pooled run | 5.71 vs 145.7, **holds** | 6.60 vs 122.2, **holds** | 4.81 vs 147.7, **holds** |
| 2 | cost-weighted 10 to 50 GeV beats the paper's | 57.8 vs 88.8, **holds** | 47.9 vs 104.6, **holds** | 19.3 vs 92.2, **holds** (26.0 at round 80) |
| 3 | 64-step pooled within a factor of two of the pilot's 8.97 | 5.71, **holds** | | |
| 4 | convergence read from the validation error | all three plateaued (59, 24, 27 rounds) | all three plateaued (54, 33, 25) | pooled 42, 3 to 8 GeV 46; the 10 to 50 GeV run reached its cap of 80 without a plateau, was extended to 160 and plateaued at 111 |

## All nine runs

| Cell | Loss | Key | Rounds | Endpoint (µm) | Below 3 GeV | 3 to 8 | 8 to 20 | 20 to 50 | Above 50 | In the loss window | Single step (µm) | Hours |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 64 / 2 | pooled | `49e5e2450eaa` | 59 | 5.71 | 21.7 | 8.7 | 3.5 | 3.0 | 3.2 | | 0.486 | 34 |
| 64 / 2 | cost-weighted 10 to 50 | `da3a1d00b33d` | 24 | 57.8 | 352 | 128 | 19.8 | 15.1 | 9.8 | 16.2 | 1.59 | 16 |
| 64 / 2 | cost-weighted 3 to 8 | `f977c7b61049` | 27 | 54.0 | 286 | 78.2 | 34.6 | 20.0 | 19.2 | 78.2 | 0.840 | 18 |
| 128 / 8 | pooled | `75200e48408c` | 54 | 6.60 | 20.5 | 8.6 | 4.5 | 4.9 | 3.8 | | 0.208 | 56 |
| 128 / 8 | cost-weighted 10 to 50 | `237dc4101ad5` | 33 | 47.9 | 432 | 114 | 20.0 | 9.8 | 8.5 | 12.8 | 0.683 | 33 |
| 128 / 8 | cost-weighted 3 to 8 | `521415f2c534` | 25 | 52.7 | 192 | 57.6 | 44.5 | 47.0 | 40.6 | 57.6 | 0.583 | 27 |
| 256 / 16 | pooled | `f38ec3e29819` | 42 | 4.81 | 13.4 | 6.0 | 3.6 | 3.4 | 2.5 | | 0.103 | 61 |
| 256 / 16 | cost-weighted 10 to 50 | `aec34bbd4934` | 111 | 19.3 | 136 | 38.4 | 9.8 | 4.3 | 2.5 | 6.4 | 0.180 | 158 |
| 256 / 16 | cost-weighted 3 to 8 | `9c3c5d86784b` | 46 | 22.4 | 116 | 35.3 | 16.1 | 12.7 | 10.0 | 35.3 | 0.156 | 57 |

The band columns are endpoint radial medians in micrometres. The 256-step 10 to 50 GeV run at its old cap, round 80, gave 26.0 µm overall and 11.1 µm in its window; the extension to round 111 lowered both by 26 and 42 per cent.

## What the figures show

- `figures/endpoint_by_momentum_band.png`: one panel per cell. The pooled curve is the lowest in every band of every cell, with one tie above 50 GeV at 256 steps. The paper's two runs of the cell are the crosses at "all momenta".
- `figures/error_along_the_track.png`: the pooled error flattens over the last quarter of the chain; the cost-weighted error grows fastest there. At 64 steps the 10 to 50 GeV run goes from 19 to 58 µm over the last nine steps, z = 7098 to 7826 mm, and the 3 to 8 GeV run from 14 to 54 µm.
- `figures/validation_per_round.png`: the validation error of every run, every round. The extended run is drawn to round 111.

## Related analysis

The round-to-round oscillation of the validation error in these nine runs was studied on 2026-10-22 in `../../../analysis/oscillations_2026-10-22/findings.md`: the optimiser converges in every restart, and the oscillation comes from the fresh draw of training states each round, which the validation error follows only weakly.
