# Results: do the literature's changes to the training improve the endpoint?

**Last updated:** 2026-10-28 · **Trust:** provisional, not yet discussed with George

## The answer, as far as the stopped tests allow

The tests stopped after 3 to 6 rounds, before any arm plateaued, and no arm has a test-split report. So the question is not answered on the test split. What the store does show:

| Change | Answer so far |
|---|---|
| 1, one fixed draw of training states | no gain for direct states (0.95 of the baseline at round 6); 0.66 to 0.68 of the baseline at round 6 for the straight line. Too early to tell whether the plateau is lower |
| 2, the round with the lowest validation error | small gain on the test split for the straight line (1.83 to 1.69 µm per-track, 2.04 to 1.91 µm global); none for direct states, whose best round is its last. Averaging the end states of the last five snapshots gains more: 1.41, 1.37 and 5.24 µm |
| 3, refit of the last layer | not run |
| 4, the self-scaled Broyden optimiser | the largest signal: at round 3 the validation error is 0.16 of the baseline for direct states and 0.23 for the straight line, and both are already below the baselines' final rounds. This is being followed up in `../../does_a_better_optimiser_improve_the_endpoint/` |

## Each arm against its baseline at the same round

Validation endpoint error, larger of x and y, median over the validation tracks, in micrometres. Source: `results/arms_against_baselines.csv`.

| Change | Form | Arm | Key | Rounds | Arm | Baseline, same round | Ratio | Baseline, final round |
|---|---|---|---|---|---|---|---|---|
| 1 | direct states | fixed draw, own predictions | `23f3bc73c61b` | 6 | 13.92 | 14.62 | 0.95 | 5.46 (round 59) |
| 1 | direct states | fixed draw, reference states | `c594bdddd98f` | 6 | 13.84 | 14.62 | 0.95 | 5.46 |
| 4 | direct states | self-scaled Broyden | `f4c86e05eafe` | 3 | 2.95 | 18.49 | 0.16 | 5.46 |
| 1 | straight line, global | fixed draw, own predictions | `1212e5c801c2` | 6 | 1.31 | 1.92 | 0.68 | 1.88 (round 22) |
| 1 | straight line, global | fixed draw, reference states | `68a9f0f91e9e` | 6 | 1.26 | 1.92 | 0.66 | 1.88 |
| 4 | straight line, global | self-scaled Broyden | `9a3df286d5fb` | 3 | 0.56 | 2.49 | 0.23 | 1.88 |

Round 1 of every fixed-draw arm equals round 1 of its baseline to the last digit, as it must: the arms differ only from round 2. The Broyden arms differ from round 1.

Baseline round-to-round noise is large: the direct-states baseline moves by up to 64 per cent between rounds (`../../../analysis/oscillations_2026-10-22/findings.md`). A ratio of 0.95 after six rounds is inside that noise. The Broyden ratios are not.

## Change 2 on the test split

Endpoint radial median on the 1,452 test tracks, in micrometres. Source: `results/best_round_and_snapshot_average.csv`, copied from the tranche-1 analysis.

| Run | Last round | Best round by validation | Mean of the last five snapshots |
|---|---|---|---|
| straight line, per-track `25dfa17ef168` | 1.83 (44) | 1.69 (33) | 1.41 (40 to 44) |
| straight line, global `9cef9c160b7d` | 2.04 (22) | 1.91 (19) | 1.37 (18 to 22) |
| direct states `49e5e2450eaa` | 5.71 (59) | 5.71 (59) | 5.24 (55 to 59) |

## What is not known

- The test-split error of any arm. A standard report of the last round of each arm would give it.
- Whether the fixed draw lowers the plateau, or only reaches it sooner.
- Whether the Broyden gain holds to a plateau, and across seeds. The optimiser experiment asks this.
- The arm `f4c86e05eafe` was resumed on the farm on 2026-10-28; its numbers here are from round 3.

## Figure

`figures/validation_per_round.png`: one panel per output form; the baseline is dashed to round 12, the arms are solid.
