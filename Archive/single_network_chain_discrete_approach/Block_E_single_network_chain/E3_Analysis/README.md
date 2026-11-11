# E3_Analysis — the sixteen Block E networks side by side

**Status (2026-09-18, corrected 2026-09-20): done for the checkpoints of 2026-09-18.** Training
was stopped by hand on 2026-09-18 because the loss on freshly drawn states had been flat for ten
rounds; every network was re-scored from the weights then on disk (`../E1_Network_grid/finalise.py`).

**Correction (found 2026-09-18 evening, README fixed 2026-09-20):** the validation error had *not*
been flat. Under the plateau rule of `../../Block_F_reweighted_loss/F2_Analysis/compare_to_blockE.py`
none of the sixteen runs had stopped improving; all eleven with twenty or more rounds were still
falling 5–24% per ten rounds. **Every number in this folder is therefore where training was
stopped, not where it converges.** Those checkpoints are kept in
`../E1_Network_grid/results/N*/stopped_2026-09-18/`; the run folders themselves now hold the
extended networks, so rerunning these scripts today gives different, better numbers (for
example N = 64, q = 2: 166 → 146 µm; N = 256, q = 2: 185 → 126 µm on the test tracks). The
extended runs are tracked in `../../Block_F_reweighted_loss/F2_Analysis/results/headline.csv`.

**The measure** (George 2026-09-18): the **radial** distance at the first SciFi plane,
z1 = 7,826 mm, between the network's endpoint and the RK6 endpoint of the same track,
r = √(Δx² + Δy²), on the 1,452 test tracks. Every comparator is measured the same way.
Earlier Block D and Block E tables used max(|Δx|, |Δy|), which runs about 10–20% lower.

## script → output

| script | what it does | output |
|---|---|---|
| [tables.py](tables.py) | error(q, dz) for the chains, with the exact scheme, the straight line, the material floor and Block D beside it; the tails; cost per track; how much the error moved round to round at the end | `results/error_qdz_chain.csv`, `tails.csv`, `cost_accuracy.csv`, `comparators.csv`, `figures/error_qdz.png` |
| [single_step_tables.py](single_step_tables.py) | the same table for ONE step: each network applied once from the RK6 state on every start plane (1,452 × N pairs), and the same against z | `results/error_qdz_single_step.csv`, `single_step_vs_z.csv`, `figures/single_step.png` |
| [errors_vs_momentum.py](errors_vs_momentum.py) | all 16 networks: x, y, tx, ty against momentum, one figure per component, 4 × 4 panels | `figures/error_vs_p_{x,y,tx,ty}.png`, `results/error_vs_p.csv` |
| [along_z.py](along_z.py) | the chain error plane by plane: where it is built up | `results/error_vs_z.csv`, `figures/error_vs_z.png` |
| [convergence.py](convergence.py) | the training loss after every restart and the validation error after every round, per network | `figures/convergence_grid.png`, `convergence_summary.png`, `results/convergence.csv` |
| [evaluate_splits.py](evaluate_splits.py) | the over-training check: every network on train, validation and test, by chain error and by the trained loss | `results/split_comparison.csv`, `overtraining.csv` |
| [case_study_3d.py](case_study_3d.py) | the same eight panels with the starting x as a third axis: surfaces of median \|error\| over (momentum, starting x), and the 10–20 GeV band as (signed error, starting x); a flat version of the same numbers beside it | `figures/case_study_components_3d.png`, `case_study_components_x0_maps.png`, `results/case_study_error_vs_p_x0.csv` |
| [case_study.py](case_study.py) | the best network (chosen on validation) on its own: headline and comparators, each component against momentum, the 10–20 GeV band signed, growth along z, convergence, tails, pseudorapidity and charge, and the loss against momentum | `results/case_study_*.csv`, `figures/case_study_*.png` |
| [error_anatomy.py](error_anatomy.py) | one network (default: the best on validation): what each step adds against RK6 taken from the state the network was given, the lever-arm model (each step's x-slope error times the distance left to z1, summed) and the coherence of the slope increments along a track. **The lever-arm model uses the x slope only, while the headline is radial:** it explains 72% of Block E's N = 64, q = 2 error but only 38% of Block F's, where y is the larger component (2026-09-20) | `results/error_anatomy.csv`, `error_anatomy_summary.json`, `figures/error_anatomy.png` |
| [analysis.ipynb](analysis.ipynb) | loads the tables and figures, computes nothing; executed, and the source for any write-up | — |

```bash
PY=/data/bfys/gscriven/conda/envs/TE/bin/python; export PYTHONNOUSERSITE=1
$PY ../E1_Network_grid/finalise.py     # re-score from the current weights first
$PY tables.py && $PY single_step_tables.py && $PY errors_vs_momentum.py
$PY along_z.py && $PY convergence.py && $PY evaluate_splits.py && $PY case_study.py
$PY -m jupyter nbconvert --to notebook --execute --inplace analysis.ipynb
```

## What the tables say (2026-09-18 checkpoints)

- **The chains sit at 137–246 µm**, with no useful dependence on N or q above N = 64. The
  best test value is N = 256, q = 16 at 137 µm; the best on validation is N = 64, q = 2.
  Between checkpoints the error wanders by ±10–20%, so the ranking inside that band is noise.
- **A single step is excellent**: 0.09–0.8 µm for N ≥ 64, which is 0.4–2% of the straight
  line's error over the same step. The chain error is therefore almost entirely
  accumulation: at N = 128, q = 8 the chain is 500 times its own single step, not 128 times.
- **The error grows faster than linearly along the crossing** (a quarter of the way 19 µm,
  half way 57 µm, at the end 166 µm for N = 64, q = 2): a slope error made early is carried
  the rest of the way.
- **The exact scheme is not the limit** at N ≥ 64 (0.0–0.5 µm), and it is the whole story at
  N = 2, q = 2 (9,271 µm against the network's 9,250 µm).
- **Against Block D at the same N and q**: 166 µm against 2,304 µm (N = 64, q = 2), 159 µm
  against 3,212 µm (N = 128, q = 8) — 14 to 20 times better, with the caveat that Block D's
  chains were stopped early by the optimiser bug.
- **No over-training**: test error is 0.947–1.026 of the training error across the 16
  networks, and the loss on unseen tracks is if anything lower.
- **Cost**: 26 µs per track at N = 2, 803 at N = 64, 1,592 at N = 128, 3,263 at N = 256
  (fp64, one thread, shared host). Accuracy is flat across those, so the cheap chains win.
- **Anatomy** (`error_anatomy.py`, N = 64, q = 2): one step adds 0.42 µm and 0.0016 mrad. Summing
  the position parts of all 64 steps gives 29 µm; summing each step's x-slope error times the
  distance left to z1 gives 120 µm, 72% of the 166 µm endpoint error. The slope increments along a
  track have coherence 0.375 against 0.125 for independent steps: aligned three times more than
  chance, but the median signed increment (−3e-5 mrad) is small against the median size (1.6e-3),
  so not a systematic bias.

## The case study (N = 64, q = 2, chosen on validation)

- 166 µm median, 1,154 µm at the 95th percentile, 3,304 µm at the 99th.
- **Momentum**: the error is U-shaped. In x it is 1,470 µm at 1–2 GeV, 72 µm at 10–15 GeV
  and 827 µm at 100–200 GeV.
- **The 10–20 GeV band** (317 tracks), median |error| with the signed median and the signed
  mean beside it:

  | | x [µm] | y [µm] | tx [mrad] | ty [mrad] |
  |---|---|---|---|---|
  | median \|error\| | 83 | 56 | 0.026 | 0.018 |
  | signed median | +19 | −16 | −0.017 | −0.004 |
  | signed mean | −59 | +15 | −0.036 | +0.005 |

  The signed median is small against the median \|error\|, so the typical track is missed in
  either direction rather than bent systematically. The signed mean is pulled by the heavy
  tails and even changes sign against the median in x and y, so it should not be read as a bias.
- **The worst 5%** are low-momentum: median 3.6 GeV, 67% below 5 GeV, median error 1,940 µm.
- **Starting x matters as much as momentum** (`case_study_3d.py`, `figures/case_study_components_3d.png`,
  `case_study_components_x0_maps.png`, `results/case_study_error_vs_p_x0.csv`). The median radial
  error against the track's x on the last UT plane:

  | \|x at z0\| | 0–75 mm | 75–150 | 150–250 | 250–400 | 400–700 |
  |---|---|---|---|---|---|
  | all tracks | 127 µm | 165 | 237 | 328 | 651 |
  | 10–20 GeV only | 101 µm | 129 | 169 | 511 (250–700 mm) | — |

  Starting x and momentum are correlated (Spearman −0.40: tracks far off-axis at z0 tend to be
  soft), but the trend survives at fixed momentum, so it is a real second variable: a track
  starting at the edge of the acceptance is missed about five times worse than one near the
  beam line of the same momentum.
- **The loss is made of the wrong tracks.** 97% of the trained residual comes from the
  2–5 GeV band; everything above 5 GeV contributes under 3%, though that is where most
  tracks are and where the error rises again at high momentum. Reweighting the loss per
  track is the obvious next lever.
