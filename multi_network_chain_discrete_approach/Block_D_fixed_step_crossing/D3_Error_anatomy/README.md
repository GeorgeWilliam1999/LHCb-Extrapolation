# D3_Error_anatomy — where the Block D error lives, and whether the legs really converged  (D3)

**Questions (George, 2026-09-16).** Is the median error carried by unrealistic or
problematic tracks? Is it x and tx, or y and ty? Are all the convergences good? What do
individual central tracks between 10 and 20 GeV look like? Did testing many collocation
points (stage counts) change anything?

Nothing in `D0_Crossing_dataset`, `D1_Chain_grid` or `D2_Comparators` is written; every
script here only reads their results.

## script → output

| script | what it does | output |
|---|---|---|
| [leg_quality.py](leg_quality.py) | reads every one of the 4,260 leg records and loss histories: restarts where the optimiser quit (a restart far shorter than the leg's first ones, loss unchanged), and each network's own-step error against the straight line's on the same leg | `results/leg_quality.csv`, `results/leg_quality_summary.csv` |
| [continue_legs.py](continue_legs.py) | rebuilds one leg's training set exactly, reloads its final weights, runs a fresh optimiser on the loss as trained (two restarts), then on the same loss multiplied by a constant so it starts at 1; scores the leg every five restarts | `results/continuation/N<NNN>_q<qq>_leg<kkk>.{csv,json}`, logs in `results/continuation/logs/` |
| [track_errors.py](track_errors.py) | every test particle's error at z1 for the best chain at each N, split into x, y, tx, ty and set beside its momentum, charge, species, eta, positions, slopes and field deflection | `results/track_errors.csv`, `results/error_distribution.csv`, `results/error_by_slice.csv`, `results/component_split.csv` |
| [example_tracks.py](example_tracks.py) | test tracks with 10–20 GeV and 3 < eta < 4; the four nearest the 25th, 50th, 75th and 95th percentile of the best single step's error; their states, deflections, errors for every best chain and the exact scheme, and the signed error on every plane | `results/example_tracks.csv`, `results/example_tracks_along_z.csv`, `figures/example_tracks_along_z.png` |

```bash
PY=/data/bfys/gscriven/conda/envs/TE/bin/python
PYTHONNOUSERSITE=1 $PY leg_quality.py
PYTHONNOUSERSITE=1 $PY track_errors.py
PYTHONNOUSERSITE=1 $PY example_tracks.py
PYTHONNOUSERSITE=1 $PY continue_legs.py --N 128 --q 7 --leg 120 --restarts 40   # one leg, ~10 min
```

Continuation legs run (2026-09-16): N = 128 q = 7 legs 2, 64, 100, 120, 127; N = 128 q = 12
leg 123; N = 128 q = 2 leg 127; N = 64 q = 7 leg 62; N = 16 q = 15 leg 8; N = 4 q = 5 leg 1;
N = 1 q = 16 leg 0.

## Why the optimiser stops early

The shared trainer's L-BFGS (`_shared/train.py`) sets `tolerance_grad=1e-13` and
`tolerance_change=1e-16`, both absolute. In PyTorch 2.9.1 the step loop also breaks when the
directional derivative is above `-tolerance_change`, and the strong-Wolfe line search is
called without a tolerance, so it uses its own absolute default of 1e-9 on the bracket width
times the direction norm. The physics loss is normalised by population spreads, so it is
about 3e-6 on the 5,178 mm crossing, 1e-7 on a 324 mm leg and 5e-9 on a 40 mm leg. The
smaller the loss, the sooner these absolute tolerances end a restart while the loss is still
falling. Multiplying the loss by a constant leaves its minimum where it is and lifts the
gradient clear of them.
