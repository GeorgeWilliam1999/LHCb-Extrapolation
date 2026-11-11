# F3_Analysis — every Block E analysis, reproduced for the Block F networks

**Status (2026-09-21):** N = 64 q = 2 and N = 128 q = 8 are finished. **N = 256 q = 16 appears here
as a checkpoint, not a result:** it hit its 1,000-restart cap (which writes a record and chain
states), the keeper reopened it with `--extend`, and it is still training (restart 1,189 and
falling 11% per ten rounds on 2026-09-21). The runner's view labels it so; rerun the one command
when it has stopped. Note that `error_anatomy.py` and the F2 `anatomy_xy.py` load the weights on
disk, which for a reopened run are *ahead* of its record — the anatomy numbers for that run and
its chain-state numbers are from different restarts until it finishes.

George (2026-09-20): "reproduce all the analysis plots from block E for Block F".

## How it is done

E3's scripts are **not copied**. [run_e3_for_block_f.py](run_e3_for_block_f.py) imports each one
from `../../Block_E_single_network_chain/E3_Analysis/` and runs it with its `HERE` (outputs) and
`E1` (run folders) pointed here, so every plot is E3's own code on Block F's runs. The exact
scheme, Block D and the material floor are the same comparators for both blocks and stay where
E3 finds them. `runs/results/` is a view of the *finished* Block F runs (symlinks, rebuilt every
call), because E3 globs run folders and expects a record in each.

Four E3 scripts draw their figures over the N × q grid and assume every N has every q; Block F's
three runs sit on a diagonal, so [grid_free_panels.py](grid_free_panels.py) holds the same
computation and per-panel drawing for those four, one panel or one series per run. The runner
also wraps matplotlib's `suptitle`/`set_title` so E3's hardcoded "Block E" reads "Block F" here;
nothing else about the drawing changes.

## script → output (the E3 names, so the two folders can be read side by side)

| E3 script | driven by | output here |
|---|---|---|
| `tables.py` | runner | `results/error_qdz_chain.csv`, `tails.csv`, `cost_accuracy.csv`, `comparators.csv`, `figures/error_qdz.png` |
| `single_step_tables.py` | grid-free | `results/error_qdz_single_step.csv`, `single_step_vs_z.csv`, `figures/single_step.png` |
| `errors_vs_momentum.py` | grid-free | `figures/error_vs_p_{x,y,tx,ty}.png`, `results/error_vs_p.csv` |
| `along_z.py` | grid-free | `results/error_vs_z.csv`, `figures/error_vs_z.png` |
| `convergence.py` | grid-free | `figures/convergence_grid.png`, `convergence_summary.png`, `results/convergence.csv` |
| `evaluate_splits.py` | runner | `results/split_comparison.csv`, `overtraining.csv` |
| `case_study.py` | runner | `results/case_study_*.csv`, `figures/case_study_{overview,components,loss_vs_p}.png` |
| `case_study_3d.py` | runner | `figures/case_study_components_3d.png`, `case_study_components_x0_maps.png`, `results/case_study_error_vs_p_x0.csv` |
| `error_anatomy.py` | runner | `results/error_anatomy.csv`, `error_anatomy_summary.json`, `figures/error_anatomy.png` |
| `analysis.ipynb` | — | E3's notebook with its header rewritten for Block F; executed; loads the above and computes nothing |
| [mirror_mini_paper.py](mirror_mini_paper.py) (George, 2026-09-21) | — | the mini paper's Figures 2 and 6 mirrored for Block F: `figures/mirror_fig2_baseline.png`, `mirror_fig6_magnet_up.png`, `results/mirror_mini_paper.json`; the panel-by-panel mapping is in its docstring |

`results/run_log.json` records which scripts ran and on which runs.

```bash
PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python run_e3_for_block_f.py
PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python -m jupyter nbconvert --to notebook --execute --inplace analysis.ipynb
```

## Read with these caveats

- **Two panels evaluate Block E's loss on the Block F network, not the loss Block F was trained
  on:** the loss column of the over-training check (`overtraining.csv`; its chain-error test/train
  ratios, 0.969 and 0.980, are what to read — no over-training) and the case study's
  loss-against-momentum panel (`case_study_loss_vs_p.png`), which therefore shows the *Block E*
  weighting's view of a Block F network (2–5 GeV dominating), not what Block F optimised.
- `error_anatomy.py` is E3's **x-slope-only** lever-arm model. For Block F it accounts for 38% of
  the endpoint error (73% for Block E), because Block F's remaining error is mostly the y slope.
  The two-slope version is `../F2_Analysis/anatomy_xy.py` (98–99% for every run); read that one.
- `error_qdz.png`'s right panel promises dotted exact-scheme lines; with one point per N there is
  nothing to join, so they do not show. The exact values are in `error_qdz_chain.csv`
  (`exact_med_um`, 0.1 µm at both settings).
- The "best network" of the case study is chosen on validation among the finished Block F runs:
  N = 64 q = 2 today.

## What the reproduced set says (2026-09-20)

Headline numbers for the two runs, all on the 1,452 test tracks, radial at z1:

| | N = 64, q = 2 | N = 128, q = 8 |
|---|---|---|
| chain median / p95 / p99 [µm] | 88.8 / 1,399 / 4,033 | 104.6 / 1,611 / — |
| single step from the RK6 state, median [µm] | 0.43 (0.46% of the straight line's 95.2) | 0.22 (0.93% of 23.8) |
| chain / single step | 205 | 475 |
| growth along z: quarter / half / end [µm] | 9.4 / 25.8 / 88.8 | 11.7 / 30.8 / 104.6 |
| over-training, test/train chain median | 0.969 | 0.980 |
| exact scheme at the same N, q [µm] | 0.1 | 0.1 |

The shape is Block E's shape — a sub-micrometre step, accumulation faster than linear along z
(end/half 3.4 against Block E's 2.9), no over-training — at a lower level. The case study's
momentum split: 1–5 GeV median 373 µm, 5–20 GeV 50 µm, 20–200 GeV 17 µm (Block E's case study:
299 / 78 / — at the 2026-09-18 checkpoint). Starting-x dependence survives in Block F and is if
anything steeper: x error 13.5 µm near the beam line against 350 µm at |x₀| ≈ 550 mm, and the y
error 17 against 250–600 µm. The comparison with Block E as such, and the two-slope anatomy, are
in `../F2_Analysis/`.

## The mini paper's Figures 2 and 6, mirrored (2026-09-21)

The networks on their own, named by N and q (George: no block labels; a first version set them
against the earlier networks and was redrawn). Neither figure maps one-to-one — Figure 2 compares
seeds and samples with a fiducial cut, Figure 6 compares the two magnet polarities — so each panel
is the nearest analogue:

| paper panel | here |
|---|---|
| Fig 2, error histogram (best seed per loss) | endpoint radial error per network; exact-scheme ceiling and straight line marked |
| Fig 2, loss histories to stall | training loss per restart, per network |
| Fig 2, error vs momentum | binned medians of the endpoint error, per network |
| Fig 2, per run (open v1 / filled v2 / squares = stages) | circles = endpoint median; **squares = the single-step error** from the RK6 state |
| Fig 2, the fiducial effect | endpoint median by band (all / 10–50 GeV / below 5 GeV), one bar per network |
| Fig 2, cut trajectories leaving the map | x(z) of test tracks under the N = 64 q = 2 chain, the worst 5% in pink, 150 others in dark grey on top |
| Fig 6, histogram, both polarities | endpoint error per network (validation-best filled), straight line grey, ceilings marked, each network's median as a thin line |
| Fig 6, scatter vs momentum | every test track, the validation-best network, binned median on top |
| Fig 6, three tracks, both polarities | three test tracks of **both charges** (the analogue of the sign check): RK6 reference solid, the network's chain states as markers |

**Every error in both figures is an endpoint error** (after the full chain, at z1) except the
squares of the per-run panel, which are single-step errors. The three tracks are the negatively
charged 2.3 GeV track that bends to x = +1,288 mm, the positive 5.9 GeV one that bends to −151 mm,
and a negative 21.9 GeV one; the network's endpoint is within 0.3, 0.08 and 0.01 mm of the
reference for the three.
