# Analysis

**Last updated:** 2026-10-28

## What this directory is

The numbers and the figure of the experiment: the step length against the single-step error and against the error at the end of the region. The scripts read the store and the archive and write nothing to either.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file | built |
| `step_length_errors.py` | the per-track errors of every series and their 16th, 50th and 84th percentiles per momentum band: the trained networks, the frozen networks with their stages summed, the exact scheme and classic fourth-order Runge–Kutta | built |
| `draw_figure.py` | the figure of four panels from the two tables (the error, and the error divided by the step length, George 2026-10-27), and the figure of the stages at 64 steps once two or more of those runs have a round | built |
| `results/` | `networks.csv` and `references.csv`, every plotted number | built |
| `figures/` | the figure, for all momenta and for 3 to 8 GeV | built |

## The contract

Every error is radial, sqrt(dx² + dy²) in micrometres, on the 1,452 test tracks, against the sixth-order reference at 0.1 mm. A network is read at its last round, and its state (plateaued, at its cap, training) is in the table. Where a standard report of that round exists, its two medians are copied beside the ones computed here, as a check.

The per-track errors are kept outside the repository, in `/data/bfys/gscriven/rkpinn_analysis_cache/does_the_error_fall_with_the_step_length/`, one file per series. A file there is never recomputed; delete it to recompute.

## How to run

From `RK_Pinn_module`:

```bash
export PYTHONNOUSERSITE=1 PYTHONPATH=src OMP_NUM_THREADS=1
PY=/data/bfys/gscriven/conda/envs/TE/bin/python
A=../experiments/does_the_error_fall_with_the_step_length/analysis
$PY $A/step_length_errors.py references 8    # 8 workers for the exact single steps
$PY $A/step_length_errors.py networks
$PY $A/draw_figure.py
```

The single-step median of an integrator is low at short steps because most steps cross no face of the field map's grid. The map is linear between grid points, so its derivative jumps at each face, and a step that crosses one carries most of the error (at 128 steps of 2 stages: median 0 on many steps, about 1e-3 micrometres on others). The end-of-region error of the integrators therefore does not fall cleanly with the step length.

The exact scheme must be in the store for every steps and stages of `REFERENCE_STEPS` and `REFERENCE_STAGES`; `rkpinn.track_data.exact_states` writes it.

## How to add to it

1. Add a script, with its row above.
2. Change the date.
