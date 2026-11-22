# Do the headline results hold on the cubic map

**Last updated:** 2026-11-22 · **State:** 39 configurations built, not submitted (waits on George)

## What this directory is

The headline networks of the paper, trained again on the larger track set with the reference on the cubic field map (`5558be31abeb`, `v8r1_up_cubic`), with five seeds each. This closes the paper gaps G2 (cubic-map runs) and G3 (seeds). A second, smaller group trains the best chained form at other numbers of steps and stages, with one seed.

The cubic map is the standard for every metric from 2026-11-22 (George agreed in the project thread). The linear-map runs keep their numbers, labelled "linear map". LHCb's own extrapolators interpolate the grid linearly (Rec v41r3 `TrackFieldExtrapolatorBase`, `UseGridInterpolation = true` gives `fieldVectorLinearInterpolation`; Allen v9r3 `MagneticField.cuh` the same), so the linear-map numbers stay in the LHCb comparison.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file: the question, the runs, the criteria | built |
| `configurations/make_configurations.py` | writes every configuration from the file it copies, and `farm/runs.txt` | built |
| `configurations/*.yaml` | one file per run, 39 in all | built |
| `farm/` | copied from `../does_a_better_optimiser_improve_the_endpoint/farm/`, with the experiment name changed in `keeper.sh`; `runs.txt` gives the cpus and memory of each job | built |
| `analysis/` | the standard reports, tables and figures, after the runs | empty |

## The runs

Every configuration is a copy of an earlier one, changed only in `tracks.key` (5558be31abeb), `equation_of_motion.field_map` (v8r1_up_cubic), the seed and, in group 2, the steps and stages. The script checks each change happens exactly once. With the seed set to 1, the copies of the baseline and of the physics layer give the keys of the runs already in the store (`400182ba3233`, `a8a4083032fd`), so those two are not run again.

### Group 1: the headline networks, 64 steps x 2 stages, seeds 1 to 5

| Network | Optimiser | Copied from | New runs | Cpus, memory | Expected hours a run |
|---|---|---|---|---|---|
| direct states | L-BFGS | `does_a_better_optimiser_improve_the_endpoint` | 5 | 1, 12 GB | about 48 |
| direct states | self-scaled Broyden | the same | 5 | 4, 16 GB | about 74 |
| straight line, per-track scale | L-BFGS | the same (seed 1 is `400182ba3233`) | 4 | 1, 12 GB | about 42 (measured) |
| straight line, per-track scale | self-scaled Broyden | the same | 5 | 4, 16 GB | about 52 |
| straight line, global scale | L-BFGS | `does_the_straight_line_form_restore_the_old_accuracy` | 5 | 1, 12 GB | about 30 |
| physics layer | L-BFGS | `can_the_network_alone_reach_the_exact_scheme` (seed 1 is `a8a4083032fd`) | 4 | 1, 12 GB | about 31 (measured) |

### Group 2: steps and stages, seed 1

| Network | Optimiser | Steps x stages | Runs | Expected hours a run |
|---|---|---|---|---|
| straight line, per-track scale | self-scaled Broyden | 16x2, 32x2, 128x2, 256x2, 64x4, 64x8, 64x16 | 7 | 25 to 100 |
| physics layer | L-BFGS | 16x2, 32x2 | 2 | about 25 |
| direct states | L-BFGS | 128x8, 256x16 (the paper's cells) | 2 | about 80 |

The hours are the old runs' wall time (`rounds.csv`) times 1.4, the measured ratio of a round on the larger set to a round on the old set (baseline: 4,300 s against 3,041 s). In all: 39 jobs, about 1,900 job-hours, about 4,900 core-hours. Most runs end in 2 to 3 days; the longest in about 4 days. A job over its time limit is started again from its last snapshot by the keeper.

## The criteria, written before the runs

All errors are `endpoint_radial_median_micrometres` of the standard report on the 13,447 test tracks, at the round where the run plateaued or reached its cap.

| # | Criterion |
|---|---|
| 1 | For each headline network, the five seeds give a median and a range. A difference between two networks is claimed only when their seed ranges do not overlap, or when it is over 2x |
| 2 | Each network is compared with the exact scheme and with RK4 at the same steps and stages on the cubic map (`../what_are_the_classical_references_on_the_cubic_map`) |
| 3 | Reported, no pass mark: each network beside its linear-map result on the old track set |
| 4 | Convergence is read from the validation error. A run at its cap without a plateau is reported as such and is not extended without George's word |

## To run

See `farm/README` of `../does_a_better_optimiser_improve_the_endpoint`; the commands, with this folder's name:

```bash
cd /data/bfys/gscriven/LHCb_Extrapolation_Project
E=do_the_headline_results_hold_on_the_cubic_map
CODE=$(experiments/$E/farm/make_code.sh <commit> /data/bfys/gscriven/rkpinn_code)
LOGS=/data/bfys/gscriven/rkpinn_farm_logs/$E
setsid nohup $CODE/experiments/$E/farm/keeper.sh \
    $CODE /data/bfys/gscriven/rkpinn_store $LOGS > /dev/null 2>&1 &
```

## Findings

None yet.
