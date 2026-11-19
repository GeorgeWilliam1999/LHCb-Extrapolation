# Does the result hold on the larger track sets

**Last updated:** 2026-11-19 · **State:** four runs submitted on 2026-11-19 (George: "you can run the jobs now")

## What this directory is

The baseline and the physics-layer network, trained again on the two track sets of 2026-11-14: about 9 times more particles, with the reference on the linear field map and on the cubic field map. Each configuration is a copy of an earlier one, changed only in `tracks.key` and `equation_of_motion.field_map`.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file: the question, the runs, the criteria | built |
| `configurations/` | one file per run, four in all | built |
| `farm/` | what puts the runs on the farm and keeps them going; copied from `../can_the_network_alone_reach_the_exact_scheme/farm/`, with the experiment name changed in `keeper.sh` and 12 GB a job in `jobs.sub` (the smoke check peaked at 6.1 GB) | built |
| `analysis/` | `smoke.py` (copied) and its log `smoke.log` | built |

## The question

The baseline gave 1.83 µm and the physics layer 0.146 µm on the 1,452 test tracks of `12a8d35c3165`. Do the same settings give the same result with 9 times more training tracks, and does the result change when the reference is on the cubic field map?

## The runs

All: 64 steps of 80.9 mm, 2 stages, pooled loss over the stages, end state summed, straight line plus a correction scaled per track, L-BFGS, seed 1, cap 80 rounds.

| File | Copied from (key) | Track set | Field map | Key |
|---|---|---|---|---|
| `baseline_linear_map.yaml` | `does_the_straight_line_form_restore_the_old_accuracy/.../64_steps_2_stages_pooled_per_track_scale.yaml` (`25dfa17ef168`) | `afc61a558211` | `v8r1_up` | `84306b16e99d` |
| `baseline_cubic_map.yaml` | the same | `5558be31abeb` | `v8r1_up_cubic` | `400182ba3233` |
| `physics_layer_64_steps_linear_map.yaml` | `can_the_network_alone_reach_the_exact_scheme/.../physics_layer_64_steps.yaml` (`857f4642d383`) | `afc61a558211` | `v8r1_up` | `ffc653fadc1e` |
| `physics_layer_64_steps_cubic_map.yaml` | the same | `5558be31abeb` | `v8r1_up_cubic` | `a8a4083032fd` |

Each track set was built with the field map named in its row (`description.json` of the set). Training tracks 108,915, validation 13,684, test 13,447.

## The criteria, written before the runs are submitted

All errors are `endpoint_radial_median_micrometres` of the standard report on the test split of the run's own track set, at the round where the run plateaued or reached its cap.

| # | Criterion |
|---|---|
| 1 | The baseline holds on the linear map when `baseline_linear_map` is between 1.46 and 2.57 µm: the three L-BFGS seeds of the baseline (1.83, 2.20, 2.10 µm), widened by their range of 0.37 µm, as in the previous experiment. The larger test set narrows the sampling error of a median; it does not widen the seed range, so the band is kept |
| 2 | The physics layer holds on the linear map when `physics_layer_64_steps_linear_map` is within 10 % of the exact two-stage scheme at 64 steps on the new test split. The store holds no exact states for the new sets yet, so this number is computed at scoring time; until then the reference value is 0.147 µm from the old test split, and the criterion reads below 0.162 µm |
| 3 | Same on the cubic map, for both networks: criteria 1 and 2 with the exact scheme on the cubic map |
| 4 | Reported, no pass mark: the difference between the linear-map and cubic-map run of each network, beside the 22.9 µm median by which the two references differ |
| 5 | Convergence is read from the validation error. A run at its cap without a plateau is reported as such and is not extended without George's word |

## What is not known before the runs

- The time of a round. A round carries all 108,915 training tracks to draw its states, and the validation chains 13,684 tracks: both about 9 times the old work. The loss of one round's 32,000 states costs about the same as before (0.42 to 0.90 s a loss and gradient on one thread, `analysis/smoke.log`).
- Whether the cubic map costs more: one loss and gradient takes 0.62 s against 0.42 s for the baseline, and 0.90 s against 0.47 s for the physics layer.

## To run

See `farm/README` of the previous experiment; the commands, with this folder's name:

```bash
cd /data/bfys/gscriven/LHCb_Extrapolation_Project
E=does_the_result_hold_on_the_larger_track_sets
CODE=$(experiments/$E/farm/make_code.sh <commit> /data/bfys/gscriven/rkpinn_code)
LOGS=/data/bfys/gscriven/rkpinn_farm_logs/$E
setsid nohup $CODE/experiments/$E/farm/keeper.sh \
    $CODE /data/bfys/gscriven/rkpinn_store $LOGS > /dev/null 2>&1 &
```

## Findings

None yet.
