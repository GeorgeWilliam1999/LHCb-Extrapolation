# Configurations

**Last updated:** 2026-10-23

## What this directory is

One file per run of the experiment. A run takes every setting from its file. The key of a run is computed from its file.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file | built |
| `8_steps_2_stages_pooled_per_track_scale.yaml` | 8 steps of 647.2 mm, 2 stages | built |
| `16_steps_2_stages_pooled_per_track_scale.yaml` | 16 steps of 323.6 mm, 2 stages | built |
| `32_steps_2_stages_pooled_per_track_scale.yaml` | 32 steps of 161.8 mm, 2 stages | built |
| `128_steps_2_stages_pooled_per_track_scale.yaml` | 128 steps of 40.5 mm, 2 stages | built |
| `256_steps_2_stages_pooled_per_track_scale.yaml` | 256 steps of 20.2 mm, 2 stages | built |
| `64_steps_4_stages_pooled_per_track_scale.yaml` | 64 steps of 80.9 mm, 4 stages | built |
| `64_steps_8_stages_pooled_per_track_scale.yaml` | 64 steps of 80.9 mm, 8 stages | built |
| `64_steps_16_stages_pooled_per_track_scale.yaml` | 64 steps of 80.9 mm, 16 stages | built |

Every file: pooled loss over the stages, end state summed from the stages, a straight line plus a correction scaled per track, seed 1, cap 80 rounds.

## The contract

Each file is `../../does_the_straight_line_form_restore_the_old_accuracy/configurations/64_steps_2_stages_pooled_per_track_scale.yaml` (run key `25dfa17ef168`), changed in `steps.number_of_steps` or `steps.number_of_stages` and in nothing else.

To see what a file changes, from `RK_Pinn_module`:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src /data/bfys/gscriven/conda/envs/TE/bin/python -c "
from rkpinn.run_record.configuration import read_configuration, difference
there = '../experiments/does_the_straight_line_form_restore_the_old_accuracy/configurations/64_steps_2_stages_pooled_per_track_scale.yaml'
here = '../experiments/does_the_error_fall_with_the_step_length/configurations/8_steps_2_stages_pooled_per_track_scale.yaml'
print(difference(read_configuration(there), read_configuration(here)))"
```

## How to add to it

1. Copy a file, change what the run asks about, and name the file by what the run is.
2. Add its row above and its line to `../farm/runs.txt`.
