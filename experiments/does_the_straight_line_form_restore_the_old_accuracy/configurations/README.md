# Configurations

**Last updated:** 2026-10-22

## What this directory is

One file per run of the experiment. A run takes every setting from its file. The key of a run is computed from its file.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file | built |
| `64_steps_2_stages_pooled_per_track_scale.yaml` | 64 steps of 80.9 mm, 2 stages, pooled loss, straight line plus a correction scaled per track: the form of the frozen networks, with the end state summed | built |
| `64_steps_2_stages_pooled_global_scale.yaml` | the same, with one global scale per component: the median of the per-track scale over the first round's states | built |

## The contract

Each file is the 64-step pooled configuration of the nine-runs experiment (`../../redo_the_papers_nine_runs_in_the_summed_form/configurations/64_steps_2_stages_pooled.yaml`, run key `49e5e2450eaa`), changed in two settings and in nothing else: `network.output_form` and `network.scale_of_outputs`. Seed 1 and the cap of 80 rounds are kept, so the three runs differ only in the output form.

To see what a file changes, from `RK_Pinn_module`:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src /data/bfys/gscriven/conda/envs/TE/bin/python -c "
from rkpinn.run_record.configuration import read_configuration, difference
nine = '../experiments/redo_the_papers_nine_runs_in_the_summed_form/configurations/64_steps_2_stages_pooled.yaml'
here = '../experiments/does_the_straight_line_form_restore_the_old_accuracy/configurations/64_steps_2_stages_pooled_per_track_scale.yaml'
print(difference(read_configuration(nine), read_configuration(here)))"
```

## How to add to it

1. Copy a file, change what the run asks about, and name the file by what the run is.
2. Add its row above and its line to `../farm/runs.txt`.
