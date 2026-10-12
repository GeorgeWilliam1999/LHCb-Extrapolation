# Configurations

**Last updated:** 2026-10-12

## What this directory is

One file per run of the experiment. A run takes every setting from its file. The key of a run is computed from its file.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file | built |
| `64_steps_2_stages_pooled.yaml` | 64 steps of 80.9 mm, 2 stages, pooled loss; the default at seed 1 | built |
| `64_steps_2_stages_cost_weighted_10_to_50_gev.yaml` | the same cell, cost-weighted loss, window 10 to 50 GeV | built |
| `64_steps_2_stages_cost_weighted_3_to_8_gev.yaml` | the same cell, cost-weighted loss, window 3 to 8 GeV | built |
| `128_steps_8_stages_pooled.yaml` | 128 steps of 40.5 mm, 8 stages, pooled loss | built |
| `128_steps_8_stages_cost_weighted_10_to_50_gev.yaml` | the same cell, cost-weighted, 10 to 50 GeV | built |
| `128_steps_8_stages_cost_weighted_3_to_8_gev.yaml` | the same cell, cost-weighted, 3 to 8 GeV | built |
| `256_steps_16_stages_pooled.yaml` | 256 steps of 20.2 mm, 16 stages, pooled loss | built |
| `256_steps_16_stages_cost_weighted_10_to_50_gev.yaml` | the same cell, cost-weighted, 10 to 50 GeV | built |
| `256_steps_16_stages_cost_weighted_3_to_8_gev.yaml` | the same cell, cost-weighted, 3 to 8 GeV | built |

## The contract

Every file is the LHCb default, `RK_Pinn_module/src/rkpinn/defaults/lhcb_64_steps_2_stages.yaml`, changed in four places and in nothing else: the cell (`steps`), the loss (`loss`), the seed (1 instead of 0) and the cap (80 rounds instead of 60). The cost-weighted files take the constants of the frozen study: roll-off ln 2, floor 0.05, clamp 5, 1,024 samples of the field integral, the lever arm, the track bend and the momentum window all on, and the reference bend measured as the median over the first round's states, as the frozen study measured it.

To see what a file changes, from `RK_Pinn_module`:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src /data/bfys/gscriven/conda/envs/TE/bin/python -c "
from rkpinn.defaults import path_of_the_default
from rkpinn.run_record.configuration import read_configuration, difference
print(difference(read_configuration(path_of_the_default('lhcb_64_steps_2_stages')),
                 read_configuration('../experiments/redo_the_papers_nine_runs_in_the_summed_form/configurations/128_steps_8_stages_pooled.yaml')))"
```

## How to add to it

1. Copy the default, change the cell or the loss, keep seed 1 and the cap, and name the file by what the run is.
2. Add its row above and its line to `../farm/runs.txt`.
