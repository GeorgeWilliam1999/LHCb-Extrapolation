# Configurations

**Last updated:** 2026-09-30

## What this directory is

One file per run of the experiment. A run takes every setting from its file. The key of a run is computed from its file.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file | built |
| `64_steps_2_stages_summed_end_state.yaml` | run A: the end state is summed from the stages; the loss sums over the stages | built |
| `64_steps_2_stages_predicted_end_state.yaml` | run B: the end state is predicted by the network; the loss sums over the stages and the end state | built |

## The contract

The two files differ in two settings, `network.end_state` and `loss.terms`, and in nothing else. They cannot differ in one alone: a predicted end state with a loss over the stages alone would leave the end state untrained, and the package refuses it.

To see the difference, from `RK_Pinn_module`:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src /data/bfys/gscriven/conda/envs/TE/bin/python -c "
from rkpinn.run_record.configuration import read_configuration, difference
d = '../experiments/does_a_summed_end_state_train/configurations/'
print(difference(read_configuration(d + '64_steps_2_stages_summed_end_state.yaml'),
                 read_configuration(d + '64_steps_2_stages_predicted_end_state.yaml')))"
```

## How to add to it

1. Copy a file, change one block, and name the file by what the run is.
2. Add its row above and its line to `../farm/runs.txt`.
