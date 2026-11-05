# Configurations of Part C

**Last updated:** 2026-11-05 · **State:** the first eight ended 2026-11-04; the second tranche of six submitted 2026-11-05 (see ../farm/README.md)

## What this directory is

One configuration file per farm run of Part C. Each copies a package default (`lhcb_64_steps_2_stages_learned_gain.yaml` or `lhcb_64_steps_2_stages_error_conditioned.yaml`) and changes only the frozen stage network it is built on and, for two runs, the scale of the correction.

## Contents

| Name | Variant | Frozen stage network (run, round) | Scale of the correction | Run key | State |
|---|---|---|---|---|---|
| `README.md` | this file | | | | built |
| `c1a_learned_gain__straight_line_per_track__seed_1.yaml` | C1a | straight line, per-track scale (`25dfa17ef168`, 44) | same as the stage network (per track) | `d986d2239b65` | built |
| `c1a_learned_gain__straight_line_global__seed_1.yaml` | C1a | straight line, one global scale (`9cef9c160b7d`, 22) | same as the stage network (four numbers) | `c6a7d5d15caa` | built |
| `c1a_learned_gain__direct_states__seed_1.yaml` | C1a | direct states (`49e5e2450eaa`, 59) | same as the stage network (state spread) | `d22c95bb773d` | built |
| `c1a_learned_gain__direct_states__per_track_scale__seed_1.yaml` | C1a | direct states (`49e5e2450eaa`, 59) | per track | `1006d1956c3c` | built |
| `c2_error_conditioned__straight_line_per_track__seed_1.yaml` | C2 | straight line, per-track scale | same as the stage network | `b38147788963` | built |
| `c2_error_conditioned__straight_line_global__seed_1.yaml` | C2 | straight line, one global scale | same as the stage network | `6c881486c499` | built |
| `c2_error_conditioned__direct_states__seed_1.yaml` | C2 | direct states | same as the stage network | `1eaf33293506` | built |
| `c2_error_conditioned__direct_states__per_track_scale__seed_1.yaml` | C2 | direct states | per track | `cac1a385740d` | built |

Second tranche, approved 2026-11-05 (seeds of C2 where it reached the exact scheme; C1a with the rates as an input). The seed sets the draw of the small network only; the frozen network is the seed-1 run in every case.

| Name | Variant | Frozen stage network | Changed from the seed-1 file | Run key | State |
|---|---|---|---|---|---|
| `c2_error_conditioned__straight_line_global__seed_2.yaml` | C2 | straight line, one global scale | seed 2 | `fbc1873b8604` | built |
| `c2_error_conditioned__straight_line_global__seed_3.yaml` | C2 | straight line, one global scale | seed 3 | `e0ab2424c5e5` | built |
| `c2_error_conditioned__direct_states__seed_2.yaml` | C2 | direct states | seed 2 | `78c986b9a154` | built |
| `c2_error_conditioned__direct_states__seed_3.yaml` | C2 | direct states | seed 3 | `46f56b85fe01` | built |
| `c1a_learned_gain__straight_line_global__with_rates__seed_1.yaml` | C1a | straight line, one global scale | `inputs_of_the_gain: state_residual_and_rates` | `1a73bba17006` | built |
| `c1a_learned_gain__direct_states__with_rates__seed_1.yaml` | C1a | direct states | `inputs_of_the_gain: state_residual_and_rates` | `5167265ab0b2` | built |

Common to all: 64 steps of 80.903 mm, 2 stages, track set `12a8d35c3165`, the stage network frozen, a small network of 2 tanh layers of 64, 32,000 states a round, 25 restarts of 200 L-BFGS iterations, the validation plateau rule, a cap of 60 rounds, seed 1. C1a: loss `distilled_analytic_gain` without the stage terms. C2: loss `pooled_on_both_calls` over the stages.

The two per-track C1a and C2 files are the package defaults to the byte, so their keys are the defaults' keys.

## How to add to it

1. Copy a default; change one block.
2. Print its key with `python -m rkpinn.run_record.configuration <file>`.
3. Add its row above and to `../farm/runs.txt`; change the date.
