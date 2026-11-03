# Configurations of the Part C smoke run

**Last updated:** 2026-11-03

## What this directory is

The two configurations of the smoke run of Part C. Each is a package default (`lhcb_64_steps_2_stages_learned_gain.yaml`, `lhcb_64_steps_2_stages_error_conditioned.yaml`) with a small draw: 4,000 states a round, 5 restarts of 100 iterations, 3 rounds.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file | built |
| `c1a_learned_gain__straight_line_per_track__smoke.yaml` | C1a on the frozen run `25dfa17ef168`, round 44; run key `61aba2432fcc` | built |
| `c2_error_conditioned__straight_line_per_track__smoke.yaml` | C2 on the same frozen run; run key `8a64dfa936fa` | built |

## How to add to it

Copy a default, make the draw small, add its row.
