# Farm

**Last updated:** 2026-11-05 · **State:** submitted 2026-11-03 23:15 (approved by George the same evening)

## What this directory is

What puts the runs of Part C on the HTCondor farm and keeps them going. The scripts are those of `../../does_a_better_optimiser_improve_the_endpoint/farm/`, copied on 2026-11-03. `make_code.sh`, `wrapper.sh` and `jobs.sub` are identical. `keeper.sh` differs in the two lines that name the experiment (`HERE`, `EXPERIMENT`). One job is one run.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file | built |
| `runs.txt` | the runs, one a line: the configuration file, 1 cpu, 8 GB; the first eight (2026-11-03) and the second tranche of six (2026-11-05). The keeper skips a run that has ended | built |
| `make_code.sh` | checks the project out at one commit in a folder of its own, for the jobs to run from | built |
| `jobs.sub` | the description of one job: the cpus and memory of `runs.txt`, the 24 hour category | built |
| `wrapper.sh` | the job: makes the run or resumes it, and hands the farm's stop signal to the trainer | built |
| `keeper.sh` | submits the runs and keeps them going; releases jobs held at the time limit; doubles the memory of a job held for its memory | built |

## The contract

A job runs the code of one commit, from a copy made by `make_code.sh`. Two jobs never train one run (the keeper's two rules and the package's lock). A run that ends at its cap is not extended here.

Each run reads the snapshot of its frozen stage network from the store. The store must hold the runs `25dfa17ef168`, `9cef9c160b7d` and `49e5e2450eaa` with the rounds named in the configurations.

## To run (after George approves the run list)

```bash
cd /data/bfys/gscriven/LHCb_Extrapolation_Project
CODE=$(experiments/does_a_kalman_like_gain_remove_the_bias/farm/make_code.sh <commit> /data/bfys/gscriven/rkpinn_code)
LOGS=/data/bfys/gscriven/rkpinn_farm_logs/does_a_kalman_like_gain_remove_the_bias
setsid nohup $CODE/experiments/does_a_kalman_like_gain_remove_the_bias/farm/keeper.sh \
    $CODE /data/bfys/gscriven/rkpinn_store $LOGS > /dev/null 2>&1 &
```

## What was measured when the runs were submitted

| | |
|---|---|
| Approved | 2026-11-03, the eight runs of `runs.txt` (the optional seeds 2 and 3 were not submitted) |
| Code | commit `ae95116d`, from the copy `/data/bfys/gscriven/rkpinn_code/ae95116d` |
| Gates | 519 passed on `stbc-i2`, 2026-11-03, at `2ccd1916` (the package is unchanged at `ae95116d`); log `../analysis/logs/part_c_gates.txt` |
| Submitted | 2026-11-03 23:15 by the keeper on `stbc-i2`; clusters 6133390 to 6133395, 6133397, 6133398 in the order of `runs.txt` |
| Logs | `/data/bfys/gscriven/rkpinn_farm_logs/does_a_kalman_like_gain_remove_the_bias/` |
| First restart | 39 to 41 s for C1a, 60 to 65 s for C2 (200 iterations, 32,000 states, `wn-lot-056`); every check before training passed |
| Expected | about 17 min a round for C1a and 27 min for C2; at 22 to 60 rounds, 6 to 17 h (C1a) and 10 to 27 h (C2) |

## The second tranche (2026-11-05)

Approved by George on 2026-11-05: C2 seeds 2 and 3 on the global straight line and on direct states, and C1a with the rates as an input on the same two networks; six runs, appended to `runs.txt`. The first eight had all plateaued by 2026-11-04 16:31. Commit, clusters and first restarts below, filled at submission.
