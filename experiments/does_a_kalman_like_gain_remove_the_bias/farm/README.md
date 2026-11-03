# Farm

**Last updated:** 2026-11-03 · **State:** prepared, NOT submitted

## What this directory is

What puts the runs of Part C on the HTCondor farm and keeps them going. The scripts are those of `../../does_a_better_optimiser_improve_the_endpoint/farm/`, copied on 2026-11-03. `make_code.sh`, `wrapper.sh` and `jobs.sub` are identical. `keeper.sh` differs in the two lines that name the experiment (`HERE`, `EXPERIMENT`). One job is one run.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file | built |
| `runs.txt` | the eight runs, one a line: the configuration file, 1 cpu, 8 GB | built |
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
