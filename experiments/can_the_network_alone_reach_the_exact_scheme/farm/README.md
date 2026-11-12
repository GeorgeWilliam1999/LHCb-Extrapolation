# Farm

**Last updated:** 2026-11-12 · **State:** not submitted

## What this directory is

What puts the runs of the experiment on the HTCondor farm and keeps them going. The scripts are those of `../../does_the_error_fall_with_the_step_length/farm/`, copied on 2026-11-12; `keeper.sh` differs in the two lines that name the experiment, and the others are identical. One job is one run.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file | built |
| `runs.txt` | the configuration files of the runs, one a line: seven | built |
| `make_code.sh` | checks the project out at one commit in a folder of its own, for the jobs to run from | built |
| `jobs.sub` | the description of one job: one thread, 8 GB, the 24 hour category | built |
| `wrapper.sh` | the job itself: makes the run or resumes it, and hands the farm's stop signal to the trainer | built |
| `keeper.sh` | submits the runs and keeps them going | built |

## The contract

That of `../../does_the_error_fall_with_the_step_length/farm/README.md`. A farm submission needs George's word in the thread first.

## To run

From the project folder, with the commit the runs are to be trained at:

```bash
cd /data/bfys/gscriven/LHCb_Extrapolation_Project
E=can_the_network_alone_reach_the_exact_scheme
CODE=$(experiments/$E/farm/make_code.sh <commit> /data/bfys/gscriven/rkpinn_code)
LOGS=/data/bfys/gscriven/rkpinn_farm_logs/$E
setsid nohup $CODE/experiments/$E/farm/keeper.sh \
    $CODE /data/bfys/gscriven/rkpinn_store $LOGS > /dev/null 2>&1 &
```

## How to add to it

1. Add the configuration file of a new run to `../configurations/` and its name to `runs.txt`.
2. Commit, make a copy of the code at that commit, and start the keeper on that copy.
