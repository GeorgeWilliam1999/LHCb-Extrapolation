# Farm

**Last updated:** 2026-10-23

## What this directory is

What puts the runs of the experiment on the HTCondor farm and keeps them going. The five scripts are those of `../../does_the_straight_line_form_restore_the_old_accuracy/farm/`, copied on 2026-10-23; `keeper.sh` differs in the two lines that name the experiment (`HERE`, `EXPERIMENT`), and the others are identical (`diff` shows it). One job is one run. The runs train at the same time; a single run cannot be split over jobs, because each round trains on the predictions of the round before.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file | built |
| `runs.txt` | the configuration files of the runs, one a line: the eight runs of this experiment | built |
| `make_code.sh` | checks the project out at one commit in a folder of its own, for the jobs to run from | built |
| `jobs.sub` | the description of one job for the farm: one thread, 8 GB, the 24 hour category | built |
| `wrapper.sh` | the job itself: makes the run or resumes it, and hands the farm's stop signal to the trainer | built |
| `keeper.sh` | submits the runs and keeps them going: releases jobs held at the time limit, submits a run that left the queue without ending | built |

The logs of the farm and of the keeper are written outside the repository, to the folder given to `keeper.sh`.

## The contract

A job runs the code of one commit, from a copy made by `make_code.sh`. Work on the package does not change what a job runs.

Two jobs never train one run. The keeper submits nothing when it cannot read the queue, and leaves a run whose lock was touched in the last 30 minutes. Under that, the package locks a run and refuses a second writer.

A run that ends at its cap without a plateau is not extended by the keeper. That is for George to rule on.

## To run

From the project folder, with the commit the runs are to be trained at:

```bash
cd /data/bfys/gscriven/LHCb_Extrapolation_Project
CODE=$(experiments/does_the_error_fall_with_the_step_length/farm/make_code.sh <commit> /data/bfys/gscriven/rkpinn_code)
LOGS=/data/bfys/gscriven/rkpinn_farm_logs/does_the_error_fall_with_the_step_length
setsid nohup $CODE/experiments/does_the_error_fall_with_the_step_length/farm/keeper.sh \
    $CODE /data/bfys/gscriven/rkpinn_store $LOGS > /dev/null 2>&1 &
```

To look:

```bash
tail $LOGS/keeper.log
condor_q -constraint 'RkpinnExperiment == "does_the_error_fall_with_the_step_length"'
cat /data/bfys/gscriven/rkpinn_store/runs/<key>/rounds.csv
```

## How to add to it

1. Add the configuration file of a new run to `../configurations/` and its name to `runs.txt`.
2. Commit, and make a copy of the code at that commit.
3. Start the keeper on that copy. Runs that have ended are left as they are.
