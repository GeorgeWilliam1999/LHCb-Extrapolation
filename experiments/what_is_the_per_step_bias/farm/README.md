# Farm

**Last updated:** 2026-10-28

## What this directory is

What puts the four runs of Part B on the HTCondor farm and keeps them going. The scripts are those of `../../does_a_better_optimiser_improve_the_endpoint/farm/`, copied on 2026-10-28. `make_code.sh`, `wrapper.sh` and `jobs.sub` are identical. `keeper.sh` differs in the two lines that name the experiment (`HERE`, `EXPERIMENT`). One job is one run.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file | built |
| `runs.txt` | the runs, one a line: the configuration file, the cpus and the memory in MB; 1 cpu and 8 GB each, as the L-BFGS runs of tranche 1 | built |
| `check_before_submission.py` | builds each run, prints its key, runs the package's checks before training and, for B.4, prints lambda; for B.3 it draws round 2 once. Writes nothing | built |
| `gates_2026-10-28.txt` | the output of the full gate suite at the commit the runs train at | built |
| `make_code.sh` | checks the project out at one commit in a folder of its own, for the jobs to run from | built |
| `jobs.sub` | the description of one job for the farm | built |
| `wrapper.sh` | the job itself: makes the run or resumes it, and hands the farm's stop signal to the trainer | built |
| `keeper.sh` | submits the runs and keeps them going | built |

The logs of the farm and of the keeper are written outside the repository, to `/data/bfys/gscriven/rkpinn_farm_logs/what_is_the_per_step_bias/`.

## The contract

A job runs the code of one commit, from a copy made by `make_code.sh`. Two jobs never train one run: the keeper submits nothing when it cannot read the queue and leaves a run whose lock was touched in the last 30 minutes; under that, the package locks a run. A run that ends at its cap without a plateau is not extended by the keeper. That is for George to rule on.

## To run

```bash
cd /data/bfys/gscriven/LHCb_Extrapolation_Project
CODE=$(experiments/what_is_the_per_step_bias/farm/make_code.sh <commit> /data/bfys/gscriven/rkpinn_code)
LOGS=/data/bfys/gscriven/rkpinn_farm_logs/what_is_the_per_step_bias
setsid nohup $CODE/experiments/what_is_the_per_step_bias/farm/keeper.sh \
    $CODE /data/bfys/gscriven/rkpinn_store $LOGS > /dev/null 2>&1 &
```

To look:

```bash
tail $LOGS/keeper.log
condor_q -constraint 'RkpinnExperiment == "what_is_the_per_step_bias"'
cat /data/bfys/gscriven/rkpinn_store/runs/<key>/rounds.csv
```
