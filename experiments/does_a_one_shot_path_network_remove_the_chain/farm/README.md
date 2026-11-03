# Farm

**Last updated:** 2026-11-03 · **State:** planned

## Contents

| Name | What it does | State |
|---|---|---|
| `make_code.sh` | checks the project out at one commit, as a git worktree, for the jobs to run from | copied from the pilot |
| `jobs.sub` | the HTCondor description of one job | copied from the pilot |
| `wrapper.sh` | the job of one run: makes or resumes the run | copied from the pilot |
| `keeper.sh` | submits the runs of `runs.txt` and keeps them going until each has ended | copied from the pilot, experiment name changed |
| `monitor.sh` | prints the state of every run on one screen; reads and changes nothing | new |
| `runs.txt` | the configurations the keeper submits | planned |

## To run

From the project folder on stbc, after the package additions are committed and the gates pass:

```bash
CODE=$(experiments/does_a_one_shot_path_network_remove_the_chain/farm/make_code.sh <commit> /data/bfys/gscriven/rkpinn_code)
STORE=/data/bfys/gscriven/rkpinn_store
LOGS=/data/bfys/gscriven/rkpinn_farm_logs/does_a_one_shot_path_network_remove_the_chain
setsid nohup $CODE/experiments/does_a_one_shot_path_network_remove_the_chain/farm/keeper.sh $CODE $STORE $LOGS > /dev/null 2>&1 &
```

George starts the keeper himself: a Remote Control session may not start it. To watch:

```bash
experiments/does_a_one_shot_path_network_remove_the_chain/farm/monitor.sh $STORE $LOGS
```
