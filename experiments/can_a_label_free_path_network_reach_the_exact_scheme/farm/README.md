# Farm

**Last updated:** 2026-11-23 · **State:** A1 and A2 submitted 2026-11-22 by a keeper running from commit 394699b3; C1 to C3 submitted 2026-11-23 by a second keeper from the commit that adds them

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file | built |
| `make_code.sh` | checks the project out at one commit, as a git worktree, for the jobs to run from | copied from the one-shot experiment |
| `jobs.sub` | the HTCondor description of one job | copied; `request_memory` raised from 8192 to 12288 MB (2026-11-20: one A1 process peaked at 7.9 GB at 4,096 states on the login node), then to 16384 MB (2026-11-23: C1 and C2 peaked at 13.4 GB) |
| `wrapper.sh` | the job of one run: makes or resumes the run | copied unchanged |
| `keeper.sh` | submits the runs of `runs.txt` and keeps them going until each has ended | copied, experiment name changed |
| `monitor.sh` | prints the state of every run on one screen; reads and changes nothing | copied unchanged |
| `runs.txt` | the configurations the keeper submits: C1 to C3 (the drift terms). A1 and A2 were in it at commit 394699b3, and the keeper of that commit keeps them. B1 to B3 are added only if criterion 1 passes | built |

## To run

From the project folder on stbc, at the head of the branch `label_free_path`, after George says "submit" in the thread:

```bash
E=can_a_label_free_path_network_reach_the_exact_scheme
CODE=$(experiments/$E/farm/make_code.sh label_free_path /data/bfys/gscriven/rkpinn_code)
STORE=/data/bfys/gscriven/rkpinn_store
LOGS=/data/bfys/gscriven/rkpinn_farm_logs/$E          # C1 to C3: $E/drift_terms, so the two keepers keep separate logs
setsid nohup $CODE/experiments/$E/farm/keeper.sh $CODE $STORE $LOGS > /dev/null 2>&1 &
```

George starts the keeper himself. To watch:

```bash
experiments/$E/farm/monitor.sh $STORE $LOGS
```
