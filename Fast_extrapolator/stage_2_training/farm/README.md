# Farm files of stage 2

Nothing is submitted. A submission needs George's word.

| File | What it is |
|---|---|
| `jobs.sub` | one job a configuration of `../configurations/` (20 runs), 8 cores each |
| `wrapper.sh` | the job: `python -m fastex.train <configuration>`; a job that is started again goes on from the last round |
| `logs/` | the logs of the farm |

The jobs run the code of this folder as it is on disk, not a copy at a commit.
The folder is not committed, so `provenance.json` of a run says that it cannot
be traced to a commit.
