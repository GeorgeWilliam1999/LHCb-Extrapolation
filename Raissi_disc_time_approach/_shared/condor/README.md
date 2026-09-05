# Running an experiment's trainings on the Nikhef HTCondor farm

Two shared files live here and are **not** copied around:

| file | what it is |
|---|---|
| [wrapper.sh](wrapper.sh) | the job executable. First argument = the experiment folder to run in; every argument after it is handed to `../train.py` unchanged. It pins the job to a single thread (`PYTHONNOUSERSITE=1`, `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1`) — multithreaded torch spin-waits on this node and runs about 68x slower. |
| [template.sub](template.sub) | the submit description to copy into an experiment. Vanilla universe, 1 CPU, 4 GB, `+UseOS "el9"`, `+JobCategory "medium"`, `getenv = False`, no file transfer (everything is on shared `/data`). |

## What each experiment owns

Every experiment folder writes its **own** two files and its own log directory:

```
<experiment>/condor/jobs.sub     a copy of template.sub (usually unedited)
<experiment>/condor/jobs.txt     one line per run = the arguments for one job
<experiment>/condor/logs/        the .out/.err/.log files land here
```

A `jobs.txt` line is the experiment folder followed by the `train.py`
arguments, exactly as you would type them by hand:

```
One_step_network_v3 --data results/frozen_leg_data.npz --mode physics --seed 0 --out results --tag physics_seed0
One_step_network_v3 --data results/frozen_leg_data.npz --mode physics --seed 1 --out results --tag physics_seed1
One_step_network_v3 --data results/frozen_leg_data.npz --mode data    --seed 0 --out results --tag data_seed0
```

Relative paths inside a line are resolved from the experiment folder, because
that is where the wrapper `cd`s to before starting python.

## Submitting

```bash
cd /data/bfys/gscriven/LHCb_Extrapolation_Project/Raissi_disc_time_approach/<experiment>
mkdir -p condor/logs results
cp ../_shared/condor/template.sub condor/jobs.sub     # once
condor_submit condor/jobs.sub
condor_q
```

## Notes

- Runs are **resumable**: `train.py` checkpoints and appends a history row after
  every L-BFGS restart, so a job that is evicted or killed can simply be
  submitted again and will continue from its checkpoint. Held or failed jobs are
  therefore safe to release or re-queue.
- One job = one `(mode, seed)`; the runs of an experiment are independent, so
  the whole grid finishes in the wall time of its slowest single run.
- `getenv = False` is deliberate: the job must not inherit an interactive
  shell's environment. The wrapper sets everything the run needs, and the
  interpreter is named by absolute path (`/data/bfys/gscriven/conda/envs/TE/bin/python`).
- Nothing is transferred: the repository, the datasets and the results all live
  on `/data`, visible from the worker nodes.
- `wrapper.sh` must keep its executable bit. Jobs run inside the el9 container,
  and a non-executable wrapper fails with a bare `FATAL: permission denied` in
  the `.err` file and nothing else — the mistake is worth recognising quickly.
  `chmod +x` fixes it; git preserves the bit.
- Gate run 2026-09-05: one job (`--outer-cap 1 --no-confirm`) on the frozen-leg
  dataset completed on `wn-sate-069.nikhef.nl` in 9 s wall, wrote its `.pt`,
  `_history.csv` and `.json`, and reproduced the same first-restart loss
  (4.927638e-03) as the identical run on the submit host.

**Gotcha (2026-09-05):** `queue args from jobs.txt` does not skip `#` comment lines; a comment line is submitted as a job. Keep jobs.txt free of comments.
