# Analysis

**Last updated:** 2026-11-12

## What this directory is

The script of the experiment and, once it has run, its tables, figures and results note. It reads the store and the field map. It writes only to the cache outside the repository, `/data/bfys/gscriven/rkpinn_analysis_cache/how_does_the_stage_residual_become_the_endpoint_error/`, one file per run.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file | built |
| `summarise.py` | the tables `results/per_run.csv` and `results/per_band.csv` and the figure `figures/step_length.png`, from the cached arrays | built |
| `results.md` | the results note: the numbers against the criteria | built |
| `decompose.py` | per run, along its own chain on the test tracks: the stage residual, the stage error and its first-order value, the local defect and its first-order value, and the carried defects (exact on 200 tracks, linear on all) | built; smoke run on 20 tracks passed (exact residual below 1e-12; the carried defects sum exactly to the network's endpoint minus the exact chain's) |

## How to run

From this folder, on stbc-i1:

```bash
THREADS=8 PYTHONNOUSERSITE=1 PYTHONPATH=../../../RK_Pinn_module/src \
    /data/bfys/gscriven/conda/envs/TE/bin/python decompose.py <run key> [<run key> ...]
```

`LIMIT=20` runs on the first 20 test tracks only, as a smoke run.
