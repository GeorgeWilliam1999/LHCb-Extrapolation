# Analysis

**Last updated:** 2026-10-28

## What this directory is

The scripts, tables and figures of the per-step bias study. They read the store and the field map. They write only to `results/`, `figures/`, `logs/` and the cache folder outside the repository.

## Contents

| Name | What it does | Writes |
|---|---|---|
| `RESULTS.md` | the results note: every number, the pass criteria, the decision point | — |
| `common.py` | the runs, tracks, single-step errors e, the chain, the scores | — |
| `stage_equations.py` | the 2-stage Gauss–Legendre stage equations in torch: fixed-point sweep, Newton step, exact solve | — |
| `part_a.py errors` | caches e of the three runs and of the exact scheme | cache |
| `part_a.py jacobians` | caches the reference Jacobian of every test step | cache |
| `part_a.py a1` | A.1 smooth models of e | `results/a1_smooth_model_r2.csv`, models in the cache |
| `part_a.py a2` | A.2 binned mean of e | `results/a2_binned_mean_of_e.csv`, `figures/a2_binned_mean_<key>.png` |
| `part_a.py a3` | A.3 e per momentum band, power of q/p | `results/a3_*.csv` |
| `part_a.py a4` | A.4 sign structure; endpoint along and across the bend | `results/a4_*.csv` |
| `part_a.py reconcile` | A.4b per-track mean against the spread along a track | `results/a4b_per_track_bias_reconciliation.csv` |
| `part_a.py a5` | A.5 e carried to the end; share per z bin | `results/a5_*.csv` |
| `part_a.py a6` | A.6 cell-face crossings | `results/a6_cell_face_crossings.csv` |
| `part_a.py a9` | A.9 bias maps of forms, snapshots and the exact scheme | `results/a9_bias_map_correlations.csv` |
| `toy_fit.py fit` | A.7 toy UT + SciFi fit | `results/a7_*.csv` |
| `seeds.py common` / `seeds.py last` | A.8 bias maps of other seeds | `results/a8_seed_correlations_<mode>.csv` |
| `part_bc.py references` | the exact scheme and RK4 at 64 steps | `results/c_comparators.csv` |
| `part_bc.py variants <key>` | B.1 and C.1 to C.4 for one run | `results/c_variants_<key>.csv` |
| `control_c2.py` | C.2b control: sweeps and Newton steps from stages without a network | `results/c2_control_*.csv`, `figures/c2_control.png` |
| `figures.py` | the figures of the note | `figures/*.png` |
| `logs/` | what each script printed | — |

The cache is `/data/bfys/gscriven/rkpinn_analysis_cache/what_is_the_per_step_bias/`. A file there is never recomputed; delete it to recompute.

## How to run

From this folder, on a login node that is not loaded (2026-10-28: `stbc-i3`):

```bash
export PYTHONNOUSERSITE=1 PYTHONPATH=../../../RK_Pinn_module/src THREADS=4
PY=/data/bfys/gscriven/conda/envs/TE/bin/python
$PY part_a.py errors; WORKERS=16 THREADS=1 $PY part_a.py jacobians
for s in a1 a2 a3 a4 reconcile a5 a6 a9; do $PY part_a.py $s; done
$PY toy_fit.py fit; $PY seeds.py common
$PY part_bc.py references
for k in 25dfa17ef168 9cef9c160b7d 49e5e2450eaa; do $PY part_bc.py variants $k; done
$PY control_c2.py
$PY figures.py
```

Order: `errors` and `jacobians` first; `a1` before `part_bc.py variants`; `jacobians` before `a5` and `toy_fit.py`. The whole set takes about 40 minutes on `stbc-i3`. Package code at commits `9a9ad1f5` to `8dd6f282` of the repository; between them only the evaluation reports changed, not the networks, the field map or the integrators.
