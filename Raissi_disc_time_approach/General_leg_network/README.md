# General_leg_network — one network for legs A, B and C  (A3a)

**Question.** The verified baseline (`../One_step_network_v2`) trains a network on
one frozen leg: every sample starts on the same plane and ends on the same plane.
That network cannot be used anywhere else. Here the leg itself becomes an input:
every sample carries its own start plane `z0` and its own step length `dz`, the
two are handed to the network as normalised extra inputs, and one network is
asked to serve all three forward-usable leg types at once —

| leg | what it is | typical step |
|---|---|---|
| A | vertex fetch: first VP state back to the primary vertex | 341 mm, backward |
| B | cross-magnet: last UT plane to first SciFi plane, both directions | 5175 mm |
| C | plane-to-plane: consecutive sensor crossings, forward | 70 mm |

Everything else is the baseline's, unchanged: q = 8 Gauss-Legendre stages, fp64,
the physics loss of the paper against the supervised data twin, full-batch L-BFGS
restarted to a genuine stall and then confirmed, the fiducial requirement, and
the same event-derived training set (`train_official_v2.npz`).

## script -> output

| script | what it does | output |
|---|---|---|
| [build_dataset.py](build_dataset.py) | times one L-BFGS restart at N = 2000/4000/8000/16000 and takes the largest N that stays under 90 s, then builds the dataset at that N | `results/general_legs.npz`, `results/general_legs_meta.json`, `results/dataset_meta.json` |
| `../_shared/train.py` via [condor/jobs.txt](condor/jobs.txt) | one job per (architecture, mode, seed): 4x50 and 4x100, physics seeds 0-9, data twin seeds 0-2 = 26 runs | `results/<tag>.json`, `<tag>.pt`, `<tag>_history.csv` |
| [aggregate.py](aggregate.py) | re-scores each checkpoint per leg type and momentum band (the run json holds only whole-split numbers) and attaches the exact-scheme ceiling | `results/summary.csv`, `results/by_leg.csv`, `results/stage_errors.csv` |
| [plot.py](plot.py) | the figures, from the csv tables only | `figures/error_by_leg_and_momentum.png`, `figures/stage_errors.png`, `figures/frozen_vs_general.png` |
| [analysis.ipynb](analysis.ipynb) | loads the tables and displays the figures; computes nothing | — |

## How the training size was chosen

The rule was fixed before any number was looked at: the largest N in
{2000, 4000, 8000, 16000} for which **one** L-BFGS restart (200 iterations,
physics loss, 4x100, one thread) runs in under 90 s, so that the 150-restart
safety cap is about four hours. Measured on the submit host, on truncated copies
of one 16,000-state pool so that only N differs:

| N | one restart | 150 restarts |
|---|---|---|
| 2000 | 13.8 s | 0.57 h |
| 4000 | 26.8 s | 1.12 h |
| 8000 | 47.3 s | 1.97 h |
| 16000 | 97.1 s | 4.05 h — over budget |

**N = 8000.** After the fiducial cut: 7935 train / 3969 val / 3961 test states
(`n_eval` 4000). The shared builder caps its pool with a plain random
permutation — it does **not** stratify by leg type or by momentum — so the mix
is the training set's own and is recorded in `results/dataset_meta.json` rather
than imposed: per leg (train) A 1317 · B 1060 · C 5558, and per momentum band
inside each leg. The sign of `dz` is recorded there too, because the chaining
experiment needs to know whether the network has ever seen a backward step: it
has — all 1317 A legs are backward and 490 of the 1060 B legs are.

## The farm

```bash
cd General_leg_network
mkdir -p condor/logs results
condor_submit condor/jobs.sub          # jobs.sub is ../_shared/condor/template.sub
```

Cluster **5781158**, 26 jobs, submitted 2026-09-05 16:25.

## Verdict

_Filled in when the runs finish; see the report._
