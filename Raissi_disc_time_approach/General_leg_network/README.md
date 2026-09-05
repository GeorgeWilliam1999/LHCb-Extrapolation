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

Cluster **5781158**, 26 jobs, submitted 2026-09-05 16:25. All 26 finished and all
26 are **converged** (55-154 restarts, longest run 1.6 h).

Getting there took a small operational detour worth recording. The shared stall
criterion is "two consecutive restarts each improving the loss by less than 1%".
On this dataset the loss creeps down by a few tenths of a percent per restart for
a long time, so that criterion fires while the endpoint medians are still moving
by more than 1% — and the confirmation pass then re-stalls immediately, records
`converged = false` and stops the run. 20 of the first 26 ended that way, at 51 to
111 restarts, still improving. Because `train.py` checkpoints after every restart,
resubmitting the identical command continues the run and gives it a fresh
optimiser and another confirmation attempt; [resubmit_unconverged.py](resubmit_unconverged.py)
does exactly that, and after five such rounds (a couple of restarts each) all 26
confirmed. Two runs additionally hit the 150-restart cap during the confirmation
pass (`--outer-cap` counts both phases together, A1's finding) and were continued
with `--outer-cap 400`.

### Second wave (overnight)

A2's architecture scan on the frozen leg selected **4x200** (physics 115 µm over
10 seeds, against 133 µm at 4x100 and 222 µm at 4x50; depth 6 hurt). The same
grid at width 200 — physics seeds 0-9, data twin seeds 0-2, same dataset, default
restart cap 400 — was submitted as **cluster 5781443** (13 jobs, submitted 2026-09-05 in the evening;
expect 3-4 h each). Its argument lines are in [condor/jobs_wave2.txt](condor/jobs_wave2.txt)
and appended to `condor/jobs.txt`. `aggregate.py`, `plot.py` and
`../Chained_legs/chain.py` read the architectures out of the result files, so
rerunning the analysis tomorrow picks the `w200_*` runs up with no code change.

## Ceilings: which number the network is being compared with

`../Simple_first_pass/results/scheme_error_vs_q.csv` measured the exact scheme on
32 legs per type drawn **stratified in momentum**, which over-weights the soft
tracks that bend hardest. This experiment's test split has the natural mix, so
[measure_ceiling.py](measure_ceiling.py) solves the same scheme with the same
solver on **these** states, cell by cell (`results/scheme_ceiling_same_population.csv`).
`by_leg.csv` carries both: `ceiling_leg_um` / `ceiling_band_um` are the published
stratified-sample values, `ceiling_own_um` is the like-for-like one.

| leg | q=8 ceiling, this population | published (stratified) |
|---|---|---|
| A vertex fetch | 0.0013 µm | 0.0007 µm |
| B cross-magnet | **46 µm** (24 µm at 5-20 GeV, 239 µm at 1-5 GeV) | 29 µm |
| C plane-to-plane | 6e-7 µm | 2e-6 µm |

## Verdict

**One label-free network does serve all three leg types — but nowhere near each
leg's ceiling, and on the short legs it is worse than doing nothing.**

Test split, median endpoint error over seeds (range across seeds in brackets):

| leg | 4x50 physics | 4x50 twin | 4x100 physics | 4x100 twin | straight line | ceiling (this population) |
|---|---|---|---|---|---|---|
| A vertex fetch | 849 [601-1491] | 653 [641-678] | **404** [286-538] | 385 [295-406] | 10.9 | 0.0013 |
| B cross-magnet | 6742 [4576-9130] | 2395 [2041-2415] | **2600** [2106-3152] | 933 [869-1055] | 444075 | 46 |
| C plane-to-plane | 535 [475-881] | 486 [484-510] | **300** [252-335] | 235 [178-262] | 6.6 | 6e-7 |

Read across that table:

1. **Distance to the ceiling.** On the leg that matters, the cross-magnet step,
   the best physics network sits at 2.6 mm against a 46 µm ceiling — a factor 56
   above the scheme it is solving. Widening from 4x50 to 4x100 buys a factor 2.6;
   the trend says width is the binding constraint, which is what the overnight
   4x200 wave will test.
2. **Generalising the leg is expensive.** The same physics loss, same q, same
   optimiser, same data source on ONE frozen cross-magnet leg reaches 177-235 µm
   (`../One_step_network_v2`). Asking one network to carry (z0, dz) as inputs and
   serve three leg geometries costs a factor 14 on that same leg
   (`figures/frozen_vs_general.png`).
3. **On short legs the network loses to a straight line.** Legs A (341 mm) and C
   (70 mm) are nearly straight: ignoring the magnet entirely gives 10.9 µm and
   6.6 µm, while the network gives 404 µm and 300 µm. A single network whose
   output scale is set by the 5 m cross-magnet step cannot also resolve a 70 mm
   one; the residual is roughly a fixed fraction of the largest step it was
   trained on, not a fixed fraction of each step.
4. **The data twin is uniformly ahead of the physics loss**, by 2.8x at 4x100 on
   leg B and by 5-25% on legs A and C. On the frozen leg the two were
   indistinguishable (182 vs 192 µm). The gap opens exactly where the leg
   geometry varies, i.e. the physics loss is the harder optimisation problem once
   the step length is an input rather than a constant.
5. **Where the error lives inside the step** (`figures/stage_errors.png`): on legs
   A and C it is flat across the eight Gauss nodes — a scale error, not an
   accumulation. On leg B it grows monotonically from the first node to the
   endpoint, roughly doubling, which is the bend being under-resolved.


## Second wave (4x200) — 2026-09-06

A2's architecture scan on the frozen leg selected 4x200, so the same grid was run
on this dataset: cluster **5781443**, 13 jobs (physics seeds 0-9, data twin seeds
0-2, `--outer-cap 400`). **12 of 13 converged.** `w200_physics_s5` stopped at 111
restarts, far from its 400-restart cap, in the same "re-stalled but the medians
moved" state described above; it was resubmitted (cluster 5781566) but the farm
had a long idle queue, so it is **reported unconverged and left out of every
median below**. Its numbers are in `summary.csv` and `by_leg.csv` with
`converged = false`, as are all the rest.

Test split, median over converged seeds, endpoint error in µm:

| leg | 4x50 physics | 4x100 physics | **4x200 physics** | 4x50 twin | 4x100 twin | **4x200 twin** | straight | ceiling q=8 |
|---|---|---|---|---|---|---|---|---|
| A vertex fetch | 849 | 404 | **229** [201-277] | 653 | 385 | **227** | 10.9 | 0.0013 |
| B cross-magnet | 6742 | 2600 | **1740** [1632-2154] | 2395 | 933 | **777** | 444075 | 46 |
| C plane-to-plane | 535 | 300 | **213** [165-235] | 486 | 235 | **182** | 6.6 | 6e-7 |

Chained along real particle paths (test, median over converged seeds, µm; see
`../Chained_legs`):

| legs walked | 4x50 phys | 4x100 phys | **4x200 phys** | 4x50 twin | 4x100 twin | **4x200 twin** |
|---|---|---|---|---|---|---|
| 1 | 440 | 246 | **174** | 388 | 217 | **165** |
| 4 | 3919 | 2293 | **1738** | 3634 | 1484 | **1490** |
| 7 | 15999 | 10248 | **10450** | 10739 | 5283 | **6476** |

Leg D, median over converged seeds (µm): composite two steps 35034 (physics) /
30351 (twin); the same leg in one giant step 4563 / **2182**. The exact q = 8
scheme on these legs is 752 µm.

**Did width alone close the gap? No.** On the one-step score width keeps paying,
but with clear diminishing returns: leg B goes 6742 -> 2600 -> 1740 µm for
50 -> 100 -> 200, i.e. factors of 2.6 then 1.5 for each doubling, and is still
**38x above the 46 µm ceiling**; on the short legs the network is still 32x
(leg C) and 21x (leg A) *worse than ignoring the magnet entirely*, which no
amount of width can fix because the defect is structural — a single output scale
set by the 5 m cross-magnet step cannot also resolve a 70 mm one. Chained, width
stops helping at all beyond four legs (10450 µm at 4x200 against 10248 µm at
4x100 after seven legs). The residual arm running in this same folder
(`residual_*`, another agent's work — untouched here) attacks exactly that
structural defect by giving each sample its own output scale from the field
integral along its leg; that, and not width, is where the gap has to close.
