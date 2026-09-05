# Network size and seed on the frozen magnet crossing

**Verdict.** The endpoint floor is set by the optimiser, not by the network's
capacity, and it does not reach the exact scheme. Growing the network from 2436
to 129,036 parameters walks the median test endpoint error down from 1059 um to
115 um, but 115 um is still **5 times** the 23 um that the exact scheme itself
achieves on these very test states, and across all 96 converged runs the final
training loss and the endpoint error move together with a Spearman rank
correlation of **+0.97** — the same number the van der Pol study measured. Every
architecture sits on one loss-error curve; what a bigger network buys is a lower
loss, not a better error at the same loss. **Depth 4 is the sweet spot**: depth 6
is worse at every width (by 4-47%) and harder to optimise, and depth 2 is
worse than depth 4 at every width (by 33-158%). For the general-leg experiment
(A3) **4x100 is the smallest adequate network** and 4x50 is not: 4x50 lands at
222 um, 1.9 times the 4x200 floor, on the steep part of the curve where the
result is still an artefact of the network size. Selecting a seed on
**validation endpoint error is reliable** (rank correlation +0.995 with test
error over the converged runs; the validation-best seed is also the test-best
seed in 9 of 12 architectures, and where it is not the penalty is at most 2.3%).

This is the experiment of `../One_step_network_v2` — the paper's one-step
discrete-time network on the frozen magnet crossing, q = 8 Gauss-Legendre
stages, MagDown, fp64, full-batch L-BFGS to a confirmed stall — with two things
varied and nothing else changed: the **network size** (widths 32, 50, 100, 200
crossed with depths 2, 4, 6) and the **seed** (0-9). 120 physics runs, plus 10
data-twin runs at the winning architecture.

## Script -> output

| script | what it does | output |
|---|---|---|
| [build_dataset.py](build_dataset.py) | one call to `_shared.prepare.frozen_leg_dataset(q=8, field='down', n_train=2000, seed=20260718, rebase_mm=60, fiducial=True)`, asserting the split counts against the verified baseline | `results/frozen_leg_q08.npz` (+ `_meta.json`); 1979 / 2062 / 2018 states, as in `../One_step_network_v2` |
| [write_job_list.py](write_job_list.py) | writes the farm argument lines: 4 widths x 3 depths x 10 seeds, mode `physics`; and, with `--mode data --widths 200 --depths 4`, the data twin at the winner | `condor/jobs.txt` (120 lines), `condor/jobs_data_twin.txt` (10 lines) |
| [write_continuation_list.py](write_continuation_list.py) | re-emits, with `--outer-cap 400`, the unconverged runs whose restart count reached their cap (see "The restart cap" below) | `condor/jobs_continuation.txt` |
| `condor/jobs*.sub` | copies of `../_shared/condor/template.sub`, one per job list | `condor/logs/*` |
| [aggregate.py](aggregate.py) | reads every `results/*.json` written by `_shared/train.py` | `results/summary.csv` (one row per run), `results/by_architecture.csv` (one row per architecture) |
| [plot.py](plot.py) | reads the two CSVs and recomputes nothing | `figures/floor_vs_architecture.png`, `figures/loss_vs_error.png`, `figures/seed_spread.png`, `figures/val_vs_test_selection.png` |
| [analysis.ipynb](analysis.ipynb) | loads the CSVs and the figures; the source for the write-up | — |

Clusters: **5781153** the 120-run grid, **5781167**, **5781442** and **5781445**
the continuation passes, **5781358** the 10 data-twin runs.

## What "converged" means here, and the restart cap

A run counts as converged only if it stalled — two consecutive L-BFGS restarts
each improving the training loss by less than 1% — **and** a fresh optimiser,
started from that point with its curvature history discarded, re-stalled within
two restarts without moving the train, validation or test endpoint medians by
more than 1%. Anything else is reported unconverged and never pooled with the
converged runs. `results/by_architecture.csv` therefore carries the median over
the converged seeds *and*, alongside it, the same median over every finished run
(`*_allruns`), so that an architecture whose seeds mostly failed the
confirmation cannot quietly report a median over a flattering subsample. The two
agree to better than 10% everywhere except the three 32-wide rows, where the
converged subsample is small.

`_shared/train.py` counts the stall phase and the confirmation pass against the
same `--outer-cap`, so a run that stalls near restart 150 hits the default cap
before it can be confirmed and is recorded unconverged although it is about to
converge. The stage-count study found this on the same day; the shared default
has since been raised to 400, but cluster 5781153 was already running with 150.
`write_continuation_list.py` re-emits exactly the runs whose restart count
reached their cap, with `--outer-cap 400`; because `train.py` checkpoints every
restart, the identical command resumes from the `.pt`. Three continuation passes
recovered 14 of the 17 truncated runs (clusters 5781167, 5781442 and 5781445);
after them no unconverged run is within 8 restarts of its cap. The runs
that stopped well short of the cap were **not** resubmitted: those failed the
confirmation on their own merits, which is a result rather than a truncation.

Convergence is itself a function of size. At depth 4 and width >= 50 every seed
converged (10/10 at 4x50, 4x100 and 4x200); the 32-wide networks converged in
only 3-5 runs of 10, and depth 6 at widths 32 and 50 in 4 and 6 of 10. A small or deep network does not
settle: it stalls, and then a fresh optimiser moves it again.

## By architecture

The endpoint error is the median over the 2018 test states of the distance
between the network's predicted endpoint and the fp64 RK4 reference; the table
reports the median, min and max of that number over the converged seeds.

| loss | depth x width | parameters | converged | test endpoint, median over the converged seeds (um) | min | max | max/min | val endpoint, median (um) | restarts, median | wall, median (s) | final loss, median |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| data | 4x200 | 129036 | 10/10 | 128.5 | 105.4 | 159.0 | 1.51 | 130.6 | 50 | 1128 | 1.34e-06 |
| physics | 2x32 | 2436 | 5/10 | 1059.0 | 852.8 | 1599.7 | 1.88 | 1116.6 | 81 | 296 | 1.83e-04 |
| physics | 4x32 | 4548 | 3/10 | 410.5 | 387.1 | 449.8 | 1.16 | 447.5 | 97 | 516 | 5.37e-05 |
| physics | 2x50 | 4686 | 9/10 | 469.5 | 339.9 | 592.9 | 1.74 | 490.6 | 74 | 288 | 4.54e-05 |
| physics | 6x32 | 6660 | 4/10 | 426.6 | 391.0 | 2724.6 | 6.97 | 433.6 | 138 | 844 | 6.51e-05 |
| physics | 4x50 | 9786 | 10/10 | 221.6 | 164.7 | 243.8 | 1.48 | 233.9 | 152 | 785 | 1.06e-05 |
| physics | 2x100 | 14336 | 10/10 | 223.1 | 201.4 | 298.9 | 1.48 | 237.0 | 92 | 600 | 1.08e-05 |
| physics | 6x50 | 14886 | 6/10 | 326.4 | 248.2 | 525.2 | 2.12 | 344.1 | 149 | 998 | 2.74e-05 |
| physics | 4x100 | 34536 | 10/10 | 133.3 | 128.0 | 155.8 | 1.22 | 144.7 | 120 | 1484 | 3.04e-06 |
| physics | 2x200 | 48636 | 10/10 | 153.0 | 127.5 | 241.4 | 1.89 | 161.2 | 100 | 1386 | 4.19e-06 |
| physics | 6x100 | 54736 | 10/10 | 191.0 | 152.2 | 242.2 | 1.59 | 202.9 | 148 | 2489 | 9.33e-06 |
| physics | 4x200 | 129036 | 10/10 | 114.7 | 100.3 | 133.5 | 1.33 | 120.6 | 96 | 2650 | 2.06e-06 |
| physics | 6x200 | 209436 | 9/10 | 159.0 | 149.5 | 208.4 | 1.39 | 169.8 | 107 | 4049 | 6.05e-06 |

The reference points the table is read against:

| reference | test endpoint, median | where it comes from |
|---|---:|---|
| exact scheme at q = 8, no network | **23 um** | `../Stage_count_sweep/measure_scheme_ceiling.py` on these same 2018 test states (22.5 um). The 29 um quoted in `../One_step_network*` was measured on a different population — 32 momentum-stratified legs — and is not the ceiling for this test set. |
| `../One_step_network_v2`, 4x50, 3 seeds | 183 um | `../One_step_network_v2/results/summary.csv`. Those runs used four BLAS threads; everything here is single-threaded, so they are not bit-comparable — the 4x50 row above (222 um over 10 seeds, range 165-244) is the like-for-like number. |
| straight line, no bending | 520,444 um | the same dataset's `straight_med_um` |

## A2.4 — the four verdict statements

### (a) Capacity or optimisation?

**Optimisation.** The floor does move with size, and by a lot at the small end:
2x32 (2436 parameters) sits at 1059 um and 4x200 (129,036 parameters) at 115 um,
a factor of **9.2 for 53 times the parameters**. But the returns collapse along
the way. From the baseline 4x50 (222 um) to 4x200 is a factor of only **1.93 for
13 times the parameters**, and the last doubling of width, 4x100 -> 4x200, buys
**14%** (133 -> 115 um) for 3.7 times the parameters and 1.8 times the wall time.
Extrapolating that trend, closing the remaining factor of 5 to the 23 um ceiling
by size alone would take several more orders of magnitude of parameters, which
is not a deployable answer.

The loss-error correlation says why. Over all 96 converged physics runs the
Spearman rank correlation between the final training loss and the test endpoint
error is **+0.965** — on van der Pol it was +0.97 — and
`figures/loss_vs_error.png` shows every architecture and every seed lying on one
band, three decades of loss mapping monotonically onto one and a half decades of
error. The endpoint error is a readout of the loss the optimiser managed to
reach, not of what the network could represent. A wider network helps only
because L-BFGS gets further down on it: 4x50 stalls at a median final loss of
1.06e-05, 4x200 at 2.06e-06. Anything that lets the optimiser reach a lower loss
— a better conditioned parameterisation, a better scaled loss, more restarts —
should be expected to pay more than more parameters do.

### (b) Does depth hurt?

**Yes, depth 6 hurts at every width, as it did on van der Pol.** Median test
endpoint error, converged seeds:

| width | depth 2 | depth 4 | depth 6 | depth 6 / depth 4 |
|---:|---:|---:|---:|---:|
| 32 | 1059.0 | 410.5 | 426.6 | 1.04 |
| 50 | 469.5 | 221.6 | 326.4 | 1.47 |
| 100 | 223.1 | 133.3 | 191.0 | 1.43 |
| 200 | 153.0 | 114.7 | 159.0 | 1.39 |

Depth 6 is never better than depth 4, and the penalty is 39-47% at widths 50,
100 and 200: 6x200 uses 1.6 times the parameters of 4x200 and lands 39% worse,
for 1.5 times the wall time. It is also much harder to train:
at depth 6 only 4, 6, 10 and 9 seeds of 10 converge at widths 32, 50, 100 and
200, against 3, 10, 10 and 10 at depth 4.
Depth 2 is worse than depth 4 at every width too (by 33% at width 200 rising to
158% at width 32), so the optimum is interior and sits at **depth 4**.

### (c) The architecture to use for the general-leg experiment (A3)

A3 is running at 4x50 and 4x100. **4x100 is adequate; 4x50 is not.**

- **4x50** — 222 um [165-244] over 10/10 converged seeds. That is 1.93 times the
  4x200 floor and 9.8 times the exact-scheme ceiling. It sits on the steep part
  of the size curve, where a general-leg result would be reporting the network
  size as much as the physics.
- **4x100** — 133 um [128-156], 10/10 converged, the tightest seed spread of any
  fully converged architecture (max/min = 1.22), 1484 s median wall time on one core. It is within
  16% of the best architecture measured, at a quarter of its parameters.
- **4x200** — 115 um [100-134], 10/10 converged, 2650 s median. The best of the
  grid, but only 14% better than 4x100.

Recommendation: run A3 at **4x100**, and treat 4x200 as the check that the
general-leg answer is not size-limited. If a single size must serve both, 4x100
is the one; the 4x50 arm should be read as a lower bound on what the general leg
can do, not as its floor.

### (d) The selection rule

**Select on validation endpoint error; it is reliable.** Over the 96 converged
physics runs the Spearman rank correlation between validation and test endpoint
error is **+0.995**, and `figures/val_vs_test_selection.png` shows the points
lying on the diagonal with validation running a few percent high (it is the
larger, unpruned split). Within an architecture — the case that matters, where
we are choosing among ten seeds that differ only by initialisation — the median
rank correlation is **+0.885**, the validation-best seed is also the test-best
seed in **9 of 12 architectures**, and in the three where it is not, the seed
validation picks is at most **2.3%** worse on test than the best one (median
penalty over all twelve: 1.000). Validation selection costs essentially nothing.

**Seed spread.** Over the architectures with at least three converged seeds the
median max/min ratio in test endpoint error is **1.54**, i.e. the luckiest seed
is about half again better than the unluckiest. It shrinks with size and with
convergence: 1.22 at 4x100 and 1.33 at 4x200, against 1.88 at 2x32 and 6.97 at
6x32, where one seed landed at 2725 um. So a single seed is not a measurement:
report a median over seeds, and at the recommended sizes three seeds bound the
spread to about 20-30%.

## The data twin at the winner (A2.2)

Ten fresh supervised runs at 4x200 — identical network, identical protocol, the
physics loss swapped for MSE against the fp64 RK4 stage and endpoint labels.
All ten were rerun rather than reused from `../One_step_network_v2`, because
those were four-thread runs at a different architecture.

| | test endpoint, median over seeds | min | max | val | median restarts | median wall |
|---|---:|---:|---:|---:|---:|---:|
| physics, 4x200 | **114.7 um** | 100.3 | 133.5 | 120.6 | 96 | 2650 s |
| data twin, 4x200 | 128.5 um | 105.4 | 159.0 | 130.6 | 50 | 1128 s |

At this size the physics loss is **12% better than its supervised twin** and the
twin has the wider seed spread (1.51 against 1.33), while reaching its stall in
about half the restarts. The factor-2 gap that the v1 and v2 experiments chased
is gone: with the fiducial cut in place and a network large enough for the
optimiser to work with, the paper's label-free loss matches and slightly beats
supervision on labels that were computed with the same clamped field. Neither
comes near the 23 um ceiling, which is consistent with (a) — both are limited by
how far L-BFGS gets, and the physics loss simply gets further.

## Provenance

Repository `LHCb_Extrapolation_Project`, branch `main`. Physics and optimiser
from `../_shared` (unchanged; its gates are `../_shared/smoke_tests.py`).
Dataset `results/frozen_leg_q08.npz`, built by `build_dataset.py` from the
official-sample training set, 1979 / 2062 / 2018 states after the fiducial cut.
Runs: HTCondor clusters 5781153 (grid), 5781167 and 5781442 (continuations),
5781358 (data twin), on `/data`, one core and one thread per job. The per-run
`.json` and `_history.csv` files in `results/` are committed; the `.pt`
checkpoints and the farm logs are gitignored and regenerate from the same
commands.
