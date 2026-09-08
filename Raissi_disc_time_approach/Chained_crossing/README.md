# Chained_crossing — every trained network walked across the whole magnet  (Block C, C6)

**What this is.** [`../Step_size_and_stage_grid`](../Step_size_and_stage_grid)
trained 720 networks and scored each of them on **one step**. A real
extrapolator does not take one step: it takes many, feeding its own output back
in, and an error that is invisible on one step can be a systematic bias that
compounds. This folder takes all 720 of those networks, chains each of them
across the whole 5.2 m magnet crossing at five step lengths and once in a single
step, and records **x, y, tx and ty separately** — for the single step and for
the chain — against two truths: the fine sixth-order reference, and where the
simulated particle actually was.

    step length in {0.1, 1, 10, 100, 1000 mm} and the whole crossing in one step
    x  720 networks (4 widths x 3 depths x 10 stage counts x 2 losses x 3 seeds)

**Nothing here trains anything, and nothing here writes into the grid.** The
checkpoints, the datasets and the run records of
[`../Step_size_and_stage_grid`](../Step_size_and_stage_grid) are opened
read-only.

---

## script → output

| script | what it does | output |
|---|---|---|
| [chain.py](chain.py) | the chaining rule, the six columns, the component statistics and the units — the module the other scripts share | — |
| [build_tracks.py](build_tracks.py) | **C6.2a.** the 1,000 whole-magnet test tracks, with their fine-reference far-plane state, the particle's real far-plane hit and the dense reference path | `results/chain_tracks.npz`, `results/chain_tracks.csv`, `results/chain_tracks_meta.json` |
| [measure_timing.py](measure_timing.py) | **C6.3a.** what one job costs, at the heaviest and a light point of the grid | `results/timing.json` |
| [chain_one.py](chain_one.py) | **C6.1 + C6.2.** one network: the per-component single step on the v3 test split, and the chain at every step length | `results/single_step/<tag>.npz`, `results/chains/<tag>.npz`, `results/records/<tag>.json` |
| [make_jobs.py](make_jobs.py) | **C6.3b.** the 720-line job list and its submit file | `condor/jobs_chain.txt`, `condor/jobs_chain.sub` |
| [resubmit.py](resubmit.py) | **C6.3c.** sends back the tags whose record has not landed and which are not already queued | `condor/jobs_chain_round<N>.txt` / `.sub` |
| [aggregate.py](aggregate.py) | **C6.1 / C6.2 summaries.** the records → the long tables | `results/single_step_components.csv`, `results/chained_components.csv`, `results/chain_growth.csv`, `results/growth_exponent.csv`, `results/reference_rows.csv`, `results/landed.csv`, `results/pending.csv` |
| [tables.py](tables.py) | **C6.4a.** the long tables → the display tables, the ranking and the component reading | `results/chained_table_<D>x<W>*.csv`, `results/single_step_table_<D>x<W>_<component>.csv`, `results/ranking.csv`, `results/ranking_extremes.csv`, `results/component_reading.csv` |
| [plot.py](plot.py) | **C6.4b.** the figures, from the csvs alone | `figures/chain_growth_<D>x<W>.png`, `figures/chained_heatmap_<D>x<W>.png`, `figures/mini_fig2_<tag>.png`, `figures/mini_fig3_<tag>.png`, `figures/components_<tag>.png`, `figures/best_vs_worst_components.png` |
| [analysis.ipynb](analysis.ipynb) | loads all of the above and displays it; computes nothing | — |

```bash
PY=/data/bfys/gscriven/conda/envs/TE/bin/python
export PYTHONNOUSERSITE=1
cd Chained_crossing
$PY build_tracks.py                                    # C6.2a, 3 s
$PY measure_timing.py                                  # C6.3a, about 6 min
$PY make_jobs.py && condor_submit condor/jobs_chain.sub  # C6.3b
$PY resubmit.py --submit                               # C6.3c, as needed
$PY aggregate.py && $PY tables.py && $PY plot.py       # C6.4
```

`results/*/*.npz` and `results/records/` are gitignored; the scripts, the
committed track list and the trained checkpoints regenerate them.

---

## The chaining rule

A grid network maps a start state and a leg to the state at the end of that leg:

    (x, y, tx, ty, qop) at z0,  extra = normalised (z0, dz)
        ->  (x, y, tx, ty) at z0 + dz,   qop carried through unchanged

A **column** of this experiment fixes a nominal step length and walks the whole
crossing with it. Each track has its own crossing length L = |dz| — 5,156 mm to
6,670 mm, median 5,176 mm — so the step is fitted to the track rather than the
track truncated to the step:

    N    = round(L / dz_nominal),  at least 1
    step = dz / N                  (signed: the track's own direction)

so the N steps land **exactly** on the far plane, every step of one track is the
same length, and the step is within a fraction of a per cent of nominal in every
column but the longest. The network's own endpoint becomes the next step's start
state; `(z0, dz)` are recomputed for every step and fed in as the extra inputs
the grid networks take; `qop` is passed through unchanged, as the scheme demands.
Everything is fp64, and the tracks are advanced as one batch inside the step
loop.

The **full-crossing column is N = 1**: one step across the magnet, which is
exactly the single-step full-crossing cell of the grid's C4 tables measured on
this folder's 1,000 tracks. It is kept as the thing every chain has to beat.

**Fifty checkpoints.** At fractions k/50 of the crossing, k = 1…50, the chained
state is compared with the fine reference's own path there, linearly
interpolated between the 10 mm dense states of
[`../Magnet_tracks_dataset`](../Magnet_tracks_dataset). For a track with N steps
the checkpoint at fraction f is taken at step round(f·N), the nearest step
boundary; when N < 50 several checkpoints collapse onto one step, and the
fraction actually reached is stored alongside the nominal one.

---

## The track list

`results/chain_tracks.csv` is the committed list — one row per track, with its
row in the v3 set, its particle (`evt`, `mckey`), direction, momentum,
pseudorapidity, PID, the two planes, and the two numbers that make it a test of
something.

The tracks are the **full-crossing stratum of the v3 test split**: the first 500
forward (the particle's last UT plane → its first SciFi plane) and the first 500
backward, in the row order
[`../Step_size_and_stage_grid`](../Step_size_and_stage_grid) itself uses.
`build_tracks.py` recovers that order from `prepare_nodes.pick_rows` and then
**asserts** that the start states, the planes and the end states are identical,
element for element, to `grid_q08.npz`'s own `test_*` arrays. The v3 split is by
particle, so no network in the grid has seen any of these tracks.

**The 0.1 mm column runs on the first 250 of each direction**; every other
column runs on all 1,000.

Each track carries four things:

| | what it is |
|---|---|
| the start state | the particle's own measured state on the near plane |
| **the fine reference** at the far plane | the v3 set's own `Y`: an RK6 march at 0.1 mm on the MagUp map. Its own accuracy over a crossing is 5e-5 µm ([`../Fine_reference`](../Fine_reference) C1.4) |
| **the particle's real hit** at the far plane | forward → its first SciFi crossing, backward → its last UT crossing, read straight out of the harvested MCHit states the way [`../MC_hit_comparison`](../MC_hit_comparison) reads them |
| the dense reference path | the fine reference's own state every 10 mm along the crossing, matched by `LEG_INDEX` and cross-checked against `EVT`, `MCKEY` and `DIRECTION` |

**The v2 training set's `Y` column is never read.** It is field-only propagation
on the MagDown map and the sample is MagUp
([`../Magnet_tracks_dataset`](../Magnet_tracks_dataset)), so it is the right
start state pushed through the wrong polarity. The real hit's z agrees with the
leg's far plane to better than 0.5 µm, which is recorded in
`chain_tracks_meta.json` rather than assumed.

---

## The tables

Every display table has the same shape: rows are the ten stage counts q, two
lines each (the physics arm and the data twin), columns are the six step
lengths, and a cell is `median [min-max over the three seeds]`.

| file | the cell |
|---|---|
| `results/chained_table_<D>x<W>.csv` | chained far-plane max(\|dx\|, \|dy\|) against the **fine reference**, µm |
| `results/chained_table_<D>x<W>_x.csv`, `_y.csv` | the signed component, \|dx\| or \|dy\|, µm |
| `results/chained_table_<D>x<W>_tx.csv`, `_ty.csv` | the slope component, mrad |
| `results/chained_table_<D>x<W>_hit.csv` | chained far-plane max(\|dx\|, \|dy\|) against the **particle's real hit**, µm |
| `results/single_step_table_<D>x<W>_<component>.csv` | the **single step** on the v3 test split: rows q, columns the six training strata, one file per component including `max_xy` |
| `results/growth_exponent.csv` | the power-law exponent of the growth curve per (network, column, direction, **component**), written by `aggregate.py` from the full per-component growth in the records: **0.5** is a random walk, **1** is a coherent bias repeating every step, **2** is a coherent bias in the *slope* integrating into position |
| `results/component_reading.csv` | per (architecture, arm, column, component): the median error, the tail, and how much of it is bias rather than spread |

Four reference rows sit beneath the network rows of every chained table:

* **single step (grid C4)** — the same architecture's own single-step
  full-crossing cell at each q, taken from
  `../Step_size_and_stage_grid/results/table_cells.csv`. It is measured on the
  grid's 2,000 full-crossing test rows and the chained "full crossing" column is
  the same measurement on this folder's 1,000 tracks, so the two differ by the
  population and by nothing else.
* **straight line** — the null step, ignoring the magnet, measured on exactly
  these tracks. Chaining it changes nothing: each sub-step keeps tx and ty and
  adds tx·step to x, and the sub-steps sum to dz, so the chained straight line
  *is* the single straight step and one row serves every column. (The grid's own
  figure on its 2,000 test rows is 456,075 µm; the number in these tables is the
  same measure on the 1,000 chained tracks.)
* **material floor** — the particle's real far-plane hit against the fine
  reference on the same tracks: the part of a crossing that no field-only method
  of any kind can predict ([`../MC_hit_comparison`](../MC_hit_comparison)).
* **reference floor** — 5e-5 µm, the fine reference's own accuracy over a
  crossing.

`results/ranking.csv` is every network's **best** chained far-plane median over
the six columns, with the column that reached it, its parameters and
multiply-adds, and the six per-column medians; `results/ranking_extremes.csv`
pulls out the best three and worst three of each arm — the twelve networks the
per-network figures are drawn for.

**`results/chain_growth.csv` carries `max_xy` only, forward and backward.**
Fifty checkpoints × five components × three directions × six columns × 720
networks is 3.2 million rows, which is not a table anybody can open. The four
signed components at every checkpoint are in `results/records/<tag>.json` and in
`results/chains/<tag>.npz`; the growth figures need the max(\|dx\|, \|dy\|)
curve and that is what the csv holds.

---

## The two figures borrowed from the mini paper, and what maps to what

The plan asks for the twelve ranked networks to be drawn "in the style of the
mini paper's Figure 2 and Figure 3". Those two figures are, in
[`../../Mini_paper/main.tex`](../../Mini_paper/main.tex):

| | the mini paper's figure | what it is there |
|---|---|---|
| **Figure 2** (`fig_baseline.png`, `\label{fig:baseline}`) | "The July baseline on the frozen magnet crossing" | a multi-panel summary of one working point: the **distribution** of held-out endpoint errors on a log axis with the exact-scheme ceiling and the straight-line baseline marked as vertical lines, the training histories, a per-run scatter and the fiducial comparison |
| **Figure 3** (`fig_stages.png`, `\label{fig:stages}`) | "Median test-split endpoint error against the number of Gauss–Legendre stages q" | **error against a swept variable** on a log vertical axis, two panels, with the label-free arm in blue, the supervised twin in red, the exact scheme in green and the straight-line baseline as a dashed horizontal line, bars for the seed spread |

Figure 3 is the "error against a swept variable with the twin and the ceiling on
the same axes" case, so `figures/mini_fig3_<tag>.png` reproduces its axes for
this experiment: **error against the step length**, log-log, two panels —
*chained across the whole magnet* and *one step of that length* — with the
network and its opposite-arm twin as the two curves, and the straight line, the
material floor and the reference floor as the horizontal reference lines that
replace the mini paper's exact-scheme ceiling and straight-line baseline.

Figure 2 is not a floor-versus-size plot (that is the mini paper's Figure 4,
`fig:architecture`), so `figures/mini_fig2_<tag>.png` keeps Figure 2's own
content and adds Figure 4's: **left**, the distribution of the chained
far-plane error at every step length on a log axis with the straight line and
the material floor marked, in Figure 2's left-panel layout; **middle**, the
same per column as medians with bars to the 95th percentile; **right**, where
this network sits on the **cost–error front** of all 720, forward-pass
multiply-adds against its best chained median, which is the floor-versus-size
panel the plan asked for.

`figures/components_<tag>.png` is this folder's own: histograms of dx and dy in
µm and dtx and dty in mrad at the far plane, one curve for the single-step full
crossing and one per chained column, with the median (dashed) and the 95th
percentile of the absolute value (dotted) marked.

---

## The farm

```bash
cd Chained_crossing
mkdir -p condor/logs results
python make_jobs.py
condor_submit condor/jobs_chain.sub
```

### What a job costs

The 0.1 mm column is the whole experiment's cost: a 5.2 m crossing at a 0.1 mm
step is about 51,700 network evaluations per track, against 5,200 at 1 mm and
one at the full crossing. `measure_timing.py` measures it twice.

**The plan's probe — the whole 0.1 mm column on ten tracks:**

| network | parameters | tracks x steps | wall | per sample-step | peak RSS |
|---|---|---|---|---|---|
| 8 x 256, q = 20 | 484,180 | 10 x 51,730 | 178.9 s | 345.8 µs | 0.60 GB |
| 4 x 32, q = 2 | 3,220 | 10 x 51,730 | 109.3 s | 211.2 µs | 0.61 GB |

**The batch probe, which is what the projection is built on.** A chain is a loop
over steps with the whole batch of tracks inside it, so the per-track cost falls
steeply with the batch — at ten tracks the fp64 matrix multiplies are far too
small to amortise torch's own per-operation overhead:

| network | batch 10 | batch 500 | batch 1000 | one step of a batch of n |
|---|---|---|---|---|
| 8 x 256, q = 20 | 3.42 ms (341.6 µs each) | 41.02 ms (82.1 µs each) | 80.82 ms (80.8 µs each) | **2.397 + 0.07819 n ms** |
| 4 x 32, q = 2 | 2.06 ms (205.8 µs each) | 5.67 ms (11.3 µs each) | 9.17 ms (9.2 µs each) | **2.017 + 0.00718 n ms** |

A whole job is 31,631,000 sample-steps — 25.9 million of them the 0.1 mm column.
At the fitted rates that is **0.73 h at 8 x 256, q = 20** and **0.10 h at
4 x 32, q = 2**. The ten-track probe, charged to every column, would have said
3.04 h and 1.86 h; the difference is the batch, and it is recorded rather than
hidden.

**The rule was fixed before measuring: over 8 h at 8 x 256 and the 0.1 mm column
is cut to 100 + 100 tracks. 0.73 h is well under it, so the column keeps its
250 + 250 tracks** and nothing was reduced. Peak resident set 0.61 GB, so the
4 GB request is generous; there is no optimiser history here, which is what made
the training jobs heavy.

### The cluster

**Cluster 5783753, 720 jobs, submitted 2026-09-08 13:20 CEST.**
One job per network, 1 CPU and 4 GB each,
`+JobCategory "medium"`, `+UseOS "el9"`, `getenv = False`, no file transfer —
the repository, the checkpoints and the results are all on `/data` and visible
from the worker nodes. `condor/jobs_chain.sub` points at
[condor/wrapper_chain.sh](condor/wrapper_chain.sh), a copy of the shared wrapper
that starts `chain_one.py` instead of `_shared/train.py`; the shared wrapper is
hard-wired to the latter and was not edited. `condor/jobs_chain.txt` holds 720
argument lines and **no comment lines** (`queue … from` does not skip `#`).

A chained job is stateless — it reads a checkpoint and the track list and writes
one record — so a failure is simply re-run rather than resumed. `resubmit.py` is
idempotent and safe on a schedule: it selects only tags with no record on disk
and no job of their own already idle or running.

**All 720 records landed and the queue emptied at 14:15 on 2026-09-08**, about
55 minutes after submission, with **no failures and no resubmission round**.
Total cost **134.3 core-hours**; a job took a median of **511 s**, from 220 s at
2 x 32, q = 2 to **2,651 s** (44 min) at 8 x 256 — against the 0.73 h the
projection gave for that point, which is the projection to within 1 %.

`results/landed.csv` carries the wall time and the worker node of every record;
`results/pending.csv` is a header and no rows.

### The single-step gate

Every job re-scores the network's single step on the whole v3 test split
through the grid's own path — `grid_model.build_grid_model` and
`_shared.evaluate.predict` — and asserts that the max(\|dx\|, \|dy\|) medians
per stratum reproduce that run's own `by_stratum` block in
`../Step_size_and_stage_grid/results/<tag>.json`. Over the 720 records and the
**5,040 stratum cells** the worst relative difference is **exactly 0.0**: the
per-component numbers here are the grid's own single-step numbers split into
components and nothing else.

---
---

# C6.4 — reading the tables

## (a) Chaining never helps. The single step across the whole magnet is the best step.

`results/ranking.csv`, every network's best chained far-plane median over the
six columns:

| the column that reached a network's best error | physics arm | data twin | all |
|---|---|---|---|
| **full crossing (one step)** | **291 of 360** | 130 of 360 | **421 of 720** |
| 1000 mm | 69 | 23 | 92 |
| 100 mm | 0 | 69 | 69 |
| 10 mm | 0 | 93 | 93 |
| 1 mm | 0 | 45 | 45 |
| **0.1 mm** | **0** | **0** | **0** |

**Not one of the 720 networks is at its best when it walks the crossing in
0.1 mm steps**, and the physics arm is at its best in one step for 291 of its
360 networks, at 1000 mm for the other 69. The 4 x 64 physics arm at q = 20 is
the shape of it: 767 µm in one step, 2,750 µm in five or six steps of 1000 mm,
14,300 µm at 100 mm, 151,000 µm at 10 mm and **207,000 µm at 0.1 mm** — which is
43 % of the straight line's 478,300 µm, i.e. the chain has thrown away most of
what the network knew about the magnet.

**713 of the 720 networks are worse at 0.1 mm than in one step**, by a median
factor of **105** — 242 for the physics arm and 20 for the twin — and the seven
exceptions are all within 10 % of breaking even rather than gaining anything.

This is the reverse of what a discretisation argument predicts and it is not
subtle: shrinking the step improves the **single** step monotonically — the same
4 x 64 network goes from 767 µm over the crossing to 1.2e-5 µm over 0.1 mm — and
makes the **chain** monotonically worse. The single step and the chain are
measured on the same networks in the same job, so the two statements are not
about different populations.

## (b) The growth law: a coherent slope bias, integrating into position

`results/growth_exponent.csv` fits log10(median error) against log10(fraction of
the crossing walked) over the checkpoints from a tenth of the crossing on. The
three hypotheses give different exponents: **0.5** if the per-step errors are
independent and add in quadrature, **1** if one bias repeats identically at every
step, and **2** if that coherent bias is in the *slope*, because a slope error
held for a distance z displaces the position by z².

Median exponent over 720 networks, forward and backward pooled:

| arm | column | x | tx |
|---|---|---|---|
| physics | 0.1 mm | **2.31** | **1.20** |
| physics | 1 mm | **2.31** | **1.20** |
| physics | 10 mm | **2.32** | **1.23** |
| physics | 100 mm | 2.11 | 1.02 |
| physics | 1000 mm | 1.41 | 0.63 |
| twin | 0.1 mm | **2.33** | **1.33** |
| twin | 1 mm | **2.08** | **1.04** |
| twin | 10 mm | **2.04** | **0.97** |
| twin | 100 mm | 1.99 | 0.92 |
| twin | 1000 mm | 1.51 | 0.74 |

**The slope exponent sits on 1 and the position exponent on 2, in both arms and
at every step length short enough for the chain to have many steps.** Over the
four columns with more than 50 steps, **99.8 %** of the 2,880 (network,
direction) fits have a position exponent above 1.5 and **not one** is below 1.0.
A random walk is excluded: it would put the slope at 0.5 and the position at
1.5.

So the mechanism is: the network makes the *same* small mistake in tx at every
step of a track, that mistake adds up linearly, and the position error it causes
grows as the square of the distance walked. Halving the step doubles the number
of steps and does not halve the per-step slope bias by enough to compensate,
which is why the error rises as the step shrinks.

The 1000 mm column's exponents fall to 1.4 and 0.6 because it has only five or
six steps: the fit there is over too short a lever arm to separate the laws, and
it is reported rather than read.

## (c) Which component: x by a factor 3 to 40, and it is spread, not offset

`results/component_reading.csv`, medians over the twelve architectures:

| arm | column | x / µm | y / µm | x / y | tx / mrad | ty / mrad | tx / ty |
|---|---|---|---|---|---|---|---|
| physics | 0.1 mm | 2.38e5 | 5,572 | **42.7** | 81.2 | 2.12 | **38.4** |
| physics | 10 mm | 1.58e5 | 5,252 | 30.0 | 55.5 | 1.88 | 29.4 |
| physics | 100 mm | 1.43e4 | 3,078 | 4.7 | 5.41 | 1.04 | 5.2 |
| physics | 1000 mm | 2,579 | 1,006 | 2.6 | 1.03 | 0.41 | 2.5 |
| physics | full crossing | **732** | **266** | 2.7 | **0.203** | **0.078** | 2.6 |
| twin | 0.1 mm | 1.26e5 | 2,393 | **52.7** | 52.8 | 0.77 | **68.8** |
| twin | 10 mm | 5,306 | 1,762 | 3.0 | 2.00 | 0.52 | 3.9 |
| twin | full crossing | 4,635 | 1,806 | 2.6 | 1.98 | 0.51 | 3.9 |

**The bending plane carries the error.** In one step the ratio x/y is 2.3–3.6 in
every stratum of both arms — the magnet bends in x, and so does the part of it
the network gets wrong. Chaining does not preserve that ratio: it *amplifies*
it, to 43 in the physics arm and 53 in the twin at 0.1 mm, because the thing
that accumulates is the bend. The slope ratio tx/ty moves the same way, from
2.6–4.3 in one step to 38–69 at 0.1 mm. **The chain is not a uniform
degradation; it is a degradation of the bend.**

**It is spread, not a shared offset.** |mean| / rms of the signed component is
**0.04–0.17** for x and 0.02–0.08 for tx everywhere: the population's mean error
is at most a sixth of its rms, so there is no common displacement that a
constant correction would remove. What is coherent is coherent *along one
track*, not across tracks — each track has its own sign and size of slope bias
and holds it for the whole crossing. The tails widen the other way: p95/median
is 4.7–5.5 for the physics arm on the short columns against 11.4 in one step, so
the chain makes the *typical* track much worse without making the tail
relatively worse.

## (d) The ranking

`results/ranking.csv` and `results/ranking_extremes.csv`, best chained far-plane
median over the six columns against the fine reference:

| | network | q | best | at | its single step over the crossing (grid C4) | rank of 720 |
|---|---|---|---|---|---|---|
| **physics, best** | 8 x 128 seed 0 | 14 | **549 µm** | full crossing | 673 µm | 1 |
| | 8 x 128 seed 2 | 12 | 573 µm | full crossing | 677 µm | 2 |
| | 8 x 128 seed 0 | 18 | 590 µm | full crossing | 674 µm | 3 |
| **physics, worst** | 2 x 256 seed 2 | 2 | 5,394 µm | 1000 mm | 10,050 µm | 562 |
| | 2 x 256 seed 1 | 2 | 5,086 µm | 1000 mm | 10,050 µm | 548 |
| | 2 x 32 seed 1 | 2 | 4,669 µm | 1000 mm | 9,831 µm | 527 |
| **twin, best** | 4 x 256 seed 1 | 2 | **894 µm** | 10 mm | 906 µm | 175 |
| | 4 x 256 seed 0 | 2 | 928 µm | full crossing | 906 µm | 196 |
| | 4 x 128 seed 1 | 4 | 955 µm | full crossing | 970 µm | 213 |
| **twin, worst** | 2 x 32 seed 2 | 10 | 10,480 µm | full crossing | 9,615 µm | 720 |
| | 2 x 32 seed 2 | 14 | 10,160 µm | 1 mm | 9,877 µm | 719 |
| | 2 x 64 seed 0 | 20 | 9,850 µm | 1000 mm | 10,180 µm | 718 |

Two things are worth saying about that table. **The best number in the whole
experiment, 549 µm, is a single step**, and it is 18 % below that network's own
single-step cell in the grid's C4 table only because it is measured on 1,000 of
the grid's 2,000 full-crossing test rows. **And the ranking is essentially the
single-step ranking**: the worst networks are the q = 2 ones, which C4 already
showed sit on the exact scheme's own q = 2 discretisation error, and no network
climbs the ranking by being chained.

## (e) Against real hits: the chain leaves the floor far behind

`results/chained_table_<D>x<W>_hit.csv`. The material floor on these 1,000
tracks — the particle's real far-plane hit against the fine reference — is
**1,643 µm** (median; 2,080 µm forward, 1,327 µm backward, p95 12,172 µm),
consistent with [`../MC_hit_comparison`](../MC_hit_comparison)'s 1,815 µm on its
own cross-magnet population.

| architecture | arm | column | vs the fine reference | vs the real hit | ratio |
|---|---|---|---|---|---|
| 4 x 64 | physics | full crossing | 802 µm | 2,038 µm | **2.54** |
| 4 x 64 | physics | 1000 mm | 2,877 µm | 3,621 µm | 1.26 |
| 4 x 64 | physics | 100 mm | 15,100 µm | 14,940 µm | 0.99 |
| 4 x 64 | physics | 0.1 mm | 204,500 µm | 205,000 µm | **1.00** |
| 8 x 256 | physics | full crossing | 726 µm | 2,008 µm | **2.77** |
| 8 x 256 | physics | 0.1 mm | 124,600 µm | 126,200 µm | **1.00** |

This is C5's reading arriving from the other side. In **one** step the network's
field-only error (726–802 µm) is *below* the 1,643 µm floor, so the vs-hit
number is dominated by the material and is 2.5–2.8 times the field-only one.
**As soon as the chain is more than about a thousand microns wrong the two
columns coincide**, and by the 0.1 mm column they agree to 1 %: a chained error
of 200 mm is a hundred times the material floor, so comparing it with a real hit
measures the chain and nothing else.

## (f) What this says about the discrete-time surrogate

1. **On this problem the one-step network should be used as a one-step
   network.** Its accuracy over the whole magnet, 549–800 µm at the good
   architectures, is not reached by any subdivision of the crossing, and the
   finest subdivision loses two and a half orders of magnitude.
2. **The obstacle is a per-track slope bias, not capacity and not noise.** The
   exponents say it plainly, and (c) says it is not a shared offset that a
   calibration could remove. A network whose *slope* output were unbiased per
   track would chain; these do not, and nothing in either loss asks them to —
   the physics loss scores the reconstruction residual and the twin scores the
   endpoint label, and both are one-step criteria.
3. **That is the experiment this suggests next**: a loss with a term over two
   or more chained steps, so the slope is trained where it is used. This folder
   measures the problem; it does not fix it.
