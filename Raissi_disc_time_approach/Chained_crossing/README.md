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
