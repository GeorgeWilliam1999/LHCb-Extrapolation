# C2_Exact_scheme_table — the scheme's own error, no network  (Block C, C2)

**What this is.** The q-stage Gauss-Legendre collocation scheme solved *exactly*
— its implicit equations driven to a residual of 1e-9 in mm and mrad by a
root-finder — on every test state of
[`../C0_Magnet_tracks_dataset`](../C0_Magnet_tracks_dataset), for ten stage counts and
six step lengths. No network appears anywhere in this folder.

**Why.** A network trained on the Raissi discrete-time loss is trying to satisfy
these equations. If it succeeded perfectly its answer would be the answer in
this table, so this table is the **ceiling**: no such network can do better, and
a network's score is only meaningful next to the ceiling for the same step
length. `../../Block_0_first_pass/S1_Simple_first_pass` and `../../Block_A_technique_works/A1_Stage_count_sweep` measured that ceiling
on one leg with one polarity; this measures it as a surface over (stage count,
step length), on the MagUp map the sample was actually simulated with, against
the fine sixth-order reference of [`../C1_Fine_reference`](../C1_Fine_reference)
rather than the old 5 mm RK4 engine.

**The headline.** The classical order 2q is **not visible anywhere on this
map**, for the opposite reasons at the two ends of the range. Below one 100 mm
field cell the scheme is already exact to the reference's own arithmetic at
q = 2, so extra stages have nothing left to buy. Above one cell the map's
trilinear kinks take over and the error falls with step length like an
order-three method whatever q is. Across the whole magnet the median error
stops improving at q = 8 and sits between **20 and 42 µm** for every stage count
from 8 to 20.

---

## script → output

| script | what it does | output |
|---|---|---|
| [exact_solver.py](exact_solver.py) | C2.1. The solver: the implicit equations for one state and one step, with the field passed in. Imported, not run. | — |
| [run_scheme_grid.py](run_scheme_grid.py) | C2.2. Solves every test state of every stratum, one file per stage count | `results/scheme_rows_q<qq>.npz`, `results/scheme_grid_meta.json` |
| [make_tables.py](make_tables.py) | C2.3 / C2.4. Aggregates the per-solve rows; also the straight-line baseline and the numbers the reading below rests on | `results/scheme_table.csv`, `results/straight_line_table.csv`, `results/summary_grid.json`, `results/reading.json` |
| [plot.py](plot.py) | the two figures, from the csv tables only | `figures/scheme_error_vs_dz_and_q.png`, `figures/scheme_convergence_fraction.png` |
| [analysis.ipynb](analysis.ipynb) | loads all of the above and displays it; computes nothing | — |

```bash
PY=/data/bfys/gscriven/conda/envs/TE/bin/python
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd C2_Exact_scheme_table
$PY run_scheme_grid.py --q all      # 2889 s = 48 min, one core
$PY make_tables.py
$PY plot.py
```

`results/*.npz` are gitignored; the scripts and the dataset regenerate them
exactly (there is no randomness in this experiment — every test state of every
stratum is used, so there is no sample to seed).

**It ran locally, not on the farm.** A timing probe on eight states per cell put
the whole grid at about 45 minutes on one core, comfortably inside the two-hour
budget, so it was run as one process. Had it not been, the split is one job per
stage count: each `--q <n>` invocation writes its own row file and `make_tables.py`
picks up whatever is there, so ten lines in a `condor/jobs.txt` (1 CPU, 4 GB, no
comment lines — see [`../../_shared/condor/README.md`](../../_shared/condor/README.md))
would have produced the same tables. The measured cost per stage count was 88 s
at q = 2 rising to 482 s at q = 20.

---

## C2.1 The solver

For a start state S0 on a plane z0 and a step dz, the q-stage scheme puts a
**stage state** Y_j on each of the planes z0 + c_j·dz and requires

>   Y_j = S0 + dz · Σ_k a_jk f(Y_k, z0 + c_k·dz),  j = 1 … q
>
>   S1  = S0 + dz · Σ_j b_j  f(Y_j, z0 + c_j·dz)

where (c, A, b) is the Gauss-Legendre tableau from
[`../../_shared/irk.py`](../../_shared/irk.py) — verified on construction, every time
— and f is the LHCb equation of motion. The first line is implicit; it is solved
directly with `scipy.optimize.root` (`hybr`, numerical Jacobian).

[`exact_solver.py`](exact_solver.py) is
[`../../Block_0_first_pass/S1_Simple_first_pass/exact_scheme.py`](../../Block_0_first_pass/S1_Simple_first_pass/exact_scheme.py)'s
`solve_leg` generalised in three ways and no others:

* **the field is an argument.** That file calls the ODE with no field, i.e. the
  MagDown map. The sample is a MagUp sample
  ([`../C0_Magnet_tracks_dataset/README.md`](../C0_Magnet_tracks_dataset/README.md)),
  and the dataset's labels were built on MagUp, so the polarity has to be the
  caller's choice rather than a default. It is passed explicitly here:
  `field.v8r1.up.bin`, md5 `9e49ddc4313b589f273540e7e0bb513b`.
* **z0 and dz are per sample**, since the Block C dataset carries a start plane
  and a step length on every row rather than one frozen leg.
* **the residual tolerance is 1e-9** rather than 1e-8, tightened because the
  strata reach down to 0.05 mm steps where a loose tolerance would be visible
  in the answer.

Everything else is that file's arithmetic unchanged: q/p is held fixed, so the
unknowns are **4 per stage** (80 at q = 20 on a full crossing); the residual is
divided by (1, 1, 1e-3, 1e-3) so that every component the solver sees is in
**mm or mrad** and the mixed-unit system is well conditioned; the initial guess
is the **straight line** through the input state evaluated at the nodes, which
uses only what a network would be given; and **convergence is judged on the
residual**, not on the solver's own flag — `root` reports failure whenever it is
asked for a tighter *step* tolerance than it can deliver, even when the
equations are already satisfied to roundoff.

**Nothing in `../../_shared/` was changed.** `_shared/irk.py`'s generic `exact_step`
already takes the right-hand side as a callable, so a caller binds whichever
field it wants into that closure; the field argument belongs in the
LHCb-specific binding, which is what this file is. `_shared/smoke_tests.py` was
re-run anyway and all six gates pass (below).

---

## C2.2 The grid

Ten stage counts **q = 2, 4, 6, 8, 10, 12, 14, 16, 18, 20** (nominal classical
orders 4 to 40) × the six |dz| strata × the v3 **test** split. The cap is 2000
states per stratum and the test split holds exactly 2000, so **n = 2000 in every
one of the 60 cells** — 120,000 solves, no subsampling. Every cell is reported
pooled over direction and split into forward (dz > 0) and backward (dz < 0)
halves, which the dataset balances 50/50 by construction.

---

## C2.3 The tables

### The scheme's error against the fine reference

`results/scheme_table.csv`. Endpoint error is the house measure —
max(|Δx|, |Δy|) in microns against the dataset's RK6 label — **median over the
2000 states**, both directions pooled. The last row is the same states with the
magnet switched off (`results/straight_line_table.csv`), the null step every
scheme has to beat.

| q | 0.05–0.2 mm | 0.5–2 mm | 5–20 mm | 50–200 mm | 500–2000 mm | full crossing |
|---|---|---|---|---|---|---|
| 2 | 1.42e-11 | 2.34e-10 | 3.18e-09 | 3.21e-03 | 12.79 | 9256 |
| 4 | 1.42e-11 | 2.35e-10 | 3.07e-09 | 7.49e-04 | 1.218 | 3617 |
| 6 | 1.42e-11 | 2.34e-10 | 3.07e-09 | 3.48e-04 | 0.8252 | 232.6 |
| 8 | 1.42e-11 | 2.34e-10 | 3.05e-09 | 2.01e-04 | 0.6958 | **32.63** |
| 10 | 1.42e-11 | 2.31e-10 | 3.04e-09 | 1.37e-04 | 0.5044 | 32.49 |
| 12 | 1.42e-11 | 2.34e-10 | 3.02e-09 | 9.71e-05 | 0.3964 | 41.50 |
| 14 | 1.42e-11 | 2.34e-10 | 3.04e-09 | 6.50e-05 | 0.3038 | **20.36** |
| 16 | 1.42e-11 | 2.34e-10 | 3.01e-09 | 5.34e-05 | 0.2189 | 36.13 |
| 18 | 1.42e-11 | 2.27e-10 | 3.00e-09 | 4.21e-05 | 0.1671 | 38.62 |
| 20 | 1.42e-11 | 2.34e-10 | 2.98e-09 | 3.41e-05 | 0.1457 | 23.56 |
| **straight line** | 1.31e-04 | 1.39e-02 | 1.415 | 141.1 | 15,986 | 456,075 |

All values in microns; median |dz| per stratum 0.098 / 0.997 / 10.0 / 101.9 /
997.9 / 5174.7 mm.

### The straight line on its own

`results/straight_line_table.csv`, pooled row:

| stratum | median \|dz\| [mm] | median [µm] | p95 [µm] | mean [µm] | slope error median [mrad] |
|---|---|---|---|---|---|
| 0.05–0.2 mm | 0.0982 | 1.31e-04 | 1.02e-03 | 2.66e-04 | 0.00285 |
| 0.5–2 mm | 0.9966 | 0.01390 | 0.1021 | 0.02756 | 0.0287 |
| 5–20 mm | 10.00 | 1.415 | 9.775 | 2.746 | 0.291 |
| 50–200 mm | 101.9 | 141.1 | 1046 | 278.5 | 2.945 |
| 500–2000 mm | 997.9 | 15,986 | 106,125 | 28,993 | 33.63 |
| full crossing | 5174.7 | 456,075 | 1,297,360 | 540,792 | 176.2 |

The straight-line slope error *is* the bend: a straight step never changes tx or
ty, so its slope error is the deflection the magnet actually applied.

### Convergence

**Every one of the 120,000 solves converged.** The converged fraction is
1.000 in all sixty cells and for both directions — nothing was dropped from any
median, and there is no cell where the table is quietly reporting a subset.
`figures/scheme_convergence_fraction.png` is a uniform green grid; that is the
result, not a formatting accident. Worth stating plainly, because "does the
implicit solve converge on a C0 field, including one giant step across the whole
magnet?" was the feasibility question `../../Block_0_first_pass/S1_Simple_first_pass` opened and this
closes it at every step length up to 80 unknowns per state.

### Cost

The median number of **residual evaluations per solve** is exactly **4q + 5** in
five of the six strata — one numerical Jacobian (4q columns) plus a handful of
Newton steps — and exactly **8q + 10**, two Jacobians, in the 5–20 mm stratum.
That stratum is the one where the straight-line guess is close enough to look
converged but not close enough to satisfy 1e-9 mm, so the solver takes a second
Newton step and `hybr` rebuilds the Jacobian; it is the only place in the grid
where the initial guess costs anything.

One residual evaluation is one call of the LHCb right-hand side at all q nodes
at once, so the field is sampled 4q + 5 times per stage-node set. Wall time per
(q, stratum) cell of 2000 solves:

| q | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|
| **whole q row [s]** | 85 | 126 | 168 | 214 | 257 | 317 | 357 | 406 | 448 | 479 |

2857 s of solve time over the sixty cells, 2889 s of wall for the whole script.
Per solve that is 7.1 ms at q = 2 and 39.9 ms at q = 20, and dividing by the
evaluation counts gives **0.40 ± 0.02 ms per residual evaluation at every stage
count** — flat, so the whole cost is the number of evaluations and the number of
evaluations is 4q + 5. That 0.40 ms is python call overhead rather than
arithmetic (one evaluation is one `deriv` call on a (q, 5) array, i.e. q field
lookups), which is why it does not grow with q. The implicit solve at 80
unknowns is not a practical obstacle: a full crossing costs 7.5 ms at q = 2 and
36.0 ms at q = 20.

### Forward against backward

The two halves agree to a few per cent everywhere except the full crossing,
where **forward legs are about twice as accurate as backward ones**:

| q | stratum | dz > 0 [µm] | dz < 0 [µm] |
|---|---|---|---|
| 8 | 50–200 mm | 2.20e-04 | 1.78e-04 |
| 8 | 500–2000 mm | 0.759 | 0.652 |
| 8 | full crossing | 25.05 | 39.19 |
| 20 | 50–200 mm | 3.41e-05 | 3.40e-05 |
| 20 | 500–2000 mm | 0.171 | 0.128 |
| 20 | full crossing | 16.69 | 34.94 |

The dataset is balanced in direction, so this is a property of the step and not
of the sample. It is reported, not explained: a forward leg starts at the UT
plane in weak field and runs into the peak, a backward leg starts at the SciFi
plane and runs into it from the other side, and the collocation nodes fall on
different parts of the field profile in the two cases. Which of those matters
is a question for a later folder.

---

## C2.4 Reading the table

### Where the classical order 2q is visible

**Nowhere, and the two halves of the grid fail to show it for opposite reasons.**

Comparing two stage counts at one step length cannot demonstrate a classical
order on its own — the error constant changes with q as well as the power — so
the measurement that can is the **step-length slope at fixed q**: a single step
of length h has classical local error O(h^(2q+1)), so a log-log plot of error
against |dz| should have slope 2q + 1. That is the left panel of
`figures/scheme_error_vs_dz_and_q.png`. Fitted over the strata whose error is
above the reference's floor:

| q | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|
| classical 2q + 1 | 5 | 9 | 13 | 17 | 21 | 25 | 29 | 33 | 37 | 41 |
| **measured slope** | **3.78** | **3.88** | **3.41** | **3.09** | **3.18** | **3.32** | **3.25** | **3.43** | **3.31** | **3.09** |

Three point something, flat in q, against a prediction that runs from 5 to 41.

**Below one field cell the scheme is already at the reference's floor at q = 2.**
The v8r1 map is trilinear on a **100 mm cubic grid**, so the three shortest
strata — median steps of 0.098, 1.0 and 10 mm, i.e. 0.001, 0.01 and 0.1 of a
cell — almost never cross a cell face, and inside a cell the field along the
path is a smooth low-degree polynomial that even a two-stage Gauss method
integrates essentially exactly. The medians in those three columns are
1.42e-11, 2.34e-10 and 3.0e-9 µm and they **do not move at all** between q = 2
and q = 20 (factors 1.00, 1.00, 1.07). Those numbers are not the scheme's error;
they are the fp64 arithmetic of the comparison — a few ulps of a coordinate of
order 100 mm, plus the reference's own rounding accumulated over its 0.1 mm
march. The steep improvement with q that a smooth short step ought to show has
nowhere to happen, because there is no error left to remove before q = 2.

Two caveats on that, both important for later folders.

* **The reference floor is length-dependent, and the quoted 5e-5 µm is the
  full-crossing figure.** `../C1_Fine_reference` established 5e-5 µm as the worst
  case over 200 whole 5.2 m crossings, where the march takes ~52,000 steps. Over
  a 0.1 mm step it takes one, and the floor is correspondingly six orders lower.
  The three short strata sit *below* 5e-5 µm, not at it; a horizontal line at
  5e-5 µm in the figure is the right reference for the right-hand end of the
  x axis only.
* **The tail still moves where the median does not.** In the 5–20 mm stratum the
  median is pinned at 3e-9 µm but the p95 falls from 7.97e-6 to 7.05e-8 µm from
  q = 2 to q = 20, a factor **113**. Those are the states in that stratum whose
  step does happen to cross a cell face (a 10 mm step crosses one about a tenth
  of the time) plus the softest tracks. So the scheme's error is genuinely
  falling with q there; it is only the typical state that has nothing left to
  give.

### Where the C0 kinks take over

**At the 50–200 mm stratum, i.e. as soon as a step spans one 100 mm cell, and at
every q.** That stratum's median |dz| is 101.9 mm — 1.02 cells — and it is the
first column in which the error moves with q at all (a factor 94 from q = 2 to
q = 20) and the first that stands above the reference's floor. From there
upwards the step-length slope is the 3.1–3.9 in the table above rather than
2q + 1.

This is the same verdict `../C1_Fine_reference` reached from the other side and it
should be: its C1.2 step ladder found the sixth-order reference converging at
**order three** on the real map while the same code on a smooth field showed
order 6.23, and its first halving ratio was 8.0 = 2³. A scheme integrating
across a derivative jump behaves like a low-order one no matter what its tableau
says, and a 5.2 m crossing passes about **52 cell faces**. The kinks are a
property of the map, not of the scheme, so they cap the Gauss-Legendre family
and RK6 alike at the same effective order.

The consequence is visible as the plateau in the last column: the full-crossing
median is 9256 µm at q = 2, falls steeply to 32.6 µm at q = 8, and then stops —
32.6, 32.5, 41.5, 20.4, 36.1, 38.6, 23.6 µm for q = 8 … 20. The scatter of that
sequence is not noise from too few states (each is a median over 2000) but the
kink-limited error itself, which has no reason to be monotone in q: adding
stages moves the collocation nodes, and where the nodes fall relative to the
~52 cell faces changes the error by tens of per cent in either direction.

### The ceiling row to carry into the network tables

For a network trained on the full cross-magnet step — the leg every one-step
network in this folder has been trained on — the number to put next to its score
is

> **exact scheme, q = 8, full crossing: 32.6 µm median, 667 µm p95, 0.012 mrad
> slope, over 2000 test states, 100 % converged.**

and, as the best the family can do at any stage count,

> **the full-crossing ceiling is 20–42 µm for every q from 8 to 20; it does not
> improve with more stages.**

q = 8 is the row to quote by default because it is the stage count every network
in `../../Block_0_first_pass/S2b_One_step_network_v2`, `../../Block_A_technique_works/A3a_General_leg_network` and `../../Block_A_technique_works/A3b_Chained_legs` was
built with, and because the plateau starts there — nothing above q = 8 buys
anything on this step. It sits comfortably next to the 46 µm that
`../../Block_0_first_pass/S1_Simple_first_pass` reported for q = 8 on leg B; the difference is population,
polarity and reference, and the two agree on the order of magnitude, which is
the check that matters.

For networks trained on the shorter strata the ceiling is not a useful number
— the scheme is exact to fp64 there — and what should be quoted instead is the
straight line, because that is what the network has to beat: 1.31e-4 µm at
0.1 mm, 0.0139 µm at 1 mm, 1.42 µm at 10 mm, 141 µm at 100 mm. That is the
comparison `../../Block_A_technique_works/A3a_General_leg_network`'s verdict was really about.

The full per-cell ceiling table is `results/scheme_table.csv`; the q = 8 and
q = 20 rows for every stratum are also lifted into `results/reading.json` under
`ceiling_row`.

### The residual-evaluation cost per cell

Per solve, **4q + 5** residual evaluations (8q + 10 in the 5–20 mm stratum) —
13 at q = 2, 37 at q = 8, 85 at q = 20. Per cell of 2000 states that is 26,000
at q = 2, 74,000 at q = 8 and 170,000 at q = 20, and per 100 mm field cell the
step crosses:

| stratum | cells crossed | evals per solve at q = 8 | evals per field cell crossed |
|---|---|---|---|
| 0.05–0.2 mm | 0.001 | 37 | 3.8e+04 |
| 0.5–2 mm | 0.010 | 37 | 3.7e+03 |
| 5–20 mm | 0.100 | 74 | 7.4e+02 |
| 50–200 mm | 1.02 | 38 | 37 |
| 500–2000 mm | 9.98 | 40 | 4.0 |
| full crossing | 51.7 | 42 | 0.81 |

The last column is the one that says why a single giant implicit step is
attractive at all: crossing the whole magnet in one q = 8 solve costs **0.81
residual evaluations per field cell traversed**, against the fine reference's
0.1 mm march at 7 stage evaluations per 0.1 mm — seven thousand per cell, four
orders of magnitude more. The exact scheme buys that by giving up four orders of
accuracy (32.6 µm against the reference's 5e-5 µm), which is the trade the whole
discrete-time line is about.

In seconds: 17.8 ms per full-crossing solve at q = 8 on one core, against the
0.12 s per track the reference needs at 0.1 mm in a batch of a few thousand
(`../C1_Fine_reference` C1.3) — a factor 7 in the scheme's favour, at four orders
worse accuracy, and that is *before* a network replaces the root-finder. The
ceiling this folder measures is what such a network would be aiming at; the
speed the root-finder needs is not, since the network's whole purpose is to
skip it.

---

## Gates

`PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python ../../_shared/smoke_tests.py`
— all six gates pass, unchanged (nothing in `_shared/` was modified by this
folder; the run is recorded because the Block C protocol asks for it whenever a
folder touches the shared solver path).

The tableau battery in `../../_shared/irk.py` runs on every call:
`gauss_legendre(q)` verifies the row sums, the quadrature exactness to degree
2q − 1, the collocation conditions, the symplecticity identity and (for
q ≤ 3) the literature tableaus before returning, so every one of the ten stage
counts in this grid was checked at the moment it was used.
