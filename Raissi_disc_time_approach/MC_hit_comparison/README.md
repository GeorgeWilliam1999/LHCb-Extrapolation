# MC_hit_comparison — steps that end on a real simulated hit  (Block C, C5)

**What this is.** Every number in Block C so far has been an error against a
*computed* truth: the fine sixth-order reference of
[`../Fine_reference`](../Fine_reference), which propagates a state through the
v8r1 field map and nothing else. This folder asks the other question. Take a
step whose start **and** end states are both places where a real simulated
particle actually crossed a sensor plane, predict the end from the start, and
compare with where the particle really was.

**Why it is a different question.** The simulated particle scatters in the
material, loses energy, and occasionally decays or interacts. A field-only
extrapolator models none of that, by design — the project's stance is that
material corrections stay with `TrackMasterExtrapolator` and the surrogate
replaces field propagation only
([`../../Data_generation_exploration/Data/README.md`](../../Data_generation_exploration/Data/README.md),
gate G2). So the fine reference itself misses the real hit, and that miss is the
**material floor**: the part of the step no field-only method of any kind can
predict. A network is worth measuring against hits only in so far as it stays
above that floor; once its field-only error drops below it, the comparison to
hits cannot tell it from the reference and stops being informative.

Three predictors are scored against the same real end states:

| | what it is | what it measures |
|---|---|---|
| **the fine reference** | `rk6_rows` at 0.1 mm on the MagUp map, from the start hit | **the material floor** |
| **the exact q = 8 scheme** | [`../Exact_scheme_table`](../Exact_scheme_table)'s root-finder, residual 1e-9 mm / mrad | the discretisation the scheme adds on top of the floor |
| **the grid networks** | the trained checkpoints of [`../Step_size_and_stage_grid`](../Step_size_and_stage_grid) at q = 8, both arms, four architectures | what the network adds on top of both |

The straight line is scored too, as the null step.

---

## script → output

| script | what it does | output |
|---|---|---|
| [build_hit_set.py](build_hit_set.py) | **C5.1.** the hit-to-hit set: consecutive sensor crossings of one particle (leg C) and UT → SciFi crossings (leg B), both directions, cut to the test split and to the domain | `results/hit_set.npz`, `results/hit_set_meta.json` |
| [score_hits.py](score_hits.py) | **C5.2.** the three predictors (plus the straight line) against the real end state and against the fine reference, by stratum, momentum band and direction | `results/hit_table.csv`, `results/hit_rows.npz`, `results/hit_scoring_meta.json` |
| [plot.py](plot.py) | **C5.3.** the four-panel figure, from `hit_table.csv` alone | `figures/hit_comparison.png` |
| [analysis.ipynb](analysis.ipynb) | loads all of the above and displays it; computes nothing | — |

```bash
PY=/data/bfys/gscriven/conda/envs/TE/bin/python
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd MC_hit_comparison
$PY build_hit_set.py     # about a second
$PY score_hits.py        # the fine reference dominates; one thread
$PY plot.py
```

`results/*.npz` are gitignored; the scripts and the recorded seed regenerate
them.

---

## The label caveat, stated first because it is the trap

The v2 training set
(`../../Data_generation_exploration/Official_xdigi/training_v2/train_official_v2.npz`)
has a `Y` column that looks like the end state of exactly these steps. **It is
not usable here.** It was computed by field-only propagation on the **MagDown**
map, and the sample is a MagUp sample: `../Magnet_tracks_dataset`'s C0
measured a median cross-magnet miss of 883 mm on MagDown against 1.73 mm on
MagUp, with the bend coming out backwards for every particle in the sample. `Y`
is therefore the right start state pushed through the wrong polarity, twice
removed from the thing this folder needs.

**Nothing in this folder reads `Y`.** The end state is the particle's own next
MCHit state, taken from
`../../Data_generation_exploration/Official_xdigi/results/states.npz` — the file
the v2 set is itself built from. The v2 npz is opened for one field only, its
per-particle train/val/test label, which is a property of the split and not of
the labels.

---

## C5.1 The set

`build_hit_set.py` sorts every particle's harvested plane crossings by z and
takes **consecutive pairs**. A pair of adjacent tracker planes is the v2
training set's leg-C geometry; a pair that jumps the magnet (a UT plane to a
SciFi plane) is its leg B. Both are kept, and both directions of each: the
reversed pair starts at the later hit and ends on the earlier one, and its
truth is that earlier hit's measured state, so the backward half is exactly as
truthful as the forward half.

| cut | pairs out | particles out | why |
|---|---|---|---|
| consecutive plane crossings of one particle | 1,080,237 | 147,609 | — |
| \|dz\| ≥ 0.05 mm | 1,072,707 | 146,983 | below this the two "hits" are the two faces of one sensor |
| both planes inside 2200 < z < 9500 mm | 688,406 | 105,170 | the network's (z0, dz) inputs were normalised on the magnet-to-magnet population; a VELO start plane is outside it. **384,301 pairs removed, almost all VP → VP** |
| 2 < eta < 5 | 510,748 | 67,477 | the domain the networks were trained on |
| 1 < p < 200 GeV | 398,317 | 38,871 | the same |
| non-electron | 317,315 | 29,535 | an electron's residual is bremsstrahlung, a different measurement |
| the particle is in the **v2 test** split | 31,640 | 2,926 | 80/10/10 by particle, seed 20260718 |
| the particle is **not** in the v3 train or val split | 24,598 | 2,436 | **5,853 pairs removed** |
| both directions | 49,196 | 2,436 | — |
| the seeded per-stratum draw | **13,228** | **2,268** | a cap, not a cut; see below |

**The second split cut is the one that had to be found rather than assumed.**
`../Magnet_tracks_dataset`'s split is 60/20/20 by particle over a 6,000-particle
subsample of the same underlying population, seeded independently of the v2
set's 80/10/10 — so a particle in the v2 *test* split can perfectly well be in
the v3 *train* split, and its whole RK6 path is what the grid networks were
fitted on. 490 of the 7,636 v2-test particles are in the v3 train or val split;
they are removed, costing 5,853 pairs. Without that cut this folder would be
scoring the networks partly on their own training particles.

The draw, seed 20260908:

| stratum | \|dz\| | available | cap | drawn |
|---|---|---|---|---|
| 0.05–2 mm | 0.05–2 mm | 436 | 4,000 | **436** |
| 2–20 mm | 2–20 mm | 382 | 4,000 | **382** |
| 20–200 mm | 20–200 mm | 36,150 | 6,000 | **6,000** |
| 200–2000 mm | 200–2000 mm | 9,818 | 4,000 | **4,000** |
| cross-magnet | > 2000 mm | 2,410 | 3,000 | **2,410** |

The strata are contiguous rather than the six the grid trained on, because a
hit-to-hit step length is set by the detector geometry and cannot be drawn:
there is nothing between 0.2 mm and 0.5 mm to put in a stratum. The caps exist
because the fine reference costs about 0.13 s per cross-magnet leg on one
thread. Two strata are population-limited (436 and 382 rows) and their numbers
should be read as such.

**What came out: 13,228 rows from 2,268 particles**, 6,669 forward and 6,559
backward, \|dz\| from 0.12 mm to 6,755 mm with a median of 70.2 mm. By plane
pair: FT → FT 8,113, UT → UT 2,705, UT → FT 1,205 and FT → UT 1,205 — the last
two are leg B, the whole magnet, in both directions. By momentum: 1–5 GeV
6,526, 5–20 GeV 5,212, 20–200 GeV 1,490.

---

## C5.2 The scoring

`score_hits.py` makes three predictions from each start state and compares each
with the measured end state. Two errors are recorded for every one of them:

* **`vs_hit`** — against the particle's real next hit. This is the number the
  folder exists for.
* **`vs_reference`** — against the fine reference's own answer from the same
  start state. This is the *field-only* error, the quantity
  `../Step_size_and_stage_grid` measures, and it is what says whether a
  comparison against hits can resolve a predictor at all.

The error is the house measure throughout, max(|Δx|, |Δy|) in microns.

**The exact scheme.** All 13,228 solves converged, worst residual 4.5e-11 mm /
mrad against the 1e-9 tolerance — the same 100 % C2 reported, now on a
population of real hit-to-hit steps rather than drawn ones.

**Which network.** Four architectures at q = 8, both arms: **4 x 64** and
**8 x 128** (the two named in the plan), **2 x 256** (the twin's best
architecture on the short columns of C4) and **8 x 256** (the physics arm's best
on the crossing). Per reporting stratum the **seed with the lowest validation
median in the matching grid stratum** is used; the match is by median |dz| in
log space and is written into `results/hit_scoring_meta.json` along with the
seed chosen for every (architecture, arm, stratum). All three seeds are also in
`results/hit_table.csv` under the predictor name `network (every seed)`, so the
choice can be audited rather than believed. The map that came out:

| this folder's stratum | median \|dz\| | matched grid stratum |
|---|---|---|
| 0.05–2 mm | 0.650 mm | 0.5–2 mm |
| 2–20 mm | 6.31 mm | 5–20 mm |
| 20–200 mm | 69.8 mm | 50–200 mm |
| 200–2000 mm | 472 mm | 500–2000 mm |
| cross-magnet | 5,183 mm | full crossing |

Total cost 807 s on one thread: 522 s the fine reference, 271 s the exact
scheme, the rest the eight networks' forward passes.

---

## C5.3 The table

`results/hit_table.csv`. Median error against the real next hit, in microns,
all momenta, both directions pooled:

| predictor | 0.05–2 mm | 2–20 mm | 20–200 mm | 200–2000 mm | cross-magnet |
|---|---|---|---|---|---|
| **fine reference = the material floor** | **0.1094** | **0.5732** | **7.463** | **51.14** | **1,815** |
| exact scheme q = 8 | 0.1094 | 0.5732 | 7.463 | 51.14 | 1,847 |
| 2 x 256 physics | 0.1090 | 0.5816 | 9.192 | 100.8 | 2,682 |
| 4 x 64 physics | 0.1089 | 0.5992 | 8.662 | 83.01 | 2,383 |
| 8 x 128 physics | 0.1087 | 0.5930 | 8.769 | 90.64 | 2,376 |
| 8 x 256 physics | 0.1085 | 0.6049 | 8.912 | 77.17 | 2,370 |
| 2 x 256 twin | 0.1094 | 0.5704 | 7.933 | 73.91 | 3,088 |
| 4 x 64 twin | 0.1094 | 0.5887 | 8.467 | 77.31 | 6,902 |
| 8 x 128 twin | 0.1094 | 0.5888 | 8.174 | 75.76 | 6,343 |
| 8 x 256 twin | 0.1094 | 0.5796 | 8.295 | 76.63 | 8,633 |
| straight line | 0.1082 | 0.6889 | 20.96 | 620.7 | 5.258e+05 |
| rows | 436 | 382 | 6,000 | 4,000 | 2,410 |

The same divided by the floor — the **excess** each predictor costs over a
perfect field propagation:

| predictor | 0.05–2 mm | 2–20 mm | 20–200 mm | 200–2000 mm | cross-magnet |
|---|---|---|---|---|---|
| exact scheme q = 8 | 1.000 | 1.000 | 1.000 | 1.000 | **1.017** |
| physics arm, the four architectures | 0.992–0.996 | 1.02–1.06 | 1.16–1.23 | 1.51–1.97 | **1.31–1.48** |
| twin, the four architectures | 1.000–1.001 | 0.995–1.03 | 1.06–1.13 | 1.45–1.51 | **1.70–4.76** |
| straight line | 0.990 | 1.20 | 2.81 | 12.1 | **290** |

### By momentum band

Median error against the real next hit, microns:

| band | predictor | 0.05–2 mm | 2–20 mm | 20–200 mm | 200–2000 mm | cross-magnet |
|---|---|---|---|---|---|---|
| **1–5 GeV** | material floor | 0.1879 | 1.791 | 17.39 | 119.1 | **6,203** |
| | exact scheme q = 8 | 0.1879 | 1.791 | 17.39 | 119.1 | 6,408 |
| | 8 x 256 physics | 0.1876 | 1.840 | 20.53 | 188.5 | 8,277 |
| | 4 x 64 physics | 0.1874 | 1.843 | 19.56 | 200.9 | 7,833 |
| | 2 x 256 twin | 0.1878 | 1.789 | 18.10 | 178.0 | 14,030 |
| | 4 x 64 twin | 0.1876 | 1.822 | 20.13 | 214.2 | 34,710 |
| | straight line | 0.1880 | 1.953 | 44.53 | 1,388 | 1.149e+06 |
| | rows | 248 | 160 | 3,042 | 2,042 | 1,034 |
| **5–20 GeV** | material floor | 0.08466 | 0.4952 | 4.657 | 29.22 | **951.9** |
| | exact scheme q = 8 | 0.08466 | 0.4952 | 4.657 | 29.27 | 972.9 |
| | 8 x 256 physics | 0.08474 | 0.5183 | 5.078 | 41.41 | 1,143 |
| | 4 x 64 physics | 0.08468 | 0.4990 | 4.835 | 44.78 | 1,217 |
| | 2 x 256 twin | 0.08466 | 0.4943 | 4.749 | 39.51 | 1,562 |
| | 4 x 64 twin | 0.08466 | 0.4959 | 4.728 | 37.86 | 3,168 |
| | straight line | 0.08497 | 0.5409 | 12.32 | 408.3 | 3.420e+05 |
| | rows | 142 | 156 | 2,309 | 1,531 | 1,074 |
| **20–200 GeV** | material floor | 0.05801 | 0.1518 | 1.164 | 6.809 | **236.9** |
| | exact scheme q = 8 | 0.05801 | 0.1518 | 1.164 | 6.829 | 244.2 |
| | 8 x 256 physics | 0.05796 | 0.1524 | 1.495 | 13.17 | 514.3 |
| | 4 x 64 physics | 0.05796 | 0.1517 | 1.358 | 13.20 | 577.3 |
| | 2 x 256 twin | 0.05801 | 0.1515 | 1.282 | 11.60 | 531.3 |
| | 4 x 64 twin | 0.05801 | 0.1518 | 1.170 | 7.966 | 1,217 |
| | straight line | 0.05781 | 0.1585 | 3.299 | 105.5 | 9.906e+04 |
| | rows | 46 | 66 | 649 | 427 | 302 |

---

## C5.4 Reading it

### The material floor is enormous compared with anything the network adds

**0.109 µm over a 0.65 mm step, 7.5 µm over 70 mm, and 1.8 mm over the whole
magnet.** For comparison, the fine reference's own numerical floor is 5e-5 µm
(`../Fine_reference` C1.4) — the material term is **36,000 times larger** on the
crossing. It is the 1/p multiple-scattering line the training set's G2 gate has
always reported, seen here per step length: at the crossing it is 6.2 mm at
1–5 GeV, 0.95 mm at 5–20 GeV and 0.24 mm at 20–200 GeV, a factor 26 across the
momentum range.

### Below about 100 mm the test cannot see the network at all

Panel 3 of the figure is the statement. The network's **field-only** error —
its error against the fine reference, the thing C4 measures — sits far below the
floor on the short steps:

| stratum | material floor | network field-only error, the eight models | ratio |
|---|---|---|---|
| 0.05–2 mm | 0.1094 µm | 5.0e-05 – 3.6e-04 µm | **300 – 2,200 × below** |
| 2–20 mm | 0.5732 µm | 0.0013 – 0.0372 µm | **15 – 440 × below** |
| 20–200 mm | 7.463 µm | 0.556 – 3.37 µm | 2 – 13 × below |
| 200–2000 mm | 51.14 µm | 32.9 – 78.5 µm | comparable |
| cross-magnet | 1,815 µm | 769 – 7,766 µm | comparable |

and the vs-hit column shows exactly what that implies: at 0.05–2 mm every
predictor — reference, exact scheme, all eight networks and even the **straight
line** — returns 0.108–0.109 µm, the same number to three digits.
**A comparison against hits cannot distinguish them, because on that step the
magnet does less than the material does.** The same holds at 2–20 mm to within
5 %. The crossover, where the network's own error reaches the floor, is at
|dz| ≈ 100–200 mm.

That is the useful negative result of this folder: **hit-level validation is
only informative for steps longer than about 100 mm**, and for anything shorter
the field-only comparison of C4 is the only measurement that means anything.

### Where it can see the network: the excess over the floor

At 20–200 mm the network costs **6–23 %** over a perfect field propagation
(twin 6–13 %, physics arm 16–23 %); the exact scheme costs 0.0 %. At
200–2000 mm it costs **45–97 %**. At the crossing the physics arm costs
**31–48 %** and the twin **70–376 %** — and there the ordering of C4 (d)
survives contact with real hits: the physics arm's 2,370–2,682 µm against the
twin's 3,088–8,633 µm, on a floor of 1,815 µm.

By momentum band the excess **grows with momentum**, and sharply:

| stratum | 1–5 GeV | 5–20 GeV | 20–200 GeV |
|---|---|---|---|
| 20–200 mm, physics arm | 1.12 – 1.18 | 1.04 – 1.12 | 1.12 – 1.28 |
| 200–2000 mm, physics arm | 1.58 – 1.92 | 1.42 – 1.74 | 1.93 – 2.39 |
| cross-magnet, physics arm | **1.26 – 1.37** | **1.20 – 1.40** | **2.17 – 6.40** |
| cross-magnet, twin | 2.26 – 6.23 | 1.64 – 4.87 | 2.24 – 5.79 |

The reason is that the two terms scale differently. The material term goes as
1/p — 6,203 → 236.9 µm from the softest band to the hardest, a factor 26 — while
the network's field-only error on the crossing falls by much less, so the
*ratio* rises. **On soft tracks the network is already within 30 % of a
material-limited measurement; on 20–200 GeV tracks it is a factor 2 to 6 above
it**, and it is there that improving the extrapolator would actually show up in
a fit.

### The exact scheme is the sanity check, and it passes

The q = 8 root-finder is identical to the fine reference to four significant
figures in the four short strata and **1.7 % above it on the crossing** — 1,847
against 1,815 µm. That is exactly what C2's ceiling predicts: its field-only
error there is 41.5 µm on this population, against a 1,815 µm floor, so it can
only add about 2 % in quadrature-free terms. The scheme is not measurable
against hits either; only the network is, and only on the long steps.

### The tails

p95 against the real hit, all momenta, microns:

| predictor | 0.05–2 mm | 2–20 mm | 20–200 mm | 200–2000 mm | cross-magnet |
|---|---|---|---|---|---|
| material floor | 0.9476 | 6.002 | 59.35 | 416 | 20,160 |
| exact scheme q = 8 | 0.9476 | 6.002 | 59.35 | 416 | 22,140 |
| physics arm | 0.944–0.945 | 6.04–6.13 | 64.6–69.3 | 901–980 | 56,300–67,500 |
| twin | 0.945–0.947 | 6.01–6.09 | 61.5–66.7 | 847–1,149 | 103,800–221,700 |

The floor's own p95 is 8.7 times its median at 0.05–2 mm and 11 times at the
crossing — the tail is decays in flight and hard hadronic interactions, states
after which the particle genuinely is not where any extrapolator would put it
(the v1 characterisation's note on the same tail). The networks do not widen
that tail on the short steps at all, multiply it by 2.0–2.8 at 200–2000 mm, and
at the crossing by 2.8–3.3 (physics arm) or **5.1–11.0** (twin) — the same
architecture-dependent instability C4 (d) saw, and larger at the tail than at
the median.

### The forward/backward split

`results/hit_table.csv` carries `direction` = forward / backward as well as
pooled, for every row of every table above. It is recorded rather than read
here: on this population the direction of a leg-C step is a property of the
geometry, and the cross-magnet half is C4 (f)'s question asked on 1,205 legs
per direction rather than 1,000 — too few to add to what C4 already settled.

---

## What this folder does and does not establish

1. **A field-only extrapolator cannot be validated against hits below ~100 mm.**
   The material floor swallows the whole comparison; the straight line and the
   fine reference are the same number there.
2. **On the whole magnet the physics-loss networks land 31–48 % above the
   floor** — 2.37–2.68 mm against 1.82 mm — so a hit-level comparison *can* see
   them, and sees them as a real but modest degradation of a material-limited
   answer.
3. **The gap that matters is at high momentum**, where the material term is
   small: a factor 2–6 over the floor at 20–200 GeV against 1.3 at 1–5 GeV.
4. It does **not** establish that any of these networks is usable in a fit. That
   needs the residual propagated through a track fit, not an endpoint median,
   and it is not what this folder measures.
