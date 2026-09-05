# Chained_legs — the network walked along real particle paths  (A3b)

**Question.** A one-step score says the network can take *a* step. An
extrapolator takes many, each starting from its own previous answer. This
experiment feeds the network its own output leg after leg along the path a real
simulated particle actually took, and asks whether the error stays bounded.

## How a chain is built

For each particle in a split we take its **forward** training rows, collect the
planes they name (every `z0` and every `z1`), sort them, and call consecutive
planes a leg. The chain starts from the state stored at the first plane, and the
truth is the fp64 RK4 path from that same start state through the same planes —
not a stored label, so the network and the reference begin in the same place and
only the propagation is being compared.

The obvious rule — row *i* is followed by row *j* when `z1(i) == z0(j)` — was
tried first and is not usable: only **15** of the 7,636 test particles have four
rows that butt together exactly, because the training set stores at most three
plane-to-plane legs per particle and they need not be adjacent. 42% of
consecutive row pairs do butt together exactly; the rest leave a gap that is
simply a piece of the particle's path no stored row covers. Using the sorted
planes as waypoints keeps every stored plane and fills those gaps with one more
leg of the same kind, which is what a real extrapolator has to do anyway.

Kept: particles with **four or more** legs — **4,563 test** (23,500 legs) and
**4,551 val** (23,516 legs); chain lengths 4 (1,649), 5 (1,433), 6 (628) and
7 (853) on test. Median leg length 70 mm, 99th percentile 5.2 m.

## Leg D — the composite test

Leg D is the downstream leg: the first T-station state taken back to the primary
vertex in one giant backward step (median -8.6 m). The exact q = 8 scheme itself
is only good to 0.74 mm on it, and q = 16 to 92 µm. The question is whether the
same network does better by *walking* it: first back to the particle's own UT
plane, then back to the vertex plane. Scored against the stored D-leg label.
2,726 test particles have a D leg. **No D leg is in the training set** — the
network was trained on A, B and C only — so this is an out-of-training test by
construction; it has seen backward steps (all A legs, and 490 of 1060 B legs),
but not this geometry.

## script -> output

| script | what it does | output |
|---|---|---|
| [build_chains.py](build_chains.py) | assembles the chains and the leg-D set from the v2 training rows | `results/chains_test.npz`, `results/chains_val.npz`, `results/leg_d_test.npz`, `results/chains_meta.json` |
| [chain.py](chain.py) | walks all 26 trained networks along every chain, scores against the RK4 path, runs the leg-D composite, and ranks the seeds | `results/chain_summary.csv`, `results/chains.csv`, `results/selection.csv`, `results/leg_d_reproduction.csv`, `results/example_paths.npz` |
| [measure_leg_d_ceiling.py](measure_leg_d_ceiling.py) | the exact scheme solved on **these** D legs, so the ceiling is like-for-like | `results/leg_d_ceiling_same_population.csv` |
| [plot_chains.py](plot_chains.py) | the figures, from those tables only | `figures/error_vs_chained_legs.png`, `figures/error_growth_examples.png`, `figures/leg_d_reproduction.png` |
| [analysis.ipynb](analysis.ipynb) | loads and displays; computes nothing | — |

`chain_summary.csv` carries every particle at every step, aggregated. `chains.csv`
carries the individual particle rows asked for (network, seed, mode, particle,
step, cumulative dz, p, endpoint error, slope error) for a fixed random
subsample of 500 test particles — the same 500 for every network, so the
networks can be compared particle by particle without a million-row file.

Seed selection: physics seeds are ranked, per architecture, by the median chain
error after four legs on the **val** particles; the test chains are then quoted
both for the selected seed and for every seed. `selection.csv` also carries the
rank correlations between the one-step val error, the val chain error and the
test chain error — i.e. whether the cheap one-step number would have picked the
same seed.

## Verdict

All 26 networks from `../General_leg_network` (cluster 5781158, all converged)
were walked along all 4,563 test and 4,551 val chains.

### 1. Chained error is not bounded — it compounds

Median endpoint error against the RK4 path, test split, by the number of legs
walked (all 4,563 particles contribute up to leg 4; beyond that only the longer
chains do, so the first four steps are the like-for-like comparison):

| legs walked | 4x50 physics | 4x50 twin | 4x100 physics | 4x100 twin | particles |
|---|---|---|---|---|---|
| 1 | 440 | 388 | **246** | 217 | 4563 |
| 2 | 1423 | 1275 | **783** | 652 | 4563 |
| 3 | 2562 | 2225 | **1443** | 1051 | 4563 |
| 4 | 3919 | 3634 | **2293** | 1484 | 4563 |
| 5 | 8163 | 6991 | 4831 | 2869 | 2914 |
| 6 | 14479 | 10087 | 8904 | 4820 | 1481 |
| 7 | 15999 | 10739 | 10248 | 5283 | 853 |

All µm. The step-to-step growth ratio is 3.2, 1.8, 1.6, 2.1, 1.8, 1.1 at 4x100
physics and essentially the same at every architecture and both losses. Two
things follow. The first step alone already costs 246 µm — the network's own
one-step error. After that the error roughly doubles per leg: this is not the
random walk of independent errors (which would grow like the square root of the
number of legs) but a systematic one, each leg starting from a state that is
already displaced and adding its own bias on top. Nothing blows up — no chain
diverges, and the p95 grows at the same rate as the median — but nothing
saturates either. Over a full particle path the network is 5-10 mm out.

### 2. The one-step error does not select the seed; the chain does

Physics seeds ranked by the median chain error after four legs on **val**, then
read out on **test** (`results/selection.csv`):

| architecture | selected seed | val chain | test chain | Spearman val-chain vs test-chain | Spearman one-step-val vs test-chain |
|---|---|---|---|---|---|
| 4x50 physics | `w50_physics_s6` | 3715 µm | 3409 µm | **0.92** | 0.76 |
| 4x100 physics | `w100_physics_s6` | 2086 µm | 1975 µm | **0.95** | **0.08** |
| 4x50 twin | `w50_data_s2` | 3695 µm | 3634 µm | -0.50 (3 seeds) | -0.50 |
| 4x100 twin | `w100_data_s2` | 1489 µm | 1403 µm | 1.00 (3 seeds) | 1.00 |

The val chain error transfers to test almost perfectly (rank correlation 0.92 and
0.95 over ten seeds). The cheap one-step val error does **not**: at 4x100 its rank
correlation with the test chain error is 0.08, i.e. no information. The best
one-step seed at 4x100 (`w100_physics_s1`, 339 µm) is only sixth best when
chained. A seed has to be selected on the chain, and the ten seeds differ by 40%
in chained error at fixed architecture — a spread as large as the gap between the
two architectures' medians.

### 3. Leg D: composite stepping makes the giant step worse, not better

Median error against the stored D-leg label, over 2,726 test particles:

| | 4x50 physics | 4x50 twin | 4x100 physics | 4x100 twin |
|---|---|---|---|---|
| two composite steps (T -> UT -> vertex) | 47353 | 34228 | 59416 | 33416 |
| the same leg in one giant step | 17947 | 6213 | **8778** | **3158** |

All µm. Reference points: the exact q = 8 scheme on these same D legs is **752 µm**
(q = 16: 216 µm; the published stratified-sample numbers are 739 and 92 µm), and
the straight line is 948 mm.

So: **no.** Walking the downstream leg in two steps is 4-10x *worse* than taking
it in one, and both are more than an order of magnitude above the one-giant-step
ceiling the composite was supposed to beat. The reason is visible in the first
step alone: the T -> UT hop (median -6.4 m, backward, longer than any B leg) is
already 1.5-10 mm out, and the second step then starts from there. The network
has seen backward steps in training — all 1,317 A legs are backward and 490 of
1,060 B legs are — so this is not a sign problem; it is that no D-leg geometry
was in the training set at all, and the network does not extrapolate to it. The
network does still beat the straight line by a factor 100-300, which is the only
thing that can be said for it here.

### In one line

One label-free network can be pointed at any leg and chained along a real
particle path without breaking, and the val chain ranks seeds reliably — but the
error compounds by roughly a factor two per leg to 5-10 mm over a full path, and
composite stepping does not recover the accuracy of a single big step, let alone
approach the exact scheme's ceiling.
