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

_Filled in when the runs finish; see the report._
