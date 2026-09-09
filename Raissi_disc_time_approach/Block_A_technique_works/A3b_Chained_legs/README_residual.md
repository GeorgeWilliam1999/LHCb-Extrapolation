# Chaining the straight-line-residual networks  (A3b, wave 3)

**Question.** `README.md` walked wave 1's 26 networks — the ones that emit
absolute stage states — along real particle paths and found the error
compounding by roughly a factor two per leg to 5–10 mm over a full path, and the
leg-D composite four to ten times *worse* than taking the giant step in one go.
Both of those were dominated by the networks' own one-step error, which on the
short legs was fifty times a straight line.

`../A3a_General_leg_network/README_residual.md` rebuilds the network to predict the
**deviation from a straight line**, scaled per sample so that the target is O(1)
on every leg type. That fixes the one-step number. This file asks the two
questions that only chaining can answer:

1. does the compounding change, or does a smaller first step simply shift the
   same growth curve down?
2. does leg D — a geometry in no training set — behave any differently when the
   network is a correction to a straight line rather than a state predictor?

## What is identical to wave 1, and what is not

Everything except which networks are loaded:

- the same chains, from [build_chains.py](build_chains.py)'s outputs
  (`results/chains_test.npz`, `results/chains_val.npz`, `results/leg_d_test.npz`)
  — 4,563 test and 4,551 val particles with four or more legs, and 2,726 test
  particles with a D leg;
- the same fp64 RK4 reference paths from the same start states through the same
  planes, so only the propagation is compared;
- the same 500-particle subsample (seed 20260905) for the per-particle rows;
- the same seed-selection rule: physics seeds ranked per architecture by the
  median chain error after four legs on **val**, then read out on test;
- the same metrics, the same leg-D ceiling
  (`results/leg_d_ceiling_same_population.csv`).

`_shared/evaluate.py`'s `chain` needed no change at all. It calls
`model(S, extra)` with the per-leg normalised (z₀, dz), and the residual network
recovers the physical leg from exactly those two numbers and builds its own
scale from the field along the straight line. That is what makes the leg-D run
meaningful: the D geometry appears in no dataset, so the scale there has to come
from the inputs, and it does.

## script → output

| script | what it does | output |
|---|---|---|
| [chain_residual.py](chain_residual.py) | walks every residual network along every chain, scores against the RK4 path, runs the leg-D composite, ranks the seeds | `results/chain_summary_residual.csv`, `results/chains_residual.csv`, `results/selection_residual.csv`, `results/leg_d_residual.csv` |

It is a new file rather than a flag on [chain.py](chain.py) because the two arms
need different model classes, and nothing the wave-1 chaining analysis depends
on should move while it is being written up. `--limit N` truncates the particle
sets for exercising the pipeline; it is never used for a result.

All 26 residual networks (`../A3a_General_leg_network`, cluster 5781469, all
converged) were walked along all 4,563 test and 4,551 val chains and over all
2,726 leg-D particles.

# Results

## 1. The whole growth curve moved down by two orders of magnitude — and then compounds faster

*Numbers corrected 2026-09-06 to match `results/*.csv` (see
`Mini_paper/README.md`): the residual 4×100 twin entry at one leg.*

Median endpoint error against the RK4 path, test split, by the number of legs
walked. All 4,563 particles contribute up to leg 4; beyond that only the longer
chains do, so the first four steps are the like-for-like comparison.

| legs walked | wave 1 4×100 phys | wave 1 4×100 twin | **residual 4×100 phys** | **residual 4×100 twin** | residual 4×50 phys | particles |
|---|---|---|---|---|---|---|
| 1 | 246 | 217 | **1.2** | **0.01** | 1.1 | 4563 |
| 2 | 783 | 652 | **60** | **2.7** | 59 | 4563 |
| 3 | 1443 | 1051 | **281** | **32** | 261 | 4563 |
| 4 | 2293 | 1484 | **572** | **118** | 568 | 4563 |
| 5 | 4831 | 2869 | 1750 | 471 | 1919 | 2914 |
| 6 | 8904 | 4820 | 3804 | 1052 | 3973 | 1481 |
| 7 | 10248 | 5284 | 4492 | 1259 | 4739 | 853 |

All µm. Two things are true at once and they pull in opposite directions.

**The chain is better everywhere.** After four legs the residual network is
4.0× better than wave 1 with the physics loss (572 against 2293 µm) and 12.6×
better with the twin (118 against 1484 µm). Over a full seven-leg path it is
still 2.3× and 4.2× better. The p95 improves by the same factor at every step
(leg 4: 13.1 mm against 38.6 mm), so this is the whole distribution moving, not
the median alone.

**But the compounding got worse, not better.** The step-to-step growth ratios,
4×100 physics:

| | leg 1→2 | 2→3 | 3→4 | 4→5 | 5→6 | 6→7 |
|---|---|---|---|---|---|---|
| wave 1 | 3.18 | 1.84 | 1.59 | 2.11 | 1.84 | 1.15 |
| **residual** | **48.6** | **4.67** | 2.04 | 3.06 | 2.17 | 1.18 |
| residual, twin | 299 | 11.8 | 3.69 | 3.99 | 2.23 | 1.20 |

Wave 1 roughly doubled per leg from a 246 µm first step. The residual network
starts 200 times lower — 1.2 µm, essentially its own one-step leg-C number —
and then multiplies by 49 on the second leg and 4.7 on the third before settling
into wave 1's own factor of two. By leg 4 the two curves are only a factor 4
apart, and by leg 7 a factor 2.3.

The reason is visible in the one-step table: a chain's early legs are short
plane-to-plane hops, where the residual network is at 1.6 µm, and its later legs
include the cross-magnet crossing, where it is at 1026 µm — the leg the redesign
barely improved. The chain error is therefore dominated, after two or three
steps, by the one leg that did not get better. **The residual redesign bought a
much better start, not a slower compounding.** Nothing diverges — no chain
blows up, and the p95 grows at the median's rate — but nothing saturates either.

## 2. The val chain still selects the seed, and now the cheap one-step number partly does too

Physics seeds ranked by the median chain error after four legs on **val**, read
out on **test** (`results/selection_residual.csv`):

| architecture | selected seed | val chain | test chain | ρ(val chain, test chain) | ρ(one-step val, test chain) |
|---|---|---|---|---|---|
| residual 4×50 physics | `residual_w50_physics_s6` | 514 µm | 487 µm | **0.98** | 0.35 |
| residual 4×100 physics | `residual_w100_physics_s0` | 561 µm | 525 µm | **0.90** | 0.72 |
| residual 4×50 twin | `residual_w50_data_s2` | 167 µm | 141 µm | 1.00 (3 seeds) | 0.50 |
| residual 4×100 twin | `residual_w100_data_s0` | 122 µm | 110 µm | 1.00 (3 seeds) | 0.50 |

Wave 1's headline finding survives: the val chain error transfers to test almost
perfectly (0.90 and 0.98 over ten seeds), so a seed can be chosen on val without
touching test. What changes is the cheap alternative. In wave 1 the one-step val
error carried **no** information about the chained error at 4×100 (ρ = 0.08);
here it carries some (ρ = 0.72 at 4×100, 0.35 at 4×50). That is consistent with
the picture above — once the one-step number is dominated by the same leg that
dominates the chain, the two start to agree — but 0.72 over ten seeds is still
not enough to select on, and the ten seeds differ by 60% in chained error at
fixed architecture (487–783 µm at 4×100 physics). **Select on the val chain.**

## 3. Leg D: still worse in two steps than in one, but the giant step is now 2.3× better

Median error against the stored D-leg label, 2,726 test particles:

| | wave 1 4×100 phys | wave 1 4×100 twin | **residual 4×100 phys** | **residual 4×100 twin** | residual 4×50 phys |
|---|---|---|---|---|---|
| two composite steps (T → UT → vertex) | 59416 | 33416 | **4252** | **2374** | 4768 |
| the same leg in one giant step | 8778 | 3158 | **3749** | **2213** | 3806 |

All µm. Reference points on these same D legs: the exact q = 8 scheme is
**752 µm** (q = 16: 216 µm) and the straight line is **948 mm**. The stored
label agrees with a fresh RK4 pass to 0.045 µm.

Three readings:

1. **The composite improves by a factor 14** (59.4 → 4.3 mm at 4×100 physics),
   far more than the single giant step's factor 2.3. The composite's first hop
   is the T → UT step (median −6.4 m, backward, longer than any B leg), and it
   went from 4.1 mm to 1.5 mm; the second hop is a vertex-fetch-like leg, which
   is exactly where the residual arm gained a factor 92 one-step.
2. **Walking it in two steps is still worse than taking it in one** — 4252
   against 3749 µm at 4×100 physics — but the gap has closed from a factor 6.8 to
   a factor 1.13. Wave 1's conclusion that composite stepping actively destroys
   accuracy is now only marginally true.
3. **Both are still five times the exact scheme's own ceiling on this geometry**
   (752 µm), and 17 times the q = 16 ceiling. Leg D is in no training set — the
   networks saw only A, B and C — so this remains an out-of-training test, and
   the answer is that the residual parametrisation extrapolates to an unseen
   geometry considerably better than the absolute-state one did (a factor 2.3 on
   the single step, 14 on the composite) without getting close to the scheme.
   The p95 barely moved (142 mm against wave 1's 149 mm): the soft-track tail
   that dominates leg D is untouched.

# Verdict

**Chaining confirms the one-step story and sharpens its limit.** The residual
networks walk real particle paths four to thirteen times more accurately than
wave 1's at every chain length, the val chain still ranks seeds essentially
perfectly, and the leg-D composite — the thing wave 1 said was actively harmful
— is now within 13% of the single giant step instead of 6.8× worse.

But the error still compounds, and it compounds *faster* than wave 1 in the
early legs, because a chain quickly reaches the cross-magnet leg and that is the
one leg the redesign left largely alone: 1026 µm one-step against a 46 µm
ceiling. Over a full seven-leg path the residual network is 4.5 mm out where
wave 1 was 10.2 mm — better, but the same order of magnitude, and for the same
reason. **The next constraint is not the output parametrisation; it is whatever
limits the network on the 5.2 m magnet crossing.**
