# The straight-line-residual redesign  (A3a, wave 3)

**Question.** Wave 1 trained one network to emit the **absolute** stage states
and endpoint of any leg, with one population-wide output scale. It worked on
the leg it was scaled for and failed on the others: on the test split, leg A
404 µm, leg B 2600 µm, leg C 300 µm at 4×100 — against a straight line of 10.9,
444075 and 6.6 µm. On the two short legs the network was **fifty times worse
than ignoring the magnet entirely**.

The hypothesis in wave 1's verdict was specific: *a 7 µm correction cannot be
resolved out of a ~400 mm output range*. The network's last layer emits numbers
in units of `out_scale` — the standard deviation of the straight-line-propagated
training states, 602 mm in x and 341 mm in y — so on a 70 mm plane-to-plane leg
the whole physics it has to get right is one part in 10⁸ of the number it is
writing down.

This wave tests that hypothesis by changing exactly one thing: **what the
network is asked to output.**

## The redesign

For a start state S = (x, y, tx, ty, q/p) at z₀, a step dz, the Gauss-Legendre
nodes z_j = z₀ + c_j·dz (j = 1…q) and the endpoint z₀ + dz,

    output_j = straight_j  +  scale ⊙ net_j
    straight_j = ( x + tx·(z_j − z₀),  y + ty·(z_j − z₀),  tx,  ty )

i.e. the network predicts the **deviation from a straight line** — the magnet's
whole contribution and nothing else — and `net_j` is the raw network output,
now a pure number instead of a length.

### The scale, and why it may not use a label

The network is never shown a label, so the scale may only be built from the
inputs. To first order in the field the equation of motion gives

    |Δtx| ≈ κ·|q/p|·I_B                 (dimensionless)
    |Δx | ≈ κ·|q/p|·I_B·|dz|/2          (mm — that slope ramped over the step)

with κ = 10⁻³ and q/p = 0.299792458·q/p[GeV] the Allen convention of
`../_shared/reference.py`, and

    I_B = ∫_{z₀}^{z₀+dz} |B(x_s(z), y_s(z), z)| dz    [T mm]

the field integral taken **along the straight line** x_s(z) = x + tx·(z − z₀),
y_s(z) = y + ty·(z − z₀), evaluated with a 16-point midpoint rule. Every symbol
on the right is an input. Floors of 10⁻³ mm and 10⁻⁶ keep the scale away from
zero for a straight track in a field-free region.

That expression is a *first-order estimate of the answer*, which is exactly what
a scale should be: it does not have to be right, it has to be right to within a
factor of a few on every leg type, so that what the network learns is the O(1)
correction to it.

### The check that had to pass first

`prepare_residual.py` forms the ratio the network will actually have to emit —
(RK4 reference − straight line) / scale — on the test split and histograms it by
leg type (`figures/residual_scale_check.png`,
`results/residual_scale_check.json`). This is the one place a label appears, and
it is a diagnostic, never an input to the scale.

Two node profiles were measured: **flat**, one scale for all q+1 outputs of a
sample, and **poly**, output j scaled by c_j² (positions) and c_j (slopes). The
rule was fixed before looking: take flat unless its endpoint medians fall
outside 0.1–10 on some leg while poly's do not.

| endpoint deviation / scale, test split | leg A | leg B | leg C |
|---|---|---|---|
| median, flat profile | **1.09** | **1.03** | **0.95** |
| median, poly profile | 1.09 | 1.03 | 0.95 |

Both pass, so the rule selects **flat**, and it is the profile stored in the
dataset. One number is worth pausing on: the same label-free expression lands
within 10% of the truth on a 341 mm backward vertex fetch, a 5.2 m cross-magnet
crossing and a 70 mm sensor hop. Whatever else this wave shows, the O(1)
requirement the redesign was built around is met by construction.

## script → output

| script | what it does | output |
|---|---|---|
| [residual_model.py](residual_model.py) | the wrapper: straight line + scale ⊙ network, the label-free scale, and the twin's loss renormalised onto that scale | — |
| [prepare_residual.py](prepare_residual.py) | wave 1's dataset plus the scale arrays, and the O(1) check | `results/general_legs_residual.npz`, `results/residual_scale_check.json`, `figures/residual_scale_check.png` |
| [train_residual.py](train_residual.py) | the shared trainer's own loop with the residual model installed; `--init-check` verifies the algebra before any farm time is spent | `results/<tag>.json`, `<tag>.pt`, `<tag>_history.csv`, `results/residual_init_check.json` |
| [resubmit_residual.py](resubmit_residual.py) | sends the runs that did not confirm back to continue from their checkpoints | `condor/jobs_residual_round<N>.txt` / `.sub` |
| [aggregate_residual.py](aggregate_residual.py) | re-scores both arms' checkpoints per leg type and momentum band on the same states, with the ceiling columns | `results/residual_summary.csv`, `results/by_leg_residual.csv`, `results/stage_errors_residual.csv` |
| [plot_residual.py](plot_residual.py) | the three A3a figures with the residual series added | `figures/*_residual.png` |

Nothing wave 1 owns was edited. The residual runs are tagged `residual_*`, which
the wave-1 `aggregate.py`, `plot.py`, `chain.py` and `resubmit_unconverged.py`
globs (`w*_s*.json`) do not match, so the two analyses cannot collide.

## What was NOT changed

The comparison only means something if everything else is wave 1's, so:

- the same dataset population — `general_legs_residual.npz` is
  `general_legs.npz` byte for byte, plus three scale arrays per split;
- the same physics loss. It is computed from the **reconstructed absolute
  states**, so it needed only the wrapper's forward and not one line of
  `_shared/model.py`;
- the same optimiser protocol: fp64, full-batch L-BFGS, `max_iter` 200, strong
  Wolfe, history 120, stall = two consecutive restarts improving by less than
  1%, a checkpoint every restart, the confirmation pass, `--outer-cap 400`.
  `train_residual.py` does not re-implement that loop — it calls
  `../_shared/train.py`'s `main` and swaps two module-level names;
- the same scoring, the same splits, the same momentum bands, the same ceilings.

The one deliberate change beyond the output parametrisation is the **data
twin's normalisation**: `_shared.model.data_loss` divides by `out_scale`, which
is the quantity being replaced, so `residual_data_loss` divides by the
per-sample residual scale instead. Without that the twin would be trained on an
objective the physics arm is not.

## The initialisation check

`train_residual.py --init-check` (`results/residual_init_check.json`), at 4×100
seed 0 on the 7935 training states:

| | |
|---|---|
| straight-line term vs an independent numpy construction | 4.5 × 10⁻¹³ mm |
| **last layer zeroed → output − straight line** | **0.0 mm, exactly** |
| raw network output at the shared trainer's own init | max 0.34, median 0.057 |
| so: offset from the straight line at init | 0.34 scale max, 0.057 median — 0.51 µm at the median endpoint |
| physics loss / max gradient at init | 4.06 × 10⁻² / 8.2 × 10⁻³, all finite |
| residual data loss / max gradient at init | 2.19 × 10⁻¹ / 7.2 × 10⁻³, all finite |

The initialisation is the shared trainer's, unchanged — the last layer is *not*
zeroed for the real runs, because a zero last layer makes every earlier layer's
gradient exactly zero on the first step. Zeroing it is used only as the algebra
check above. With the standard init the model therefore starts a fraction of one
residual scale from the straight line, which is the point: wave 1's network
started tens of millimetres away from it.

An untrained residual network already scores **15.5 µm** whole-split on the test
set against a 16.1 µm straight line. Wave 1's converged 4×100 physics networks
scored ~300 µm.

## The farm

```bash
cd General_leg_network
mkdir -p condor/logs results
condor_submit condor/jobs_residual.sub
```

`condor/jobs_residual.sub` is `../_shared/condor/template.sub` with the
executable pointed at [condor/wrapper_residual.sh](condor/wrapper_residual.sh)
— a copy of the shared wrapper that starts `train_residual.py` instead of
`_shared/train.py`; the shared wrapper is hard-wired to the latter and was not
edited. `condor/jobs_residual.txt` holds 26 argument lines and **no comment
lines** (`queue args from` does not skip `#`).

Cluster **5781469**, 26 jobs (4×100 and 4×50, physics seeds 0–9, data twin seeds
0–2), submitted 2026-09-05 21:14.

All 26 confirmed. The harness issue A1 recorded — the shared stall criterion
firing while the endpoint medians are still moving, so the confirmation pass
re-stalls at once and records `converged = false` — hit only the three 4×50 data
twins here, and [resubmit_residual.py](resubmit_residual.py) continued them from
their checkpoints in three rounds
([condor/jobs_residual_round2.txt](condor/jobs_residual_round2.txt) …`round4`).
That is a much lighter touch than wave 1, where 20 of the first 26 ended that way.

| arm | seeds | restarts | wall per run |
|---|---|---|---|
| residual 4×50 physics | 10, all converged | 48–64 | 13–28 min |
| residual 4×50 twin | 3, all converged | 100–112 | 24–27 min |
| residual 4×100 physics | 10, all converged | 40–59 | 26–48 min |
| residual 4×100 twin | 3, all converged | 95–110 | 33–80 min |
| wave 1 4×100 physics | 10, all converged | 84–121 | 43–79 min |
| wave 1 4×200 physics | 10, 9 converged | 101–131 | 123–259 min |

The residual runs also **stall sooner** — 40–64 restarts against wave 1's 84–121
at the same width — at a physics loss an order of magnitude lower (9 × 10⁻⁷
against 1.8 × 10⁻⁵ at 4×100 seed 0). The optimisation problem got easier, not
just the answer better.

# Results

*Numbers corrected 2026-09-06 to match `results/*.csv` (see
`Mini_paper/README.md`): the wave-1 4×200 physics cells in the whole-split and
by-leg tables, the residual 4×100 twin column in the by-leg and by-momentum-band
tables, the derived twin factors in reading 4 and in the tails section, and the
per-band improvement range quoted under the by-momentum-band table.*

`aggregate_residual.py` re-scored **65** checkpoints — the 26 residual runs and
the 39 wave-1 runs on disk (4×50, 4×100 and A2's overnight 4×200 wave) — on the
same test states with the same scorer, so every number below is like for like.
Re-scoring is not why the wave-1 4×200 column moved in the 2026-09-06
correction: every median here pools **converged seeds only**, per the protocol,
and the earlier values pooled all ten, including the unconverged
`w200_physics_s5` (all-seed medians 234 / 1781 / 213 µm against the
converged-only 229 / 1740 / 213 µm).

## Whole test split, median over seeds

| arm | 4×50 physics | 4×50 twin | 4×100 physics | 4×100 twin | 4×200 physics | 4×200 twin |
|---|---|---|---|---|---|---|
| wave 1 (absolute states) | 721 | 637 | 397 | 321 | 273 | 237 |
| **residual** | **2.12** | **0.06** | **2.64** | **0.04** | — | — |

µm, medians over converged seeds. The 4×200 physics cell is 273 µm over the
nine converged seeds; pooling all ten, including the unconverged
`w200_physics_s5`, gives 274 µm, which is what this table said before the
2026-09-06 correction. The straight line on this split is 16.1 µm. Wave 1 never
beat it at any width; the residual arm beats it by a factor 6 with the physics loss and 400
with the twin, and a 4×50 residual network beats a 4×200 wave-1 network by a
factor 130.

## By leg type, all momenta (test split, median over seeds [range])

| leg | wave 1 4×100 physics | wave 1 4×200 physics | **residual 4×100 physics** | **residual 4×50 physics** | residual 4×100 twin | straight line | ceiling (this population) |
|---|---|---|---|---|---|---|---|
| A vertex fetch (341 mm, backward) | 404 [286–538] | 229 [201–277] | **4.4** [2–6] | **4.1** [3–6] | 0.058 [0.047–0.063] | 10.9 | 0.0013 |
| B cross-magnet (5.2 m) | 2600 [2106–3152] | 1740 [1632–2154] | **1026** [878–1209] | **1136** [945–1627] | 512 [486–566] | 444075 | 46.4 |
| C plane-to-plane (70 mm) | 300 [252–335] | 213 [165–235] | **1.6** [1–2] | **1.2** [1–1] | 0.012 [0.0118–0.0122] | 6.6 | 5.8 × 10⁻⁷ |

All µm. Read across:

1. **The two short legs are transformed.** Leg A goes 404 → 4.4 µm, a factor
   **92**; leg C goes 300 → 1.6 µm, a factor **190**. Both now sit *below* the
   straight line (2.5× on A, 4× on C) where wave 1 sat thirty to fifty times
   above it. This is the hypothesis's own prediction, and it is confirmed with
   room to spare.
2. **The cross-magnet leg improves by a factor 2.5**, 2600 → 1026 µm. Real, but
   an order of magnitude short of the same transformation. Leg B is the one leg
   where wave 1's output scale was roughly right, so it had least to gain — the
   redesign fixed the conditioning problem, and what is left on leg B is not a
   conditioning problem.
3. **Width has stopped mattering** on legs A and C: 4×50 and 4×100 are within
   20% of each other, and on both legs the 4×50 residual network is the better
   of the two. Wave 1's verdict read the 4×50 → 4×100 → 4×200 trend as "width is
   the binding constraint"; at 4×200 wave 1 still had not reached 200 µm on leg
   C. It was never width. It was the parametrisation.
4. **The data twin is now extraordinary on the short legs** — 0.058 µm on A and
   0.012 µm on C, i.e. **6,600×** and **2.0 × 10⁴** better than wave 1's twin —
   and still twice the physics loss on leg B (512 against 1026 µm). The factors
   are (wave 1 4×100 twin) ÷ (residual 4×100 twin), each the median over the
   three converged twin seeds on the test split at all momenta, from
   `results/by_leg_residual.csv`: 384.774 / 0.0583 = 6,605 on leg A and
   235.242 / 0.0119 = 19,792 on leg C. The physics-vs-twin
   gap that opened in wave 1 when the leg geometry became an input has *widened*
   in relative terms on the easy legs and narrowed on the hard one.
5. **Distance to the ceiling, which is the honest scorecard.** Against the exact
   q = 8 scheme measured on these same states:

   | leg | residual 4×100 physics | ceiling | factor above |
   |---|---|---|---|
   | A | 4.4 µm | 0.0013 µm | 3,400 |
   | B | 1026 µm | 46.4 µm | 22 |
   | C | 1.6 µm | 5.8 × 10⁻⁷ µm | 2.8 × 10⁶ |

   Wave 1's factors were 310,000 / 56 / 5 × 10⁸. So the gap closed by two orders
   of magnitude on A and C and by a factor 2.5 on B — but the network is still
   nowhere near the scheme it is solving on any leg, and on the short legs the
   scheme is so nearly exact that no network is going to reach it.

## By momentum band (test split, median over seeds, physics loss)

| leg | band | wave 1 4×100 | **residual 4×100** | residual 4×100 twin | straight | ceiling |
|---|---|---|---|---|---|---|
| A | 1–5 GeV | 702 | **15.2** | 0.152 | 31.1 | 0.0028 |
| A | 5–20 GeV | 221 | **3.1** | 0.034 | 9.8 | 0.00051 |
| A | 20–200 GeV | 183 | **2.3** | 0.031 | 6.0 | 0.00091 |
| B | 1–5 GeV | 6582 | **2641** | 2575 | 1197128 | 239 |
| B | 5–20 GeV | 2120 | **775** | 316 | 355193 | 23.8 |
| B | 20–200 GeV | 1370 | **628** | 110 | 104911 | 6.7 |
| C | 1–5 GeV | 463 | **2.3** | 0.044 | 20.7 | 6.8 × 10⁻⁶ |
| C | 5–20 GeV | 181 | **1.2** | 0.005 | 4.0 | 1.8 × 10⁻⁶ |
| C | 20–200 GeV | 134 | **1.2** | 0.003 | 0.74 | 4.8 × 10⁻⁷ |

All µm. The improvement is uniform in momentum on legs A and C — a factor 46 to
200 in every band — so it is not an artefact of the momentum mix. The one cell
where the residual network is still *worse* than a straight line is
**C at 20–200 GeV** (1.2 µm against 0.74 µm): a stiff 70 mm hop is so nearly
straight that there is essentially nothing to predict, and the network's own
noise floor is above the thing it is correcting. Every other cell it wins.

## Where the error lives inside the step

Median position error at each of the eight Gauss nodes and the endpoint, test
split, 4×100 physics, median over seeds
(`results/stage_errors_residual.csv`, `figures/stage_errors_residual.png`):

| | node 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | endpoint |
|---|---|---|---|---|---|---|---|---|---|
| wave 1, leg A | 365 | 340 | 353 | 320 | 312 | 408 | 349 | 381 | 404 |
| **residual, leg A** | **0.06** | 0.10 | 0.43 | 1.01 | 1.74 | 2.81 | 3.64 | 4.34 | **4.42** |
| wave 1, leg C | 289 | 240 | 249 | 270 | 260 | 271 | 268 | 278 | 300 |
| **residual, leg C** | **0.03** | 0.05 | 0.29 | 0.65 | 0.88 | 1.12 | 1.34 | 1.52 | **1.57** |
| wave 1, leg B | 987 | 905 | 1054 | 1436 | 1907 | 2326 | 2536 | 2519 | 2600 |
| **residual, leg B** | **108** | 143 | 277 | 456 | 616 | 812 | 924 | 988 | **1026** |

This is the clearest single piece of evidence that the diagnosis was right.
Wave 1's error on legs A and C was **flat across the step** — the same few
hundred micrometres at the first node as at the last — which is the signature of
a scale error, an offset the network cannot resolve, not of physics being got
wrong. The residual arm's error instead **grows monotonically from the first
node to the endpoint on every leg**, by a factor 50–70 on A and C: the flat
offset is gone, and what is left is the honest accumulation of an
under-resolved bend. Leg B kept its monotonic growth from wave 1 and simply
scaled down by 2.5.

## What did not improve: the tails

Median is not the whole story. The 95th percentile of the endpoint error, all
momenta, 4×100:

| leg | wave 1 physics | residual physics | wave 1 twin | residual twin |
|---|---|---|---|---|
| A | 30693 | 23640 | 10445 | 20128 |
| B | 41542 | 27080 | 18395 | 17425 |
| C | 2862 | **292** | 2193 | **32** |

µm. Leg C's tail improves by a factor 10–70 with the median. Legs A and B do
not: the residual twin's p95 on leg A is twice wave 1's, so a small population
of states got worse while the bulk got 6,600× better. Those are the soft tracks
that bend hardest — the residual scale is a *first-order* estimate, and where the
true deviation is many times that estimate the network is back to writing down a
large number. Fixing the median did not fix the tail, and the tail is what an
extrapolator with a bounded time budget actually has to survive.

# Verdict

**The hypothesis held, decisively, on the legs it was about, and not at all on
the leg that matters most.**

Wave 1 said: *a 7 µm correction cannot be resolved out of a ~400 mm output
range*. Making the network predict the deviation from a straight line, on a
per-sample scale built from nothing but the inputs, improved the two short legs
by factors of **92 (leg A)** and **190 (leg C)**, moved both from thirty to
fifty times *worse* than ignoring the magnet to two to four times *better* than
it, flattened the width dependence that wave 1 read as its binding constraint,
and turned a flat scale error along the step into a monotonic accumulation. On
the whole test split one label-free network now gives 2.1 µm with the physics
loss and 0.04 µm with the twin, against 16.1 µm for a straight line and 274 µm
for the best wave-1 network at any width. Nothing else changed: same states,
same loss, same optimiser, same q, same scoring.

But the cross-magnet leg improved only by a factor 2.5, to 1026 µm against a
46 µm ceiling, and it is the cross-magnet leg the extrapolator exists for. Two
things follow. First, wave 1's diagnosis was **right about the symptom and
incomplete about the cause**: output conditioning was crippling legs A and C and
was never leg B's problem. Second, whatever limits leg B — 22× above the exact
scheme, with the error still growing monotonically along the step — is a
different failure, and it is now the only one left to find. It is not width
(4×50 and 4×100 are within 10% of each other), and it is not the output scale.

The remaining honest caveats: the p95 on legs A and B did not improve with the
median, and the 20–200 GeV plane-to-plane cell is still marginally worse than a
straight line. Chaining is in
[../Chained_legs/README_residual.md](../Chained_legs/README_residual.md).

## Figures

- `figures/residual_scale_check.png` — the O(1) check on both node profiles
- `figures/error_by_leg_and_momentum_residual.png` — the table above, both arms
- `figures/stage_errors_residual.png` — the error along the step, both arms
- `figures/frozen_vs_general_residual.png` — leg B: frozen leg, wave 1, residual
