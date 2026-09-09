# C1_Fine_reference — the sixth-order reference and how far it can be trusted  (Block C, C1)

**What this is.** A reference integrator for the LHCb equation of motion that is
much finer than the one everything so far has been built on, and a measurement
of its own accuracy. Everything in `Raissi_disc_time_approach/` up to now has
been scored against fp64 RK4 at a 5 mm step. That engine is fine for labels at
the tens-of-micron level, but Block C needs to talk about steps of 50 µm and
about errors below a micron, and at that level "the reference" has to be a
number with a bar on it rather than a convention.

**What was added.** `../../_shared/reference.py` now carries

```python
rk6_rows(S0, z0, z1, step=0.1, field=None) -> (N, 5)
rk6_dense_rows(S0, z0, z1, sample_mm=10.0, step=0.1, field=None)
```

Butcher's seven-stage explicit method of order six, vectorised over rows exactly
like `rk4_rows`: fp64, per-row `(z0, z1)`, masked stepping, the last step of
each row shortened so the row lands exactly on `z1`, and `z1 < z0` integrating
backwards. The coefficients are copied from the van der Pol study's reference
machinery, `/data/bfys/gscriven/Van_Der_Pole/RK_Truth/rk6.py` (read-only there).
Nothing that existed before was changed.

---

## script → output

| script | what it does | output |
|---|---|---|
| [check_tableau.py](check_tableau.py) | C1.1. The five-part verification battery on the tableau **and** on `rk6_rows` itself | `results/tableau_checks.json` |
| [measure_convergence.py](measure_convergence.py) | C1.2 / C1.3. The step ladder, the smooth-field control, the closure, the RK4 comparison and the cost, on 200 momentum-stratified cross-magnet legs | `results/reference_convergence.csv`, `results/reference_convergence_per_leg.csv`, `results/reference_convergence_meta.json` |
| [plot.py](plot.py) | the four-panel figure, from the csv tables only | `figures/reference_convergence.png` |
| [analysis.ipynb](analysis.ipynb) | loads the tables and the figure; computes nothing | — |

```bash
PY=/data/bfys/gscriven/conda/envs/TE/bin/python
export PYTHONNOUSERSITE=1
cd C1_Fine_reference
$PY check_tableau.py
$PY measure_convergence.py     # about 40 min, one thread
$PY plot.py
```

---

## C1.1 The tableau battery

A copied tableau is worth nothing until it has been checked on the machine that
will use it. `results/tableau_checks.json`; all five pass.

**1. Provenance.** The seven coefficient arrays in `_shared/reference.py` are
compared with the arrays in `rk6.py`, imported directly from that file, element
by element. Exact equality, no tolerance. `C`, `A`, `B`, `ORDER` and `N_STAGES`
all identical.

**2. Identities.** Each row of `A` sums to its node `c` (max error
2.2e-16, one bit), `b` sums to one (error exactly 0.0), and `A` is strictly
lower triangular, so the method is explicit.

**3. Order on three problems with exact solutions.** `rk6.py`'s own test set,
re-run here with a generic stepper driven by the *same* coefficient arrays that
`rk6_rows` uses. A linear problem alone cannot catch every mistake and an
autonomous one cannot catch a wrong node vector, so the set includes both.

| problem | fitted order | points above roundoff |
|---|---|---|
| `dy/dt = -y` | **6.06** | 6 of 8 |
| `dy/dt = -y²` | **6.27** | 6 of 8 |
| `dy/dt = y cos t` | **5.75** | 5 of 8 |

These reproduce the numbers in `RK_Truth/README.md` (6.06, 6.27, 5.75) exactly,
on this node.

**4. Order of `rk6_rows` itself, on the LHCb equation of motion.** The three
problems above cannot be pushed through `rk6_rows`, which is hard-wired to the
LHCb ODE — so the ODE was kept and the *field* was replaced by a smooth analytic
stand-in with the same call signature (a gaussian blob peaking at 1.05 T near
z = 4700 mm). On a smooth field the scheme must show its order, and any mistake
in the masked stepping, the shortened last step or the direction handling would
cost an order.

| step [mm] | 640 | 320 | 160 | 80 |
|---|---|---|---|---|
| max abs error vs the 0.5 mm run [mm] | 1.66e-5 | 2.12e-7 | 2.93e-9 | 4.20e-11 |

**Fitted order 6.23.** Each halving divides the error by 70–78, against the
ideal 64. The ladder has to be this coarse: on a smooth field `rk6_rows` is
already at the fp64 floor (~2e-12 mm) by a 40 mm step, so a finer ladder would
be measuring rounding.

The real v8r1 map is **trilinear on a 100 mm grid**, i.e. continuous but with a
derivative that jumps at every cell face. A scheme of order six cannot show
order six through a kink, so this order check *has* to be done on a smooth
field — and the level at which the real map's convergence flattens is what C1.2
measures.

**5. Mechanics.** Four properties that are not about order, on a leg of
5177.8 mm — deliberately not a whole number of steps:

| | |
|---|---|
| forward then back, 1 mm step, smooth field | 1.8e-12 mm |
| a row alone vs the same row inside a mixed batch of five different legs | **0.0, exactly** |
| a leg shorter than one step, and a zero-length leg | finite; the zero-length leg returns the start state bit-for-bit |
| q/p passthrough | exact |

---

## C1.2 Convergence on the real map

**The legs.** 200 cross-magnet legs drawn by `../../_shared/prepare.magnet_leg_rows`
— the same selection `../C0_Magnet_tracks_dataset` builds on — 20 per momentum band
per direction, so the soft tracks that bend hardest are not swamped by the stiff
ones. Spans 5159 to 6537 mm, median 5174 mm. **The field is v8r1.up**, because
the sample is a MagUp sample: see
[`../C0_Magnet_tracks_dataset/README.md`](../C0_Magnet_tracks_dataset/README.md). All
800 screened candidates stay inside the map on that polarity.

`figures/reference_convergence.png`, `results/reference_convergence.csv`.

### The step sequence

Endpoint position, worst of x and y, over the 200 legs. Left: what one halving
moves the answer. Right: the distance from the finest run on the ladder.

| step h [mm] | \|S(h) − S(h/2)\| median | p95 | | \|S(h) − S(0.05)\| median | p95 | max |
|---|---|---|---|---|---|---|
| 0.8 | **9.4e-5 µm** | 6.4e-4 | | 1.00e-4 µm | 6.8e-4 | 1.1e-3 |
| 0.4 | **1.2e-5 µm** | 1.5e-4 | | 1.0e-5 µm | 1.4e-4 | 2.5e-4 |
| 0.2 | **7.8e-6 µm** | 5.0e-5 | | 7.0e-6 µm | 3.5e-5 | 5.9e-5 |
| 0.1 | **2.1e-6 µm** | 1.6e-5 | | 2.1e-6 µm | 1.6e-5 | 2.4e-5 |

Successive ratios **8.0, 1.5, 3.8** against the 64 an order-six method would
give. So the answer is *not* converging at order six on this map — as C1.1
predicted it could not.

### Where it flattens, and at what level

Two things are happening, and the smooth-field control separates them.

1. **The trilinear grid costs about two orders of magnitude.** At h = 0.8 mm the
   real map moves the endpoint by 9.4e-5 µm per halving while the *same code on
   a smooth field* moves it by 9.9e-7 µm — a factor 95. The map is C0: its
   derivative jumps at every one of the ~52 cell faces a crossing passes, and a
   sixth-order scheme integrating across a kink behaves like a low-order one.
   The first ratio, 8.0 = 2³, is what an order-three method would give.
2. **Below about 1e-5 µm nothing is left but arithmetic.** The smooth control is
   the tell: its 0.8 → 0.2 mm difference is 9.9e-7 µm but its 0.2 → 0.05 mm
   difference is *larger*, 4.0e-6 µm. On a smooth field this scheme is exact to
   fp64 at any of these steps, so that rise is pure rounding accumulating over
   more steps — 103,500 of them at 0.05 mm — and it sets a floor of a **few
   times 1e-6 µm** that the real map's series reaches at h = 0.1 mm.

**The flattening is therefore at h ≈ 0.2 mm and at a level of about
1e-5 µm (10 pm) in the median, 3e-5 µm at p95.** Finer than that the series is
inside the arithmetic floor and the numbers stop meaning "truncation error".

By momentum band, per halving (median, µm) — the 1/p bend is visible throughout:

| band | h = 0.8 | 0.4 | 0.2 | 0.1 |
|---|---|---|---|---|
| 1–2 GeV | 4.3e-4 | 6.1e-5 | 4.5e-5 | 1.4e-5 |
| 2–5 GeV | 2.1e-4 | 2.7e-5 | 2.1e-5 | 5.6e-6 |
| 5–10 GeV | 9.6e-5 | 1.4e-5 | 1.0e-5 | 2.6e-6 |
| 10–25 GeV | 5.2e-5 | 8.0e-6 | 4.7e-6 | 1.3e-6 |
| 25–200 GeV | 2.2e-5 | 3.0e-6 | 2.0e-6 | 5.9e-7 |

### Forward-then-back closure at 0.1 mm

Integrate each leg to the far plane and back, and compare with the start state.
This is not an accuracy test — both halves carry the same truncation error — but
it is a floor: the reference cannot be trusted below the level at which it fails
to undo itself.

| | median | p95 | worst |
|---|---|---|---|
| all 200 legs | **3.7e-6 µm** | 2.9e-5 | 4.4e-5 |
| 1–2 GeV | 1.7e-5 | 4.0e-5 | 4.4e-5 |
| 25–200 GeV | 1.1e-6 | 1.6e-6 | 2.0e-6 |
| UT → SciFi | 3.3e-6 | 1.8e-5 | 2.6e-5 |
| SciFi → UT | 4.0e-6 | 3.0e-5 | 4.4e-5 |

Which agrees with the ladder: both say a few times 1e-6 µm in the median and a
few times 1e-5 µm at the tail.

### The incumbent 5 mm RK4 engine

Every label in the training set and every experiment before Block C was built
with fp64 RK4 at 5 mm. Against RK6 at 0.05 mm on these same legs:

| | median | p95 | worst | 1–2 GeV median | 25–200 GeV median |
|---|---|---|---|---|---|
| **RK4 at 5 mm** | **0.0127 µm** | 0.077 | 0.291 | 0.057 | 0.0032 |
| **RK4 at 1 mm** | **0.00082 µm** | 0.0049 | 0.0097 | 0.0042 | 0.00021 |

So the old engine was already good to about **13 nm** in the median and
**0.3 µm** at worst across the magnet — four orders of magnitude below the
q = 8 exact-scheme ceiling of 46 µm and six below the networks' millimetre
errors. **RK4 at 5 mm was never what was limiting anything**, and no earlier
result is disturbed by this. What the fine reference buys is headroom: Block C
wants to talk about 50 µm steps and sub-micron residuals, and at 0.29 µm worst
case a 5 mm RK4 label is only a factor 160 below the ceiling it is being used to
measure.

### The other polarity

The same two rungs on v8r1.down, on the same 200 start states, to show the
verdict is about the map's grid and not its sign:

| | v8r1.up | v8r1.down |
|---|---|---|
| \|S(0.1) − S(0.05)\| median | 2.1e-6 µm | 2.8e-6 µm |
| closure at 0.1 mm, median | 3.7e-6 µm | 3.6e-6 µm |
| closure at 0.1 mm, **p95** | 2.9e-5 µm | **1.9e-3 µm** |
| closure at 0.1 mm, **worst** | 4.4e-5 µm | **5.1e-3 µm** |

The medians are the same — the step verdict transfers. The tails are not: on the
wrong polarity the soft backward legs bend out to the edge of the map, where the
loader clamps the field, and the round trip stops closing. That is the same
effect, seen from a third direction, that
[`../C0_Magnet_tracks_dataset`](../C0_Magnet_tracks_dataset) measures head-on.

---

## C1.3 Cost

One thread, fp64, the field call is essentially the whole cost, and it amortises
over the batch — so the per-track figure falls with batch size until the map's
working set stops fitting in cache.

| batch | wall for a 20 mm step at 0.1 mm | extrapolated per track, 5174 mm crossing |
|---|---|---|
| 200 | 0.675 s | 0.87 s |
| 1,000 | 1.007 s | 0.26 s |
| 5,000 | 2.587 s | 0.134 s |
| **20,000** | 9.276 s | **0.120 s** |

Measured directly on the 200-leg batch: **1.04 s per track** at 0.1 mm. Peak
resident memory for the whole script, 200 legs and every ladder rung, was
**146 MB**; the cost is time, not memory. `rk6_rows` holds one `(7, N, 5)` stage
buffer, so 20,000 rows is 5.6 MB of stages and the batch is limited only by the
caller's own state array — `../C0_Magnet_tracks_dataset` runs 12,000 legs with the
whole 10 mm-sampled path in memory, about 320 MB.

**Practical figure: about 0.12 s of one core per track for a full crossing at
0.1 mm, in batches of a few thousand.** The whole 200-leg ladder, the smooth
control, the closure, the RK4 comparison and the cost measurement together took
2229 s.

---

## C1.4 Verdict

**Use a 0.1 mm step.** The sequence does not force it — the endpoint has
effectively stopped moving by 0.2 mm — but 0.1 mm is where the real map's series
finally reaches the arithmetic floor set by the smooth-field control, it costs
0.12 s per track in a batch, and it makes the reference's accuracy a
non-question rather than a thing to argue about. That is the step
`../C0_Magnet_tracks_dataset` is built with.

**Its demonstrated accuracy at 0.1 mm.** Halving the step again to 0.05 mm moves
the endpoint by 2.1e-6 µm in the median, 1.6e-5 µm at p95 and 2.4e-5 µm at
worst; the round trip closes to 3.7e-6 µm median and 4.4e-5 µm worst.

**The number to quote as the reference's own floor: 5 × 10⁻⁵ µm (0.05 nm).**
That is the worst case over 200 momentum-stratified legs of both the
step-halving and the closure measurements, rounded up. It is a *floor*, not an
error bar on any single number: it says nothing built on this reference can
claim accuracy better than 0.05 nm, and it sits

* 1/2400 of the fp32 rounding step on a metre-scale coordinate (0.12 µm),
* 1/250 of the 0.0127 µm the 5 mm RK4 engine differs from it,
* 1/900,000 of the q = 8 exact-scheme ceiling on leg B (46 µm), and
* 1/20,000,000 of the best network so far on that leg (~1 mm).

In other words the reference is five to seven orders of magnitude below anything
Block C will be asked to resolve, which is the whole point of building it.
