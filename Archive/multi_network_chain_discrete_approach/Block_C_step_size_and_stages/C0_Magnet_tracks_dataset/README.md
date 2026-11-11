# C0_Magnet_tracks_dataset — cross-magnet steps of every length  (Block C, C0)

**What this is.** One dataset of magnet-to-magnet track states: real particles
from the official simulated sample, their paths across the LHCb magnet computed
by the fine sixth-order reference of [`../C1_Fine_reference`](../C1_Fine_reference),
and steps of every length along those paths — from 50 µm to the whole 5.2 m
crossing — in six equally-sized strata.

**Why.** Every one-step network so far has been trained on one step length, or
on a mixture dominated by one. `../../Block_A_technique_works/A3a_General_leg_network`'s verdict was that a
single output scale set by the 5 m cross-magnet step cannot also resolve a
70 mm one: the network was 32× *worse than ignoring the magnet entirely* on the
short legs. That is a statement about a training set as much as about a
network, and it cannot be tested without a set in which every step length is
equally represented on the same trajectories. This is that set.

---

## The thing that had to be settled first: the sample is MagUp

Building C0 required propagating a particle from its last UT plane to its first
SciFi plane and asking whether the answer is where the particle actually was.
That question had never been asked of leg B before — the training set's own
gates check the short legs (G2) and the engine's self-consistency (G1, G3), not
this. The answer was not the one the repository assumed.

`check_polarity.py` runs it on all **15,257** selected particles. For each one,
the forward row starts at its last UT plane and the backward row of the *same*
particle starts at that particle's **actual** first SciFi state, so the truth is
available for free.

| forward cross-magnet leg vs the particle's real SciFi state | v8r1.**up** | v8r1.**down** |
|---|---|---|
| median miss, all 15,257 particles | **1.73 mm** | 883 mm |
| median miss, 1–2 GeV | 14.8 mm | 3973 mm |
| median miss, 2–5 GeV | 5.13 mm | 1856 mm |
| median miss, 5–10 GeV | 1.62 mm | 892 mm |
| median miss, 10–25 GeV | 0.65 mm | 418 mm |
| median miss, 25–200 GeV | 0.28 mm | 169 mm |
| **fraction of particles whose bend has the right sign** | **100.0 %** | **0.0 %** |
| legs whose path leaves the field map (4000-leg subset) | **0** | 80 (57 % of the 1–2 GeV band) |

The two maps are exact negatives of one another — `max |B_up + B_down| = 0.0`
on 200,000 random in-map points, re-confirming `../../Block_A_technique_works/A4_Magnet_up_field`'s A4.1 — so
the only thing the polarity changes is the sign of the bend. With MagDown the
bend comes out backwards for **every particle in the sample** and the endpoint
misses by roughly twice the deflection; with MagUp the residual collapses onto a
clean 1/p line, which is the multiple-scattering-and-energy-loss residual that
the labels exclude by design and that the training set's own G2 gate reports for
the short legs.

The sample's conditions tag is `sim-20231017-vc-mu100`.

**Consequences.**

1. The v2 training set (`Official_xdigi/training_v2/train_official_v2.npz`) was
   labelled with the MagDown map. Its `Y` column is the field-only propagation
   of the right start state through the **wrong polarity**.
2. Experiments that only ever compared a network with that same engine
   (`../../Block_0_first_pass/S2b_One_step_network_v2`, `../../Block_A_technique_works/A3a_General_leg_network`, `../../Block_A_technique_works/A3b_Chained_legs`, the
   residual wave) are internally consistent, and their conclusions about how
   well a network solves the scheme stand unchanged. What does not stand is any
   claim that those labels are where the simulated particle went.
3. `../../Block_A_technique_works/A4_Magnet_up_field` is framed as training "on a field polarity for which no
   labelled sample has ever been produced". It is in fact the polarity of the
   sample — and its own A4.2 table already carries the tell that nobody read:
   the fiducial cut removes **0** states on MagUp and 21 / 20 / 15 on MagDown.
4. **Block C therefore builds on MagUp**, and this file is the reason. Whether
   the rest of the line follows is George's call, not this folder's.

`figures/field_polarity.png`, `results/polarity_check.json`.

---

## script → output

| script | what it does | output |
|---|---|---|
| [check_polarity.py](check_polarity.py) | the question above, on all 15,257 particles and both maps | `results/polarity_check.json`, `figures/field_polarity.png` |
| [build_dataset.py](build_dataset.py) | the whole build; a thin driver over `../../_shared/prepare.magnet_tracks_dataset` | `results/magnet_tracks_v3.npz`, `results/magnet_tracks_v3_meta.json`, `results/dense_states.npz`, `results/dataset_meta.json` |
| [check_path_consistency.py](check_path_consistency.py) | the one assumption C0.3 rests on: that a stored dense state can be integrated onwards without re-basing | `results/path_consistency.json` |
| [plot_schematic.py](plot_schematic.py) | the detector side view, the four leg classes and 20 real tracks | `figures/lhcb_legs_schematic.png`, `results/detector_planes.csv`, `results/schematic_meta.json` |
| [plot_population.py](plot_population.py) | momentum, pseudorapidity, step length per stratum, direction and split | `figures/dataset_population.png` |
| [analysis.ipynb](analysis.ipynb) | loads all of the above and displays it; computes nothing | — |

Order: `check_polarity` → `build_dataset` → `check_path_consistency` →
`plot_schematic` → `plot_population`.

```bash
PY=/data/bfys/gscriven/conda/envs/TE/bin/python
export PYTHONNOUSERSITE=1
cd C0_Magnet_tracks_dataset
$PY check_polarity.py
$PY build_dataset.py --particles 6000     # about 40 min, one thread
$PY check_path_consistency.py
$PY plot_schematic.py
$PY plot_population.py
```

`results/*.npz` are gitignored; the scripts, the seeds and the meta json
regenerate them exactly.

---

## C0.1 The selection, and what each cut costs

`../../_shared/prepare.magnet_leg_rows` — the same function `../C1_Fine_reference`
draws its 200 legs from, so the two studies are on one population by
construction.

| cut | rows in | removed | rows out | particles left |
|---|---|---|---|---|
| leg-B rows in the v2 training set (both directions) | — | — | 41,398 | 21,932 |
| the pre-magnet plane is a **UT** plane, not a VELO one | 41,398 | 2,476 | 38,922 | 20,655 |
| both directions survive for the particle | 38,922 | 2,388 | 36,534 | 18,267 |
| 2 < eta < 5 | 36,534 | 1,240 | 35,294 | 17,647 |
| 1 < p < 200 GeV | 35,294 | 0 | 35,294 | 17,647 |
| non-electron | 35,294 | 4,780 | 30,514 | 15,257 |
| both directions still a pair after the cuts | 30,514 | 0 | 30,514 | 15,257 |
| *compute cap: 6,000 particles drawn at random, seed 20260718* | 30,514 | 18,514 | 12,000 | 6,000 |
| fiducial: the whole RK6 path stays inside the field map | 12,000 | **0** | **12,000** | **6,000** |

**The fiducial cut removes nothing.** Not one of the 12,000 RK6 paths leaves the
field map. On the MagDown map 2 % of the same legs do, rising to 57 % in the
1–2 GeV band — which is the polarity finding again, arriving from a third
direction.

Three of those are worth a sentence.

* **The UT-plane requirement.** The training set builds leg B as "the last
  tracker plane the particle crossed before z = 2800 mm → the first one after
  z = 7000 mm". For a particle with no UT hit that first plane is its last VELO
  module, which makes a 7 m leg through the whole of the UT instead of the
  magnet-to-magnet step this dataset is about. The two populations are cleanly
  separated in z — the VELO ends at 751 mm, the UT starts at 2307 mm, and
  nothing lies between — so a boundary at 1500 mm splits them with no row near
  it. The 2,476 rows removed are the VELO-start ones.
* **The compute cap is a budget, not a physics cut.** The RK6 path at 0.1 mm
  costs about a tenth of a second per leg, and the fiducial requirement cannot
  be applied until the path exists. 15,257 particles were available and 6,000
  were integrated; the cap is seeded and recorded as its own cascade row so it
  is never mistaken for a selection.
* **The fiducial cut is applied per leg, not per particle.** A particle whose
  backward leg leaves the map still contributes its forward one. Dropping the
  whole particle would remove the soft tracks preferentially — exactly the ones
  the extrapolator finds hardest — for a reason that is about the map's edge
  rather than about the physics.

`p_min` removes nothing because 1 < p < 200 GeV is already the training set's
own domain cut; it is applied and reported anyway so the cascade is complete.

---

## C0.2 The paths

Every surviving leg is integrated with `rk6_rows` at **0.1 mm** from its start
plane to its far plane — forward legs from the UT plane to the SciFi plane,
backward legs from the particle's own SciFi state back to the UT plane — and the
state is stored **every 10 mm** along the way, plus the exact endpoint
(`rk6_dense_rows`). `results/dense_states.npz` holds them flat: `EVT`, `MCKEY`,
`DIRECTION`, `LEG_INDEX`, `SPLIT`, `P`, `Z`, `S` (N, 5), plus per-leg arrays
`leg_*`.

10 mm is exactly 100 steps of 0.1 mm, so a stored state is a step boundary of
the march and can be integrated onwards without re-basing. That is the one
assumption C0.3 rests on and `check_path_consistency.py` tests it directly.

---

## C0.3 / C0.4 What came out

**60,000 rows: six strata × 10,000, each 6,000 train / 2,000 val / 2,000 test.
Every stratum hit its target exactly.** 12,000 legs from 6,000 particles, split
3,600 / 1,200 / 1,200 particles.

| stratum | \|dz\| | train (dz>0 / dz<0) | val | test | total |
|---|---|---|---|---|---|
| 0.05–0.2 mm | log-uniform | 6,000 (2,997 / 3,003) | 2,000 | 2,000 | 10,000 |
| 0.5–2 mm | log-uniform | 6,000 (2,976 / 3,024) | 2,000 | 2,000 | 10,000 |
| 5–20 mm | log-uniform | 6,000 (2,964 / 3,036) | 2,000 | 2,000 | 10,000 |
| 50–200 mm | log-uniform | 6,000 (3,002 / 2,998) | 2,000 | 2,000 | 10,000 |
| 500–2,000 mm | log-uniform | 6,000 (3,040 / 2,960) | 2,000 | 2,000 | 10,000 |
| full crossing | the leg itself | 6,000 (3,014 / 2,986) | 2,000 | 2,000 | 10,000 |

The direction balance is 50/50 to within 1 % in every cell, including the full
crossing, where it is the track's own direction and each particle contributes
both of its legs.

**What the population allowed.** Everything, at the target. The binding stratum
was always going to be the full crossing, which has exactly one row per leg and
so cannot exceed 2 × (particles in the split): 3,600 train particles give 7,200
available and 6,000 were drawn. The interior strata draw from 6.26 million dense
states and were never close to a limit. Had the cap been set at 5,000 particles
instead of 6,000 the full-crossing train cell would have come out at 6,000
exactly with no headroom, which is why 6,000 was used.

**Clipping.** The rule is: use the drawn direction if the step fits inside
[UT plane, SciFi plane]; if not, flip it; clip \|dz\| only if neither direction
fits. Of the 50,000 interior-stratum rows, **2,390 (4.8 %) were flipped** — all
of them in the 500–2,000 mm stratum, where a start state near one end of a
5.2 m window cannot always take a 2 m step in the drawn direction — and
**none were clipped**, so every row's \|dz\| is exactly the value drawn from its
stratum.

Momentum, all 60,000 rows: 1–2 GeV 327 · 2–5 GeV 20,692 · 5–10 GeV 17,545 ·
10–25 GeV 15,445 · 25–200 GeV 5,991. The 1–2 GeV band is thin because the leg-B
population itself is (188 legs before the cap): a 1–2 GeV particle rarely leaves
a hit in both the UT and the SciFi.

**Cost.** 2,942 s in total on one thread — 1,607 s of it the RK6 paths (0.134 s
per leg, matching `../C1_Fine_reference`'s C1.3 figure of 0.12–0.13 s per track at
a few thousand rows), the rest the strata and the file writes.
`results/dense_states.npz` is 464 MB and `results/magnet_tracks_v3.npz` 5.6 MB.

### The assumption the strata rest on

C0.3 draws a start state from the stored dense states and integrates onwards
from it, because that state is already on the RK6 path. `check_path_consistency.py`
tests exactly that on 240 rows, 40 per stratum
(`results/path_consistency.json`):

| | median | p95 | worst |
|---|---|---|---|
| restart from the stored state vs marching the whole way from the leg's start plane | **8.9e-9 µm** | 4.1e-8 | 8.6e-8 |
| the stored dense state vs RK6 from the leg's start plane to that z0 | 4.5e-9 µm | 3.3e-8 | 7.1e-8 |
| each row integrated back from its own end state | 3.6e-10 µm | 6.1e-6 | 3.1e-5 |

The first line is the claim, and it holds three orders of magnitude below the
reference's own floor of 5e-5 µm. q/p passes through bit-exactly on all 60,000
rows.

---

## C0.5 The figures

**`figures/lhcb_legs_schematic.png`** — the side view. Every z in it is
measured: the sensor planes are the z clusters of the harvested MCHit states
(`Official_xdigi/results/states.npz`, the file the v2 set is built from), cut
wherever a gap exceeds 15 mm, and written out to `results/detector_planes.csv`;
each detector's transverse extent is the 99.9th percentile of |x| of its own
hits; the magnet shading is where |B| on the beam line exceeds 5 % of its peak,
read from the v8r1 map, with the peak marked. Over that, twenty real particles —
their own hit states joined, and the RK6 reference path across the magnet —
coloured by truth momentum, and the four leg classes as labelled arrows.

**`figures/dataset_population.png`** — momentum, pseudorapidity, |dz| per
stratum, start plane, direction split and rows per stratum per split.

---

## Reading the file

```python
import use_shared                                  # noqa: F401
from _shared.prepare import load_magnet_tracks

d = load_magnet_tracks("results/magnet_tracks_v3.npz", split="train",
                       stratum="500-2000 mm", direction=-1)
d["X"]    # (N, 7) fp64: x, y, tx, ty, qop, z0, dz
d["Y"]    # (N, 5) fp64: the RK6 end state at z0 + dz, qop passed through
```

| array | meaning |
|---|---|
| `X` | (x, y, tx, ty, qop, z0, dz), fp64 |
| `Y` | the state at z0 + dz, fp64 |
| `STRATUM` | 0–5, indexing `stratum_names` |
| `DIRECTION` | sign(dz) |
| `LEG_DIR` | +1 if the parent leg runs UT → SciFi, −1 back |
| `P`, `ETA`, `PID`, `EVT`, `MCKEY` | the particle's truth |
| `SPLIT` | 0 train / 1 val / 2 test, **by particle**, seed 20260718 |
| `LEG_INDEX` | the row in `dense_states.npz`'s `leg_*` arrays |

**The split is 60/20/20, not the v2 set's 80/10/10.** A particle in this set's
train split may well be in the v2 test split. The two must not be intersected
casually; the v2 label travels with the legs as `SPLIT_V2` inside
`magnet_leg_rows`' output for anyone who needs to.
