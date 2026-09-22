# A4_Magnet_up_field — training the one-step network where no labels exist

The physics loss of the discrete-time method never looks at a reference
trajectory. It asks only that the network's own stage states satisfy the
Gauss–Legendre equations of the LHCb equation of motion, and those equations
need nothing but the field map. If that is really what the loss is doing, the
network should train on a field polarity for which no labelled sample has ever
been produced.

MagUp is that polarity. This experiment trains the identical 4×50 one-step
network on the magnet-up map, with the identical protocol, and asks two
questions:

1. does the label-free network reach the same floor on MagUp as on MagDown?
2. is that floor the same distance above the exact scheme's own ceiling on
   each polarity?

Nothing about the network, the optimiser, the leg, the number of stages or the
training population changes between the two. The only difference is the field
the loss is built on — and, for scoring only, the field the reference
trajectories were integrated with.

The up-field references in `results/frozen_leg_up.npz` are used **only** to
score. The `--mode physics` runs never see them; that is the whole point. The
three `--mode data` runs are the supervised control: the same network on the
same states fitted directly to those references, which is what the label-free
runs have to match.

## Script → output

| script | what it does | outputs |
|---|---|---|
| [check_field.py](check_field.py) | A4.1. Identity (path, md5) of both maps; the torch twin's parity gate on the up map, which is what the physics loss takes its gradients through; and the shape of the two fields along the frozen leg's axis and along 300 real cross-magnet legs | `results/field_up_parity.json`, `figures/field_up_vs_down.png` |
| [build_dataset.py](build_dataset.py) | A4.2. `frozen_leg_dataset(q=8, field=…)` for both polarities, and the comparison of counts, scales and fiducial removals | `results/frozen_leg_{up,down}.npz` (+ `_meta.json`), `results/dataset_meta.json` |
| [check_population.py](check_population.py) | A4.2. Goes behind the state arrays and checks that both polarities are built from the same event rows, by reproducing the (field-independent) selection | `results/population_check.json` |
| [check_loss_uses_the_field.py](check_loss_uses_the_field.py) | A4.3 pre-submit gate. One L-BFGS restart on the same states and seed with the loss built on each polarity in turn; the two losses must differ | `results/loss_field_probe.json`, `results/probe_{up,down}.json` |
| [condor/jobs.txt](condor/jobs.txt) | the 13 farm runs: physics seeds 0–9, data twin seeds 0–2, q = 8, 4×50, `--field up` | `results/up_*.json`, `results/up_*_history.csv` |
| [condor/jobs_extended.txt](condor/jobs_extended.txt) | the three runs re-submitted with `--outer-cap 300` after hitting the 150-restart cap (see *Problems*) | the same files, resumed |
| [ceiling_up.py](ceiling_up.py) | A4.4. The exact Gauss–Legendre scheme solved with a root-finder, no network, on both polarities and on two populations | `results/ceiling_up.csv`, `results/ceiling_test_states.csv`, `results/ceiling_summary.json` |
| [aggregate.py](aggregate.py) | A4.4. Collects the 13 runs and builds the comparison table | `results/summary.csv`, `results/up_vs_down.csv` |
| [matched_subset.py](matched_subset.py) | A4.4 control. Re-scores the up networks on only the 2018 tracks MagDown also kept | `results/matched_subset.json` |
| [plot.py](plot.py) | A4.4. The three-panel figure | `figures/magnet_up_results.png`, `results/test_errors.npz` |
| [analysis.ipynb](analysis.ipynb) | loads all of the above; recomputes nothing | — |

Order: `check_field` → `build_dataset` → `check_population` →
`check_loss_uses_the_field` → submit → `ceiling_up` → `aggregate` →
`matched_subset` → `plot`.

## A4.1 The field

| | MagUp | MagDown |
|---|---|---|
| file | `…/FieldMap/v8r1/cdf/field.v8r1.up.bin` | `…/field.v8r1.down.bin` |
| md5 | `9e49ddc4313b589f273540e7e0bb513b` | `af284c6954d2273c637a5e766b82b58e` |
| torch twin vs numpy loader, 200k points | **1.332e-15 T** (pass, tol 1e-12) | 1.332e-15 T (pass) |

The two maps are **exactly** one another's negative. On 200,000 random in-map
points, `max |B_up + B_down|` is **0.0 in all three components**, and the
magnitude `|B|` differs by 0.0 everywhere. So By flips sign — and so do Bx and
Bz — while the magnitude profile is bit-for-bit identical, and the map bounds
are the same. This is a stronger statement than the figure alone can make, and
it means every difference reported below comes from the *sign* of the bending,
never from a different field shape.

## A4.2 The dataset

Both polarities are built from the **same event rows**: the selection (leg B,
forward, own plane within 60 mm of z0, then the seeded 2000-state cap) uses no
field at all, and `results/population_check.json` confirms 2000 / 2082 / 2033
identical rows. The field enters twice afterwards: the exact RK4 transport of
each state to z0 (which moves it by a median 0.5 µm and at worst 68 µm), and
the reference trajectory across the leg.

| | train | val | test |
|---|---|---|---|
| states, MagUp | **2000** | **2082** | **2033** |
| states, MagDown | 1979 | 2062 | 2018 |
| removed by the fiducial cut, MagUp | **0** | **0** | **0** |
| removed by the fiducial cut, MagDown | 21 | 20 | 15 |

The down-field build reproduces `../../Block_0_first_pass/S2b_One_step_network_v2/results/frozen_leg_data.npz`
bitwise, so this is the baseline builder and not a variant of it.

**The fiducial asymmetry is real and it is the one genuine polarity effect in
the experiment.** The cut removes states whose reference trajectory leaves the
field map — soft tracks bending out past |x| = 4 m, where the ODE is not
defined and the physics loss is being asked to satisfy equations built on a
clamped field. On MagDown 1% of states do that. On MagUp **none** do. The map is
symmetric, so the asymmetry is in the tracks: the sample's charge and angular
distribution is not symmetric in x, and the polarity that bends these particular
soft tracks outward is MagDown. The same asymmetry shows up in the ceiling
measurement below.

## A4.3 The runs

Before submitting, `check_loss_uses_the_field.py` established that `--field up`
really reaches the loss and is not silently ignored — one L-BFGS restart from
the same seed on the same up-field states gives **2.6215e-03** with the loss
built on MagUp and **8.0317e-03** with it built on MagDown (a factor 3.06), and
the resulting networks score 8.4 mm and 1039 mm against the up-field references.

Cluster **5781156**, 13 jobs, all completed with empty `.err` files.
Clusters **5781163** and **5781164** re-ran three of them from their checkpoints
against a higher restart cap (see *Problems*). **13 of 13 converged**, meaning
each stalled and then a fresh optimiser re-stalled within two restarts with the
endpoint medians unchanged. Restarts 114–154 (physics), 94–134 (data twin).

## A4.4 The comparison

Full table in `results/up_vs_down.csv`. Endpoint error is the median over the
test split of max(|dx|, |dy|) against the fp64 RK4 reference, in µm.

| | MagUp | MagDown |
|---|---|---|
| **network, physics loss (label-free)** | **229 µm** [171–296], 10 seeds, 10/10 converged | **221 µm** [163–244], 10 seeds (A1, 50×4) |
| network, data loss (supervised twin) | 204 µm [187–230], 3 seeds | 192 µm [162–208], 3 seeds (v2) |
| exact-scheme ceiling, on the test states | 22.1 µm | 22.5 µm |
| exact-scheme ceiling, 32 stratified legs | 33.2 µm (32/32 solved) | 32.0 µm (31/32 solved) |
| straight line (ignore the magnet) | 525,184 µm | 520,444 µm |
| **network / its own ceiling** | **10.4×** [7.7–13.4] | **9.8×** [7.2–10.8] |

Notes on the comparators, which are not interchangeable:

* The headline MagDown row is `../A2_Network_size_and_seed_study` at width 50,
  depth 4 — the same architecture, the same ten-seed protocol and the same
  single-thread arithmetic as these runs. `../../Block_0_first_pass/S2b_One_step_network_v2`'s three-seed
  183 µm [177–235] is also in the table but was run with four BLAS threads,
  which moves the optimiser's path in the fourth digit.
* The ceiling is quoted two ways because the two populations give different
  numbers and only one of them is like-for-like. The **test-states** row is the
  scheme solved on the very states the networks are scored on, and is the
  comparator used for the ratio. The MagDown value it produces, 22.544094616
  µm, reproduces `../A1_Stage_count_sweep/results/scheme_ceiling_same_population_q08.json`
  to the last digit — the cross-check that the solver copied into
  `ceiling_up.py` is character-for-character the shared one. The **stratified**
  row is the `../../Block_0_first_pass/S1_Simple_first_pass` population, 32 leg-B legs drawn stratified
  in momentum on their own start planes, which over-weights the hardest tracks.
  (The 29 µm figure published in that experiment is on neither population
  measured here: it was taken on the v1 self-generated training sample, while
  everything in this folder stands on the v2 official-sample one.)
* **Control for the fiducial asymmetry.** The MagUp test split contains 15
  tracks MagDown discards, and they are the hardest ones, so the two medians are
  not taken over the same tracks. Re-scoring the up networks on only the common
  2018 (`results/matched_subset.json`) moves the physics median from 229.4 µm to
  **226.6 µm** and the twin from 204.2 µm to 202.7 µm — about 1%. The extra
  tracks are not the explanation for anything.

## A4.5 Verdict

**Yes. The label-free network reaches the same floor on the magnet-up field as
on magnet-down, and it sits the same distance above the up-field ceiling as it
does above the down-field one.**

* 229 µm [171–296] over ten converged MagUp seeds against 221 µm [163–244] over
  ten MagDown seeds at the same architecture. The two ten-seed ranges overlap
  almost completely; the difference between their medians, 4%, is far inside a
  seed-to-seed spread that is itself a factor 1.7 wide on MagUp and 1.5 on
  MagDown. Panel 1 of the figure shows the two error distributions lying on top
  of one another.
* On MagUp the label-free network also matches its own supervised twin —
  229 µm against 204 µm, again inside the seed spread — exactly as it did on
  MagDown (221 µm against 192 µm). The loss that never sees a label is doing
  the same job as the loss that does, on a polarity where the labels do not
  exist.
* The ceiling is the same on both polarities to 2% (22.1 vs 22.5 µm on the test
  states), so the ratio is the same: **10.4×** on MagUp against **9.8×** on
  MagDown. Whatever holds the network above the exact scheme is a property of
  the network and the optimiser, not of the field it was trained on.
* Both are three orders of magnitude below the straight line (525 mm), so the
  network has learned the bending and not merely the geometry — and panel 3
  shows it has learned it with the correct sign, the stage states tracking a
  trajectory that curves the opposite way from MagDown's.

**Asymmetries found.** Three, all pointing the same way and all traceable to
the track sample rather than to the map:

1. the fiducial cut removes 21/20/15 states on MagDown and **none** on MagUp;
2. the exact scheme's root-finder solved 32/32 stratified legs on MagUp and
   31/32 on MagDown, and its worst state is 38 mm on MagUp against 258 mm on
   MagDown;
3. those are the same tracks in both cases — soft, large-|x| ones that MagDown
   bends out of the map and MagUp bends back into it.

None of these affect the medians (see the matched-subset control), but they mean
the *tails* are not symmetric, and any future statement about p95 or worst-case
behaviour must say which polarity it is about. Convergence and restart counts
show no asymmetry: 114–154 restarts on MagUp against A1's comparable MagDown
range, and the momentum dependence in panel 2 is the same curve for both.

## Problems, and the fix needed in `_shared`

**`--outer-cap` counts both phases, so runs that stall late never get to
confirm.** The cap limits the stall phase and the confirmation pass together.
The physics runs here stall at 110–150 restarts, so with the default cap of 150
three of thirteen (`seed 4`, `seed 9`, and `seed 1` for a related reason) came
back `converged = false` while being about to converge. Re-submitting the
identical line with `--outer-cap 300` resumed each from its `.pt` and confirmed
it within a few restarts; all three then reported `converged = true` with
endpoint medians within 1% of what they had. This is the same finding A1
reported independently, and the default in `_shared/train.py` has since been
changed to 400. **No further fix is needed**, but the cap is still shared
between the two phases, and the cleaner fix would be to give the confirmation
pass its own small budget (it needs at most a handful of restarts) so that a
late-stalling run cannot be denied its confirmation by the stall phase's cap.

Two smaller notes, neither of which needed a workaround:

* `../../Block_0_first_pass/S1_Simple_first_pass/exact_scheme.py` cannot be imported for an up-field
  measurement: its `deriv` comes from `S0_Baseline_data_exploration/reference_card.py`,
  which hard-wires the MagDown map and takes no field argument. `solve_leg` is
  therefore copied into `ceiling_up.py` with the single edit of passing the
  field through. The copy is validated by reproducing A1's down-field ceiling to
  the last digit. If `_shared` ever adopts the exact-scheme solver, it should
  take the field as an argument the way `deriv` and `rk4_rows` already do.
* `_shared/evaluate.py::predict` returns a numpy array, but the docstring's
  phrasing ("network outputs, no grad") reads as though it might return a
  tensor. Not a bug; noted only because it cost a first run of two scripts here.
