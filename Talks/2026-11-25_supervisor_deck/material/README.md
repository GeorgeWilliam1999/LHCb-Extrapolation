# Material for the supervisor catch-up deck (2026-11-25)

Numbers and figures for the RK-PINN deck (Marcel Merk, Jochen Schultz). Everything was made on stbc-i2 on 2026-11-25. The work was read-only: nothing was trained, no farm job or keeper was touched, nothing was written into the run store, and the package defaults are unchanged. The scripts are in `_scripts/`. The large intermediate files (the test-split caches and the scored end states) stayed in `/data/bfys/gscriven/LHCb_Extrapolation_Project_deck_material/_work/` and are not committed.

## Conventions used in every file

- **Benchmark (reference)**: the sixth-order explicit Runge–Kutta method at a fixed step of 0.1 mm through the field map. It starts from the true MC state on the first plane (2648.2 mm) and is carried plane to plane to 7826.0 mm. Every "error" is a prediction minus this benchmark, unless the file says otherwise.
- **Endpoint radial error**: √(Δx² + Δy²) on the last plane (7826.0 mm), in µm. "Median" and "95 %" are taken over the tracks.
- **Maps**: *cubic* is the v8r1 magnet-up grid read through a tricubic B-spline (track set `5558be31abeb`). *Linear* is the same grid read trilinearly, which is what LHCb does (track sets `afc61a558211` and the older `12a8d35c3165`). The cubic map is the metric standard from 2026-11-22. Linear-map rows are marked "linear map".
- **Test split**: 13,447 tracks (cubic and linear sets: the same particles). The old linear set `12a8d35c3165` has 1,452 test tracks, which are a subset of the same 13,447 with identical benchmarks.
- **Momentum bands**: < 3, 3–8, 8–20, 20–50, > 50 GeV. A track is in a band when lower ≤ p < upper.
- **Exact scheme**: the Gauss–Legendre scheme with the stated steps × stages, solved to a stage residual below 1e-10 by Newton sweeps. It is what a network with a loss of zero would give, so it is the floor of that layout.
- **"Scored here"**: I carried the test split through the run's last-round weights with the package code of the run's own commit (`_scripts/eval_run.py`). Check: run `2382b5d314a5` reproduces its stored report exactly (3.5454 µm). Path network `25934ed22604` also reproduces its report: 196.9 µm alone, 0.1455 µm after 1 sweep, 0.1465 µm after 2.
- **Python ms per track**: the whole test split in one batch, one CPU thread on the shared login node stbc-i2 (load average about 18). Use these as indicative only. They are not the compiled timings of item 10.

## Files

| File | Shows | Map | Split / tracks | Runs (key, round) | Provisional |
|---|---|---|---|---|---|
| `polarity.txt` | the field-map file of each track set and of the fast-extrapolator experiment, with the evidence | both | n/a | n/a | no |
| `field_map_By_along_z.png/.csv` | By against z over the whole map, on the beam axis and on the line x = 1000, y = 500 mm. The crossing 2648.2–7826 mm is shaded | cubic reading (the linear reading looks the same at this scale) | n/a | n/a | no |
| `field_map_By_xy_slice.png/.csv` | By in x–y at z = 4711 mm (the plane nearest the on-axis peak at 4720 mm), with 3,000 test tracks overlaid at that plane. The colour scale is clipped at 0–1.6 T: inside the iron the map holds values up to about 5 T | linear reading | test, 3,000 tracks | n/a | no |
| `field_map_linear_against_cubic.png/.csv` | 4 grid cells along z (4500–4900 mm) on x = 1000, y = 500 mm. Top: By. Bottom: dBy/dz. The linear reading has a constant slope in each cell and jumps at every cell face; the cubic slope is continuous | both | n/a | n/a | no |
| `example_event_paths.png/.csv` | event 1369 (21 tracks): benchmark x(z) and y(z) across the crossing, coloured by p | cubic | test | n/a | no |
| `example_event_single_tracks.png/.csv` | 3 single positive tracks of 2.0, 8.0 and 39.9 GeV. Left: x minus the straight line of the start state. Right: x | cubic | test | n/a | no |
| `population.png`, `population_*.csv` | momentum spectrum, start slopes tx and ty, charge; counts per band | cubic set (the linear set has the same particles) | test, 13,447 | n/a | no |
| `gauss_legendre_accuracy.png/.csv` | endpoint median (solid or dashed) and 95 % (dotted) against steps, for the exact scheme with 2, 4, 8 and 16 stages and for RK4; right panel: the median per band at 64 steps | cubic | test, 13,447 | classical, no runs | no. **Medians are the stored numbers** of `what_are_the_classical_references_on_the_cubic_map`; **the 95 % and the per-band 95 % are new**, from its per-track cache |
| `best_per_type.csv` | item 6: the best run of each model type, see the table below | as marked | as marked | in the file | see the `state` and `note` columns |
| `all_test_scores.csv` | the stored `test_scores.csv` of the cubic experiment (39 runs + 2 earlier seed-1 runs), plus each run's state now, its rounds, its last validation median, and **my test score of every run that has ended but has no report** (`*_scored_here`) | cubic | test, 13,447 | all | runs still training have no test score |
| `by_momentum.png/.csv` | item 7: median (and 95 %) endpoint error per band for the best run of each type; per component: median \|Δ\| and signed median of x, y (µm), tx, ty. Grey lines: RK4 at 16, 32 and 64 steps and the exact scheme 64×2. Left panel cubic, right panel linear (grey line = linear exact 64×2) | both, separate panels | test | as in `best_per_type.csv` | no |
| `resolution_<model>.png`, `resolution_all_models.csv` | \|Δx\|, \|Δy\|, \|Δtx\|, \|Δty\| against p: the binned median with the 16–84 % band | as for the model | test | as in `best_per_type.csv` | no |
| `against_true_state.png/.csv` | item 8, see below | both | test | as in `best_per_type.csv` | no |
| `benchmark_against_true_state.png/.csv` | \|benchmark − true\| on the particle's own fibre-tracker plane against p, for x and tx: scatter, binned median, 16–84 % band, and a 1/p guide through the 10 GeV bin | cubic | test, 13,447 | n/a | no |
| `steps_and_stages.png/.csv` | item 9: endpoint median against steps (2 stages) and against stages (64 steps), per model class, with the exact scheme as the floor (dotted) | cubic where finished, linear otherwise (dashed); labelled | cubic: 13,447; linear: the 1,452 old test tracks (stored reports; their exact floor is computed on the same 1,452) | in the CSV | the cubic runs at 256×2 and at 64×4, 64×8, 64×16 (straight line) and at 128×8 and 256×16 (direct) are **still training**, so they are not in the figure |
| `timing_test_state.md/.csv` | item 10: the compiled timing test (fast extrapolator), see below | as marked | test, 13,447 | stage-2 run names | **all stage-2 network numbers are not final** |
| `open_runs.csv` | item 11: every run still training or queued | as marked | n/a | keys | the expected end is the latest possible end (at the cap of rounds); a run stops earlier when its validation error plateaus |

## Item 1: polarity

All three use **magnet up**, `field.v8r1.up.bin` (md5 9e49ddc4…, which is the hash the store records for both track sets). The simulation sample is `sim-20231017-vc-mu100` (magnet up). Details are in `polarity.txt`.

## Item 6: the best run of each type (test split)

| Type | Map | Run | Round | State | Median µm | 95 % µm | Exact (same layout) | Ratio | Python ms/track |
|---|---|---|---|---|---|---|---|---|---|
| direct states (summed end state) | cubic | 29510f525130 | 37 | plateaued | 7.64 | 93.7 | 0.0183 | 418 | 0.75 |
| straight line, per-track scale (self-scaled Broyden) | cubic | 713f38afa7a3 | 22 | plateaued | 0.712 | 8.80 | 0.0183 | 39 | 3.45 |
| straight line, global scale | cubic | 2382b5d314a5 | 22 | plateaued | 3.55 | 36.1 | 0.0183 | 194 | 0.70 |
| physics layer | cubic | cdd676e19320 | 22 | plateaued | 0.0182 | 0.144 | 0.0183 | 0.99 | 3.87 |
| supervised path network | linear (1,452) | 25934ed22604 | 40 | at cap | 196.9 | 1263 | 0.1465 | 1344 | 0.06 |
| … + 1 sweep | linear (1,452) | same | | | 0.1455 | 1.23 | 0.1465 | 0.99 | 0.68 |
| … + 2 sweeps | linear (1,452) | same | | | 0.1465 | 1.17 | 0.1465 | 1.00 | 1.24 |
| label-free path network | cubic | 3ccd44d61cd8 | 40 | at cap | 523 | 3098 | 0.0183 | 28600 | 0.53 |
| … + 1 sweep | cubic | same | | | 0.0321 | 1.14 | 0.0183 | 1.76 | 2.60 |
| … + 2 sweeps | cubic | same | | | 0.0183 | 0.134 | 0.0183 | 1.00 | 4.72 |
| whole-crossing network | linear | dd9acf59ac88 | 40 | at cap | 79.4 | 774 | n/a | n/a | 0.018 |
| learned gain on a frozen network | linear | c6a7d5d15caa | 25 | plateaued | 0.340 | 3.73 | 0.1455 | 2.33 | 0.63 |
| error-conditioned corrector | linear | 1eaf33293506 | 22 | plateaued | 0.145 | 1.36 | 0.1455 | 1.00 | 0.70 |
| direct states, predicted end state | linear | 4cc888dcc710 | 45 | plateaued | 671 | 5275 | 0.1455 | 4600 | 0.43 |
| analytic gain (no training) | linear (1,452) | evaluation | | | 0.147 (quoted from the Kalman experiment) | | 0.147 | 1.0 | |

Notes:

- **How the best run was chosen.** For each type, I took the lowest test median among its finished runs. The candidates are in `_work/candidates.csv`.
  - Direct states: five L-BFGS seeds on the cubic map, 7.64–10.1 µm. The self-scaled Broyden seeds of direct states are still training; their validation medians, about 1.7–2.1 µm (larger of x and y), suggest they will win.
  - Straight line, per-track scale: five L-BFGS seeds (2.40–3.51 µm) and five self-scaled Broyden seeds (0.71–1.0 µm, scored here).
- **Linear-map rows** were trained on the old 1,452-track linear set. I scored them on the 13,447 linear test tracks of `afc61a558211`.
  - The supervised path network is the exception: it needs stored exact states of its training split, which `afc61a558211` does not have, so it is scored on its own 1,452 tracks.
- **Linear exact scheme**: I computed it new on the 13,447 linear test tracks (`_scripts/exact_linear.py`, the same method as the cubic references; every cell converged to below 1e-10). At 64×2 it gives 0.1455 µm on the 13,447 tracks and 0.1465 µm on the old 1,452, which agrees with the stored 0.147.
- **Ratios near 1.0 to the exact scheme**: the physics layer, both path networks after sweeps, and the error-conditioned corrector are at the floor of 64×2.
- **"At cap"** means the run reached its cap of rounds without a validation plateau.

## Item 8: the three references

**The true state** is the MC truth (Geant4: field, multiple scattering, energy loss) that the simulation sample stores on the particle's own plane of the fibre tracker (SciFi). That plane lies within 60 mm of 7826 mm. Benchmarks and predictions end on 7826.0 mm, so I first carried them from 7826.0 mm to each particle's own plane with the benchmark integrator (`three_references.carry_to_own_planes`).

**The error magnitudes** in `against_true_state.csv` are, per band:

- for each of x, y (µm), tx and ty: the signed median, the **68 % half-width** = (p84 − p16)/2 of the signed difference, and the median of the absolute difference;
- the median radial distance √(Δx² + Δy²) in µm.

The figure shows the 68 % half-width of Δx and of Δtx.

All momenta, the 68 % half-width of Δx (µm) and the radial median (µm):

| Comparison | Δx half-width | Radial median | Δtx half-width |
|---|---|---|---|
| benchmark against true state (floor), cubic set | 2715 | 1861 | 8.7e-4 |
| physics layer against benchmark | 0.031 | 0.018 | 1.5e-8 |
| straight line, per-track scale, against benchmark | 0.70 | 0.71 | 2.6e-8 |
| straight line, global scale, against benchmark | 4.5 | 3.5 | 9.2e-8 |
| direct states against benchmark | 10.3 | 7.6 | 2.6e-7 |
| label-free path network (alone) against benchmark | 328 | 523 | 9.1e-6 |
| every model above against true state | 2713–2719 | 1858–1862 | 8.67e-4 |

What the numbers mean:

- **The floor is set by physics outside the field model.** Multiple scattering and energy loss move the true state by about 2.7 mm (68 % half-width in x) away from any field-only extrapolation. In 3–8 GeV it is 3.7 mm; below 3 GeV it is about 12 mm.
- **Every model except the un-swept path networks and the predicted end state matches the floor.** Its "prediction against true state" equals the floor to 3–4 significant figures.
- **The floor's p-dependence.** `benchmark_against_true_state.png` shows that it falls roughly as 1/p above about 8 GeV and more steeply below that.

### Item 8b: the LHCb adaptive Nyström as an extra reference

The LHCb adaptive Runge-Kutta-Nyström integrator (the copy of Rec's TrackSTEPExtrapolator, tolerance 5 µm, in the fast-extrapolator program `fx`, method `rec_step`) carried the same 13,447 test tracks from 2648.2 mm to 7826.0 mm. It reads the linear v8r1 magnet-up map, as LHCb does. It started from the start states of the set it is compared with. (The cubic and linear sets hold the same tracks, but their start states differ by up to 0.13 µm.) I then carried its end states to each particle's own plane with the benchmark integrator of that set, as above. Script: `_scripts/nystrom_three_references.py` (about 15 s on one core). Output: `nystrom_three_references.csv` (also the last-plane rows), `against_true_state_with_nystrom.png`, `benchmark_against_true_state_with_nystrom.png`.

All momenta, on the particle's own plane:

| Comparison | Δx signed median (µm) | Δx half-width (µm) | Δx median abs (µm) | Δy half-width (µm) | Radial median (µm) | Δtx half-width | Δty half-width |
|---|---|---|---|---|---|---|---|
| Nyström against the cubic benchmark | −0.19 | 41.9 | 24.2 | 1.76 | 25.9 | 1.3e-5 | 1.3e-6 |
| Nyström against the linear benchmark | −0.00 | 16.3 | 9.9 | 0.73 | 10.1 | 5.4e-6 | 5.8e-7 |
| Nyström against the true state | 24.5 | 2712 | 1222 | 1402 | 1848 | 8.7e-4 | 4.0e-4 |
| (cubic benchmark against the true state, the floor) | | 2715 | | | 1861 | 8.7e-4 | |
| (linear benchmark against the true state) | | 2708 | | | 1851 | 8.7e-4 | |

Radial median (µm) per band:

| Comparison | < 3 | 3–8 | 8–20 | 20–50 | > 50 |
|---|---|---|---|---|---|
| Nyström against the cubic benchmark | 83.7 | 41.7 | 17.8 | 7.3 | 3.1 |
| Nyström against the linear benchmark | 16.2 | 14.2 | 9.4 | 2.6 | 1.6 |
| Nyström against the true state | 9630 | 3052 | 883 | 373 | 167 |

What the numbers mean:

- **Linear against cubic map.** The Nyström reads the linear map. Against the cubic benchmark, its error (25.9 µm) is mostly the difference between the two maps, not the integrator. Against the benchmark of its own map it is 10.1 µm. These two numbers agree with the timing test (item 10).
- **Against the true state the Nyström matches the floor.** Its 68 % half-width in x (2712 µm) and its radial median (1848 µm) lie within 0.7 % of the two benchmarks' values. Its field-model error is about 100 times smaller than the effect of scattering and energy loss. The true-state rows are the same (to 0.1 %) whichever set's start states and carry I use.
- **The Nyström is a weaker reference than the cubic benchmark.** In the figure every cubic network except the un-swept label-free path network lies below the Nyström line against the cubic benchmark.

## Item 10: the compiled timing test (state at about 12:30 UTC)

Read `timing_test_state.md`. In short (µs per track, one core of stbc-i2, single-precision networks, test tracks):

| Method | Map | Median µm | 95 % µm | µs/track | Final? |
|---|---|---|---|---|---|
| HLT2 adaptive Nyström (LHCb reference) | linear | 10.1 | | 5.42 | yes |
| HLT2 adaptive Cash–Karp | linear | 22.2 | | 2.53 | yes |
| HLT1 parametrised step | linear | 55.6 | | 0.52 | yes |
| HLT2 Nyström as LHCb runs it, against the cubic benchmark | cubic | 25.9 | | 5.41 | yes |
| straight line + polynomial of degree 9 (no network) | cubic | 4.35 | 386 | 1.13 | yes |
| straight line + network 2×128 | cubic | 3.63 | 187 | 2.20 | **no**: scored at round 91, still training |
| straight line + network 2×64 | cubic | 4.50 | 279 | 0.52 | **no**: scored at round 179; plateaued at round 196 |
| straight line + network 3×64 | cubic | 4.80 | 227 | 0.91 | **no**: still training |
| straight line + network 2×128 | linear | 25.7 | 241 | 2.23 | **no**: still training |

- The 95 % column and the worst-band medians are in the CSV.
- The 11 integrator-plus-network runs (Nyström 4, 6, 8 or 12 steps + network) are training at 9–44 rounds. They have no test score yet.
- `cubic_straight_line_2x32` did not train (10.4 mm). Its second seed is training.

## Item 11: open runs

There are 14 open runs in the rkpinn store:

- 11 of the cubic headline experiment: direct states with self-scaled Broyden, seeds 1–5; the straight line at 256×2 and at 64×4, 64×8 and 64×16; direct states at 128×8 and at 256×16;
- 3 label-free path networks.

There are also 17 open fast-extrapolator stage-2 runs. Details are in `open_runs.csv`.

## What I could not produce

- **Cubic best runs for 6 of the types.** No cubic run of these types has finished, so their rows use the linear map: the whole-crossing network, the learned gain, the error-conditioned corrector, the predicted end state, the analytic gain, and the supervised path network.
- **The analytic gain is quoted, not re-scored.** It is an evaluation, not a run.
- **Sweep times for the path-network rows.** I took these from the stored reports' sweep timings, not from a new measurement.
