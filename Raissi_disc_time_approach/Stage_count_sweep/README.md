# Stage_count_sweep — how many stages the magnet crossing needs

**Verdict: q = 8 is the working point.** Below it the one-step network is limited by the
Runge-Kutta scheme itself and reproduces the scheme's own error almost exactly; at q = 8
the scheme's error drops an order of magnitude below the network's floor and stops being
the limiter; going on to q = 16 buys nothing.

The experiment varies one thing and nothing else: `q`, the number of Gauss-Legendre stages
the network predicts across the frozen magnet crossing (leg B, z = 2648.2 → 7826.0 mm),
over q ∈ {2, 4, 8, 16}, both training modes, three seeds each — 24 runs. The population,
the fiducial cut, the network shape (4 × 50, fp64), the L-BFGS protocol and the split seed
are the shared package's, unchanged, so a difference between two points on these curves is
a difference in the number of stages.

## Results

Test-split median endpoint error, in micrometres, median over the three seeds with the
seed range in brackets. "Ceiling" is the same q-stage scheme solved directly by a
root-finder **on these same 2018 test states** (see the note below on why the number in
`../Simple_first_pass` is not the one to compare against).

| q | physics loss | data twin | ceiling (exact scheme) | ceiling p95 |
|---:|---|---|---:|---:|
| 2 | **5202** [5199 – 5218] | 174 [168 – 223] | 5198 | 37728 |
| 4 | **5371** [5352 – 5394] | 181 [157 – 181] | 5401 | 54057 |
| 8 | **181** [165 – 240] | 176 [162 – 204] | 23 | 1610 |
| 16 | **187** [186 – 280] | 170 [147 – 176] | 20 | 325 |

Straight line through the same leg (no magnet at all): 520 444 µm, at every q.
Interior stage states, same measure: physics 16182 / 5360 / 247 / 306 µm, data twin
278 / 258 / 250 / 262 µm, exact scheme 15557 / 1138 / 10.5 / 5.8 µm.
All 24 runs converged (genuine stall plus a confirmation pass with a fresh optimiser).

### What the table says

**At q = 2 and q = 4 the physics loss is scheme-limited, and it hits its ceiling.** 5202 µm
against a ceiling of 5198 µm, and 5371 µm against 5401 µm — within 0.1% and 0.6%. The
network is not failing to learn; it is faithfully learning equations that are themselves
5 mm wrong on this leg. The data twin at the same q is at 174 and 181 µm, thirty times
better, because its labels are the true RK4 states and it never sees the scheme.

**The scheme stops being the limiter between q = 4 and q = 8.** The ceiling falls from
5401 µm to 23 µm across that step and crosses below the network's own floor. For q ≥ 8 the
limiting factor is the 4 × 50 network and the optimiser.

**q = 8 and q = 16 agree within seed spread.** 181 µm [165 – 240] against 187 µm
[186 – 280]: medians 3% apart, ranges overlapping over most of their length. Doubling the
stages costs 68 network outputs per sample instead of 36 and returns nothing at the
endpoint, even though it halves the scheme's error again (23 → 20 µm).

**The floor belongs to the network, not the discretisation.** The data twin is flat at
170 – 181 µm at *every* q, q = 2 included. That is the same number the physics loss reaches
at q = 8 and 16. Lowering it is a question about width, depth or protocol.

**The q = 8 runs reproduce the verified baseline.** Physics 165 – 240 µm against
`One_step_network_v2`'s 177 – 235 µm, data twin 162 – 204 µm against 162 – 208 µm, on a
dataset checked bitwise identical to v2's. (Exact agreement is not expected: these run on
one BLAS thread and v2 ran on four, which changes the last bits of every reduction — the
point documented in `../_shared/smoke_tests.py`.)

### Three things that were not expected

1. **q = 4 is no better than q = 2** — the ceiling is 5401 µm against 5198 µm, slightly
   *worse*. A q-stage Gauss-Legendre scheme is formally of order 2q, but that is asymptotic
   in the step length, and one step of 5178 mm straight through the magnet is nowhere near
   that regime. The extra stages buy nothing until there are enough of them to resolve the
   field variation along the leg; between q = 4 and q = 8 they suddenly do. The root-finder
   converged to residual < 1e-8 on 99.9% of legs at every q, so these are genuine solutions
   of the discrete equations — the equations are simply far from the ODE.

2. **q = 16 is not the hard case.** The concern going in was that 68 outputs per sample
   would train badly. It trains *faster* than q = 8: 113 – 115 restarts and ~15 min against
   154 restarts and ~20 min. The hardest optimisation in the grid is q = 4, whose loss
   grinds down through two decades and is still moving at restart 150. The one wobble at
   q = 16 is seed 2, at 280 µm against 186 – 187 µm for the other two — a seed outlier, not
   a trend, and it is inside the reported range.

3. **The pre-stated ceiling numbers (~14 mm at q = 2, ~4.3 mm at q = 4) do not describe
   this population.** They come from `../Simple_first_pass`, which measured leg B on 32
   legs drawn *stratified in momentum* — deliberately over-weighting soft, hard-bending
   tracks — each on its own start plane rather than rebased to the common one. On the 2018
   frozen-leg test states the ceiling is 5198 µm and 5401 µm instead. Both panels of the
   figure carry both curves so the difference is visible rather than argued about. The
   qualitative claim the criterion was built on survives intact: the physics network tracks
   the ceiling at low q and leaves it at q = 8.

## Figures

- [figures/error_vs_stages.png](figures/error_vs_stages.png) — median endpoint error against
  q (log y, left) and the same for the interior stage states (right), with both losses,
  both ceiling measurements, and the straight-line reference. Bars are the seed spread.
- [figures/convergence.png](figures/convergence.png) — training loss per L-BFGS restart for
  all 24 runs, one panel per q; the tick marks where the confirmation pass took over.

## script → output

| script | what it does | writes |
|---|---|---|
| [build_datasets.py](build_datasets.py) | `frozen_leg_dataset(q)` for each q, with two assertions: the split counts must be 1979 / 2062 / 2018 for every q (the fiducial cut is a property of the leg and the field, not of q), and at q = 8 every array the v2 baseline holds must be reproduced bitwise | `results/frozen_leg_q{02,04,08,16}.npz` + `_meta.json`, `results/dataset_check_q<qq>.json` |
| [measure_scheme_ceiling.py](measure_scheme_ceiling.py) | solves the exact q-stage scheme on **this experiment's own test states**, so the ceiling is like-for-like with what the networks are scored on. `solve_leg` is imported from `../Simple_first_pass/exact_scheme.py`, not copied | `results/scheme_ceiling_same_population_q<qq>.json` |
| [condor/jobs.txt](condor/jobs.txt) + [condor/jobs.sub](condor/jobs.sub) | the 24 farm jobs: {physics, data} × q ∈ {2,4,8,16} × seeds {0,1,2}, each `../_shared/train.py` at 4 × 50 with the converged protocol and confirmation | `results/q<qq>_<mode>_s<seed>.{json,pt,_history.csv}` |
| [condor/jobs_extended.txt](condor/jobs_extended.txt) + [condor/jobs_extended.sub](condor/jobs_extended.sub) | the eight physics runs that had not converged, resumed from their checkpoints with `--outer-cap 400` | the same files, continued |
| [aggregate.py](aggregate.py) | reads the run jsons and both ceiling measurements; never re-scores a model, and never pools an unconverged run into a median | `results/summary.csv`, `results/error_vs_stages.csv` |
| [plot.py](plot.py) | the two figures, from the CSVs and the history files only | `figures/*.png` |
| [analysis.ipynb](analysis.ipynb) | loads the CSVs and the figures and narrates the result | — |

`use_shared.py` is the standard path helper copied from `../_shared/`.

Datasets (`.npz`), model weights (`.pt`) and `condor/logs/` are gitignored and regenerate
from the scripts, the seeds and the committed metadata.

## Provenance

- Repo `LHCb_Extrapolation_Project`, branch `main`.
- Shared package `../_shared/` (physics, model, losses, optimiser protocol, farm wrapper),
  unchanged — nothing outside this folder was edited.
- Training population: the official TestFileDB sample via
  `Data_generation_exploration/Official_xdigi/` (`train_official_v2.npz`), the same one
  `One_step_network_v2` used; 1979 train / 2062 val / 2018 test states after the fiducial
  cut, identical for all four q.
- Field: v8r1 MagDown, vendored in `../_shared/field_v8r1.py`.
- Farm: Nikhef HTCondor, **cluster 5781154** (the 24 runs, submitted 2026-09-05 16:16,
  all complete by 16:37) and **cluster 5781162** (the eight resumed runs, 16:37).
- Every run single-threaded fp64 on one core, per `../_shared/condor/wrapper.sh`.

## A note on `../_shared/train.py`

The default `--outer-cap 150` is too tight for the physics loss under this protocol, and
it is worth changing. In `One_step_network_v2` the 150 was the cap on the *stall* phase
alone and the confirmation pass ran afterwards as a separate script; in the shared driver
the one cap covers both phases. A v2-style physics run stalls at 147 – 152 restarts and then
needs two or three more to confirm, so it hits the cap and is recorded as `converged =
false` when it is in fact about to converge. Six of the 24 runs here did exactly that —
every physics seed at q = 4 and q = 8 — and all six converged within four further restarts
once resumed with `--outer-cap 400`.

**Suggested fix (not applied — `_shared` was not touched):** raise the default `--outer-cap`
to 300, or make the cap apply per phase rather than to the total. Either would have made
this grid converge on the first submission. Nothing about the results changes: a larger cap
cannot affect a run that stopped on the stall criterion below it, so the eight resumed runs
are what the whole grid would have produced under the larger cap from the start.
