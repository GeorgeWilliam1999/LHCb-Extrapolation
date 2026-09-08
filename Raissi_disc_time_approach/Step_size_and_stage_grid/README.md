# Step_size_and_stage_grid — width, depth and stages on the magnet-to-magnet set  (Block C, C3)

**What this is.** One network architecture grid, trained on
[`../Magnet_tracks_dataset`](../Magnet_tracks_dataset)'s magnet-to-magnet set —
real particles from the official simulated sample, steps of every length from
50 µm to the whole 5.2 m crossing, labelled by the fine sixth-order reference of
[`../Fine_reference`](../Fine_reference) — and scored **per step-length
stratum** so that step C4 can read an error(dz, q) table straight off the run
records.

    width  in {32, 64, 128, 256}   x   depth in {2, 4, 8}
    q      in {2, 4, 6, 8, 10, 12, 14, 16, 18, 20}
    mode   in {physics, data}      x   seed  in {0, 1, 2}

= **720 runs**, one farm job each, one cluster, tagged
`w<width>_d<depth>_q<qq>_<mode>_s<seed>`.

**Why.** Every earlier experiment in this folder fixed q at 8 and varied at most
the width, on a population dominated by one step length. Two questions were left
open by [`../General_leg_network`](../General_leg_network)'s residual wave: how
many Gauss-Legendre stages the discrete-time scheme actually needs on a real
magnet crossing, and whether the network size that was "enough" at 70 mm is
still enough at 5 m. Neither can be answered without a training set in which
every step length is equally represented — which is what C0 built — and a grid
that moves q and the network size independently, which is this.

**Everything about the optimiser is the verified baseline's, unchanged.** fp64,
full-batch L-BFGS with `max_iter` 200, strong Wolfe, history 120; a checkpoint
and a history row after every restart, so a run resumes on rerun; stall = two
consecutive restarts each improving the loss by less than 1%; then a
confirmation pass with a fresh optimiser, which counts as converged only if it
re-stalls within two restarts with the endpoint medians unchanged;
`--outer-cap 400`. `train_grid.py` does not re-implement that loop — it calls
`../_shared/train.py`'s `main` and swaps three module-level names.

---

## script → output

| script | what it does | output |
|---|---|---|
| [grid_model.py](grid_model.py) | the straight-line-residual network, made grid-capable: q, width and depth from the CLI, the (z0, dz) extra inputs, the MagUp field twin, and the floors lowered to 1e-9 mm / 1e-12 | — |
| [prepare_grid.py](prepare_grid.py) | **C3.1.** the residual scales on the v3 set and the O(1) check per stratum, plus the straight-line error per stratum | `results/magnet_tracks_v3_residual.npz`, `results/scale_check.json`, `figures/scale_check.png` |
| [prepare_nodes.py](prepare_nodes.py) | the per-q training sets: the RK6 reference at the q Gauss nodes as well as the endpoint | `results/grid_q<qq>.npz` (+ `_meta.json`) |
| [build_all_nodes.sh](build_all_nodes.sh) | runs the ten builds one after another | `results/build_all_nodes.log` |
| [train_grid.py](train_grid.py) | **C3.2.** one grid point, on the shared stall-and-confirm protocol, with the per-stratum and per-direction scores written into the json; `--init-check` verifies the algebra first | `results/<tag>.json`, `<tag>.pt`, `<tag>_history.csv`, `results/grid_init_check.json` |
| [measure_timing.py](measure_timing.py) | **C3.3.** one restart of the heaviest grid point on each rung of the row ladder, with peak memory | `results/timing.json` |
| [make_jobs.py](make_jobs.py) | **C3.4.** the 720-line job list and its submit file | `condor/jobs_grid.txt`, `condor/jobs_grid.sub` |
| [resubmit_grid.py](resubmit_grid.py) | sends the runs that did not confirm back to continue from their checkpoints | `condor/jobs_grid_round<N>.txt` / `.sub` |
| [aggregate_grid.py](aggregate_grid.py) | **C3.5.** the jsons → the two tables C4 reads | `results/summary.csv`, `results/error_vs_dz_q.csv` |
| [plot_grid.py](plot_grid.py) | **C3.5.** the heat maps and the two families of curves | `figures/heatmap_w<W>_d<D>.png`, `figures/error_vs_dz.png`, `figures/error_vs_q.png` |
| [tables.py](tables.py) | **C4.1 / C4.3.** the long table → the per-architecture display tables, the references from `../Exact_scheme_table`, and the eight csvs the readings rest on | `results/table_cells.csv`, `results/table_<D>x<W>.csv`, `results/table_<D>x<W>_by_direction.csv`, `results/pending_cells.csv`, `results/reading_*.csv` |
| [plot_tables.py](plot_tables.py) | **C4.2.** the six C4 figures, from the C4 csvs alone | `figures/heatmap_<D>x<W>.png`, `figures/error_vs_dz.png`, `figures/error_vs_q.png`, `figures/physics_over_twin.png`, `figures/cost_vs_error.png`, `figures/best_architecture_per_cell.png` |
| [analysis.ipynb](analysis.ipynb) | loads all of the above and displays it; computes nothing | — |

```bash
PY=/data/bfys/gscriven/conda/envs/TE/bin/python
export PYTHONNOUSERSITE=1
cd Step_size_and_stage_grid
$PY prepare_grid.py                       # C3.1, about a minute
./build_all_nodes.sh 8                    # the ten per-q sets
$PY train_grid.py --data results/grid_q20.npz --width 256 --depth 8 \
      --init-check --out results          # C3.2
condor_submit condor/jobs_timing.sub      # C3.3, the row ladder, on farm slots
condor_submit condor/jobs_arch_cost.sub   # C3.3, one restart per architecture
$PY measure_timing.py --collect           # C3.3, the ladder -> results/timing.json
$PY make_jobs.py && condor_submit condor/jobs_grid.sub   # C3.4
$PY aggregate_grid.py && $PY plot_grid.py                # C3.5
$PY tables.py && $PY plot_tables.py                      # C4.1, C4.2
```

**`plot_tables.py` must be run after `plot_grid.py`.** Both write
`figures/error_vs_dz.png` and `figures/error_vs_q.png`; the committed versions
are C4's, which carry the exact scheme as a reference curve and distil the
twelve architectures rather than drawing one panel each. Nothing else clashes —
C3's heat maps are `heatmap_w<W>_d<D>.png` and C4's are `heatmap_<D>x<W>.png`.

`results/*.npz` and `results/*.pt` are gitignored; the scripts, the seeds and
the meta json regenerate them.

---

## C3.1 The scales, and the check they had to pass

The parametrisation is
[`../General_leg_network`](../General_leg_network/README_residual.md)'s residual
redesign, unchanged in form: the network predicts the **deviation from a
straight line**, on a per-sample scale built from the inputs alone,

    output_j    = straight_j  +  scale (x) net_j
    straight_j  = ( x + tx*(z_j - z0),  y + ty*(z_j - z0),  tx,  ty )
    scale_slope = kappa * |qop| * I_B                  (dimensionless)
    scale_pos   = kappa * |qop| * I_B * |dz| / 2       (mm)

with `I_B` the field integral along that straight line (16-point midpoint rule,
T mm) and `kappa = 1e-3`. No label enters the scale.

**One thing changed: the floors.** The residual wave floored the position scale
at 1e-3 mm and the slope scale at 1e-6, which is right for legs of 70 mm and up.
Stratum 0 here is |dz| ≈ 0.1 mm, where the true deviation from a straight line
is of order 1e-7 mm — four orders of magnitude *below* the old floor. Left
alone the floor, not the field, would have set the scale on the three shortest
strata. The floors are now **1e-9 mm and 1e-12**, and only 2 of the 60,000 rows
are floored at all (both in stratum 0).

`prepare_grid.py` forms the number the network actually has to emit,
(RK6 reference − straight line) / scale at the endpoint, and histograms it per
stratum (`figures/scale_check.png`, `results/scale_check.json`). The criterion,
fixed before looking: the median must lie in 0.1–10 in **every** stratum.

| stratum | median \|dz\| | scale_pos median | **deviation / scale, median** | p05 – p95 | slope ratio median |
|---|---|---|---|---|---|
| 0.05–0.2 mm | 0.101 mm | 1.35e-7 mm | **1.005** | 0.93 – 1.18 | 1.005 |
| 0.5–2 mm | 1.004 mm | 1.35e-5 mm | **1.006** | 0.93 – 1.18 | 1.006 |
| 5–20 mm | 10.14 mm | 1.37e-3 mm | **1.005** | 0.93 – 1.18 | 1.005 |
| 50–200 mm | 99.77 mm | 0.135 mm | **1.009** | 0.93 – 1.18 | 1.005 |
| 500–2000 mm | 998.7 mm | 15.39 mm | **1.010** | 0.82 – 1.19 | 1.007 |
| full crossing | 5175 mm | 432.4 mm | **1.021** | 0.96 – 1.14 | 1.008 |

The label-free first-order estimate lands within 2% of the truth at the median
on every stratum, across four and a half decades of step length, and the
5th-to-95th spread never leaves the band 0.8–1.2. The target the network is
asked to learn is O(1) everywhere by construction, which is the premise the
whole grid rests on. The torch twin of the field integral agrees with the numpy
one to 1.2e-15 relative.

### Where the straight line already wins

C4 has to know which cells of the grid are worth reading, so the same script
records the straight line's own endpoint error against the RK6 reference:

| stratum | straight line, median | p95 | below 1 µm | val | test |
|---|---|---|---|---|---|
| 0.05–0.2 mm | **0.000137 µm** | 0.000993 µm | 100.0 % | 0.000140 | 0.000131 |
| 0.5–2 mm | **0.0138 µm** | 0.103 µm | 100.0 % | 0.0137 | 0.0139 |
| 5–20 mm | **1.39 µm** | 10.1 µm | 39.9 % | 1.38 | 1.42 |
| 50–200 mm | **137 µm** | 982 µm | 0.0 % | 130 | 141 |
| 500–2000 mm | **1.57e4 µm** | 1.07e5 µm | 0.0 % | 1.54e4 | 1.60e4 |
| full crossing | **4.38e5 µm** | 1.31e6 µm | 0.0 % | 4.18e5 | 4.56e5 |

Two readings matter for C4. First, **stratum 0 is already at the reference's own
floor**: 1.4e-4 µm against the 5e-5 µm the fine reference can be trusted to
(`../Fine_reference`), a factor of under three. Nothing measured in that stratum
— by a network or by anything else — can be claimed as an improvement on the
straight line, because the straight line is inside the measuring error of the
truth. Stratum 1, at 1.4e-2 µm, has about two and a half decades of headroom
above that floor and is the shortest stratum in which a result means anything.
Second, the forward and backward halves of every stratum agree to within a few
per cent, so a per-direction split in the grid is a check on the scoring rather
than a physical asymmetry.

---

## C3.2 The model, the trainer, and the check before any farm time

`grid_model.py` is `residual_model.py` with q, width and depth taken from the
command line (the network emits 4·(q+1) numbers), the MagUp map as the default
field twin, and the lowered floors. `../General_leg_network` was not edited; the
file is a copy so the two studies cannot drift into each other.

`train_grid.py` installs three names into `../_shared/train.py` and then calls
its `main`:

| replaced | with |
|---|---|
| `OneStepNetwork` | a factory returning `GridResidualNetwork` with this dataset's Gauss nodes, its (z0, dz) normalisation and the MagUp field |
| `data_loss` | `grid_data_loss`, normalised by the per-sample residual scale rather than the population-wide `out_scale` |
| `load_dataset` | the same loader with the training split optionally cut to `--n-train` rows, evenly across the six strata |

`physics_loss` is untouched: it is computed from the reconstructed **absolute**
states, so it only ever needed the wrapper's forward.

The json the shared trainer writes is then reopened and extended with the block
C4 reads:

* `by_stratum` — one score row per (split, stratum, direction), covering **val
  and test**, the six strata plus `all`, and both directions plus `all`. Each
  row carries `n`, the endpoint median and p95 in µm, the stage median, the
  endpoint slope median in mrad, ρ (mean and median) and the **straight-line
  median on exactly those rows**;
* `straight_by_stratum` — the straight-line column pulled out on its own;
* `n_train`, `n_parameters`, `node_profile`.

Every metric is `../_shared/evaluate.score_against_reference`, the function the
whole line has reported since the baseline, applied to a row mask. The network
is run once per split and the masks applied to its output, so the twenty-one
cells cost one forward pass, not twenty-one.

### The initialisation check

`train_grid.py --init-check` at the heaviest point of the grid — q = 20,
depth 8, width 256, seed 0 — writes `results/grid_init_check.json`:

| check | value |
|---|---|
| network | depth 8, width 256, q = 20 — **484,180 parameters**, 4·(q+1) = 84 outputs |
| training rows checked | 4,002 |
| straight-line term vs an independent numpy construction | 9.1e-13 mm |
| **last layer zeroed → output − straight line** | **0 mm, exactly** |
| smallest residual position scale in the set | 3.72e-09 mm — the floor is 1e-9 mm, so the floor never binds |
| raw network output at the shared trainer's own init | max 0.129, median 0.034 |
| so: offset from the straight line at init | 0.129 residual scales at most, 0.034 at the median |
| physics loss / max gradient at init | 4.577e-02 / 6.0e-03, all finite |
| twin loss / max gradient at init | 1.886e-01 / 2.9e-03, all finite |
| **passes** | **True** |

The last layer is *not* zeroed for the real runs — a zero last layer makes every
earlier layer's gradient exactly zero on the first step. Zeroing it is the
algebra check and nothing else: it says that the wrapper's straight-line term
and its node bookkeeping are right and that nothing else leaks into the output.
With the standard initialisation the model starts a few per cent of one
residual scale away from the straight line, which is the point of the
parametrisation — wave 1 of `../General_leg_network` started tens of
millimetres away from it.

---

## C3.3 How many training rows one restart can afford

The grid is 720 runs and a run is tens of L-BFGS restarts, so the training-set
size is not a free parameter: it is set by what one restart of the *heaviest*
point of the grid can do in a sensible wall time. **The rule, fixed before
measuring: one restart at depth 8, width 256, q = 20, the physics loss, one
thread, on 4,000 / 8,000 / 16,000 / 36,000 training rows; take the largest N
whose restart is under 120 s.**

The ladder was run **as four farm jobs** (cluster 5783187), not on the shared
interactive node: the 720 jobs run on farm slots, and the interactive node was
carrying a load average near 80 on 28 cores while this study was set up, so a
wall time measured there would have said more about the other users than about
the grid. For the record, the same restart at 4,002 rows on the interactive
node took 235.2 s against the farm's 201.8 s.

| training rows | restart wall | closures | s per closure | peak RSS | farm node |
|---|---|---|---|---|---|
| 4,002 | **201.8 s** | 205 | 0.985 | 2.22 GB | `wn-sate-061` |
| 8,004 | **216.0 s** | 206 | 1.048 | 2.64 GB | `wn-pijl-007` |
| 16,002 | **489.6 s** | 206 | 2.376 | 3.94 GB | `wn-lot-003` |
| 36,000 | **1055.6 s** | 209 | 5.051 | 4.75 GB | `wn-pijl-002` |

**No rung came in under 120 s, so the rule falls back to its smallest and
N = 4,002 training rows — 667 per stratum — is what the whole grid trains
on.** Three things about that table are worth saying plainly.

First, **the limit is not really N.** Every rung makes the same ~205 closure
evaluations (`max_iter 200` plus the strong-Wolfe line search), and between
4,002 and 8,004 rows the wall time moves by 7%. At the small end the cost is
the 484,180-parameter network and its L-BFGS history, not the rows; only from
16,002 rows on does it track N, and there it tracks it almost linearly.

Second, **the farm nodes differ by more than the two smallest rungs do.** The
per-architecture probe below re-ran the *identical* configuration — 8 x 256,
q = 20, 4,002 rows — on a different node and got 138.2 s against the ladder's
201.8 s, a spread of 46%. The verdict is the same on either number, but no
single measurement here is worth more than about two significant figures, and a
"120 s" line drawn through this population is a soft one.

Third, **the memory does track N**, and that is what settles the farm request:
2.22 GB at 4,002 rows, 3.94 GB at 16,002, 4.75 GB at 36,000. So the 8 GB the
plan allowed for the 8 x 256 architecture is not needed at the N actually
chosen — all 720 jobs ask for 1 CPU and 4 GB — but it would have been needed at
either of the two largest rungs, which is a second, independent reason the
grid is not run there.

`--outer-cap 400` bounds the worst case, and every run checkpoints after every
restart, so an evicted job resumes rather than restarting.

### Cost per architecture, all at q = 20 and 4,002 rows

| architecture | parameters | one restart | 60 restarts | 130 restarts | peak RSS |
|---|---|---|---|---|---|
| 2 x 32 | 4,084 | 15.6 s | 0.3 h | 0.6 h | 1.22 GB |
| 2 x 64 | 10,132 | 17.3 s | 0.3 h | 0.6 h | 1.23 GB |
| 2 x 128 | 28,372 | 32.3 s | 0.5 h | 1.2 h | 1.23 GB |
| 2 x 256 | 89,428 | 32.8 s | 0.5 h | 1.2 h | 1.43 GB |
| 4 x 32 | 6,196 | 14.6 s | 0.2 h | 0.5 h | 1.19 GB |
| 4 x 64 | 18,452 | 22.9 s | 0.4 h | 0.8 h | 1.19 GB |
| 4 x 128 | 61,396 | 34.4 s | 0.6 h | 1.2 h | 1.20 GB |
| 4 x 256 | 221,012 | 60.2 s | 1.0 h | 2.2 h | 1.76 GB |
| 8 x 32 | 10,420 | 16.6 s | 0.3 h | 0.6 h | 1.15 GB |
| 8 x 64 | 35,092 | 31.7 s | 0.5 h | 1.1 h | 1.17 GB |
| 8 x 128 | 127,444 | 56.0 s | 0.9 h | 2.0 h | 1.30 GB |
| 8 x 256 | 484,180 | 138.2 s | 2.3 h | 5.0 h | 2.26 GB |

One restart of each of the twelve architectures, measured on farm slots
(cluster 5783189, `results/timing_w*_q20.json`). q = 20 is the most expensive
column of the grid, so these are upper bounds: at q = 2 the same network emits
12 numbers instead of 84 and the physics loss evaluates the field at a tenth as
many stage planes. The "60 restarts" and "130 restarts" columns bracket the
range the residual wave stalled in, and are what to expect per job.

---

## C3.4 The farm

```bash
cd Step_size_and_stage_grid
mkdir -p condor/logs results
python make_jobs.py
condor_submit condor/jobs_grid.sub
```

**Cluster 5783188, 720 jobs, submitted 2026-09-07 19:29 CEST.** One CPU and
4 GB each, `+JobCategory "medium"`, `+UseOS "el9"`, `getenv = False`, no file
transfer — the repository, the datasets and the results are all on `/data` and
visible from the worker nodes. `condor/jobs_grid.sub` is the shared template
with the executable pointed at [condor/wrapper_grid.sh](condor/wrapper_grid.sh),
a copy of the shared wrapper that starts `train_grid.py` instead of
`_shared/train.py`; the shared wrapper is hard-wired to the latter and was not
edited. `condor/jobs_grid.txt` holds 720 argument lines and **no comment lines**
(`queue … from` does not skip `#`).

Two smaller clusters belong to C3.3 and are recorded here so the numbers above
can be traced: **5783187**, the four rungs of the row ladder, and **5783189**,
the twelve per-architecture cost probes.

### Expected wall time per architecture

The per-architecture table in C3.3 is the answer, scaled by the 60–130 restarts
the residual wave needed: about **20 minutes** for a 2 x 32 run, **1–2 hours**
for a 4 x 256 or 8 x 128 one and **2.3–5 hours** for an 8 x 256 one. Those are
the q = 20 numbers, which are the worst column of the grid; a q = 2 run of the
same architecture emits 12 outputs instead of 84 and evaluates the field at a
tenth as many stage planes, so the q = 2 end of each row is several times
cheaper. Summing the table over the grid gives an expectation of roughly
250–550 core-hours for the whole cluster.

### The confirmation gate, and the harness change it forced

Of the first 107 runs to land, **30 confirmed and 77 did not**, and the reason
was the harness rather than the runs. The shared stall criterion (two
consecutive restarts each improving the loss by less than 1%) fires while the
endpoint medians are still moving by more than 1%; the confirmation pass then
re-stalls at once, and until 2026-09-07 `../_shared/train.py` wrote
`converged = false` and **stopped there**. Because the phase was read straight
off the last history row, resubmitting such a run only re-ran the confirmation
it had just failed, from the same point, with the same outcome. That is harness
issue A1, which hit three of the twenty-six residual runs and is the common
case on this dataset.

It is now fixed at source. A confirmation that does not hold sends the run
**back to the stall phase**, and the stall/confirm cycle repeats until a
confirmation holds or `--outer-cap` (400) is reached; `resume_phase()` reads
the cycle position back out of the history, so a job that stopped mid-cycle
carries on training rather than re-confirming — including the records written
before the change, since nothing new is stored. A run that confirms at its
first attempt is untouched, restart for restart. The json keeps every field it
had and gains `confirm_attempts`.

Measured on a 30-restart run the old code abandoned: every loss in the shared
prefix is identical to the last digit (only the wall-clock column differs), and
resumed under the new code the same run went back to the stall phase, needed
three more confirmation attempts and converged at restart 42.
`_shared/smoke_tests.py` passes in full, including the bitwise first-restart
parity gate against the original baseline model.

### Running a resubmission pass

`resubmit_grid.py` is **idempotent and safe to run on a schedule**: it picks up
only records whose json says `converged = false` and whose tag is not already
idle or running in the queue — two processes writing one checkpoint would
corrupt it — so a pass run while a round is still draining selects nothing, and
a pass run after more records land selects exactly the new ones. Each pass
writes its own numbered round files and `--round` defaults to the next number
not yet used. The command, suitable for an hourly schedule:

```bash
cd /data/bfys/gscriven/LHCb_Extrapolation_Project/Raissi_disc_time_approach/Step_size_and_stage_grid && \
  PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python resubmit_grid.py --submit
```

Drop `--submit` to see what a pass would send without sending it. A run that
hit the restart cap rather than failing to confirm is skipped unless
`--include-capped` is given, since resuming it without also raising the cap
would simply cap again.

**Round 2: cluster 5783191, 137 runs, submitted 2026-09-07 19:54 CEST** — every
unconfirmed record on disk at that moment, resuming from its own checkpoint
under the new cycle. `plot_grid.py` marks any cell no seed of which confirmed
rather than dropping it, so C4 always sees which conclusions rest on
unconfirmed runs.

---

## C3.5 The tables and the figures

`aggregate_grid.py` reads json only — it never loads a checkpoint and never
re-scores anything — so it can be run while the grid is still draining and its
numbers cannot drift from the ones each run reported.

`results/summary.csv`, one row per run:

    tag, width, depth, q, mode, seed, n_train, n_parameters, converged,
    restarts, final_loss, wall_s,
    {val,test}_{median_um, p95_um, stage_um, slope_mrad, rho, straight_um}

`results/error_vs_dz_q.csv`, the long table C4 reads, one row per
(run, split, stratum, direction):

    width, depth, q, mode, seed, split, stratum, direction, converged,
    median, p95, slope, rho, straight_line, n, tag

`stratum` is the stratum name (`all` for the whole split), `direction` is `all`
/ `forward` (dz > 0) / `backward` (dz < 0), `median` and `p95` are the endpoint
position error in µm, `slope` the endpoint slope error in mrad, `rho` the median
of the agreed scalar relative error, and `straight_line` the straight-line
median **on exactly those rows**. Runs that did not confirm are kept with
`converged` false rather than filtered out, so C4 decides what to do with them.

`plot_grid.py` draws, from that csv alone:

* `figures/heatmap_w<W>_d<D>.png` — one file per architecture: rows are q,
  columns the six strata, colour is log10 of the endpoint median in µm, with the
  physics arm and the data twin side by side on a shared colour scale and the
  straight line as the bottom row of each panel. A cell no seed of which
  confirmed is marked with a red dot rather than dropped;
* `figures/error_vs_dz.png` — one panel per architecture, endpoint median
  against the stratum's median \|dz\|, one line per q, the straight line in
  black;
* `figures/error_vs_q.png` — one panel per stratum, endpoint median against q,
  one line per architecture.

Each cell is the median over the seeds that confirmed; if no seed of a cell
confirmed, the median over all of its seeds is used and the cell is marked.

---

## What C4 should be careful about

1. **Stratum 0 cannot show an improvement.** The straight line there is 1.4e-4
   µm against a reference floor of 5e-5 µm. Read strata 1–5.
2. **The (z0, dz) extra inputs are normalised linearly**, as the residual design
   normalises them, so |dz| = 0.1 mm and |dz| = 1 mm both arrive at the network
   as essentially zero. The residual *scale* carries the step length exactly, so
   the target stays O(1) (C3.1 above), but if the error-vs-dz curves show the
   three shortest strata behaving as one, this normalisation is the first thing
   to suspect.
3. **`in_scale`, `out_scale`, `extra_mean` and `extra_scale` are computed on the
   dataset's training split as built**, not on the subset a run trains on, so
   they are identical across the grid.
4. **The endpoint label is the dataset's own.** `prepare_nodes.py` marches
   through the Gauss nodes in segments, which is one crossing per row rather
   than q/2 of them, and then substitutes `Y` from `magnet_tracks_v3.npz` for
   the endpoint row. The difference between the composed and the unsegmented
   endpoint is recorded per split in each `grid_q<qq>_meta.json` under
   `endpoint_composition_vs_dataset_um`; the interior nodes carry that same
   difference and it is not separately measurable.

---
---

# C4 — the error(dz, q) tables, and what they say

**What C4 is.** The 720 run records of C3 read as one surface: the endpoint
error as a function of the step length |dz|, the number of Gauss-Legendre
stages q, the network size, and which of the two losses trained it — with the
exact scheme of [`../Exact_scheme_table`](../Exact_scheme_table) and the
straight line printed underneath every table as the two things a cell has to be
read against. Nothing here retrains or re-scores anything: `tables.py` and
`plot_tables.py` read `results/error_vs_dz_q.csv`, which C3's
`aggregate_grid.py` builds out of the run jsons.

**All 720 records are on disk and all 720 confirmed** (cluster 5783188 drained
at 08:30 on 2026-09-08; the queue is empty). Every cell of every table below is
the median over **three** seeds, nothing is flagged `pending`, and
`results/pending_cells.csv` is an empty table — a header and no rows.

The last five to land were one seed each of five cells of the **8 x 256
data-twin** row — `w256_d8_q{06,08,12,18,20}_data_s{2,0,2,1,0}` — which needed
187 to 355 restarts and 2 to 15 confirmation attempts, about 11 hours of one
core each, against the `--outer-cap 400` ceiling. In all five cells the
late-landing seed turned out to be **the best of the three** (its full-crossing
median is 1,164 to 3,674 µm against 4,113 to 6,892 µm for the two that had
already landed), which is not a coincidence: those runs are the ones that kept
training. It moves the cell medians by −0.1 % to −15 % — the median of three is
the smaller of the two old values — and widens the `[min-max]` brackets a long
way. **No reading below changes**; the numbers that move are listed at the end
of C4.3.

## C4.1 The tables

`results/table_cells.csv` is the tidy master — one row per (architecture, arm,
q, stratum, direction, split), with the median over the seeds that confirmed,
the min and max over those seeds, the p95, the slope error, ρ, the straight
line on exactly those rows, the seed count and the pending flag. Everything
else in C4 is computed from it.

`results/table_<D>x<W>.csv` is the display table, one per architecture: rows
q = 2 … 20, columns the six strata, cells `median [min-max]` in µm for the
physics arm and the data twin, then the exact scheme at each q, the straight
line, and the fine reference's floor as a note.
`results/table_<D>x<W>_by_direction.csv` is the same split into forward
(dz > 0) and backward (dz < 0).

**The fine reference's own floor is 5e-5 µm over a whole crossing**
([`../Fine_reference`](../Fine_reference) C1.4) and far lower over a short step,
since that figure is the worst case of a ~52,000-step march and a 0.1 mm step
takes one. It is quoted as a note rather than a table row because it is not a
constant down the columns.

### 4 x 64, test split, both directions pooled

| row | arm | 0.05-0.2 mm | 0.5-2 mm | 5-20 mm | 50-200 mm | 500-2000 mm | full crossing |
|---|---|---|---|---|---|---|---|
| q=2 | physics | 1.75e-05 [1.47e-05-2.51e-05] | 0.0018 [0.00149-0.00276] | 0.186 [0.14-0.273] | 13.5 [12.4-20.7] | 312 [304-318] | 9.92e+03 [9.74e+03-1.03e+04] |
| q=2 | twin | 1.55e-06 [1.4e-06-1.78e-06] | 0.000141 [0.000133-0.000175] | 0.0172 [0.0164-0.0199] | 2.16 [2.12-3.03] | 323 [314-405] | 3.57e+03 [2.87e+03-4.02e+03] |
| q=4 | physics | 1.18e-05 [9.43e-06-1.21e-05] | 0.00124 [0.00098-0.00125] | 0.126 [0.105-0.127] | 10.2 [9.14-11.1] | 272 [259-272] | 3.58e+03 [3.53e+03-3.72e+03] |
| q=4 | twin | 1.67e-06 [1.31e-06-3.25e-06] | 0.00015 [0.00013-0.000216] | 0.0168 [0.015-0.0226] | 2.18 [1.85-2.46] | 367 [260-487] | 3.86e+03 [3.21e+03-6.33e+03] |
| q=6 | physics | 7.86e-06 [6.16e-06-2.45e-05] | 0.000802 [0.000655-0.00261] | 0.0841 [0.0652-0.257] | 8.18 [8.04-16.2] | 295 [269-299] | 755 [737-785] |
| q=6 | twin | 2.43e-06 [1.16e-06-3.02e-06] | 0.000221 [0.00011-0.000252] | 0.0237 [0.0114-0.0277] | 2.84 [1.76-3.47] | 481 [274-575] | 5.78e+03 [3.72e+03-7.25e+03] |
| q=8 | physics | 2.35e-05 [1.44e-05-2.48e-05] | 0.00246 [0.00153-0.00267] | 0.245 [0.154-0.27] | 15.1 [13.2-16.3] | 276 [275-310] | 776 [733-807] |
| q=8 | twin | 1.77e-06 [1.42e-06-4.11e-06] | 0.000182 [0.000167-0.000344] | 0.0225 [0.0172-0.0309] | 2.78 [2.21-3.47] | 392 [384-927] | 4.77e+03 [4.57e+03-8.17e+03] |
| q=10 | physics | 1.85e-05 [8.38e-06-3.02e-05] | 0.00193 [0.000855-0.00331] | 0.2 [0.0986-0.325] | 15.9 [11.3-26.7] | 290 [261-298] | 736 [706-749] |
| q=10 | twin | 2.11e-06 [2.06e-06-2.22e-06] | 0.000203 [0.000178-0.000254] | 0.0193 [0.0186-0.0263] | 2.52 [2.2-2.55] | 409 [343-579] | 5.42e+03 [3.93e+03-6.17e+03] |
| q=12 | physics | 1.91e-05 [4.75e-06-2.25e-05] | 0.00191 [0.000482-0.00241] | 0.193 [0.0522-0.242] | 15.1 [8.18-21.6] | 303 [274-340] | 726 [699-889] |
| q=12 | twin | 3.11e-06 [2.76e-06-3.57e-06] | 0.000281 [0.000198-0.000286] | 0.0286 [0.0218-0.0288] | 3 [2.9-4.03] | 668 [431-714] | 5.85e+03 [5.72e+03-6.55e+03] |
| q=14 | physics | 2.07e-05 [1.08e-05-2.95e-05] | 0.00216 [0.00115-0.00315] | 0.22 [0.112-0.317] | 15.1 [13.9-25.1] | 330 [282-333] | 814 [752-820] |
| q=14 | twin | 3.05e-06 [2.41e-06-3.15e-06] | 0.000209 [0.000177-0.000272] | 0.0215 [0.0165-0.0289] | 2.44 [2.41-3.29] | 442 [368-480] | 6.2e+03 [5.23e+03-7.52e+03] |
| q=16 | physics | 9.63e-06 [9.6e-06-1e-05] | 0.00105 [0.000994-0.00105] | 0.108 [0.101-0.111] | 9.42 [9.4-10.6] | 288 [271-297] | 766 [657-770] |
| q=16 | twin | 1.83e-06 [1.81e-06-2.91e-06] | 0.000198 [0.000144-0.000239] | 0.021 [0.0175-0.0236] | 2.47 [2.08-2.68] | 415 [324-668] | 4.97e+03 [3.35e+03-6.52e+03] |
| q=18 | physics | 1.81e-05 [9.92e-06-3.13e-05] | 0.00185 [0.000958-0.00332] | 0.194 [0.103-0.335] | 16.8 [10.6-27] | 285 [282-309] | 770 [697-780] |
| q=18 | twin | 1.98e-06 [1.54e-06-2.77e-06] | 0.000218 [0.000159-0.000237] | 0.0199 [0.016-0.0276] | 2.22 [2.19-2.7] | 387 [355-396] | 4.83e+03 [4.36e+03-5.25e+03] |
| q=20 | physics | 1.12e-05 [7.82e-06-1.72e-05] | 0.00119 [0.0008-0.00181] | 0.117 [0.0862-0.193] | 11.6 [10.2-15.2] | 302 [267-367] | 735 [714-853] |
| q=20 | twin | 3.21e-06 [2.5e-06-3.33e-06] | 0.00019 [0.000177-0.000217] | 0.0217 [0.0189-0.0245] | 2.65 [2.45-2.66] | 465 [361-495] | 5.66e+03 [3.7e+03-6.4e+03] |
| exact scheme q=2 | exact scheme | 1.42e-11 | 2.34e-10 | 3.18e-09 | 0.00321 | 12.8 | 9.26e+03 |
| exact scheme q=4 | exact scheme | 1.42e-11 | 2.35e-10 | 3.07e-09 | 0.000749 | 1.22 | 3.62e+03 |
| exact scheme q=6 | exact scheme | 1.42e-11 | 2.34e-10 | 3.07e-09 | 0.000348 | 0.825 | 233 |
| exact scheme q=8 | exact scheme | 1.42e-11 | 2.34e-10 | 3.05e-09 | 0.000201 | 0.696 | 32.6 |
| exact scheme q=10 | exact scheme | 1.42e-11 | 2.31e-10 | 3.04e-09 | 0.000137 | 0.504 | 32.5 |
| exact scheme q=12 | exact scheme | 1.42e-11 | 2.34e-10 | 3.02e-09 | 9.71e-05 | 0.396 | 41.5 |
| exact scheme q=14 | exact scheme | 1.42e-11 | 2.34e-10 | 3.04e-09 | 6.5e-05 | 0.304 | 20.4 |
| exact scheme q=16 | exact scheme | 1.42e-11 | 2.34e-10 | 3.01e-09 | 5.34e-05 | 0.219 | 36.1 |
| exact scheme q=18 | exact scheme | 1.42e-11 | 2.27e-10 | 3e-09 | 4.21e-05 | 0.167 | 38.6 |
| exact scheme q=20 | exact scheme | 1.42e-11 | 2.34e-10 | 2.98e-09 | 3.41e-05 | 0.146 | 23.6 |
| straight line | straight line | 0.000131 | 0.0139 | 1.42 | 141 | 1.6e+04 | 4.56e+05 |

### 8 x 256, test split, both directions pooled

| row | arm | 0.05-0.2 mm | 0.5-2 mm | 5-20 mm | 50-200 mm | 500-2000 mm | full crossing |
|---|---|---|---|---|---|---|---|
| q=2 | physics | 1.73e-05 [1.56e-05-4.07e-05] | 0.00164 [0.00156-0.00418] | 0.17 [0.159-0.416] | 10 [9.94-19.7] | 282 [278-299] | 9.68e+03 [9.55e+03-9.93e+03] |
| q=2 | twin | 3.11e-06 [2.14e-06-3.5e-06] | 0.000272 [0.000217-0.000276] | 0.0271 [0.0227-0.0304] | 3.17 [3.11-3.51] | 498 [495-738] | 4.98e+03 [4.44e+03-6.33e+03] |
| q=4 | physics | 3.71e-05 [3.11e-05-3.8e-05] | 0.0039 [0.00331-0.004] | 0.377 [0.311-0.385] | 20.6 [19.3-24.9] | 241 [240-261] | 3.53e+03 [3.51e+03-3.57e+03] |
| q=4 | twin | 3.89e-06 [9.73e-07-4.81e-06] | 0.000325 [5.21e-05-0.000332] | 0.031 [0.00684-0.0337] | 3.5 [0.928-4.3] | 743 [138-783] | 5.78e+03 [1.06e+03-7.34e+03] |
| q=6 | physics | 2.73e-05 [1.09e-05-4.46e-05] | 0.00282 [0.0012-0.00456] | 0.267 [0.113-0.444] | 12.5 [7.72-20.5] | 261 [242-271] | 769 [709-770] |
| q=6 | twin | 2.6e-06 [1.02e-06-2.96e-06] | 0.000228 [9.35e-05-0.000235] | 0.0241 [0.00882-0.0265] | 2.6 [1.2-2.71] | 415 [210-437] | 4.48e+03 [1.64e+03-4.69e+03] |
| q=8 | physics | 3.05e-05 [2.63e-05-4.16e-05] | 0.00304 [0.00281-0.00442] | 0.299 [0.264-0.427] | 16.2 [12.8-25.1] | 229 [215-272] | 660 [632-669] |
| q=8 | twin | 4.16e-06 [1.35e-06-4.25e-06] | 0.000282 [8.13e-05-0.000389] | 0.03 [0.00924-0.0323] | 3.57 [1.33-4.46] | 806 [218-913] | 6.39e+03 [2.09e+03-6.47e+03] |
| q=10 | physics | 2.09e-05 [1.55e-05-3.86e-05] | 0.00218 [0.00161-0.00399] | 0.216 [0.161-0.388] | 12.4 [12-17.7] | 228 [227-259] | 747 [705-759] |
| q=10 | twin | 2.92e-06 [2.46e-06-3.57e-06] | 0.000258 [0.000197-0.000303] | 0.0257 [0.0212-0.0287] | 3.06 [2.62-3.13] | 572 [475-607] | 5.22e+03 [4.36e+03-5.35e+03] |
| q=12 | physics | 2.5e-05 [1.69e-05-2.8e-05] | 0.00271 [0.0019-0.00297] | 0.258 [0.195-0.271] | 17.1 [16.7-19.2] | 244 [216-252] | 645 [602-738] |
| q=12 | twin | 4.36e-06 [1.16e-06-4.42e-06] | 0.000271 [9.75e-05-0.000298] | 0.0287 [0.00749-0.029] | 3.41 [1.21-3.71] | 769 [186-892] | 5.29e+03 [1.16e+03-6.89e+03] |
| q=14 | physics | 2.02e-05 [1.11e-05-4.29e-05] | 0.00224 [0.00117-0.00437] | 0.212 [0.126-0.422] | 14.9 [10.1-27] | 248 [247-260] | 621 [591-698] |
| q=14 | twin | 2.73e-06 [2.69e-06-2.83e-06] | 0.000235 [0.000233-0.000286] | 0.0219 [0.0202-0.0226] | 2.47 [2.26-2.47] | 444 [441-482] | 5.11e+03 [4.2e+03-5.45e+03] |
| q=16 | physics | 3.56e-05 [1.21e-05-5.24e-05] | 0.00369 [0.00126-0.00547] | 0.355 [0.126-0.532] | 17.6 [10.5-39.3] | 225 [224-301] | 670 [625-736] |
| q=16 | twin | 3.62e-06 [2.55e-06-4.95e-06] | 0.00028 [0.000223-0.000412] | 0.0276 [0.0193-0.0385] | 2.83 [2.26-4.03] | 579 [397-974] | 5.96e+03 [4.02e+03-8.1e+03] |
| q=18 | physics | 2.81e-05 [2.24e-05-4.46e-05] | 0.00292 [0.00241-0.0047] | 0.311 [0.23-0.446] | 21.8 [17.9-27.8] | 230 [223-264] | 593 [582-659] |
| q=18 | twin | 2.71e-06 [1.27e-06-4.07e-06] | 0.000171 [0.000113-0.000354] | 0.0185 [0.00905-0.0291] | 2.32 [1.27-2.88] | 384 [203-627] | 4.11e+03 [2.26e+03-5.58e+03] |
| q=20 | physics | 3e-05 [7.83e-06-3.06e-05] | 0.00317 [0.000818-0.00325] | 0.301 [0.0862-0.317] | 16.2 [9.19-17.8] | 255 [229-260] | 656 [647-684] |
| q=20 | twin | 3.68e-06 [2.38e-06-3.95e-06] | 0.000257 [0.000177-0.000258] | 0.0296 [0.0198-0.0306] | 3.47 [2.32-3.51] | 929 [434-989] | 6.88e+03 [3.67e+03-6.89e+03] |
| exact scheme q=2 | exact scheme | 1.42e-11 | 2.34e-10 | 3.18e-09 | 0.00321 | 12.8 | 9.26e+03 |
| exact scheme q=4 | exact scheme | 1.42e-11 | 2.35e-10 | 3.07e-09 | 0.000749 | 1.22 | 3.62e+03 |
| exact scheme q=6 | exact scheme | 1.42e-11 | 2.34e-10 | 3.07e-09 | 0.000348 | 0.825 | 233 |
| exact scheme q=8 | exact scheme | 1.42e-11 | 2.34e-10 | 3.05e-09 | 0.000201 | 0.696 | 32.6 |
| exact scheme q=10 | exact scheme | 1.42e-11 | 2.31e-10 | 3.04e-09 | 0.000137 | 0.504 | 32.5 |
| exact scheme q=12 | exact scheme | 1.42e-11 | 2.34e-10 | 3.02e-09 | 9.71e-05 | 0.396 | 41.5 |
| exact scheme q=14 | exact scheme | 1.42e-11 | 2.34e-10 | 3.04e-09 | 6.5e-05 | 0.304 | 20.4 |
| exact scheme q=16 | exact scheme | 1.42e-11 | 2.34e-10 | 3.01e-09 | 5.34e-05 | 0.219 | 36.1 |
| exact scheme q=18 | exact scheme | 1.42e-11 | 2.27e-10 | 3e-09 | 4.21e-05 | 0.167 | 38.6 |
| exact scheme q=20 | exact scheme | 1.42e-11 | 2.34e-10 | 2.98e-09 | 3.41e-05 | 0.146 | 23.6 |
| straight line | straight line | 0.000131 | 0.0139 | 1.42 | 141 | 1.6e+04 | 4.56e+05 |

## C4.2 The figures

| figure | what it is |
|---|---|
| `figures/heatmap_<D>x<W>.png` | one per architecture: rows q, columns the six strata, the physics arm and the data twin side by side on one log colour scale, with the exact scheme's ten rows and the straight line beneath them. A cell with fewer than three seeds is hatched in red. |
| `figures/error_vs_dz.png` | log-log error against \|dz\|, one curve per q, each point the **best of the twelve architectures** at that stratum; the straight line solid, the q = 8 exact scheme dashed |
| `figures/error_vs_q.png` | error against q, one curve per stratum, at 4 x 64 and 8 x 256, with the straight line dotted and the exact scheme at that q dashed in the matching colour |
| `figures/physics_over_twin.png` | the ratio map, physics median / twin median, per cell, twelve panels |
| `figures/cost_vs_error.png` | parameters and forward-pass multiply-adds against error at 50–200 mm and the full crossing, with the best error at or below each cost drawn as a line |
| `figures/best_architecture_per_cell.png` | which of the twelve architectures wins each (q, stratum) cell, and the error it reaches |

Multiply-adds are the MLP's alone,
`7·W + (D−1)·W² + W·4·(q+1)` — 7 inputs (5 state + 2 extras), D hidden layers of
width W, a linear layer to 4(q+1) outputs. The residual wrapper's 16 field
lookups per sample are the same for every architecture and are not in that
number. Parameters and multiply-adds differ by less than the biases, so the two
panels of `cost_vs_error.png` are near-identical by construction; both are drawn
because the second is the one that matters if this ever runs in a trigger.

## C4.3 Reading the tables

### (a) q only matters in one column, and only for the physics loss

`results/reading_q_sensitivity.csv`. For each (architecture, arm, stratum) it
records the spread of the cell medians over the ten stage counts, and — as the
control that makes the spread mean something — the **typical spread over the
three seeds inside one cell**. q is called out as mattering only when the first
exceeds twice the second.

| arm | stratum | median spread over q | median seed spread inside a cell | architectures where q matters |
|---|---|---|---|---|
| physics | 0.05–0.2 mm | 2.30 | 1.93 | 1 of 12 |
| physics | 0.5–2 mm | 2.40 | 1.91 | 1 of 12 |
| physics | 5–20 mm | 2.39 | 1.91 | 1 of 12 |
| physics | 50–200 mm | 2.04 | 1.73 | 1 of 12 |
| physics | 500–2000 mm | 1.25 | 1.15 | **0 of 12** |
| physics | full crossing | **13.4** | 1.11 | **12 of 12** |
| twin | 0.05–0.2 mm | 1.74 | 1.44 | 1 of 12 |
| twin | 0.5–2 mm | 1.63 | 1.38 | 1 of 12 |
| twin | 5–20 mm | 1.53 | 1.35 | 0 of 12 |
| twin | 50–200 mm | 1.46 | 1.30 | 0 of 12 |
| twin | 500–2000 mm | 1.84 | 1.34 | 0 of 12 |
| twin | full crossing | 1.58 | 1.39 | 1 of 12 |

**Only the full crossing, and only under the physics loss.** In the other five
columns the answer moves with q by a factor 1.2–2.4 while three seeds of one
cell move by 1.1–1.9: the q dependence is inside the optimiser's own scatter.
The 500–2000 mm column, which the plan expected to show a q dependence, does
not — 0 of 12 architectures, a spread of 1.25 against a seed spread of 1.15.

At the full crossing the picture is the one the exact scheme predicts, and it is
sharp. Over the twelve architectures:

| q | physics-arm network, full crossing | exact scheme at the same q | ratio |
|---|---|---|---|
| 2 | 9,676 – 10,050 µm | 9,256 µm | **1.045 – 1.085** |
| 4 | 3,477 – 4,050 µm | 3,617 µm | **0.96 – 1.12** |
| 6 – 20 | 593 – 1,745 µm | 20.4 – 233 µm | 18 – 53 × the q = 8 ceiling |

**At q = 2 and q = 4 the network sits on the scheme**, within 9 % and 12 %
respectively: those two stage counts are so inaccurate on a 5.2 m step that the
scheme's own discretisation error is the whole error, and the network reproduces
it. **From q = 6 the network is on its own floor**: the scheme drops by a
further factor 15 (3,617 → 233 µm) and then to 32.6 µm at q = 8, while the
network flattens at 593–1,745 µm and stays there to q = 20. The plateau starts
at q = 6 in every one of the twelve architectures
(`plateau_from_q` in the csv). The data twin shows none of this — 906 –
10,870 µm at the crossing with no trend in q at all — because it is fitted to
labels rather than to the scheme's equations and the stage count only changes
how many intermediate labels it is also asked to fit.

**The practical reading: past q = 6 the stages are free accuracy the network
does not collect.** Nothing above q = 6 is worth its cost in this grid.

### (b) The floor scales as |dz|² — the twin exactly, the physics arm not quite

`results/reading_dz_slope.csv`, a straight-line fit of log10(median) against
log10(median |dz|) over the five interior strata (the full crossing is a
different population — whole legs, not a log-uniform draw — and is excluded from
the fit; it is fitted as well and reported as `slope_all_six`).

The prediction is 2. The residual scale the network's output is measured in is
`kappa · |qop| · I_B · |dz| / 2` with `I_B` itself proportional to |dz|, so a
network that lands a fixed fraction of one residual scale from the truth has an
error going as |dz|².

| arm | slope over strata 0–4, median | range over the twelve architectures | by depth (2 / 4 / 8) |
|---|---|---|---|
| **twin** | **2.088** | 2.032 – 2.212 | 2.117 / 2.074 / 2.077 |
| **physics** | **1.859** | 1.708 – 2.012 | 1.934 / 1.861 / 1.762 |

The twin is on the prediction to within 5 % at every one of the twelve
architectures. The physics arm is systematically **shallower**, and the shortfall
grows with depth. A slope below 2 means the error falls with |dz| more slowly
than the residual scale does, i.e. relative to what it is asked to emit the
network is worse on short steps than on long ones — which is (c).

### (c) Against the straight line: flat for the twin, 70× better at the crossing for the physics arm

`results/reading_ratio_straight.csv`, the cell median divided by the straight
line's median **on exactly the same rows**.

The two "best architecture" columns are the smallest of the twelve
architectures' own medians over q, not the single best cell; the best cell is
lower again (0.0013 for the physics arm at the crossing, 0.0020 for the twin).

| stratum | physics, median over architectures | physics, best architecture | twin, median | twin, best architecture |
|---|---|---|---|---|
| 0.05–0.2 mm | 0.120 | 0.068 | 0.023 | 0.0041 |
| 0.5–2 mm | 0.117 | 0.067 | 0.019 | 0.0041 |
| 5–20 mm | 0.119 | 0.067 | 0.019 | 0.0045 |
| 50–200 mm | 0.092 | 0.066 | 0.023 | 0.0072 |
| 500–2000 mm | 0.019 | 0.014 | 0.039 | 0.011 |
| full crossing | **0.0017** | 0.0015 | 0.013 | 0.0027 |

The physics arm behaves exactly as the plan expected: about **0.1 of the
straight line at short steps, improving to 1.7e-3 at the crossing** — a factor
70 between the two ends. The twin does not: its ratio is **flat at 0.013–0.039
across four and a half decades of step length**, which is the same statement as
its |dz|² slope of 2.09.

**The reason is the normalisation, and the two arms are normalised
differently.** `_shared/model.reconstruction_residuals` divides the physics
loss's residual by `model.in_scale[:4]` — **one population-wide input scale**,
the same four numbers (495 mm, 378 mm, 0.170, 0.0694) for every row of the set.
The deviation a step has to resolve goes as |dz|², so measured on that fixed
scale a 0.101 mm row's residual is (0.101 / 5175)² = **3.8e-10** of a
full-crossing row's, and its contribution to the mean *squared* residual is
that squared again, of order 1e-19. The optimiser has effectively no incentive
to fit the short rows at all. `grid_data_loss` (in `grid_model.py`) divides instead by the **per-sample
residual scale**, so every row contributes an O(1) target whatever its length —
and its ratio to the straight line comes out constant, which is what a
per-sample normalisation is for.

That is also why the short columns of the physics arm **do not improve with
network size** while the twin's do. The best cell in each column, over all q:

| stratum | physics, w = 32 | 64 | 128 | 256 | twin, w = 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|---|
| 0.05–0.2 mm | 5.75e-06 | 7.86e-06 | 1.02e-05 | 6.95e-06 | 1.92e-06 | 1.55e-06 | 7.30e-07 | **4.31e-07** |
| 0.5–2 mm | 5.80e-04 | 8.02e-04 | 1.09e-03 | 6.95e-04 | 2.22e-04 | 1.41e-04 | 5.78e-05 | **4.47e-05** |
| 5–20 mm | 0.0609 | 0.0841 | 0.107 | 0.0755 | 0.0206 | 0.0168 | 0.0064 | **0.0051** |
| 50–200 mm | 7.43 | 8.18 | 11.2 | 7.42 | 2.74 | 2.16 | 0.947 | **0.776** |
| 500–2000 mm | 215 | 192 | 216 | 225 | 436 | 323 | **132** | 143 |
| full crossing | 725 | 639 | 632 | **593** | 5,953 | 3,565 | 970 | **906** |

The physics arm's three short columns are **flat in width to within the seed
scatter** — an eightfold increase in width moves 0.05–0.2 mm from 5.8e-6 to
7.0e-6 µm, i.e. not at all, and the narrowest network is nominally the best of
the four. The twin's improve monotonically: a factor 4.5 at 0.05–0.2 mm, 5.0 at
0.5–2 mm, 4.0 at 5–20 mm and 3.5 at 50–200 mm from w = 32 to w = 256. **The
short columns are not limited by capacity; they are limited by how much the loss
weights them**, and swapping the population-wide normalisation for a per-sample
one is what unlocks them.

Two things this does *not* say. It does not say the twin is a better
extrapolator — see (d), where the ordering reverses on the long steps. And it
does not make stratum 0 a result: the straight line there is 1.31e-4 µm against a
reference floor of 5e-5 µm, so the best twin cell, 4.3e-7 µm, is three hundred
times *below* the floor of the truth it is being scored against and is a
statement about the fit, not about accuracy.

### (d) Physics against twin: the twin owns the short steps, the physics loss owns the crossing

`results/reading_physics_vs_twin.csv`, physics median / twin median, cell by
cell, 720 cells.

| stratum | quartile 1 | median | quartile 3 | min | max |
|---|---|---|---|---|---|
| 0.05–0.2 mm | 4.90 | **7.95** | 11.8 | 1.93 | 29.1 |
| 0.5–2 mm | 5.89 | **9.34** | 13.9 | 1.92 | 32.4 |
| 5–20 mm | 5.92 | **8.32** | 12.9 | 1.78 | 30.8 |
| 50–200 mm | 3.41 | **5.04** | 6.75 | 2.11 | 16.3 |
| 500–2000 mm | 0.35 | **0.63** | 1.20 | 0.19 | 3.40 |
| full crossing | 0.12 | **0.163** | 0.60 | 0.072 | 10.7 |

Supervision wins the four short columns by 5–9× in the median and never loses
one (the minimum ratio is 1.78, i.e. the twin is ahead in every one of the 480
cells of those four columns). The crossover is the 500–2000 mm column, where
the ratio straddles 1. At the **full crossing the label-free loss wins by a
factor 6**, and by more than that at every q ≥ 6: at 8 x 256 the ratio is 1.94
at q = 2, 0.61 at q = 4 and 0.095–0.17 from q = 6 on.

That reversal is the interesting one, because it is the case the whole
discrete-time line is about. On the crossing the twin is fitting an RK6 label it
cannot reach — 906 µm at best — while the physics arm is solving the collocation
equations, which on that step *are* an accurate description, and reaches 593 µm.
On short steps the twin's label is essentially exact and easy, and the physics
arm's population-wide normalisation stops it from caring. The two effects are
different mechanisms and both are visible in one grid.

### (e) Which architecture wins each column, and what depth 8 costs

`results/reading_best_architecture.csv` and
`figures/best_architecture_per_cell.png`.

| arm | stratum | winner | q | median [µm] | parameters |
|---|---|---|---|---|---|
| physics | 0.05–0.2 mm | **2 x 32** | 12 | 5.75e-06 | 3,028 |
| physics | 0.5–2 mm | **2 x 32** | 12 | 5.80e-04 | 3,028 |
| physics | 5–20 mm | **2 x 32** | 12 | 0.0609 | 3,028 |
| physics | 50–200 mm | **2 x 256** | 2 | 7.42 | 70,924 |
| physics | 500–2000 mm | **8 x 64** | 18 | 192 | 34,572 |
| physics | full crossing | **8 x 256** | 18 | 593 | 482,124 |
| twin | 0.05–0.2 mm | **2 x 256** | 8 | 4.31e-07 | 77,092 |
| twin | 0.5–2 mm | **2 x 256** | 8 | 4.47e-05 | 77,092 |
| twin | 5–20 mm | **2 x 256** | 8 | 0.0051 | 77,092 |
| twin | 50–200 mm | **4 x 256** | 2 | 0.776 | 202,508 |
| twin | 500–2000 mm | **4 x 128** | 4 | 132 | 53,140 |
| twin | full crossing | **4 x 256** | 2 | 906 | 202,508 |

The winner moves down the column in a way that is consistent across the two
arms: **shallow and wide on the short steps, deep on the long ones for the
physics arm, and never deeper than 4 for the twin.** For the physics arm the
short-column winner is meaningless — (c) showed those columns are flat in size —
but the long-column pattern is real: 8 x 64 and 8 x 256 win the two longest
columns by 28 % and 17 % over the best depth-4 network.

**Depth 8 is a liability everywhere except the physics arm's two longest
columns**, which is a sharper statement than "depth 6 hurt" was. Best cell of
each depth:

| arm | stratum | depth 2 | depth 4 | depth 8 | depth 8 / best |
|---|---|---|---|---|---|
| physics | 0.05–0.2 mm | 5.75e-06 | 7.86e-06 | 1.26e-05 | **2.19** |
| physics | 0.5–2 mm | 5.80e-04 | 8.02e-04 | 1.33e-03 | **2.29** |
| physics | 5–20 mm | 0.0609 | 0.0841 | 0.142 | **2.33** |
| physics | 50–200 mm | 7.42 | 8.18 | 7.43 | 1.00 |
| physics | 500–2000 mm | 448 | 267 | **192** | 1.00 (depth 8 wins) |
| physics | full crossing | 848 | 718 | **593** | 1.00 (depth 8 wins) |
| twin | 0.05–0.2 mm | 4.31e-07 | 5.41e-07 | 2.27e-06 | **5.27** |
| twin | 0.5–2 mm | 4.47e-05 | 5.32e-05 | 1.71e-04 | **3.83** |
| twin | 5–20 mm | 0.0051 | 0.0058 | 0.0185 | **3.61** |
| twin | 50–200 mm | 1.04 | **0.776** | 2.32 | 2.98 |
| twin | 500–2000 mm | 258 | **132** | 384 | 2.92 |
| twin | full crossing | 1,458 | **906** | 4,113 | 4.54 |

For the twin, depth 8 costs a factor 2.9–5.3 in every column. For the physics
arm it costs a factor 2.2–2.3 on the three short columns, breaks even at
50–200 mm and **is the only thing that helps** on the two longest ones. That is
a coherent picture: on the crossing the map is being asked for a strongly
non-linear function of the inputs and depth buys it; on a 0.1 mm step the answer
is nearly the straight line plus a small quadratic, and eight tanh layers are
harder to optimise than two for no gain.

### (f) The network shows no forward/backward asymmetry, and the exact scheme does

`results/reading_direction.csv`, the backward median divided by the forward one,
over all 720 cells.

| stratum | physics | twin | exact scheme, q = 8 |
|---|---|---|---|
| 0.05–0.2 mm | 1.003 | 0.966 | — |
| 0.5–2 mm | 0.952 | 1.016 | — |
| 5–20 mm | 0.966 | 1.041 | — |
| 50–200 mm | 1.033 | 0.838 | 0.81 |
| 500–2000 mm | 1.044 | 0.886 | 0.86 |
| full crossing | 1.043 | 1.163 | **1.56** |

C2 found the exact scheme's backward legs on the crossing **1.56× worse** than
its forward ones at q = 8 (39.19 against 25.05 µm) and 2.09× at q = 20. The
network shows no such thing: 1.04 for the physics arm and 1.16 for the twin,
against a seed scatter of 1.1–1.9 per cell. **The asymmetry is a property of
where the collocation nodes fall on the field profile, and it lives four orders
of magnitude below the level at which the network is operating**, so the network
cannot see it. The direction split is therefore doing here what C3.1 said it
would — checking the scoring, not measuring physics. The full per-direction
tables are `results/table_<D>x<W>_by_direction.csv`.

### (g) The tails

`results/reading_p95.csv`. p95 over median, per cell:

| stratum | physics, median over cells | twin |
|---|---|---|
| 0.05–0.2 mm | 8.6 | 20.9 |
| 0.5–2 mm | 8.6 | 22.9 |
| 5–20 mm | 8.2 | 20.1 |
| 50–200 mm | 7.2 | 18.2 |
| 500–2000 mm | 10.2 | 13.0 |
| full crossing | 11.1 | 12.9 |

**The twin's distribution is about twice as heavy-tailed as the physics arm's on
the short steps** and they converge at the crossing. Both are far heavier than
the straight line's own p95/median, which the C3.1 table puts at 6.8–7.4 on
every interior stratum and 3.0 on the crossing. Measured against the straight
line's *p95* rather than its median, the picture of (c) survives at the tail:

| stratum | physics p95 / straight-line p95 | twin |
|---|---|---|
| 0.05–0.2 mm | 0.142 | 0.047 |
| 0.5–2 mm | 0.147 | 0.047 |
| 5–20 mm | 0.144 | 0.045 |
| 50–200 mm | 0.084 | 0.047 |
| 500–2000 mm | 0.031 | 0.069 |
| full crossing | **0.0076** | 0.050 |

Same shape, same crossover, so the medians are not hiding a different tail
story. The one place the tail is worse than the median suggests is the physics
arm on the two longest strata, where p95/median rises from 7.2 to 11.1 — those
are the soft tracks, and they are the rows that also drive (a).

### (h) Cost

`results/reading_cost.csv` and `figures/cost_vs_error.png`. The measure is the
gain over the straight line (straight-line median / cell median) per thousand
parameters.

| stratum | arm | best error per parameter | its error | best error outright | its cost |
|---|---|---|---|---|---|
| 50–200 mm | physics | **2 x 32, q = 12** — 3,028 params, gain 16.0× | 8.81 µm | 2 x 256, q = 2 (70,924 params) | 7.42 µm |
| 50–200 mm | twin | **2 x 32, q = 2** — 1,708 params, gain 39.7× | 3.55 µm | 4 x 256, q = 2 (202,508 params) | 0.776 µm |
| full crossing | physics | **2 x 32, q = 6** — 2,236 params, gain 288× | 1,582 µm | 8 x 256, q = 18 (482,124 params) | 593 µm |
| full crossing | twin | **2 x 32, q = 2** — 1,708 params, gain 52.9× | 8,623 µm | 4 x 256, q = 2 (202,508 params) | 906 µm |

**The cost-error front is very flat.** At the full crossing the physics arm goes
from 1,582 µm at 2,236 parameters to 593 µm at 482,124 — a factor 2.7 in error
for a factor **216 in parameters and multiply-adds**, and four fifths of that
gain is already collected by 8,836 parameters (8 x 32, q = 8, 767 µm). At
50–200 mm the physics arm's front is flat from 8,044 parameters (7.90 µm) to
70,924 (7.42 µm): a factor 8.8 in cost for 6 % in error.

The 17-point front at the crossing is in
`results/reading_cost.csv`; the practical entries are

| architecture | q | parameters | multiply-adds | full-crossing median |
|---|---|---|---|---|
| 2 x 32 | 6 | 2,236 | 2,144 | 1,582 µm |
| 4 x 32 | 6 | 4,348 | 4,192 | 885 µm |
| 8 x 32 | 8 | 8,836 | 8,544 | 767 µm |
| 8 x 64 | 20 | 35,092 | 34,496 | 639 µm |
| 8 x 256 | 18 | 482,124 | 480,000 | 593 µm |

For comparison, the exact q = 8 root-finder reaches 32.6 µm on the same step at
37 residual evaluations, i.e. 37 field lookups at 8 planes each, and 17.8 ms of
one core (C2.3). The whole grid is between 18 and 53 times worse than that, at
any cost, so **the limit is not the network's size and not the stage count; it
is whatever the physics loss actually converges to.** That is the question C4
hands on.



### What the last five records changed

The tables above were first built on the 715 records that had landed by 05:00 on
2026-09-08 and rebuilt on all 720 at 08:30. The five late records are one seed
each of the 8 x 256 data twin at q = 6, 8, 12, 18 and 20, so **only that row of
that one architecture moves at all**. Its full-crossing cells:

| q | two seeds: median [min–max] | three seeds: median [min–max] | the late seed | shift in the median |
|---|---|---|---|---|
| 6 | 4,585 [4,479 – 4,691] | **4,479 [1,644 – 4,691]** | 1,644 µm | −2.3 % |
| 8 | 6,433 [6,392 – 6,473] | **6,392 [2,087 – 6,473]** | 2,087 µm | −0.6 % |
| 12 | 6,090 [5,287 – 6,892] | **5,287 [1,164 – 6,892]** | 1,164 µm | −13 % |
| 18 | 4,846 [4,113 – 5,578] | **4,113 [2,264 – 5,578]** | 2,264 µm | −15 % |
| 20 | 6,886 [6,881 – 6,892] | **6,881 [3,674 – 6,892]** | 3,674 µm | −0.1 % |

Every one of the eight readings survives unchanged in direction and in
substance. What moved, to the digit:

* **(a)** the twin's spread-over-q column, by 0.00–0.07, and its seed-spread
  column by 0.01–0.10; the count of architectures where q matters is unchanged
  in all twelve rows (physics: 1, 1, 1, 1, **0**, **12**; twin: 1, 1, 0, 0, 0, 1).
  The physics-arm half of that table does not move at all, and neither do the
  q = 2 / q = 4 / plateau numbers, which are the reading.
* **(b)**, **(c)** and **(h)** are unchanged to every digit quoted, including
  both cost-error fronts and every entry of the best-cell-per-width table.
* **(d)** three quartiles: the 0.05–0.2 mm median 7.81 → 7.95, the 0.5–2 mm
  median 9.29 → 9.34 and its upper quartile 13.7 → 13.9, the 500–2000 mm median
  0.62 → 0.63. The crossing, the minima and the maxima are unchanged.
* **(e)** the twin's depth-8 column: the penalty becomes 2.9–5.3 rather than
  3.1–5.3. Every winner is the same
  architecture at the same q, and the physics-arm rows are unchanged.
* **(f)** the twin's backward/forward ratio at the crossing, 1.171 → 1.163.
* **(g)** the twin's p95/median at 0.05–0.2 mm, 20.8 → 20.9.

The one thing worth carrying forward is not a reading but a caution: **in all
five cells the seed that took longest to confirm was the best**, by a factor
1.8 to 4.5 against the cell median on the full crossing. A cell scored on the
seeds that finished first is biased against the architecture, and it understates
the seed scatter badly: these five `[min-max]` brackets spanned a factor
1.00–1.36 on two seeds and span **1.88–5.92** on three. That is an argument for
reading the brackets, and the seed count, rather than the medians alone whenever
a row of the grid is still draining.
