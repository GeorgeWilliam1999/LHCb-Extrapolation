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
```

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
