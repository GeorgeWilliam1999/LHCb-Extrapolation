# G2 — judging the trained networks, in isolation

The five scripts that turn a folder of finished runs into the numbers the study is judged on.
Each one reads run folders and writes `results/` and `figures/` beside itself. Nothing here
trains anything, and nothing here writes outside its own `--out` folder.

Every script takes the same two flags:

| flag | default | meaning |
|---|---|---|
| `--runs DIR` | `../G1_Training/results/p03-08/full` | the folder holding the `N<NNN>_q<qq>` run folders |
| `--out DIR` | this folder | where `results/` and `figures/` are written |
| `--only A,B` | all runs | restrict to named run folders (not accepted by `error_tables_by_component.py`, see below) |

Python, single-threaded, as everywhere in this project:

```
cd G2_Analysis
PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python errors_near_5gev.py
```

## Which script produces what

| script | what it answers | outputs |
|---|---|---|
| `errors_near_5gev.py` | the primary measure: the signed endpoint deviation around 5 GeV, per component, with a tail-robust width beside the RMS, and how much of the RMS is the beyond-1 mm tail | `results/errors_near_5gev.csv`, `results/tail_near_5gev.csv`, `figures/signed_errors_4-6GeV.png` |
| `error_tables_by_component.py` | the endpoint error per state component, one table per component, over 5–30 GeV and over 3–8 GeV | `results/error_by_component_5-30GeV.csv` and `_3-8GeV.csv`, `figures/error_by_component_5-30GeV.png` and `_3-8GeV.png` |
| `momentum_bands.py` | the endpoint error against momentum, band by band — the secondary measure, what the moved window costs at 10–50 GeV | `results/momentum_bands.csv`, `figures/momentum_bands.png` |
| `anatomy_xy.py` | where the endpoint error is made: position errors, slope errors carried over their lever arms, x against y | `results/anatomy_xy_<N064_q02>.json`, one per network |
| `convergence.py` | has each run stopped improving, and what is its headline number | `results/convergence_check.csv`, `figures/convergence_check.png` |
| `run_discovery.py` | not a script: the shared helper the others import (which run folders are analysable, the checkpoint flag, the loss window, and the wording of every network's name) | — |

### `errors_near_5gev.py`

Per network and per component (x, y, tx, ty), on the test tracks, in the bands **3–5, 4–6,
5–7, 5–8 and 10–20 GeV**: `n`, median |d|, RMS, standard deviation, signed mean and median,
the 68 % half-width (half the 16th–84th percentile range) and the 95th percentile of |d|.
Those columns are computed by the `stats` function imported from the earlier block's script, so
they are the same arithmetic, not a retyping of it.

Added here: `rms_no_tail` (the RMS with the tracks more than 1 mm from the RK6 endpoint
removed) beside the full `rms`, the tail fraction, and the median |x₀| — the distance of the
track's starting point from the beam line on the last UT plane — of that tail against the rest
of the band. `tail_near_5gev.csv` carries the same tail numbers one row per network and band,
with the radial median and the radial RMS with and without the tail.

### `error_tables_by_component.py`

A driver. The table is the earlier block's script, imported by path with its output folder and
its run folder pointed here, and called once for 5–30 GeV and once for 3–8 GeV. `--only` is
refused rather than silently ignored, because the imported code reads every run folder under
`--runs`; to table one network, point `--runs` at a folder holding only that run.

### `convergence.py`

The rule is imported (`plateaued_now`) so there is only one copy of it: **the median validation
error of the last ten rounds no more than 5 % below the median of the ten before, at each of the
last three rounds.** The headline is the median of the last ten rounds with its round-to-round
spread, never the final checkpoint. The script also prints the ten-round medians round by round
and the last-ten-over-previous-ten ratio.

**It de-duplicates.** A run that two farm jobs have been training at once has the same round
number twice in `rounds.csv` and the same restart number twice in `history.csv`. Both files are
de-duplicated by number, keeping the last row written for each number and sorting by it; what
was dropped is printed and recorded (`dup_rounds`, `n_dup_rounds`, `dup_restarts_from`,
`n_dup_restarts`, `interleaved`). Anything after the first duplicate in such a run is two jobs'
work interleaved and should be read as a checkpoint.

## Caveats

- **Endpoint, not single step.** Everything in G2 is the error after the network has been
  applied N times from the track's real state on the last UT plane, at the first SciFi plane
  z1 = 7,826 mm. The single-step numbers (one application from an RK6 state) are G3's.
- **`anatomy_xy.py` recomputes from `network.pt`**, not from the stored `chain_states.npz`. For
  a run still training the two disagree, because the weights on disk have moved on since the
  states were stored. Its `final_band_med_um` and `local_step_band` are for **10–50 GeV**, a
  fixed reference band inherited from the imported code, not for this study's loss window.
  About 30–40 s a network.
- **Checkpoints.** A run that hit its restart cap and was reopened has a `record.json` that is
  a checkpoint; `progress.json`'s `phase` is what says so, and every table and legend marks it.
  Its numbers are a snapshot, not a result.
- **RMS against the 68 % half-width.** Around 5 GeV a few percent of the tracks land beyond
  1 mm and carry most of the RMS; they start at the edge of the acceptance. Read the RMS and
  the half-width side by side, and the no-tail RMS beside both.
- **In isolation.** No script here draws or tabulates another study's networks.
