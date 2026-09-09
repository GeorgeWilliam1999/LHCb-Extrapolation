# Block C — the error as a function of step length and stage count

7–8 September 2026. This block exists to answer a question my supervisors asked
directly: for a range of network widths and depths, give the error
`error(dz, q)` as a table — stage count `q` from 2 to 20 against step length
`dz` from 0.1 mm to the whole magnet crossing — measured against a
super-fine, high-order reference **and** against the real simulated hits.

Read the folders in order; each consumes the one before it.

| Folder | Step | What it produces |
|---|---|---|
| [`C0_Magnet_tracks_dataset/`](C0_Magnet_tracks_dataset/) | C0 | The data set: magnet-to-magnet steps of every length, both directions, cut to the LHCb acceptance 2 < η < 5, on the magnet-**up** map the official sample was actually generated with. 60,000 rows. |
| [`C1_Fine_reference/`](C1_Fine_reference/) | C1 | The sixth-order reference integrator at 0.1 mm and the evidence for how far it can be trusted — its own accuracy over a crossing is 5e-5 µm. |
| [`C2_Exact_scheme_table/`](C2_Exact_scheme_table/) | C2 | The collocation scheme solved *exactly*, no network: the ceiling any network of a given `q` is working under. |
| [`C3_Step_size_and_stage_grid/`](C3_Step_size_and_stage_grid/) | C3, C4 | The grid itself — 4 widths × 3 depths × 10 stage counts × 2 losses × 3 seeds = **720 trained networks** — and the twelve single-step tables plus their eight readings. |
| [`C5_MC_hit_comparison/`](C5_MC_hit_comparison/) | C5 | The same predictors scored against the particles' real simulated hits, which brings in the material the extrapolator does not model. |
| [`C6_Chained_crossing/`](C6_Chained_crossing/) | C6 | Every one of the 720 networks walked across the whole magnet at each step length, per component, with the growth law and the ranked-network figures. |

There is no C4 folder: C4 is the *reading* of the C3 grid and lives in
`C3_Step_size_and_stage_grid/tables.py` and its `reading_*.csv` outputs.

## What the tables say

**Single steps (C3/C4).** Stage count only matters on the whole crossing, and
only for the label-free loss: at q = 2 and q = 4 the network sits on the
scheme's own discretisation error, and from q = 6 it is on its own floor of
593–1745 µm and stays there to q = 20 — 18 to 54× above the exact scheme's
32.6 µm ceiling. The floor goes as `dz²`. The short columns are limited not by
capacity but by how much the loss weights them: the label-free residual is
normalised by one population-wide input scale, so a 0.1 mm row contributes
about 4e-10 of a crossing row. Depth 8 hurts the short steps and helps the
crossing. The cost–error front is flat: 8,836 parameters already collect four
fifths of the available gain.

**Chained crossings (C6).** Chaining never beats one step. 713 of the 720
networks are worse at 0.1 mm than in a single step, by a median factor of 105,
and **not one** of the 720 is at its best at 0.1 mm. The growth exponents — 1
for the slope, 2 for the position — identify the cause as the same relative
bend error repeating coherently at every step of a track, not a random walk,
which would have given 0.5 and 1.5. The error is a spread rather than a shared
offset, so no constant calibration removes it. Best single step anywhere:
549 µm, the 8×128 label-free network at fourteen stages.

**Against real hits (C5).** The material floor — the part of a crossing no
field-only method can predict — is 0.11 / 0.57 / 7.5 / 51 / 1815 µm across the
step-length bands. Below about 100 mm every predictor, including the straight
line, returns the same number to three digits, so hit-level validation cannot
see the network there at all. On the crossing the label-free arm is 1.3× the
floor on soft tracks but 2.4–6× on 20–200 GeV tracks, which is where a better
extrapolator would actually show up in a fit.

## What this points at next

The clearest actionable finding is the normalisation. The supervised twin
divides each row's residual by that row's *own* bending-integral scale, so every
row contributes an order-one target whatever its length; the label-free loss
divides by one population-wide scale. That single difference explains the flat
short columns, the fitted exponent of 1.86 against the twin's 2.09, and why the
twin chains an order of magnitude better. Applying the per-sample normalisation
to the label-free loss costs nothing — the scale is already computed for every
row — and is the obvious next experiment. **It has not been run.**

## Write-ups

Published in Notion, Trust = Provisional, every table cell machine-transcribed
from the CSVs in these folders:

- Part 1, single steps — https://app.notion.com/p/3d55d544b9d981178af8d235c20bdf5c
- Part 2, chained crossings and the components — https://app.notion.com/p/3d55d544b9d98161b1c2dcc29b974eba

## Reproducing

Each folder's own README carries the script → output map. The two farm steps:

```bash
cd C3_Step_size_and_stage_grid && python make_jobs.py && condor_submit condor/jobs_grid.sub   # 720 runs
cd C6_Chained_crossing        && python make_jobs.py && condor_submit condor/jobs_chain.sub   # 720 walks
```

`make_jobs.py` writes the folder's path relative to `Raissi_disc_time_approach/`
into each job line, which is what `_shared/condor/wrapper.sh` resolves against.
The `.npz` and `.pt` files are not committed; the scripts, the recorded seeds
and the committed metadata regenerate them.
