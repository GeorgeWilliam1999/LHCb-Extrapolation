# Fast extrapolator

Question: what is the cheapest method that carries a track across the magnet
(2,648 mm to 7,826 mm) with a median error of about 1 um, and can it replace
what LHCb runs now?

Status (2026-11-24): the plan is agreed (decisions 1 to 7). Stage 0 (the compiled
program and the LHCb baselines) and stage 1 (what is possible with no training) are
done and reported. Stage 2 waits for George: the proposal is to train only a
network with no field read and a coarse integrator with a network correction. No
farm job is submitted and no network is trained. See PLAN.md for the log.

| Folder | What it holds |
|---|---|
| `cpp/` | the compiled program `fx`: field reads, classical integrators, copies of the LHCb integrators, the forward pass of the networks, a start plus a correction, the timing loop |
| `python/fastex/` | the export of maps, tracks and weights to flat files; Python copies of the LHCb integrators; the calls of `fx` for scoring and timing |
| `lhcb_software/` | the extrapolator test of Rec, run in the LHCb software, and its output |
| `stage_0_baselines/` | stage 0: what LHCb, a field read and a network call cost (README, RESULTS.md, notebook) |
| `stage_1_no_training/` | stage 1: the classical frontier and the floor of each layout |
| `stage_2_training/` | stage 2, in preparation: what a correction must learn, the polynomial baseline, the draft of the design (nothing is trained) |

This folder is separate from the present package, experiments and run store.
The present package is read and never edited. The flat files of this experiment
are in /data/bfys/gscriven/fast_extrapolator_store (`exports/`, `scratch/`).

The numbers that started this experiment are in
../experiments/how_long_does_one_track_take/README.md.
