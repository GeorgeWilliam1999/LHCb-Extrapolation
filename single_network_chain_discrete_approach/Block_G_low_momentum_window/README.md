# Block G — the momentum window in the loss moved to the region around 5 GeV

**Status (2026-09-21, late evening): built and gated, nothing trained.** The windowed loss,
the trainer, the farm harness and the analysis scripts are in place and verified; the window
and the pre-registration are with George (`PLAN.md` §6–7). No farm job has been submitted.
Nothing is committed. The plan and worklog live in the Notion to-do "Retrain the chained
network with the loss weighted for tracks around 5 GeV"; `PLAN.md` is the working copy and
`HANDOFF_PROMPT.md` George's brief.

## The one change

Block F's loss (`../Block_F_reweighted_loss/`) measures every residual as the displacement it
would cause at the SciFi plane, as a fraction of the track's own bend, with a momentum window
`W(p)` = 1 on 10–50 GeV. Block G moves that window down (anchor candidate **3–8 GeV**, George's
region of interest) and changes nothing else: same tracks, network, seed, optimiser, rounds,
clamp, lever weighting, roll-off and floor.

Block F's `per_track_factor` called `band_window(p)` with the module defaults and ignored the
run's stored `p_lo`/`p_hi`; a Block G run on that path would have trained silently on 10–50 GeV.
Block G's `windowed_loss.py` imports Block F's module and re-implements only the three
functions that need the run's constants. Block E, Block F and `_shared/` are read-only.

## Layout

| folder | what it holds | state |
|---|---|---|
| [G0_Weighting/](G0_Weighting/) | `windowed_loss.py`; `check_windowed_weights.py` (gates W1–W3 and the pre-flight W4 over six candidate windows on Block F's trained N = 64, q = 2 network) | all gates pass; pre-flight in its README |
| [G1_Training/](G1_Training/) | `train_windowed.py` (Block F's trainer with `--p-lo/--p-hi`, diff in `results/`); `condor/` harness with the resubmit fix | gates I1–I3 pass; dry-run only |
| [G2_Analysis/](G2_Analysis/) | the judging scripts for the new networks: around 5 GeV signed and with the > 1 mm tail removed, per-component tables (5–30 and 3–8 GeV), momentum bands, two-slope anatomy, convergence by the plateau rule | smoke-tested on Block F's runs; byte-identical reproduction |
| [G3_Analysis/](G3_Analysis/) | E3's figure set driven on the new networks | smoke-tested, all nine analyses run |
| [worklog/](worklog/) | dated agent worklogs (mirrored into the Notion to-do) | |

Outputs name every network by N, q and dz and say whether an error is an endpoint error
(after the full chain, at z1) or a single-step one; no block letters, no comparison rows.

## What the pre-flight says (G0, 2026-09-21)

On Block F's trained N = 64, q = 2 network, 8,000 states, the loss share inside 3–8 GeV with
the most extreme 1 % of states removed: 61 % for a 3–8 window, 59.5 % (3–10), 57 % (3–20),
56.5 % (2–15), 67 % (4–6, but only a quarter inside its own band), 49 % for the present 10–50
window. Raw, one state carries 41–69 % of the objective whichever the window, so the window is
a modest lever on a heavy-tailed residual. Recommendation to George: 3–8 GeV, with 3–20 as the
fallback if the 10–50 GeV cost matters more. Full tables in `G0_Weighting/README.md`.

## To run (after George's decision)

```bash
cd G1_Training
PY=/data/bfys/gscriven/conda/envs/TE/bin/python; export PYTHONNOUSERSITE=1
$PY condor/make_jobs.py --p-lo 3 --p-hi 8            # three lines into results/p03-08/
condor_submit condor/jobs.sub
cp condor/jobs.txt condor/jobs_active.txt
tmux new-window -t claude-phone -n blockG-keeper "$PWD/condor/keeper.sh"
```

Expected wall at Block F's measured medians: 22 h (N = 64, q = 2), 33 h (N = 128, q = 8),
50 h (N = 256, q = 16). Convergence is judged by `G2_Analysis/convergence.py` (the plateau rule
of `../Block_F_reweighted_loss/F2_Analysis/compare_to_blockE.py`), never by the loss.
