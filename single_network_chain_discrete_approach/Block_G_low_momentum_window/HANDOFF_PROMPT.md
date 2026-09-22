# Block G — retrain the chained network with the loss weighted for tracks around 5 GeV

Read `/data/bfys/gscriven/CLAUDE.md` first; Notion is the source of truth and the to-do
"Retrain the chained network with the loss weighted for tracks around 5 GeV" is where the plan
and the worklog go. Discuss the plan with me before building; results in chat before any
write-up; never commit or push; never label outputs with block letters.

## Where we are

`/data/bfys/gscriven/LHCb_Extrapolation_Project/single_network_chain_discrete_approach/`

- `Block_E_single_network_chain/` — one network per step length (N = 2…256 steps of
  dz = L/N across the 5,178 mm magnet crossing, q Gauss–Legendre stages), applied to itself N
  times, trained label-free with the discrete RK-PINN loss. E0 = tracks (`E0_Track_dataset/
  results/tracks.npz`, 11,567 / 1,463 / 1,452 train/val/test, RK6 at 0.1 mm), E1 = network +
  round trainer, E3 = analysis. Sixteen networks, extended to their plateau on 2026-09-19/20.
- `Block_F_reweighted_loss/` — the same networks with ONE change: the weights inside the loss
  (`F0_Weighting/weighted_loss.py`). Every residual is measured as the displacement it would
  cause at the SciFi plane, as a fraction of that track's total bend, and a momentum window
  `W(p)` = 1 on **10–50 GeV** with a Gaussian roll-off in log p (width ln 2) outside, floored
  at 0.05; per-track factor clamped to [1/5, 5] of its median. Three runs, N = 64 q = 2,
  N = 128 q = 8, N = 256 q = 16 (`F1_Training/results/full/`); N = 256 q = 16 may still be a
  checkpoint of a run in training — check `progress.json` phase, not the existence of
  `record.json`. F2 has the comparison and judging scripts, F3 the full E3 figure set.
- Result of Block F on the test tracks, endpoint after the full chain, N = 64 q = 2: 89 µm
  radial overall, 25 µm in 10–50 GeV, but **370 µm at 2–5 GeV** and at 4–6 GeV: |Δx| median
  93 µm, 68% half-width 146 µm, RMS 618 µm; |Δy| 130 / 223 / 425 µm; tx 0.078 / 0.103 / 0.284
  mrad; ty 0.095 / 0.161 / 0.329 mrad (`F2_Analysis/results/errors_near_5gev.csv`). The RMS is
  six times the median because 6–7% of that band is beyond 1 mm; those tracks start far from
  the beam line (median |x₀| 190–230 mm vs 114 mm). Signed medians are within ±30 µm: no offset.

## Why Block G

My supervisor's reply (2026-09-21): the region that matters is **around 5 GeV** — the expected
deviation there (RMS in x, or |Δx|), and the signed per-component distributions. Block F's
window was set to 10–50 GeV and rolls off below 10, so the 5 GeV core is seven times wider
than at 10–20 GeV. Block G moves the window.

## The task

Block G = Block F with the momentum window moved, and nothing else changed. Same tracks,
network, seed, optimiser, rounds, states per round, clamp, lever weighting.

1. **The one code trap.** In `weighted_loss.py`, `per_track_factor` calls `band_window(p)` with
   the module defaults `P_LO`/`P_HI`, not `const["p_lo"]`/`const["p_hi"]`, while the numpy twin
   in `check_weights.py` reads the constants. Make the torch path honour the run's constants
   (pass `p_lo`, `p_hi`, `rolloff`, `floor` from `const`), add `--p-lo`/`--p-hi` to
   `train_weighted.py` (stored in `scale.json["weighting"]`), and prove with the twin gate and
   the isolation gate (`--weighting blockE` still bit-identical to Block E; the default window
   still reproduces Block F's round-1 losses bit for bit) that nothing else moved. Block F's
   folders are read-only; Block G imports from them.
2. **Propose the window to me before training.** My default: 3–20 GeV (keeps 10–20 in and
   centres on 5); alternatives 3–10 or 2–15. Run the F0 pre-flight (`check_weights.py` gate 4)
   for each candidate on Block F's trained N = 64 q = 2 network: loss share by momentum band,
   rank correlation of loss share with true endpoint cost inside the band, and the share taken
   by the >100 GeV tail — that is how the 10–50 window and the clamp were chosen.
3. **Pre-register the judgement**, then run the three settings on HTCondor (Block F's
   `make_jobs.py`/`resubmit.py`/`keeper.sh` pattern, ~20 h each, keepers in tmux `claude-phone`).
   Primary: at 4–6 GeV on the test tracks, the 68% half-width and median |Δx|, |Δy| and the
   signed distributions per component, against Block F's numbers above. Secondary: what it
   costs in 10–50 GeV. Convergence: the `plateaued_now` rule in
   `Block_F_reweighted_loss/F2_Analysis/compare_to_blockE.py` (validation median of the last ten
   rounds no more than 5% below the ten before, three rounds running) — the trainer's own 1%
   rule never fires at N ≥ 64 with the rescaled loss.
4. **Analysis in isolation**: reproduce `F2_Analysis/errors_near_5gev.py`,
   `error_tables_by_component.py` (5–30 GeV per component) and the F3 set for the new networks;
   name networks by N, q, dz; no block labels, no comparison rows unless I ask.
5. Report in chat with the numbers and the figures; log everything dated in the to-do body.

## Things to keep in mind

- The tail beyond 1 mm at 5 GeV comes from tracks starting at the edge of the acceptance; the
  momentum window will not touch it. If it dominates the RMS after Block G, the next lever is a
  weight on starting |x|, which is a separate block.
- What remains of Block F's error is in the y plane (y and ty larger than x and tx above
  5 GeV; two-slope anatomy in `F2_Analysis/anatomy_xy.py`). Report y and ty separately.
- Memory notes for this work: `~/.claude/projects/-data-bfys-gscriven/memory/
  lhcb-extrapolator-block-a.md`, `outputs-no-block-labels.md`, `supervisor-interest-5gev.md`.
