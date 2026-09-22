# Plan: the Block F write-up (hand-off to an Opus agent, 2026-09-22)

Read first: `../../_shared/WRITEUP_RULES.md`, then `../README.md`, `../F0_Weighting/README.md`,
`../F1_Training/README.md`, `../F2_Analysis/README.md` (all sections, including the 2026-09-22
addition on the true state), `../F3_Analysis/README.md`, the docstrings of
`../F0_Weighting/weighted_loss.py`, `../F2_Analysis/compare_to_blockE.py`,
`../F2_Analysis/anatomy_xy.py`, `../F2_Analysis/errors_near_5gev.py`,
`../F2_Analysis/against_true_state.py`, and the Notion to-do
https://app.notion.com/p/3dd5d544b9d98144bdcbd17c5a66e3e0 (entries from 2026-09-18 evening on).
The Block E write-up is being written in parallel by another agent; refer to it as "the companion
write-up on the single-network chain" and leave a placeholder link "[link to be filled by George]"
if its URL is not in `../../Block_E_single_network_chain/E4_Writeup/README.md` when you finish.

**Title:** "Reweighting the label-free loss by what an error costs at the SciFi plane: the same
chained network, three step lengths, and where the remaining error lives (September 2026)"

**Todos relation:** ["https://app.notion.com/p/3dd5d544b9d98144bdcbd17c5a66e3e0",
"https://app.notion.com/p/3e25d544b9d981849f9cf8817f6bb379"]

**Note:** one paragraph: one change to the previous study, the weights in the loss (each residual
as the displacement it causes at the SciFi plane as a fraction of the track's own bend, the
10–50 GeV band weighted up); three networks; endpoint radial 88.8 / 104.6 / 92.2 µm against
145.7 / 122.2 / 147.7 for the same networks under the pooled loss; 24.8 / 27.3 / 34.6 in
10–50 GeV; the price is below 5 GeV and in the y slope; against the Geant4-true state the
networks and the field-only reference are indistinguishable.

## Sections

### Intro
Roadmap; what the previous study found (the chain error is accumulation of early slope error;
97 % of the loss from 2–5 GeV; 0.6 % of attention on the first quarter of the crossing); the
question: if the loss is asked for the right thing, how much of the error goes away? Folder
sentence. The reference sentence (field-only RK6; link to the reference-versus-truth write-up).

### Aims
The pre-registered rules verbatim from `../README.md` "What would count as a result" (success /
partial / null, the plateau rule applied to both sides, the ablations held back). Define the
plateau rule.

### Method (numbered)
1. **What is kept fixed.** Same tracks and splits (14,482 crossings, 11,567 / 1,463 / 1,452), same
   network (imported, not copied), same seed, same L-BFGS protocol, same rounds (25 restarts),
   same field map. The isolation gate F-1 (bit-identical losses with `--weighting blockE`, the
   command and the result).
2. **The weight, derived.** Start from the pooled loss; write the new loss exactly as in
   `../README.md` (the code block) and derive each factor: the lever arm (distance left to z1 plus
   one step; why additive), the track factor a_n = √W(p)·D_ref/D_n with D_n = κ|q/p|·Ī·L the
   track's total bend (why a constant of the track, not the step), the window W(p) (1 on
   10–50 GeV, Gaussian roll-off in log p, floor 0.05), the clamp [1/5, 5] of the batch median;
   both factors come from the input state and the field alone, so the loss stays label-free.
   Worked micro-example with two states.
3. **Why not weight by the field** (`F0_Weighting/field_correlations.py`,
   `results/field_correlations.json`): slope error vs |B| −0.57 (Spearman), cost vs distance
   left +0.996, cost vs |B| +0.12.
4. **The gates** (F0 README table): the four gates with their numbers; the torch/numpy median trap.
5. **The pre-flight and the clamp** (F0 README tables; `figures/weighting_preflight.png`): share of
   the loss by band and by quarter of z for each weighting mode; ρ(share, cost); the clamp scan;
   the 83.8 % top-1 % concentration that no weight removes.
6. **Training and the farm** (F1 README): three runs, caps 40 rounds / 1,000 restarts, cluster
   5809660, the keeper; the extension of N = 256 q = 16 to 1,500 restarts / 60 rounds (clusters
   5823724 and 5833416, two jobs on one run from restart 1,198, the interleaved history and how
   the convergence check de-duplicates it: `../../Block_G_low_momentum_window/G2_Analysis/convergence.py`);
   wall time per restart and per run (F2 headline.csv train_wall_h: 21.6 / 32.8 / 100.4 h).
7. **Stopping.** Plateau verdicts from `F2_Analysis/results/headline.csv` (plateaued_now,
   last10_vs_prev10_pct) and the de-duplicated headline (89.3 ± 12 %, 107.0 ± 11 %, 86.4 ± 26 %
   validation, from the G2 convergence run of 2026-09-22; cite the to-do worklog entry and rerun
   the script into `F4_Writeup/results/` to have the CSV beside the page).
8. **Scoring.** As E3, reproduced through `F3_Analysis/run_e3_for_block_f.py` (imports, not
   copies); the grid-free panels; F2's own measures: compare_to_blockE (same rule both sides),
   components_and_momentum, error_tables_by_component (5–30 GeV, endpoint), errors_near_5gev
   (signed, RMS, 68 % half-width), anatomy_xy (both slopes), against_true_state, mirror_mini_paper.
   Define each statistic.

### Results (all on the finished weights of 2026-09-22; every file in F2/F3 results/ is current)
- Headline table: three networks × (validation headline ± spread, test radial median, 10–50 GeV,
  p95, per-step slope errors, wall hours). From headline.csv and the G2 CSV.
- The two pre-registered criteria for N = 64, q = 2: both hold (numbers, F2 README 2026-09-20).
- Per component (components.csv) with signed medians; the reweighting bought x (110.7 → 37.5 µm)
  and left y (52.3 → 55.2); comparison with the pooled-loss counterpart is allowed HERE as a table
  (this is the study's question), but figures must not carry block letters: use
  `error_by_component_5-30GeV.png` and `signed_errors_4-6GeV.png` (already relabelled), regenerate
  `blockF_vs_blockE.png` through a title wrapper if used, or replace it by a table.
- Momentum bands (momentum_bands.csv): the price below 5 GeV (373 vs 242 µm at 1–5 GeV) and the
  gain above 10 GeV.
- 5–30 GeV per component, endpoint (error_by_component_5-30GeV.csv): x 20.1 / 31.1 / 19.0, y
  31.5 / 33.3 / 40.9 µm, tx 0.019 / 0.022 / 0.014, ty 0.023 / 0.020 / 0.020 ×10⁻³.
- Around 5 GeV, signed (errors_near_5gev.csv): the 4–6 GeV table for all three networks (median
  |Δ|, 68 % half-width, RMS, signed median), the seven-fold width against 10–20 GeV, why RMS and
  median disagree (the 6–7 % tail beyond 1 mm; |x0| 189 vs 114 mm; RMS without the tail 184 µm),
  the 68 % half-width as the honest single number.
- Two-slope anatomy (anatomy_xy_*.json): per-step tx/ty errors, x part / y part at z1, coherence;
  "the reweighting bought x and cost ty".
- The F3 set: single step 0.43 / 0.22 / (N = 256 value from error_qdz_single_step.csv) µm, chain /
  single step, growth along z (quarter / half / end), no over-training, exact scheme 0.1 µm,
  starting-x dependence (case_study_error_vs_p_x0.csv), the worst 5 %. Figures from F3
  (`error_vs_z.png`, `single_step.png`, `error_vs_p_*.png`, `case_study_components_x0_maps.png`,
  `convergence_grid.png`); F3's titles were relabelled "Block F" by its runner, so regenerate the
  ones you use through a wrapper that drops the block word (write `F4_Writeup/make_figures.py`
  modelled on the F3 runner, output `F3_Analysis/figures_writeup/`).
- Against the Geant4-true state (against_true_state.csv / .png; F2 README 2026-09-22 section):
  the three-column table per band; the 0–4 % statement; what "true" means (MCHit; scattering and
  energy loss in, no detector resolution); defer the interpretation to the reference write-up
  with a link.
- The mirror figures (mirror_fig2_baseline.png, mirror_fig6_magnet_up.png) with their
  panel-by-panel mapping from the docstring; the charge-sign check.
- Cost per track (cost_accuracy.csv): 657 / 1,322 / 3,655 µs, fp64, one thread.

### Conclusion and next steps
What the reweighting did and did not do; the ablations (no_lever / no_track / no_window) not run
and when they would be; the window move to 3–8 GeV (the follow-up to-do, its pre-flight numbers if
you cite them come from `../../Block_G_low_momentum_window/G0_Weighting/results/preflight_windows.csv`);
supervising the step Jacobian as the lever after that; the heavy tail (starting-|x| weight) as a
separate lever; the reference question (link).

### Provenance
As in the rules: every F0/F1/F2/F3/G2 script by path, clusters 5809660 / 5823724 / 5833416,
restarts and rounds per run, the tracks file and its counts, the date each results file was
produced (F2/F3 2026-09-22 16:27–16:43).

## Practicalities
Save `F4_Writeup/page.md` (+ children), `F4_Writeup/README.md` with URLs and verification;
`F4_Writeup/numbers.py` for any computed number; `F4_Writeup/results/` for the G2 convergence CSV
rerun.
