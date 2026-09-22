# Plan: the Block E write-up (hand-off to an Opus agent, 2026-09-22)

Read first: `../../_shared/WRITEUP_RULES.md`, then `../README.md`, `../E0_Track_dataset/README.md`,
`../E1_Network_grid/README.md`, `../E2_Comparators/README.md`, `../E3_Analysis/README.md`, and the
Notion to-do "Train one network per step length and chain it to itself across the magnet"
(https://app.notion.com/p/3dd5d544b9d98144bdcbd17c5a66e3e0): its body is the dated worklog with
George's decisions (2026-09-16), the gates, the training history, the stopping lesson and the
analysis. Also fetch the Block D write-up (https://app.notion.com/p/3dc5d544b9d9813fa4a1dfba8da3cc98)
for the conventions and for what this study follows.

**Title (no block letters):** "One network per step length, chained to itself across the LHCb
magnet: a label-free Runge–Kutta network at every step count and stage count (September 2026)"

**Todos relation:** ["https://app.notion.com/p/3dd5d544b9d98144bdcbd17c5a66e3e0"]

**Note (abstract):** one paragraph: one 2×128 network per (step length, stage count), applied N
times from the last UT plane to the first SciFi plane, trained label-free on the discrete-time
Runge–Kutta residual over the whole crossing; 16 networks; endpoint 118–166 µm radial at N ≥ 64
(name the file), a single step under a micrometre; the error is early slope error carried to the
end; the loss is 97 % about 2–5 GeV tracks; convergence must be read from the validation error,
never the loss.

## Sections and what goes in each

### Intro
- The problem in one paragraph for a reader with no background: a charged particle crosses the
  LHCb dipole between the UT and the SciFi (z0 = 2,648.2 mm to z1 = 7,826.0 mm, L = 5,177.8 mm);
  the track state (x, y, tx, ty, q/p) must be carried across; the classical answer is numerical
  integration of the equation of motion in the measured field map.
- What came before and why this study: the previous study (Block D write-up, link) trained a
  different network for every step of a chain; George asked on 2026-09-16 for ONE network of
  step dz applied N times (precedent Wang & Perdikaris, arXiv 2106.05384; the paper's network
  takes the state and the start plane, not dz). One sentence mapping the study to the folder
  `single_network_chain_discrete_approach/Block_E_single_network_chain/`.
- The reference this study is measured against: the field-only RK6 track. State plainly that it
  contains no material effects and link the companion write-up on the reference versus the
  Geant4 truth (the agent writing that page will report its URL; if not yet available, name it and
  leave the link for George).

### Aims
The question verbatim from `../README.md`, then the pre-set rules: what counts as converged (the
plateau rule), what comparators exist (exact scheme chained the same way, the straight line, the
per-step-network chains of the previous study), what will be reported (endpoint at z1 by
component and momentum band; single step; growth along z; anatomy; cost).

### Method (the long section; numbered protocols)
1. **The tracks (E0).** From `E0_Track_dataset/README.md` and `results/tracks_meta.json` (if present)
   or the D0 README cut cascade: the official sample (expected_2024_minbias_xdigi, production
   00212966, 200 events, MagUp, sim-20231017-vc-mu100), the cut cascade with row counts, 14,482
   crossings, splits 11,567 / 1,463 / 1,452, the 257 planes, RK6 at 0.1 mm as the reference, the
   real last-UT state transported to z0, the field map v8r1 up (md5 9e49ddc4313b589f273540e7e0bb513b),
   the equation of motion written out (from `_shared/reference.py` deriv: dS/dz with
   N = sqrt(1 + tx² + ty²), κ = 1e-3, q/p in Allen units), the momentum bands and how many tracks in each.
   Figure: `E0_Track_dataset/figures/tracks_overview.png`.
2. **The discrete-time construction.** Derive it: an implicit Runge–Kutta step with q Gauss–Legendre
   stages, the stage equations, why a network that outputs the stage states can be trained by the
   residual of those equations without labels (the RK-PINN idea of Raissi 2019 §3.2), the exact
   collocation solution as a guardrail (its loss is machine zero). Worked micro-example with q = 2.
3. **The network (E1).** Inputs, normalisation by the round-1 spread, the start-plane input mapped to
   [−1, 1]; outputs as deviation from the straight line times the per-track scale; the x/tx scale
   κ|q/p|∫|B|dz along the straight line and the separate y/ty scale
   κ|q/p|∫√(1+tx²+ty²)((1+ty²)|Bx| + |tx·ty·By| + |tx·Bz|)dz floored at 10⁻³ of the x scale
   (why: gate 3 numbers, y_scale_check.png); 2×128 tanh; parameter counts (18,956 at q = 2;
   from `results/N*/record.json` n_parameters for others). `carry`: the same weights applied N
   times, advancing only the start-plane input; the bit-identity check of 2026-09-20.
4. **The loss.** The shared label-free RK-PINN loss (`_shared/model.py physics_loss`), each
   component divided by its POOLED spread (this is the fact the next study changes), stage planes
   set per track.
5. **Training (E1 trainer).** Rounds: round 1 on RK6 states at the N start planes; later rounds on the
   network's own predictions for all training tracks; 32,000 (track, plane) pairs a round, evenly
   per plane; 25 restarts a round (George, 2026-09-17); L-BFGS 200 iterations, history 120, strong
   Wolfe; the loss rescaled to 1 at the start of each round and after each tenfold fall, with a
   fresh optimiser; checkpoints and resume; memory 8 GB for q = 16.
6. **Gates before the farm** (E1 README "Gate results"): the five gates with their numbers.
7. **The grid and the farm.** 16 networks (N ∈ {2, 64, 128, 256} × q ∈ {2, 4, 8, 16}, dz = 2,588.9 /
   80.9 / 40.5 / 20.2 mm), clusters 5805460, 5809659, 5815090, 5816352; the keeper; wall hours per
   run from `F2_Analysis/results/headline.csv` train_wall_h.
8. **Stopping, and the lesson.** The trainer's 1 % rule never fired at N ≥ 64; the runs were stopped
   by hand on 2026-09-18 with the loss flat; the plateau rule (from
   `Block_F_reweighted_loss/F2_Analysis/compare_to_blockE.py`) applied afterwards showed 11 of 11
   runs with ≥ 20 rounds still improving 5–24 % per ten rounds (p = 0.0005 under noise), so all were
   put back; final state per run (restarts, rounds, plateaued_now) from headline.csv. Table.
   State the rule in words and as a formula.
9. **Comparators (E2).** The exact scheme chained at N = 2 and 256 for every q (E2 results JSON),
   the straight line, the previous study's chains, the material floor (real SciFi state vs RK6;
   define it, and defer its discussion to the companion write-up).
10. **Scoring (E3).** Which splits, endpoint vs single step, radial vs max metric, per plane, per
    band, the anatomy method (per-step local error against RK6 from the state the network was
    given; Σ(dx + dtx·lever); coherence definition), the case-study selection on validation.

### Results
Every number from the E3 `results/*.csv` (2026-09-18 checkpoints; say so) AND the extended-run
headline from `Block_F_reweighted_loss/F2_Analysis/results/headline.csv` (Block E rows: the runs
as on disk after extension; say which table is which). Sub-sections:
- The 16 endpoints (table: N × q, test radial median, p95, val spread; E3 error_qdz_chain.csv
  and headline.csv) + `error_qdz.png` (regenerate without the block label).
- Single step (`single_step.png`, error_qdz_single_step.csv): 0.09–0.8 µm at N ≥ 64; chain/single
  step ratios.
- Growth along z (`error_vs_z.png`, error_vs_z.csv).
- Against the exact scheme and the straight line (comparators.csv); the N = 2, q = 2 case equals
  the scheme's own error.
- Momentum and starting x (case study: `error_vs_p_{x,y,tx,ty}.png`, `case_study_*.png`,
  case_study_error_vs_p_x0.csv): the U-shape, the |x0| dependence, the worst 5 %.
- Per component in 10–20 GeV (E3 README table) with signed medians.
- Anatomy (`error_anatomy.png`, error_anatomy_summary.json): 72 % from early x-slope error; coherence
  0.375 vs 0.125; note the two-slope version exists in F2 (`anatomy_xy_blockE_*.json`, 98–99 %).
- No over-training (overtraining.csv, split_comparison.csv).
- Convergence (`convergence_grid.png`, `convergence_summary.png`) and the lesson.
- Cost (cost_accuracy.csv).
- Where the loss comes from: 97 % of the trained residual from 2–5 GeV (case_study_loss_vs_p.csv).
- One sentence on the previous study at the same N, q (14–20 times better), with its caveat.

### Conclusion and next steps
The chains sit at 118–166 µm because a sub-micrometre step error, mostly in the slopes, is carried
to the end; the loss weights states by their pooled residual and so spends itself on the 2–5 GeV
tracks and on the last quarter of the crossing where an error costs least. Next steps as they
happened: the reweighted loss (link to that write-up when its URL is known), the window move to
about 5 GeV, and the reference question (link).

### Provenance
As in the rules; list every script by path; input rows; clusters; the two checkpoint dates.

## Practicalities
- Figures with "Block E" in the title: regenerate via a runner `E4_Writeup/make_figures.py` that
  imports E3's scripts and wraps matplotlib's title/suptitle (copy the technique from
  `../../Block_F_reweighted_loss/F3_Analysis/run_e3_for_block_f.py`), writing to
  `E3_Analysis/figures_writeup/`. Do NOT edit E3's scripts. Use those paths in the page.
- Numbers script: `E4_Writeup/numbers.py` for anything computed (ratios, ranges).
- Save the final markdown you send to Notion as `E4_Writeup/page.md` (and children as
  `E4_Writeup/child_<n>.md`) so the page is reproducible; write `E4_Writeup/README.md` with the
  page URL(s), the verification result and the figure list.
