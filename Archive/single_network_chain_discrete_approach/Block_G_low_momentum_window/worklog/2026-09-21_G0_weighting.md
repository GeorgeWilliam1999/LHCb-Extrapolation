# 2026-09-21 — Block G phase 1 (G0): the windowed loss and its gates

Agent A. Scope: build `G0_Weighting/` exactly as `PLAN.md` §3 specifies. No training, no farm
submission, no commit, no Notion write. Block E, Block F and `_shared/` treated as read-only.

Python throughout:
`PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 /data/bfys/gscriven/conda/envs/TE/bin/python`.

## What was built

```
Block_G_low_momentum_window/G0_Weighting/
  use_shared.py                       copied verbatim from Block F's F0_Weighting/
  windowed_loss.py                    the fixed torch path; imports the rest from F0
  check_windowed_weights.py           gates W1, W2, W3 + the pre-flight W4
  results/check_windowed_weights.json
  results/preflight_windows.csv       78 rows: 6 windows x (9 momentum bands + 4 quarters of z)
  figures/window_preflight.png
  README.md
Block_G_low_momentum_window/worklog/2026-09-21_G0_weighting.md   (this file)
```

`windowed_loss.py` re-exports `MODES, QOP_TO_GEV, P_LO, P_HI, ROLLOFF, W_FLOOR, CLAMP,
N_IBAR_SAMPLES, momentum_gev, band_window, i_bar, track_bend, lever_arms, weighted_loss,
loss_for` from `Block_F_reweighted_loss/F0_Weighting/weighted_loss.py` untouched, and
re-implements exactly three functions:

- `reference_constants(..., p_lo=P_LO, p_hi=P_HI, rolloff=ROLLOFF, w_floor=W_FLOOR)` — calls
  F0's `reference_constants` and then overwrites the four window keys from the arguments (F0
  hard-codes the module constants into them). Wrapping F0's rather than copying its body is
  deliberate: `D_ref` and `lev_ref` then cannot drift from Block F by accident.
- `per_track_factor(qop, const, switches, clamp=None)` — F0's body with
  `band_window(p, const["p_lo"], const["p_hi"], const["rolloff"], const["w_floor"])`.
- `weights(model, S, z_start, const, mode=None)` — F0's body verbatim, present only so that it
  binds to the `per_track_factor` above.

Interface matches `PLAN.md` §3.1 exactly, so agent B can code against it.

## The trap, restated for the record

`F0_Weighting/weighted_loss.py` line 166: `w = band_window(p) if switches["window"] else …`.
No arguments, so `p_lo`/`p_hi`/`rolloff`/`floor` fall back to the module constants 10/50/ln2/
0.05. `const["p_lo"]` etc. are never read on the torch path, but they ARE what goes into
`scale.json["weighting"]`. The numpy twin in `F0_Weighting/check_weights.py` (`weights_numpy`,
lines 118–127) reads the dict. Block F's runs are unaffected — their dict equals the defaults.
A Block G run built on that path would have trained the 10–50 GeV objective while its run
folder claimed the new window, silently.

## Bit-identity check, before anything else

Ad-hoc script (scratchpad): N = 64, q = 2, 500 states, defaults on both sides.
`WL.reference_constants(...) == GL.reference_constants(...)` → `True`; `np.array_equal` of the
weights under all five modes → `True` for every mode. So the premise of the training-side
isolation gate I2 holds: with `--p-lo 10 --p-hi 50` Block G's path is Block F's, bit for bit.

## Gate results

```
PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 /data/bfys/gscriven/conda/envs/TE/bin/python \
  check_windowed_weights.py
```
8.5 s single-threaded. `ALL WINDOW CHECKS PASS`.

- **W1**, N = 2 and 256 x q = 2 and 16 x 6 windows x 5 modes, 2,000 states:
  - Block G's torch weights vs the numpy twin (F0's `weights_numpy`, imported): worst
    **1.47e-15**, tolerance 1e-13. Pass.
  - The trap demonstrated: F0's own `weights`, handed the same non-default `const`, vs the same
    twin — relative difference **3.07 to 3.47** (order unity) for every moved window; for the
    10–50 GeV row it is **1.02e-15**, i.e. F0 agrees with the twin on its own window. Both
    halves asserted in the gate (`trap_demonstrated`, and `< TWIN_TOL` required on the default
    row).
  - Lever arm: min = dz, max = L + dz(1 − c₁), exact to 1e-9 relative.
- **W2**, `blockE` mode through Block G's `weights`: `physics_loss` vs `weighted_loss`,
  relative difference **0.0** in all four (N, q) cases. Pass.
- **W3**, exactly solved collocation states, window 3–8 GeV, all five modes: exact/straight
  ratio **6.2e-30 … 2.0e-29**, tolerance 1e-10; 0 non-converged solves. Pass.
- **W4**: the pre-flight, below.

## W4 — the pre-flight

Network: `Block_F_reweighted_loss/F1_Training/results/full/N064_q02`, `progress.json` phase
`done`, round 40, restart 1000 (checked before loading, per PLAN). Loaded with
`chain_network.load_network` from `Block_E_single_network_chain/E1_Network_grid`. N = 64,
q = 2, dz = 80.90 mm. 8,000 (track, plane) states, `draw_pairs(..., even=True)`, 125 per plane.

Draw seed: 20260918, but from a generator created immediately before the draw. It is therefore
NOT the same 8,000 states as F0's gate 4, whose generator had already been consumed by three
gates. Noted in the script docstring. The comparison of interest is between windows on one
draw, so this costs nothing.

### Dead end found on the first run: the raw shares are uninterpretable

First run of W4 gave, for all six windows, 84.7–89.7 % of the loss inside 3–8 GeV and shares
that differed by under 1 point between windows — a result that would have said "the window
does not matter". Diagnosed with a scratch script over the per-state shares:

```
residual rms per component (mm, mm, -, -): [9.34e-4  1.64e-3  1.32e-5  2.14e-5]
residual magnitude quantiles (50/90/99/99.9/100 %):
    1.57e-4  1.07e-3  3.84e-3  1.10e-2  1.56e-2
top state under the 10-50 GeV window: share 68.8 %, p = 6.66 GeV, plane k = 18,
    |residual| 1.03e-2 (66x the median), x0 = 325 mm
```

One state out of 8,000 carries 68.8 % of the objective under the 10–50 GeV window and ~41 %
under the low windows. The raw band shares are a statement about where that one state sits.
The residual median by band confirms the picture (rms |r| 2.57e-3 at 1–2 GeV, 1.16e-3 at
3–5, 5.26e-4 at 5–8, 1.21e-4 at 10–20): after training with the 10–50 GeV window the network
is already accurate at high p, so whatever weight is applied, the loss now lives at low p.

Fix, added to the script: every W4 quantity is reported twice, raw and with the most extreme
1 % of states (80) removed and the rest renormalised. Trimmed, the windows separate cleanly
(61.0 / 59.5 / 57.1 / 56.5 / 66.6 / 48.7 % in 3–8 GeV). The figure gained a second momentum
panel and dashed trimmed curves on the cumulative-cost panel, which was flat-to-the-axis
before. Also recorded per candidate: `largest_single_state_share_pct`.

### Numbers (share of the loss, raw / trimmed)

| loss window | own band | 3–8 GeV | > 50 GeV | > 100 GeV | top 1 % | largest single state | ρ all | ρ own band | ρ 3–8 GeV |
|---|---|---|---|---|---|---|---|---|---|
| 3–8 (anchor) | 85.4 / 61.0 | 85.4 / 61.0 | 0.05 / 0.85 | 0.05 / 0.78 | 93.8 | 41.1 | +0.949 | +0.942 | +0.942 |
| 3–10 | 85.7 / 63.5 | 85.1 / 59.5 | 0.06 / 0.89 | 0.05 / 0.83 | 93.6 | 41.0 | +0.937 | +0.932 | +0.942 |
| 3–20 | 86.7 / 68.8 | 84.9 / 57.1 | 0.06 / 0.87 | 0.05 / 0.79 | 93.4 | 40.8 | +0.887 | +0.898 | +0.942 |
| 2–15 | 99.5 / 92.6 | 84.7 / 56.5 | 0.06 / 0.84 | 0.05 / 0.79 | 93.3 | 40.8 | +0.911 | +0.921 | +0.942 |
| 4–6 (narrow) | 21.9 / 25.2 | 88.2 / 66.6 | 0.04 / 0.77 | 0.04 / 0.69 | 94.4 | 42.3 | +0.952 | +0.963 | +0.941 |
| 10–50 (Block F's) | 3.6 / 20.8 | 89.7 / 48.7 | 0.13 / 0.99 | 0.09 / 0.39 | 94.4 | 68.8 | +0.620 | +0.798 | +0.757 |

Per-band and per-quarter tables: `G0_Weighting/README.md` and `results/preflight_windows.csv`.
Cost left in 10–50 GeV, trimmed: 5.2 (3–8), 6.2 (3–10), 7.7 (3–20), 7.5 (2–15), 3.6 (4–6),
14.1 % (10–50).

### Clamp scan, 3–8 GeV window, report only

| clamp | 3–8 GeV raw (trimmed) | 10–50 GeV raw (trimmed) | > 50 GeV | ρ | tracks clipped low / high |
|---|---|---|---|---|---|
| off | 85.4 (60.9) | 0.9 (5.5) | 0.07 | +0.949 | — |
| 2 | 81.7 (51.8) | 0.8 (4.5) | 0.01 | +0.958 | 15.1 % / 1.4 % |
| 3 | 85.3 (59.7) | 0.9 (5.4) | 0.02 | +0.951 | 2.4 % / 0.3 % |
| 5 | 85.4 (61.0) | 0.9 (5.6) | 0.05 | +0.949 | 0.1 % / 0.0 % |
| 10 | 85.4 (60.9) | 0.9 (5.5) | 0.07 | +0.949 | 0.0 % / 0.0 % |

The clipped-fraction columns were added to the script after the scan came out flat, to show
why: at clamp 5 the 3–8 GeV window clips 0.1 % of tracks from below and none from above, so
the clamp is inert. `a_n ∝ sqrt(W(p)) · p`; a low window shrinks precisely the high-momentum
factors the clamp existed to catch (the 0.05 floor gives sqrt(W) = 0.224 above 50 GeV). Only
clamp 2 changes anything and it moves weight out of 3–8 GeV. Decision: clamp stays 5, as
instructed; it is a no-op rather than a choice.

## Decisions taken

- Pre-flight statistics reported raw AND trimmed; the trimmed column is the one to read. Not
  in PLAN §3.2, added because without it the pre-flight answers nothing.
- `sys.dont_write_bytecode = True` at the top of `check_windowed_weights.py`. The first run
  dropped `check_weights.cpython-310.pyc` into `Block_F_reweighted_loss/F0_Weighting/
  __pycache__/`; it was deleted and that folder is back to its two pre-existing files
  (`use_shared`, `weighted_loss`, untouched mtimes). Nothing else was written under Block E or
  Block F.
- `reference_constants` wraps F0's rather than copying it (see above).
- Candidate roles kept as PLAN §3.2: 3–8 anchor, 3–10 / 3–20 / 2–15 candidates, 4–6 narrow
  reference, 10–50 baseline.

## Recommendation to George (for the §6 decision gate)

**3–8 GeV.** Reasoning:

1. It is the band he named on 2026-09-21 evening, and among the four serious candidates it puts
   the most weight there (61.0 % trimmed, against 59.5 / 57.1 / 56.5 %).
2. Widening buys nothing in the region of interest and only a little back in 10–50 GeV
   (5.2 → 7.7 % trimmed going 3–8 → 3–20). If the secondary measure (radial median at 10–50
   GeV, currently 24.8 µm) turns out to matter more than the primary, 3–20 GeV is the fallback,
   not a co-equal.
3. Inside 3–8 GeV all four low windows rank states by their true endpoint cost identically
   (ρ = +0.942 against +0.757 for 10–50 GeV), so there is no aiming argument for a wider one.
4. 4–6 GeV is not recommended despite the highest 3–8 GeV share: it holds only a quarter of the
   loss inside its own band, halves the weight on 10–20 GeV, and concentrates below 5 GeV
   rather than at it.

### Caveats George should weigh before the gate

- **The lever is modest.** The 10–50 GeV window already gives 48.7 % (trimmed) of the loss to
  3–8 GeV, because that is where the residual now is. Moving the window buys +12 points. On
  this evidence the pre-registered success bar in PLAN §7 — median |Δx| and the 68 % half-width
  at 4–6 GeV both to two thirds of Block F's — is ambitious. Worth confirming or amending the
  bar at the gate rather than after the runs.
- **The pre-flight is measured on the network Block F's own objective produced.** It describes
  where a new run's first rounds would place their attention, not where a converged run ends
  up. The residual distribution will move as training proceeds.
- **The heavy tail is untouched by any of this.** One state in 8,000 at 41 % of the objective,
  top 1 % at 93.8 %. That is the same phenomenon as the > 1 mm radial tail at 4–6 GeV (tracks
  starting at the edge of the acceptance, median |x₀| 190–230 mm against 114 mm). The window
  will not move it; a starting-|x| weight is the separate lever George already identified.
- **The clamp is no longer a design knob** for a low window (0.1 % of tracks clipped at 5).
  Nothing to change, but the reasoning that set it to 5 no longer applies.
- **Not checked here**, by design: that a trainer built on this module reproduces Block F's
  round-1 losses (gate I2) and Block E's bit-identity (gate I1). Those are agent B's, phase 2.

## Not done (out of scope, as instructed)

No commit, no push, no `condor_submit`, no Notion write, nothing written under Block E or
Block F, Block F's running jobs untouched. The duplicate Block F job noted in PLAN §9
(5833416) was not acted on; it needs George's `condor_rm`.
