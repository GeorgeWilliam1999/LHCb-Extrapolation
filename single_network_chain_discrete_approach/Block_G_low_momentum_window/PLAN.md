# Block G — the momentum window moved to the region around 5 GeV

Plan drafted 2026-09-21 (Claude, for George). The Notion to-do "Retrain the chained network
with the loss weighted for tracks around 5 GeV" (3e25d544-b9d9-8184-9f9c-f8817f6bb379) is the
system of record; this file is the working copy the agents build from. `HANDOFF_PROMPT.md`
beside it is George's brief and takes precedence where the two differ.

**George's added instruction (2026-09-21 evening): focus the optimisation on 3–8 GeV.** That
makes 3–8 GeV the anchor candidate for the window; his earlier default of 3–20 GeV and the
alternatives 3–10 and 2–15 stay in the pre-flight as the comparison rows.

## 0. Standing rules (binding for every agent)

- `Block_E_single_network_chain/`, `Block_F_reweighted_loss/` and `_shared/` are **read-only**.
  Block G **imports** from them; it never copies physics code and never monkey-patches a
  Block F module in place. (Farm harness scripts are not physics: those are copied and adapted.)
- **Never commit, never push.** The folder is uncommitted; leave it so.
- **No block letters in any output** a human reads: figure titles, legends, axis text, printed
  tables, CSV columns. Name a network by N, q and dz ("N = 64, q = 2, dz = 81 mm"), the study by
  what it is ("one network per step length, chained across the magnet; loss window 3–8 GeV").
  Say whether an error is an **endpoint** error (after the full chain, at the first SciFi
  plane z1) or a **single-step** error. Block letters stay in folder names, docstrings and worklogs.
- **Analysis in isolation**: no comparison rows or panels against Block E or Block F unless
  George asks. The Block F numbers to beat are quoted in the report text, not drawn in figures.
- **Nothing is submitted to the farm before George has chosen the window** (§5).
- **No Notion write-up.** Results go to chat first; the worklog goes into the to-do body.
  Agents write their worklog to `worklog/<date>_<topic>.md` in this folder (dated, append-only:
  commands, numbers, dead-ends, decisions); the coordinating chat mirrors it into Notion.
- Python: `PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python`, single-threaded
  (`OMP_NUM_THREADS=1` etc., as every Block F script sets at import).

## 1. What Block G is

Block F's loss with **one** change: the momentum window `W(p)` is moved from 10–50 GeV to a
band around 5 GeV. Same tracks and splits (E0, 11,567 / 1,463 / 1,452), same network (imported
from E1 `chain_network.py`), same seed, same L-BFGS protocol and rescaling, same 25 restarts a
round and caps (40 rounds / 1,000 restarts), same states per round (32,000), same clamp
([1/5, 5] of the batch median), same lever weighting, same track-bend normalisation, same
roll-off (Gaussian in log p, width ln 2) and floor (0.05). Three settings: N = 64 q = 2,
N = 128 q = 8, N = 256 q = 16.

The one code trap (George): in `F0_Weighting/weighted_loss.py`, `per_track_factor` calls
`band_window(p)` with the module defaults `P_LO`/`P_HI`, ignoring `const["p_lo"]`/`const["p_hi"]`,
while the numpy twin in `check_weights.py` reads the constants. Block F's runs were unaffected
(their constants equal the defaults), but a Block G run built on that path would silently train
with the 10–50 window. The fix lives in Block G's own module (§3), not in Block F's file.

## 2. Layout (mirrors Block F)

| folder | role | agent |
|---|---|---|
| `G0_Weighting/` | `windowed_loss.py` (the fixed torch path, imports the rest from F0), `check_windowed_weights.py` (gates W1–W3 + the pre-flight W4 over the candidate windows), `results/`, `figures/` | A |
| `G1_Training/` | `train_windowed.py` (F1's trainer with the import swapped and `--p-lo/--p-hi`), `results/train_windowed.diff`, the farm harness in `condor/`, the isolation gates I1–I2 | B |
| `G2_Analysis/` | the judging scripts for the new networks: around-5-GeV signed table, per-component tables, momentum bands, two-slope anatomy, convergence check | C |
| `G3_Analysis/` | the E3 figure set reproduced for the new networks (F3's runner pattern) | C |
| `worklog/` | dated agent worklogs | all |
| `README.md` | the block's front page (written by the coordinating chat, updated at each phase) | — |

Every script takes the run folder(s) as an argument or module constant that defaults to
`G1_Training/results/<window>/full/`, so a smoke test can point at Block F's runs without
writing into Block F.

## 3. Phase 1 — G0: the windowed loss and its gates (agent A)

### 3.1 `windowed_loss.py` — the interface (agent B codes against this; keep it exact)

```python
# G0_Weighting/windowed_loss.py
import use_shared                             # copied helper, as in every F folder
sys.path.insert(0, <F0_Weighting>)            # F0 on the path; nothing in it is modified
from weighted_loss import (MODES, QOP_TO_GEV, P_LO, P_HI, ROLLOFF, W_FLOOR, CLAMP,
                           momentum_gev, band_window, i_bar, track_bend, lever_arms,
                           weighted_loss, loss_for)          # re-exported unchanged

def reference_constants(model, S, z_start, z1, ibar, L, mode="full", clamp=CLAMP,
                        p_lo=P_LO, p_hi=P_HI, rolloff=ROLLOFF, w_floor=W_FLOOR) -> dict
    # F0's dict with p_lo/p_hi/rolloff/w_floor set from the ARGUMENTS

def per_track_factor(qop, const, switches, clamp=None)
    # F0's function with band_window(p, const["p_lo"], const["p_hi"], const["rolloff"],
    #                                 const["w_floor"])

def weights(model, S, z_start, const, mode=None)
    # F0's function, re-implemented here (≈15 lines) so it calls THIS per_track_factor;
    # lever_arms / track_bend / MODES imported from F0
```

With `p_lo=10, p_hi=50` every function must return exactly what F0 returns (that is gate I2's
premise). No other behaviour changes.

### 3.2 Gates (all in `check_windowed_weights.py`; JSON + CSV under `results/`, figure under `figures/`)

- **W1 — twin gate with a moved window.** F0's gate 1 (numpy twin `weights_numpy`, imported
  from `check_weights.py`), but with `const` built for each candidate window. Also assert the
  *negative*: F0's own `weights` with the same non-default `const` **disagrees** with the twin
  (that is the trap, demonstrated), and Block G's agrees (< 1e-13 relative).
- **W2 — `blockE` path unchanged**: F0's gate 2 through Block G's `weights`.
- **W3 — the minimum does not move**: F0's gate 3 under every mode for the anchor window.
- **W4 — the pre-flight, on Block F's trained N = 64, q = 2 network**
  (`F1_Training/results/full/N064_q02`, finished, `progress.json` phase = done), 8,000
  (track, plane) states drawn evenly over the planes, seed 20260918 as in F0, the same
  "cost" reference (local error against RK6 from the same state, position plus slope times
  distance left; diagnostic only).

  Candidate windows (p_lo–p_hi, GeV): **3–8 (anchor)**, 3–10, 3–20, 2–15, 4–6 (narrow
  reference), and 10–50 (Block F's, the baseline row). Clamp 5 throughout.

  Per candidate, report: loss share by momentum band with edges 1, 2, 3, 5, 8, 10, 20, 50,
  100, 200 GeV; the share inside 3–8 GeV and inside the candidate's own band; Spearman
  ρ(share, cost) overall, inside the candidate's band and inside 3–8 GeV; the share taken by
  > 50 GeV and by > 100 GeV; the top-1 % concentration; the first/last quarter of z. Then a
  clamp scan (off, 2, 3, 5, 10) for the anchor window, **report only** — the clamp stays at 5
  unless George moves it.

  One caution to measure, not assume: the track factor is `a_n ∝ sqrt(W(p)) · p` (D_n ∝ |q/p|),
  so it pulls the other way from the window: at p_hi = 8 GeV, W(20) ≈ 0.17 and a(20)/a(5) ≈ 1.7
  *before* the clamp. Whether a 3–8 window actually puts most of the loss into 3–8 GeV is the
  pre-flight's question. If no candidate gets the majority of the loss into its band, say so
  plainly; the knobs that would (roll-off width, floor, clamp) are George's to turn, not the agent's.

Output: `results/check_windowed_weights.json`, `results/preflight_windows.csv` (one row per
candidate × band), `figures/window_preflight.png` (share-by-momentum per candidate, share
along z, cumulative share vs cost for the anchor and the baseline), and a worklog entry with
the printed table.

## 4. Phase 2 — G1: the trainer and the harness (agent B)

- `train_windowed.py` = `F1_Training/train_weighted.py` with (a) the import swapped to
  `G0_Weighting.windowed_loss`, (b) `--p-lo`, `--p-hi` (floats, defaults 10 and 50 so the
  default reproduces Block F), passed into `reference_constants`, so `scale.json["weighting"]`
  carries `p_lo`/`p_hi`; (c) the same start-up guard as `--weighting`: a run folder whose
  `scale.json` window differs from the command line exits. **Nothing else.**
  `results/train_windowed.diff` = the diff against F1's trainer.
- Run folders: `<out>/<weighting>/N<NNN>_q<qq>/`, with `--out results/p03-08` (the window in the
  path, zero-padded), so a second window could never overwrite a first. `record.json` carries
  `weighting_constants` (already does, via `sc["weighting"]`).
- **Gate I1 — Block E bit-identity**: F1's gate F-1 recipe (`--weighting blockE --n-train 300
  --n-eval 200 --stop-after 3`) against E1's `train_network.py`: `loss_before`/`loss_after`
  equal to the last digit.
- **Gate I2 — Block F round-1 bit-identity**: `--N 64 --q 2 --p-lo 10 --p-hi 50 --stop-after 2`
  on the **full** data into a scratch `--out`; `scale.json` (`in_scale`, `weighting`) equal
  to `F1_Training/results/full/N064_q02/scale.json` field for field, and `history.csv` rows
  0–1 (`loss_before`, `loss_after`, `n_iter`, `func_evals`) equal to Block F's rows 0–1 to the
  last digit. About 3 minutes (78 s a restart). Repeat with one restart at N = 256, q = 16
  (about 4 minutes) if time allows.
- Harness in `G1_Training/condor/`: `wrapper.sh` (points at `train_windowed.py`),
  `make_jobs.py` (adds `--p-lo/--p-hi/--out`), `resubmit.py`, `keeper.sh`, `prune_active.py`
  (imports `plateaued_now` from `F2_Analysis/compare_to_blockE.py`; the trainer's own 1 % rule
  never fires at N ≥ 64). Keep Block F's pattern and its (N, q, weighting, wrapper) queue match,
  **plus the window in the match**.
  **Fix the bug that bit Block F today (2026-09-21 13:35):** `resubmit.queued()` returns an
  empty set when `condor_q` fails, and the keeper then submitted a second copy of the
  N = 256, q = 16 job, which has been double-writing that run's checkpoint since restart 1198.
  In Block G's `resubmit.py` a `condor_q` failure (non-zero exit, exception, or empty output
  while `condor_q` reports an error) means "unknown", and **nothing is submitted**. Also refuse
  to submit a line whose run folder has a `progress.json` modified in the last 10 minutes
  (a live process is writing it). Dry-run the whole harness (`make_jobs.py`, `resubmit.py`
  without `--submit`) and show the job lines; **do not `condor_submit`**.

## 5. Phase 3 — G2/G3: analysis in isolation, scaffolded now, run later (agent C)

Scripts that read a run folder and produce the report set; smoke-tested against Block F's
finished runs with outputs into the session scratchpad, then the smoke outputs deleted (nothing
written under Block F). Each takes `--runs <dir>` (default `../G1_Training/results/p03-08/full`).

- `G2_Analysis/errors_near_5gev.py` — F2's, reproduced: per component (x, y, tx, ty), bands
  3–5, 4–6, 5–7, 5–8 (added), 10–20 GeV: n, median |d|, RMS, std, signed mean and median,
  68 % half-width, p95; the signed-distribution figure at 4–6 GeV; **and the RMS with the
  > 1 mm-radial tracks removed** beside the full RMS, with the tail fraction and the median
  |x₀| of the tail against the rest (George: the tail is the edge of the acceptance and the
  window will not touch it; measure whether it still dominates).
- `G2_Analysis/error_tables_by_component.py` — F2's, reproduced, for 5–30 GeV and for 3–8 GeV.
- `G2_Analysis/momentum_bands.py` — radial median and p95 in the bands 2–3, 3–5, 5–8, 8–10,
  10–20, 20–50, 50–100, 100–200 GeV, per network (the secondary measure, cost in 10–50 GeV).
- `G2_Analysis/anatomy_xy.py` — F2's two-slope anatomy driven on the new runs (import, do not
  copy, if its `main` allows a run-folder argument; otherwise a thin driver).
- `G2_Analysis/convergence.py` — `plateaued_now` (imported from F2) over each run's
  `rounds.csv`, the ten-round medians and the last-ten-over-previous-ten ratio; the headline
  is the median of the last ten rounds with its spread, not the final checkpoint.
- `G3_Analysis/run_e3_for_block_g.py` — F3's runner pattern (import E3's scripts, point `HERE`
  and `E1` here, `runs/results/` view of the finished runs, the four grid-free panels), with the
  title patch reading "one network per step length, chained; loss window 3–8 GeV" (never a
  block letter).
- Every figure and table names networks by N, q, dz and says "endpoint" or "single-step".

## 6. Decision gate — George chooses (nothing below runs before this)

Presented in chat with the pre-flight table: the window (anchor 3–8 GeV, or 3–10 / 3–20 /
2–15), and the pre-registration below, confirmed or amended. Also the Block F duplicate job
(§9), which needs a `condor_rm` the agent is not permitted to run.

## 7. Pre-registration (draft; fixed once George confirms, before any training)

All on the **1,452 test tracks, endpoint after the full chain at z1**, N = 64, q = 2 as the
case-study setting; the other two reported the same way.

- **Primary, 4–6 GeV (286 test tracks):** median |Δx|, 68 % half-width of Δx, the same for Δy,
  and the signed distributions per component (x, y, tx, ty). Block F's numbers: |Δx| 93 / 146 /
  RMS 618 µm; |Δy| 130 / 223 / 425 µm; tx 0.078 / 0.103 / 0.284 mrad; ty 0.095 / 0.161 /
  0.329 mrad; signed medians within ±30 µm.
  - **Success**: median |Δx| and the 68 % half-width of Δx at 4–6 GeV both at most two thirds of
    Block F's (≤ 62 µm and ≤ 97 µm), and |Δy| not worse than Block F's (≤ 130 / 223 µm), with the
    signed medians still within ±30 µm.
  - **Partial**: x clears, y does not, or only one of the two x measures clears: the window
    reached x, the y plane needs its own lever (the two-slope anatomy says which).
  - **Null**: neither x measure clears two thirds. Then the momentum window is not the binding
    constraint at 5 GeV; the next lever is the starting-|x| weight (a separate block).
- **Secondary, the cost in 10–50 GeV:** radial median at z1 (Block F: 24.8 µm). Acceptable at
  ≤ 50 µm (still half of Block E's 108 µm); reported either way.
- **RMS and the tail:** RMS in x at 4–6 GeV reported with and without the > 1 mm-radial tracks;
  the expectation, stated now, is that the tail fraction (6–7 %) does not move.
- **Convergence:** `plateaued_now` — the median validation error of the last ten rounds no more
  than 5 % below the ten before, for three rounds running — as of the latest round; result =
  median of the last ten rounds with its round-to-round spread. A run that hits the
  1,000-restart cap while still falling is extended (`--extend`) the way Block F's N = 256 was,
  and is reported as a checkpoint until the rule holds.

## 8. Phase 4 — the runs and the report (after §6)

1. Write the confirmed pre-registration and the chosen window into the to-do body, dated.
2. `make_jobs.py --p-lo <lo> --p-hi <hi> --out results/p<lo>-<hi>`; `condor_submit`; copy the
   job lines to `condor/jobs_active.txt`; keeper in tmux: `tmux new-window -t claude-phone -n
   blockG-keeper <G1>/condor/keeper.sh`. Expected wall: about 22 h (N = 64), 33 h (N = 128),
   50 h (N = 256) at Block F's measured median 78 / 118 / 182 s a restart (the 226 s first
   quoted for N = 256 was measured while two jobs shared that run, §9).
3. Watch `rounds.csv` with `G2_Analysis/convergence.py`; never call convergence from the loss.
4. When a run holds the rule: G2 tables and figures, G3 set, the two-slope anatomy; worklog.
5. Report in chat: the primary numbers against Block F's, y and ty separately, the secondary
   cost, the tail, and the figures. Then stop; George decides about a write-up.

## 9. Found on the way (2026-09-21, not Block G's to fix)

Block F, N = 256, q = 16: two HTCondor jobs (5823724, started 01:05, and 5833416, submitted by
the keeper at 13:35) have been training the same run at once. `history.csv` has duplicate
restart numbers from 1198 on and `rounds.csv` has rounds 48–51 twice; `network.pt` and
`progress.json` are written alternately by two processes. Cause: `resubmit.queued()` read an
empty queue (a `condor_q` failure returns an empty set) and resubmitted. Recommended:
`condor_rm 5833416` (the younger, at restart 1295 against 1298), then treat that run's history
after restart 1198 as interleaved when it is analysed. The `condor_rm` was refused to the agent
by the permission policy; George's call.
