# 2026-09-21 — G1: the windowed trainer and the farm harness (agent B)

Phase 2 of `PLAN.md` §4. Built `G1_Training/` from Block F's `F1_Training/`. Nothing was
committed, nothing was pushed, nothing under `Block_E_single_network_chain/`,
`Block_F_reweighted_loss/` or `_shared/` was written to, and no `condor_submit`,
`condor_rm`, `condor_hold` or `condor_release` was run.

Python throughout: `PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python`, with
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1`.
Scratch for every gate:
`/tmp/claude-11298/-data-bfys-gscriven/30d9e3ab-fc22-4442-94eb-4fb80478ffed/scratchpad/`.

## 1. `train_windowed.py`

`cp Block_F_reweighted_loss/F1_Training/train_weighted.py train_windowed.py`, then four code
changes and a rewritten docstring. `results/train_windowed.diff` is
`diff -u` against F1's trainer. The four code hunks:

1. `sys.path.insert(0, ".../F0_Weighting")` → `".../G0_Weighting"`, and
   `from weighted_loss import (MODES, i_bar, loss_for, reference_constants, weights)` →
   `from windowed_loss import (...)` — the same five names, agent A's module re-exporting
   `MODES`, `i_bar` and `loss_for` from F0 unchanged and redefining `reference_constants`,
   `per_track_factor` and `weights` so the window comes from the run's constants.
   The trainer does **not** need F0 on its own path: `windowed_loss` puts it there.
2. `--p-lo` / `--p-hi`, floats, defaults 10.0 and 50.0.
3. the start-up guard, immediately after the existing `--weighting` one:

   ```python
   w_lo, w_hi = sc["weighting"].get("p_lo"), sc["weighting"].get("p_hi")
   if (w_lo, w_hi) != (None, None) and (float(w_lo), float(w_hi)) != (a.p_lo, a.p_hi):
       raise SystemExit("this run was started with the window %g-%g GeV, not %g-%g" % ...)
   ```

   The `.get` is deliberate: a `scale.json` written before the window existed as a field has
   no `p_lo`, and that is not a mismatch, it is an older file. The guard sits before the
   checkpoint is loaded and before anything is written, so a rejected command touches nothing.
4. `reference_constants(..., mode=a.weighting, p_lo=a.p_lo, p_hi=a.p_hi)`, which is what puts
   the window into `scale.json["weighting"]` and from there into `record.json`
   (`weighting_constants`).

Nothing else changed. `--out` and the run-folder convention `<out>/<weighting>/N<NNN>_q<qq>/`
are Block F's, unchanged; the window goes into `--out` (`results/p03-08`) by convention, not
by code, except in `make_jobs.py` which derives that default.

Dead end, corrected: the first patch left the import's continuation line one space out of
alignment (`windowed_loss` and `weighted_loss` are the same length, so the original
indentation was right). Fixed before the diff was written, so the diff shows a one-line
change there rather than two.

## 2. The harness, in `G1_Training/condor/`

All five scripts live in `condor/` (PLAN §4), unlike Block F where `make_jobs.py` and
`resubmit.py` sit a level up. `HERE` is therefore `condor/` and `ROOT` is `G1_Training/`;
job and submit files are written to `ROOT/condor/…` and named `condor/jobs.txt` inside the
submit file, so `condor_submit` is still run from `G1_Training/` and the log paths resolve.

- `wrapper.sh` — Block F's, pointing at `train_windowed.py`.
- `make_jobs.py` — Block F's, plus `--p-lo` / `--p-hi` (defaults 3 and 8, the anchor window),
  `--out` defaulting to `results/p<lo>-<hi>` via `window_tag()` (zero-padded two digits a
  side; a non-integer edge becomes e.g. `07p50`), and the window written into every job line.
  Refuses `p_lo >= p_hi`.
- `resubmit.py` — Block F's, with the queue key extended from (N, q, weighting) to
  (N, q, weighting, p_lo, p_hi), the run folder taken from the line's own `--out` instead of
  a hard-coded `results/`, and the two safety rules below.
- `prune_active.py` — new here (Block F's lives in `F2_Analysis/`). It **imports**
  `plateaued_now` from `Block_F_reweighted_loss/F2_Analysis/compare_to_blockE.py`; the rule is
  not restated. It also imports `key_of` and `run_dir` from `resubmit.py` so the two agree on
  what a job line means.
- `keeper.sh` — Block F's, with the prune step pointed at `condor/prune_active.py` and the
  "still to train" count read from `jobs_active.txt` (the run folders are now a level deeper,
  so Block F's `results/*/N*_q*/progress.json` glob would not have matched).

### The resubmit fix

Block F's `queued()` returned `set()` when `condor_q` failed, which reads as "none of our jobs
are running" and resubmits everything. That is what put a second job on the N = 256, q = 16 run
at 13:35 today. Here:

- `queued()` returns `None` — *unknown* — on a non-zero exit, on any exception, or when
  `stderr` matches `error|failed|unable|cannot|can't|denied|refused`. A clean read with empty
  output is still an empty set, so an idle queue behaves normally.
- `plan()` returns an empty to-do list when `inq is None`, so **nothing is submitted**. The
  keeper's next pass, half an hour later, reads the queue again.
- second, independent rule: a job line whose run folder has a `progress.json` modified in the
  last `LIVE_SECONDS = 600` is held back with the age printed. The trainer writes that file
  after every restart (78 s at N = 64, 182 s at N = 256), so a file younger than ten minutes
  means a process is in that run whatever the queue says.

## 3. Gate I1 — `--weighting blockE` is still bit-identical to the unweighted trainer

The recipe of `Block_F_reweighted_loss/F1_Training/README.md` (its gate F-1), with
`train_windowed.py` in place of `train_weighted.py`:

```bash
cd Block_E_single_network_chain/E1_Network_grid
$PY train_network.py  --N 64 --q 2 --n-train 300 --n-eval 200 --stop-after 3 --out $S/I1/blockE
cd Block_G_low_momentum_window/G1_Training
$PY train_windowed.py --N 64 --q 2 --weighting blockE --n-train 300 --n-eval 200 --stop-after 3 --out $S/I1/blockG
```

| restart | loss_before | loss_after | n_iter | func_evals |
|---|---|---|---|---|
| 0 | 0.0001001095011696207 | 8.685510495777045e-08 | 200 | 204 |
| 1 | 8.685510495777045e-08 | 3.259393684560016e-08 | 200 | 203 |
| 2 | 3.259393684560016e-08 | 1.9375225163375078e-08 | 200 | 204 |

Both sides, identical to the last digit. `diff` over `loss_before,loss_after` is empty, and so
is a `diff` over every logged field except `wall_s` (60/58/60 s against 60/62/60 s — the clock,
not the arithmetic). **I1 passes.**

## 4. Gate I2 — the default window reproduces the 10–50 GeV runs bit for bit

`$PY train_windowed.py --N 64 --q 2 --p-lo 10 --p-hi 50 --stop-after 2 --out $S/I2/p10-50`,
on the full 11,567 training tracks, compared with
`Block_F_reweighted_loss/F1_Training/results/full/N064_q02/` field by field.

`scale.json`, outside the weighting: `N`, `q`, `L`, `z0`, `dz`, `seed`, `width`, `depth`,
`y_floor_rel`, `field`, `in_scale` (all five entries), `states_per_round`, `n_train_tracks`,
`tracks` — 14 of 14 the same. `scale.json["weighting"]`, all 11 constants the same:

| constant | this run | reference |
|---|---|---|
| D_ref | 861.5556435120753 | 861.5556435120753 |
| lev_ref | 2656.3192708333336 | 2656.3192708333336 |
| i_bar | 3752.054296250222 | 3752.054296250222 |
| z1 | 7826.0 | 7826.0 |
| L | 5177.8 | 5177.8 |
| mode | full | full |
| clamp | 5.0 | 5.0 |
| p_lo | 10.0 | 10.0 |
| p_hi | 50.0 | 50.0 |
| rolloff | 0.6931471805599453 | 0.6931471805599453 |
| w_floor | 0.05 | 0.05 |

`history.csv`, restarts 0 and 1, all four fields the same on both:

| restart | loss_before | loss_after | n_iter | func_evals |
|---|---|---|---|---|
| 0 | 7.881635580414379e-06 | 5.4113796501607666e-09 | 200 | 206 |
| 1 | 5.4113796501607666e-09 | 2.186204448970496e-09 | 200 | 205 |

0 of 33 compared fields differ. **I2 passes at N = 64, q = 2.**

Repeated at N = 256, q = 16 (`--stop-after 2`, about 8 minutes), same comparison against
`results/full/N256_q16/`. Again 0 of 33 fields differ:

| constant | both |
|---|---|
| D_ref | 858.1525365132031 |
| lev_ref | 2608.5309053308824 |
| p_lo, p_hi | 10.0, 50.0 |

| restart | loss_before | loss_after | n_iter | func_evals |
|---|---|---|---|---|
| 0 | 3.610290958103318e-07 | 3.120692771081614e-10 | 200 | 212 |
| 1 | 3.120692771081614e-10 | 2.0177449431229676e-10 | 200 | 216 |

That reference run is the one with two processes writing it (PLAN §9), but its interleaving
starts at restart 1198, so restarts 0 and 1 are clean and the comparison stands.

## 5. Gate I3 — the moved window really does change the objective

Two three-restart runs on the same 300 tracks and the same seed, differing only in the window:

```bash
$PY train_windowed.py --N 64 --q 2 --p-lo 3  --p-hi 8  --n-train 300 --n-eval 200 --stop-after 3 --out $S/I3/p03-08
$PY train_windowed.py --N 64 --q 2 --p-lo 10 --p-hi 50 --n-train 300 --n-eval 200 --stop-after 3 --out $S/I3/p10-50
```

| restart | loss_before, window 3–8 GeV | loss_before, window 10–50 GeV | ratio |
|---|---|---|---|
| 0 | 9.7176e-06 | 8.2297e-06 | 1.181 |
| 1 | 5.8939e-09 | 2.6819e-09 | 2.198 |
| 2 | 2.2991e-09 | 1.2820e-09 | 1.793 |

Round-1 restart 0 starts from the same untrained network on the same states, so the 18 %
difference at restart 0 is the weighting alone. After that the two diverge because they are
optimising different objectives (a factor 2.2 by restart 1). The same run at the full 11,567
tracks and the default window (gate I2) starts at 7.8816e-06; the 300-track number 8.2297e-06
is the sample, not the window, which is why the comparison above is run at matched sample size.

`scale.json["weighting"]` records the window as asked: `p_lo = 3.0`, `p_hi = 8.0` in the first,
`10.0` / `50.0` in the second. Every other constant is identical between the two
(`D_ref = 797.7488347363303`, `lev_ref = 2656.319270833333`, `i_bar`, `z1`, `L`, `mode`,
`clamp = 5.0`, `rolloff`, `w_floor`) — `D_ref` is a property of the tracks, not of the window,
so it must not move, and it does not. **I3 passes.**

The start-up guard, checked on the same folder:

```
$ $PY train_windowed.py --N 64 --q 2 --out $S/I3/p03-08        # i.e. the default 10-50
this run was started with the window 3-8 GeV, not 10-50        # exit 1
```

`history.csv` was not appended to: the guard runs before the checkpoint is read and before
anything is written.

Dead end: the contrast case was first written as `--weighting no_lever` on the same `--out`,
expecting the existing `--weighting` guard to fire. It does not, and should not — the weighting
is a level of the run folder, so that command starts a **new** run rather than colliding with
the old one. It was killed after two minutes (it was writing only into the scratch area).
The window is not a level of the path in the same way, which is exactly why it needs the guard.

## 6. The harness, dry-run

`$PY condor/make_jobs.py --p-lo 3 --p-hi 8`:

```
3 jobs, loss window 3-8 GeV, runs under results/p03-08/
-> condor/jobs.txt; submit with: cd <...>/Block_G_low_momentum_window/G1_Training && condor_submit condor/jobs.sub
   --N 256 --q 16 --weighting full --p-lo 3 --p-hi 8 --out results/p03-08 --round-restarts 25 --round-cap 40 --outer-cap 1000
   --N 128 --q 8 --weighting full --p-lo 3 --p-hi 8 --out results/p03-08 --round-restarts 25 --round-cap 40 --outer-cap 1000
   --N 64 --q 2 --weighting full --p-lo 3 --p-hi 8 --out results/p03-08 --round-restarts 25 --round-cap 40 --outer-cap 1000
```

`condor/jobs.sub` is Block F's submit file with the executable pointing at this block's
`condor/wrapper.sh`, `request_cpus = 1`, `request_memory = 8192`, `+UseOS = "el9"`,
`+JobCategory = "medium"`, `queue args from condor/jobs.txt`. Nothing was submitted.

`$PY condor/resubmit.py` (no `--submit`, and no `condor/jobs_active.txt` yet, so it falls back
to the three default lines). The real `condor_q` was readable and returned the running
10–50 GeV job; that job is under the other block's wrapper, so it is correctly not matched:

```
3 runs unfinished and not queued
   --N 256 --q 16 --weighting full --p-lo 3 --p-hi 8 --out results/p03-08 ...
   --N 128 --q 8 --weighting full --p-lo 3 --p-hi 8 --out results/p03-08 ...
   --N 64 --q 2 --weighting full --p-lo 3 --p-hi 8 --out results/p03-08 ...
not submitted: condor/jobs_round1.sub
```

`condor/jobs_round1.txt` and `.sub` from that dry run were deleted afterwards, so the first
real resubmit pass will be round 1.

`$PY condor/prune_active.py --active <a scratch list pointing at the finished 10–50 GeV runs>`
— a read-only smoke test of the imported rule, no `--write`:

```
  keep  N = 256, q = 16, window 10-50 GeV (round 56): the validation error is still falling
  DONE  N =  64, q =  2, window 10-50 GeV (round 40): the validation error has stopped falling
  1 of 2 runs still to train
  not written (pass --write)
```

which is what `F2_Analysis/compare_to_blockE.py` says about those two runs, so the import is
wired up correctly.

## 7. The resubmit self-test

`$PY condor/resubmit.py --self-test` monkeypatches `subprocess.run` (inside the test only, via
the `run=` parameter of `queued()`) and uses a temporary directory as the run root. 12 of 12
checks pass:

```
  condor_q exits non-zero -> unknown                         pass
  condor_q raises -> unknown                                 pass
  condor_q exits 0 with an error on stderr -> unknown        pass
  unknown queue -> nothing is submitted                      pass
  empty queue, read cleanly -> the line IS submitted         pass
  a warning on stderr that is not an error -> still read     pass
  the line's own job in the queue -> held back               pass
  the same N, q at another window -> NOT a match             pass
  the same arguments under another wrapper -> not ours       pass
  progress.json written just now -> held back                pass
  progress.json an hour old -> submitted                     pass
  finished at its restart cap -> held back                   pass
  12 of 12 checks pass
```

The fifth and sixth checks are the ones that keep the fix honest: an empty queue that was read
cleanly must still resubmit, or the keeper would never restart anything, and a `condor_q` that
prints a warning is not a failure.

## 8. Wall time, measured

Medians of `wall_s` over the finished 10–50 GeV runs, which is the best estimate for the new
runs (same node pool, same states per round, same optimiser):

| setting | dz | restarts logged | median restart | 1,000 restarts |
|---|---|---|---|---|
| N = 64, q = 2 | 80.9 mm | 1,000 | 77.7 s | 21.6 h |
| N = 128, q = 8 | 40.5 mm | 1,000 | 118.3 s | 32.9 h |
| N = 256, q = 16 | 20.2 mm | 1,405 | 181.6 s | 50.4 h |

The N = 256 number is the median over its first 1,198 restarts only. From restart 1198 — when
the second process joined it — the median is 232.9 s, 28 % slower, which is two jobs sharing a
node and not the cost of the work. PLAN §8 quotes 226 s and 63 h for this setting; that figure
came from the contended period. **21.6 / 32.9 / 50.4 h** is the corrected expectation, about
105 h of farm time for the three runs, and they run in parallel.

My own gate timings are not a wall-time estimate: they were taken with six processes on the
node at once (112 s a restart at N = 64, 256 s at N = 256).

## 9. Caveats and things left for the coordinator

- **Nothing is submitted.** `condor/jobs.txt` and `condor/jobs.sub` exist for the 3–8 GeV
  anchor window as a dry run. The window is George's to choose (PLAN §6); if he picks another,
  rerun `make_jobs.py --p-lo <lo> --p-hi <hi>` and the folder name follows automatically.
  `condor/jobs_active.txt` is deliberately absent — it is created by copying `jobs.txt` at
  submission, and until it exists `resubmit.py` falls back to the default three lines.
- **`prune_active.py` reaches into the other block** for `plateaued_now`. That is the
  instruction (import, do not copy), but it means `Block_F_reweighted_loss/F2_Analysis/` has to
  stay where it is for this block's keeper to run.
- The duplicate job on the 10–50 GeV N = 256 run (PLAN §9) is **still running** — both
  5823724 and 5833416 were writing that checkpoint during this session. `condor_rm` is George's
  to run; this block's harness cannot cause a repeat but it cannot fix that one either.
- The 3–8 GeV smoke run in the scratch area was stopped after three restarts; it is not a
  result, only evidence that the objective moved.
- `--weighting` is a level of the run path but the window is not, which is why the window needs
  a guard and gets one. If a future block adds a third dimension to the objective, it needs the
  same treatment.

## 10. Files written by this agent

```
Block_G_low_momentum_window/G1_Training/train_windowed.py
Block_G_low_momentum_window/G1_Training/use_shared.py                 (copy of F1's)
Block_G_low_momentum_window/G1_Training/README.md
Block_G_low_momentum_window/G1_Training/results/train_windowed.diff
Block_G_low_momentum_window/G1_Training/condor/wrapper.sh
Block_G_low_momentum_window/G1_Training/condor/make_jobs.py
Block_G_low_momentum_window/G1_Training/condor/resubmit.py
Block_G_low_momentum_window/G1_Training/condor/prune_active.py
Block_G_low_momentum_window/G1_Training/condor/keeper.sh
Block_G_low_momentum_window/G1_Training/condor/jobs.txt               (dry run, 3-8 GeV)
Block_G_low_momentum_window/G1_Training/condor/jobs.sub               (dry run)
Block_G_low_momentum_window/G1_Training/condor/logs/                  (empty, for the farm)
Block_G_low_momentum_window/worklog/2026-09-21_G1_trainer_harness.md  (this file)
```

Nothing under `Block_E_single_network_chain/`, `Block_F_reweighted_loss/` or `_shared/` was
written by this agent (checked with `find -newermt`; the only files that moved there are the
other block's own running job and the byte-compile caches of modules other agents imported).
