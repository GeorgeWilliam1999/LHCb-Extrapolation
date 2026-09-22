# E0_Track_dataset — the tracks every Block E network trains on and is scored against

**Status (2026-09-16): built and gated.**

**What.** Block D's particles, unchanged: 14,482 forward particles of the official
MagUp sample (2 < η < 5, 1 < p < 200 GeV, no electrons). Each particle's real last-UT
state is moved to z0 = 2648.2 mm with RK6 (0.1 mm steps), and RK6 then carries it
across the crossing. Its state is stored on the 257 planes z0 + k·L/256, which include
every plane of N = 2, 64, 128 and 256. The track is also carried on to the particle's
own SciFi plane and set beside its real state there. The split by particle is Block D's:
training 11,567 (not capped), validation 1,463, test 1,452.

## script → output

| script | what it does | output |
|---|---|---|
| [build_tracks.py](build_tracks.py) | Block D's `build_dataset.py` with the 257-plane grid and no training cap; nothing else changed | `results/tracks.npz` (116 MB), `results/tracks_meta.json`, `figures/tracks_overview.png`; log `build_tracks.log` |
| [check_against_block_d.py](check_against_block_d.py) | the gate: validation and test particles identical to Block D's, in the same order, with bit-identical start states, real states, planes and momenta; RK6 states on the 129 shared planes within 1e-3 µm and 1e-6 mrad; Block D's 2,000 training particles all present | `results/check_against_block_d.json` |

```bash
PY=/data/bfys/gscriven/conda/envs/TE/bin/python; export PYTHONNOUSERSITE=1
$PY build_tracks.py --workers 16      # 5.5 min on a loaded submit host
$PY check_against_block_d.py
```

## Results (2026-09-16)

- **Counts:** training 11,567, validation 1,463, test 1,452. RK6 took 325 s on 16 workers.
- **Gate: PASS.**
  - Validation and test are the same particles as Block D, with every exact field bit-identical.
  - On the shared planes the RK6 states differ from Block D's by at most 1.5e-4 µm in position, 6.3e-8 mrad in slope and 0 in q/p. RK6 restarts its 0.1 mm steps at every stored plane, and Block E stores twice as many, so these tiny differences are expected.
  - All 2,000 of Block D's training particles are present, with bit-identical start states.
- **Material floor** (real SciFi state against the field-only RK6 track), identical to Block D's: 5,136 / 1,608 / 622 / 251 µm median at 2–5 / 5–10 / 10–25 / 25–200 GeV.
