# Official_xdigi — training data v2 from the official TestFileDB sample

George's directive (2026-07-21): use centrally-produced TestFileDB samples instead of
generating events ourselves — small config errors in home-grown generation can cause huge,
silent issues. This folder re-harvests the training population from the official
`expected_2024_minbias_xdigi` sample and rebuilds the training set (v2) with the same
pipeline and gates as v1, so the two are directly comparable.

## The sample

- **TestFileDB entry**: `expected_2024_minbias_xdigi` (PRConfig), production 00212966.
- **Access**: local CVMFS mirror — no EOS, kerberos, or DIRAC:
  `/cvmfs/lhcbdev.cern.ch/testfiledb-mirror/lhcb/swtest/expected_2024_minbias_xdigi/`
  (5 of the entry's 8 files mirrored, 19 GB).
- **Qualifiers** (travel with the entry — the point of TestFileDB): Format XDIGI
  (extended digi = digitised banks **plus** packed MC truth `pSim/...`), Simulation,
  DDDB `dddb-20231017`, CondDB `sim-20231017-vc-mu100`.
- Caveat for consumers: the PRConfig bundled with Moore v59r4 predates the CVMFS-mirror
  rewrite; use `options.set_input_and_conds_from_testfiledb(...)` for conditions and point
  `options.input_files` at the mirror paths by hand.

## Pipeline (same as v1 from the harvest onwards)

| file | what it does |
|---|---|
| [dump_xdigi.py](dump_xdigi.py) | GaudiPython MC-truth dump of the .xdigi stream (SimConf unpack recipe from `First_Pass/dump_event.py`) -> `truth_official/{particles,vertices,hits,collisions}.csv` |
| [run_dump.sh](run_dump.sh) | LbEnv wrapper: `bash run_dump.sh truth_official 200` |
| `../Data/harvest_states.py truth_official results` | states from MCHits -> `results/states.npz` |
| `../Data/make_training_set.py results/states.npz training_v2 train_official_v2 official_xdigi` | legs + RK4 labels + gates -> `training_v2/train_official_v2.npz` (+ gates.json in `results/`). The 4th argument names the sample description written into the meta json (`gauss_mb100` = v1 default, `official_xdigi` = this sample). |
| [compare_v1_v2.py](compare_v1_v2.py) | population comparison v1 (ours) vs v2 (official) -> `figures/v1_vs_v2_population.png` |

The harvest and training-set builders are the **v1 scripts parameterised** (CLI args added
2026-07-21 with v1 defaults unchanged) — deliberately shared code, so v1/v2 differences can
only come from the input events, not the pipeline.

## Metadata correction (2026-09-05)

`training_v2/train_official_v2.meta.json` was written before the sample description was
parameterised, so its `sample` block still described the v1 Gauss run (100 self-generated
events) while every other field described this build. The block now describes this sample
(TestFileDB `expected_2024_minbias_xdigi`, production 00212966, DDDB `dddb-20231017`,
CondDB `sim-20231017-vc-mu100`, 5 mirrored files, 200 events), and the file records the fix
under `meta_corrected`. `../Data/make_training_set.py` now takes the sample key as its
4th argument, so a rebuild writes the right block by itself. No data changed.

## The larger harvest (2026-11-13/14)

George asked for more tracks of the same sample (2026-11-13). The first 2,000 events of the same
concatenated stream were read with the same scripts; events 0 to 199 are those of v2.

| Step | Command | Output |
|---|---|---|
| truth dump | `bash run_dump.sh truth_official_2000 2000` (35 minutes) | `truth_official_2000/` (4.7 GB, not in git) |
| states | `python ../Data/harvest_states.py truth_official_2000 results_2000` | `results_2000/states.npz` (11.7 million states) |
| sample | `python ../Data/make_training_set.py results_2000/states.npz training_v3 train_official_v3 official_xdigi_2000 training_v2/train_official_v2.npz` | `training_v3/train_official_v3.npz` (3,029,075 rows; 387,972 across the magnet against 41,398 in v2) |

The fifth argument keeps the split of v2: each of the 76,353 particles of v2 keeps its split, and
the 638,910 new particles are split at random in the same proportions. Checked: the rows of v2
across the magnet are in v3 unchanged, to the last bit, with the same split. The four gates give
the same values as in v2 (`results_2000/gates.json`).
