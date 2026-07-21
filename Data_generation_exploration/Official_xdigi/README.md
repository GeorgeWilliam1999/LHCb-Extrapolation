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
| `../Data/make_training_set.py results/states.npz training_v2 train_official_v2` | legs + RK4 labels + gates -> `training_v2/train_official_v2.npz` (+ gates.json in `results/`) |
| [compare_v1_v2.py](compare_v1_v2.py) | population comparison v1 (ours) vs v2 (official) -> `figures/v1_vs_v2_population.png` |

The harvest and training-set builders are the **v1 scripts parameterised** (CLI args added
2026-07-21 with v1 defaults unchanged) — deliberately shared code, so v1/v2 differences can
only come from the input events, not the pipeline.
