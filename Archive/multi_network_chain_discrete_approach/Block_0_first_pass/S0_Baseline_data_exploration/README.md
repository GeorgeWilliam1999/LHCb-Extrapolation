# S0_Baseline_data_exploration — step 0 of the discrete-time port

The reference card and the data-look that everything later in the port imports
and builds on. No new physics, no networks.

| file | what it does |
|---|---|
| [reference_card.py](reference_card.py) | THE importable module: the ODE (`deriv`), the fp64 RK4 reference (`rk4_rows`), the canonical v8r1 field (loaded through the vendored copy `../../_shared/field_v8r1.py`, parity-gated against the archive original in `../../_shared/vendoring_parity.py`; MagUp path recorded and verified to exist), the agreed scalar metric `rho`, the frozen step-2 leg, and `load_training()` for the event-derived sample with its by-particle splits. `python reference_card.py` prints the card. |
| [data_look.py](data_look.py) | measures the loss-normalisation scales and the field along real legs -> `results/scales.{csv,json}`, `figures/field_along_legs.png` |
| [baseline_data_exploration.ipynb](baseline_data_exploration.ipynb) | loads the results, displays the figure |

## The numbers that matter downstream

- **Loss normalisation** (`results/scales.json`): the q+1 reconstruction
  residuals of the paper's loss are made dimensionless by dividing each state
  component by its training-population spread — x 655.6 mm, y 360.2 mm,
  tx 0.2592, ty 0.1152, qop 0.1190 (train split, all legs). Per-leg values in
  `results/scales.csv`; `sigma_corr` there documents the physical correction
  scale per leg (for reading results, not for the loss).
- **Frozen step-2 leg**: z0 = 2648.2 mm (last UT plane) -> z1 = 7826.0 mm
  (first SciFi plane), the modal plane pair of the 11,046 forward cross-magnet
  legs in the training set (measured 2026-07-18).
- **Field along legs** (`figures/field_along_legs.png`): the dipole peaks at
  |By| ~ 1.05 around z = 4.7 m and is remarkably uniform across real leg paths
  (narrow 10-90% band); bending integrals separate the leg types by three
  orders of magnitude (cross-magnet/downstream ~ 4e3 vs plane-to-plane ~ 1);
  the B-leg x-correction over the straight line falls as 1/p from metres at
  1 GeV to ~30 mm at 200 GeV — this is the function the technique must learn.
