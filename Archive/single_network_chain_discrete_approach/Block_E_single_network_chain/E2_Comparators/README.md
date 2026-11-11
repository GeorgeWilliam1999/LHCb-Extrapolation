# E2_Comparators — what each Block E network is measured against

**Status (2026-09-20): built.** `exact_chain.py` ran as cluster 5805461 (2026-09-17): the exact
collocation scheme chained at N = 2 and 256 for q = 2, 4, 8, 16 on the 1,452 test tracks, every
solve converged. At N = 2 the median error at z1 is 9,270 / 195 / 7.3 / 5.1 µm for q = 2 / 4 / 8 / 16
(95th percentile 38,521 / 1,009 / 82 / 46 µm). The planned `compare.py` was never written: the
comparator columns (exact scheme, straight line, material floor, Block D) are produced by
`../E3_Analysis/tables.py` into `comparators.csv` and `error_qdz_chain.csv`.

- **The exact collocation scheme** chained at the same N and q, solved without a
  network: the ceiling. New runs at N = 2 and 256; Block D's runs at N = 64 and 128
  are reused (the E0 gate guarantees the same test particles).
- **The straight line**, the null.
- **The particle's real SciFi state**: the material floor, which no field-only method
  can beat.
- **Block D's one-network-per-step chains** at N = 64 and 128 for q = 2, 4, 8, 16.
  Caveat: those networks stopped training early (the optimiser's absolute
  tolerances).

## script → output

| script | what it does | output |
|---|---|---|
| [exact_chain.py](exact_chain.py) | the exact scheme at N = 2 and 256 for q = 2, 4, 8, 16 on the test tracks | `results/exact_N<NNN>_q<qq>.json`, `results/exact_N<NNN>_q<qq>_states.npz` |
| `compare.py` | not written; its role is in `../E3_Analysis/tables.py` | — |
