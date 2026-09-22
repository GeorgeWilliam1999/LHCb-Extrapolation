# D2_Comparators — the ceiling, the twin and the comparison  (D2)

**Question.** How far is each chain of D1 from what the equations allow at
that (N, q), from a supervised network, from the straight line, and from the
real detector?

## script → output

| script | what it does | output |
|---|---|---|
| [exact_chain.py](exact_chain.py) | the q-stage Gauss–Legendre scheme solved exactly (C2's root-finder, no network) on every test particle, leg after leg across the magnet, against the RK6 truth: the ceiling at that (N, q) | `results/exact_N<NNN>_q<qq>.json` |
| [make_jobs.py](make_jobs.py) | one farm job per (N, q) for the above | `condor/jobs_exact.*` |
| [train_twin.py](train_twin.py) | the one supervised twin: start state at z0 → end state at z1, no stages (q = 0), the same class and trainer, residual-normalised MSE against the RK6 endpoint | `results/twin/twin.{pt,json,_history.csv}`, `twin_scores.json` |
| [compare.py](compare.py) | joins D1's table with the exact scheme and the twin | `results/comparison_table.csv`, `results/comparison_summary.json` |

```bash
PY=/data/bfys/gscriven/conda/envs/TE/bin/python; export PYTHONNOUSERSITE=1
$PY make_jobs.py && condor_submit condor/jobs_exact.sub
$PY train_twin.py
$PY compare.py            # after ../D1_Chain_grid/aggregate.py
```

## Findings

(filled in when the grid has run)
