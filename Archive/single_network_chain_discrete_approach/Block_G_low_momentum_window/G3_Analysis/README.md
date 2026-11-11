# G3 — the full analysis set, reproduced for these networks

One script. It runs the nine analyses of the earlier blocks' figure set on this block's runs by
**importing** them, never by copying them: the analysis code has one home, and this folder only
points it at different runs and at a different output folder.

```
cd G3_Analysis
PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python run_e3_for_block_g.py
```

| flag | default | meaning |
|---|---|---|
| `--runs DIR` | `../G1_Training/results/p03-08/full` | the folder holding the `N<NNN>_q<qq>` run folders |
| `--out DIR` | this folder | where `runs/`, `comparators/`, `results/` and `figures/` are written |
| `--only A,B` | all nine | a subset of `convergence, tables, single_step_tables, along_z, evaluate_splits, errors_vs_momentum, case_study, case_study_3d, error_anatomy` |
| `--case N,q` | the best mean validation error over the last eight rounds | which network the in-depth study, the 3D study and the error anatomy use |
| `--no-relabel` | off | leave the imported code's own wording in place (for checking what it said) |

## What it produces

| analysis | outputs | about |
|---|---|---|
| `convergence` | `figures/convergence_grid.png`, `convergence_summary.png`, `results/convergence.csv` | training loss and validation error against restart, one panel per network |
| `tables` | `results/error_qdz_chain.csv`, `tails.csv`, `cost_accuracy.csv`, `comparators.csv`, `figures/error_qdz.png` | the endpoint error at the SciFi plane, its tails, its cost per track, and the comparators |
| `single_step_tables` | `results/error_qdz_single_step.csv`, `single_step_vs_z.csv`, `figures/single_step.png` | **single-step** error: one application from the RK6 state on every start plane |
| `along_z` | `results/error_vs_z.csv`, `figures/error_vs_z.png` | how the error grows plane by plane across the crossing |
| `evaluate_splits` | `results/split_comparison.csv`, `overtraining.csv` | train / validation / test, recomputed from `network.pt` |
| `errors_vs_momentum` | `results/error_vs_p.csv`, `figures/error_vs_p_{x,y,tx,ty}.png` | |error| against momentum per component — the figures the supervisor asked for |
| `case_study` | `results/case_study_*.csv`, `figures/case_study_{components,overview,loss_vs_p}.png` | one network in depth, including the loss share by momentum |
| `case_study_3d` | `figures/case_study_components_3d.png`, `case_study_components_x0_maps.png`, `results/case_study_error_vs_p_x0.csv` | the error against momentum **and** starting x |
| `error_anatomy` | `results/error_anatomy.csv`, `error_anatomy_summary.json`, `figures/error_anatomy.png` | the x-slope lever-arm decomposition, step by step |

`results/run_log.json` records, for every analysis, whether it ran and how long it took; a
partial re-run with `--only` keeps what the earlier runs recorded.

Wall clock on three networks (N = 64, 128, 256): about 3 and a half minutes for the whole set —
`evaluate_splits` 90 s, `error_anatomy` 37 s, `case_study` 29 s, `single_step_tables` 14 s, the
rest a few seconds each.

## How it points the imported code here

- `HERE` → `--out`, so `results/` and `figures/` are written there, and the scripts that read
  `results/convergence.csv` to pick the network to study in depth read this block's one.
- `E1` → `<out>/runs/results/N<NNN>_q<qq>`, a view of symlinks to the runs that have both a
  record and stored chain states, rebuilt on every call. The imported scripts glob run folders
  and expect a record in each, and a run that has not written one yet must not appear.
- `BD` → `<out>/comparators/`, a view holding **only** `D2_Comparators` (the q-stage
  collocation scheme chained without a network — a numerical-scheme comparator, kept). The
  other study's trained chains are not linked, so the row that would compare this block's
  networks with another block's networks is simply absent: this block is being looked at in
  isolation. The straight line and the material floor are unaffected.
- The four analyses that assume the full N × q grid (`convergence`, `errors_vs_momentum`,
  `along_z`, `single_step_tables`) come from the earlier block's `grid_free_panels.py`, again
  imported, with its own `HERE` and `RUNS` pointed here: the same computation, one panel or one
  series per run.

## Wording

The imported titles and printed tables carry block letters in their source. An output a human
reads must not, so the runner wraps matplotlib's `suptitle` and `set_title` and the built-in
`print`, and rewrites the CSV headers and cells it produced, so that they read **"one network
per step length, chained; loss window &lt;lo&gt;–&lt;hi&gt; GeV"** — the window read from each
run's `scale.json`, never typed in. "error at the SciFi plane" becomes "endpoint error at the
SciFi plane, after the full chain", because an output has to say which kind of error it shows.
A title longer than 110 characters is wrapped, since the replacement is much longer than the
two words it replaces. The drawing itself is not changed, and `--no-relabel` turns all of it
off. The one column that held the other study's chains is empty here and is dropped from the
CSV (`run_log.json` lists which files were rewritten).

## Caveats

- **`figures/error_qdz.png` assumes the full N × q grid.** With three runs on a diagonal it is
  a mostly-empty matrix with three filled cells. It is inherited as it is; the four worst
  offenders are already replaced by the grid-free panels, this one is not.
- **`convergence` counts rows, not rounds.** The imported reader does not de-duplicate, so for
  a run two farm jobs have been writing at once its round and restart counts are inflated.
  `G2_Analysis/convergence.py` is the one that de-duplicates, and it is the one to read for the
  plateau verdict and the headline number.
- **`evaluate_splits`, `case_study`, `case_study_3d`, `error_anatomy` and `single_step_tables`
  load `network.pt`**, so for a run still training they describe the weights as they are at that
  moment, which are ahead of the stored `chain_states.npz` the table analyses use.
- **The in-depth study picks the best mean validation error over the last eight rounds**, which
  can be a run that is still training. `--case N,q` overrides it.
- **`overtraining.csv` evaluates the earlier block's unweighted loss** on the network, which is
  not the loss these networks were trained on; its chain-error test/train ratio is the column to
  read, not its loss columns.
