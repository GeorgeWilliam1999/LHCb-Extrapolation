# E4_Writeup — the Notion write-up of this study

**Status (2026-09-22): published, Trust = Provisional, verified.** Eleven Notion pages: one main page in
the write-up database and ten child pages under it. Nothing in this repository folder is committed,
so **every figure on those pages is blank until George commits and pushes** `single_network_chain_discrete_approach/`
(local HEAD `1040ee9`, GitHub main `a062bd8`). No commit hash is claimed anywhere on the pages.

## The pages

| page | URL | local copy |
|---|---|---|
| **One network per step length, chained to itself across the LHCb magnet: a label-free Runge–Kutta network at every step count and stage count (September 2026)** (main) | https://app.notion.com/p/3e35d544b9d98117b366f3283549b60d | `page.md` |
| Method, protocols 1 to 3: the tracks, the discrete-time construction, and the network | https://app.notion.com/p/3e35d544b9d981c786c0e1f5ef0d4a64 | `child_1.md` |
| Method, protocols 4 to 6: the loss, the training protocol, and the gates before the farm | https://app.notion.com/p/3e35d544b9d981a6bf27f177860a78f0 | `child_2.md` |
| Method, protocols 7 to 10: the grid and the farm, stopping and the lesson, the comparators, and the scoring | https://app.notion.com/p/3e35d544b9d981cfb5c3c8bc84aaff66 | `child_3.md` |
| Results: the sixteen endpoints, a single step, growth along the crossing, and the comparators | https://app.notion.com/p/3e35d544b9d981879044eb068cb66edb | `child_4.md` |
| Results: momentum, starting position, the components, and what the endpoint error is made of | https://app.notion.com/p/3e35d544b9d981948dcbe7695b50f5ab | `child_5.md` |
| Results: convergence, over-training, cost, and where the loss goes | https://app.notion.com/p/3e35d544b9d9817d944ad09318a22103 | `child_6.md` |
| The sixteen networks run by run: training, stopping and the endpoint | https://app.notion.com/p/3e35d544b9d981cd8b2bd703b87aa5b7 | `child_7.md` |
| Figure gallery: the endpoint error against momentum, per component, for all sixteen networks | https://app.notion.com/p/3e35d544b9d98149af8cca8ee06927f0 | `child_8.md` |
| The case study in detail: 64 steps of 80.9 mm with two stages | https://app.notion.com/p/3e35d544b9d981358900f38702601880 | `child_9.md` |
| Provenance in full | https://app.notion.com/p/3e35d544b9d981c7b567cb93d89cb04d | `child_10.md` |

Properties on the main page: Type = Write up, Status = Done, Date = 2026-09-22, **Trust = Provisional**
(never Verified; only George sets that), Provenance, Note, Project → Track Extrapolation, Todos →
"Train one network per step length and chain it to itself across the magnet".

The `.md` files are the pages as Notion returned them on a read-back, so they are the published form,
not the submitted form; they round-trip a few inline escapes (for example `\[µm\]` inside a table cell).

## Verification (2026-09-22)

Every one of the eleven pages was fetched back and its raw markdown checked for the two markers that
mean Notion literalised a block instead of parsing it: a backslash-escaped `\<table` and a
backslash-escaped `\![`. **All pages PASS: neither marker occurs anywhere.** Across the set the pages
carry **19 real `<table …>` blocks and 16 real `![caption](https://raw.githubusercontent.com/…)` image
lines.** Note that a plain `<table` substring count is not a valid check: the main page contains
`<table_of_contents/>`, which is not a table block.

## Figures

All figures are pinned to `main` under
`https://raw.githubusercontent.com/GeorgeWilliam1999/LHCb-Extrapolation/main/single_network_chain_discrete_approach/…`
and **render only after George pushes this folder.** Fifteen of the sixteen image lines point at
`Block_E_single_network_chain/E3_Analysis/figures_writeup/`, produced by `make_figures.py` below; the
sixteenth is `Block_E_single_network_chain/E0_Track_dataset/figures/tracks_overview.png`, used as it is.

| figure | page |
|---|---|
| `E0_Track_dataset/figures/tracks_overview.png` | method 1 to 3 |
| `figures_writeup/error_qdz.png` | results: endpoints |
| `figures_writeup/single_step.png` | results: endpoints |
| `figures_writeup/error_vs_z.png` | results: endpoints |
| `figures_writeup/case_study_components_3d.png` | results: momentum and anatomy |
| `figures_writeup/case_study_components_x0_maps.png` | results: momentum and anatomy |
| `figures_writeup/case_study_components.png` | results: momentum and anatomy |
| `figures_writeup/error_anatomy.png` | results: momentum and anatomy |
| `figures_writeup/convergence_grid.png` | results: convergence, cost, loss |
| `figures_writeup/convergence_summary.png` | results: convergence, cost, loss |
| `figures_writeup/case_study_loss_vs_p.png` | results: convergence, cost, loss |
| `figures_writeup/error_vs_p_x.png`, `_y.png`, `_tx.png`, `_ty.png` | figure gallery |
| `figures_writeup/case_study_overview.png` | the case study |

`E1_Network_grid/figures/y_scale_check.png` is **not** used: its axis titles name the earlier study by
its block letter, and regenerating it would have meant rerunning a gate script and overwriting a
result file. Gate 3 is reported in numbers instead.

## script → output

| script | what it does | output |
|---|---|---|
| [make_figures.py](make_figures.py) | imports the nine E3 scripts **unedited** and runs them against the kept 2026-09-18 snapshots (`E1_Network_grid/results/N*/stopped_2026-09-18/`, reached through a symlink view), with matplotlib's `suptitle`/`set_title` wrapped so the block label is stripped from the titles E3 hard-codes; then copies the figures into `../E3_Analysis/figures_writeup/` and checks the regenerated tables against E3's published ones cell by cell | `runs_stopped_2026-09-18/`, `regen/results/*.csv`, `regen/figures/*.png`, `regen/run_log.json`, `../E3_Analysis/figures_writeup/*.png` |
| [numbers.py](numbers.py) | every ratio, range, share and correlation the pages quote that is not already a cell of a results file | `results/numbers.json` |

```bash
PY=/data/bfys/gscriven/conda/envs/TE/bin/python; export PYTHONNOUSERSITE=1
$PY make_figures.py        # ~9 min, evaluate_splits is most of it
$PY numbers.py
```

**Reproduction check: 0 mismatches over 192 compared cells** (`error_qdz_chain.csv`,
`error_qdz_single_step.csv`, `cost_accuracy.csv`, `tails.csv`), so the figures on the pages and the
tables in `../E3_Analysis/results/` describe the same networks.

## Which numbers came from where

- **Everything labelled "as stopped on 18 September 2026"** is `regen/results/*.csv`, identical to
  `../E3_Analysis/results/*.csv` where both exist. `overtraining.csv` and `split_comparison.csv` are the
  exception: E3's copies were written earlier that day at the 400-restart cap, so the write-up quotes the
  regenerated versions, which are at the same 2026-09-18 stop as everything else.
- **Everything labelled "after the extension"** is `../../Block_F_reweighted_loss/F2_Analysis/results/headline.csv`
  (written 2026-09-22 16:27), rows with `weighting = blockE`.
- **The two-slope anatomy** is `../../Block_F_reweighted_loss/F2_Analysis/results/anatomy_xy_blockE_*.json`,
  which describes the **extended** networks, and the pages say so.
- **The starting-x bands and the Spearman correlation** are computed in `numbers.py` from
  `tracks.npz` and the stored chain states, because E3's `case_study_3d.py` bins signed `x0` per
  component and the write-up wanted the radial error against `|x0|`.

## Two Notion gotchas found on the way (2026-09-22)

1. **`create-pages` with a `data_source_id` parent rejects long content.** Anything from about 7,800
   characters of markdown upward comes back as `validation_error: "Content must be read and
   manipulated separately: content"`. The same content with a **page** parent, or as a workspace
   draft, goes through at 9,500 characters without complaint, so the limit is specific to the
   database-parented path and is not about tables, images, equations, mentions or `allow_async`.
   **Route that works:** create the page as a draft with `creation_mode: "draft"`, move it into the
   data source with `notion-move-pages`, then set its properties with `update-page`
   `update_properties`. The rich content is still built by `create-pages` in one call, so the
   tables-and-images rule of `_shared/WRITEUP_RULES.md` is kept; only properties go through
   `update-page`. That is how the main page here was published.
2. **Child pages are unaffected**: `create-pages` with a `page_id` parent took 13 kB of markdown with
   tables and images in a single call.

Three scratch probe pages were created while finding that limit and are now private pages in the
workspace titled "DELETE ME - scratch size probe from the Block E write-up (2026-09-22)":
https://app.notion.com/p/3e35d544b9d981ae96cddc2d8dbeec43,
https://app.notion.com/p/3e35d544b9d98129a4c1d073f15955bd and
https://app.notion.com/p/3e35d544b9d98125814fc513a9f1b448. They can be deleted.
