# F4 — the Notion write-up for the reweighted-loss study

Built 2026-09-22 to the plan in [PLAN.md](PLAN.md) and the rules in
[../../_shared/WRITEUP_RULES.md](../../_shared/WRITEUP_RULES.md). **Trust = Provisional; only
George sets Verified.**

## The pages

| page | URL |
|---|---|
| main | https://app.notion.com/p/3e35d544b9d981518a83cf80ee9f612c |
| child 1 — the error broken down: per component, per momentum band, and around 5 GeV | https://app.notion.com/p/3e35d544b9d981d091ffec654fd321c6 |
| child 2 — the reproduced analysis set: one step, growth along the crossing, over-training, cost and the case study | https://app.notion.com/p/3e35d544b9d98181b959ee95c89e66dc |
| child 3 — the networks against the Geant4-true SciFi state | https://app.notion.com/p/3e35d544b9d981849ec9c4a3b18ba4d2 |

Title: *Reweighting the label-free loss by what an error costs at the SciFi plane: the same chained
network, three step lengths, and where the remaining error lives (September 2026)*.
Properties: Type = Write up, Status = Done, Date = 2026-09-22, **Trust = Provisional**, Project =
the track-extrapolation project, Todos = the chained-network to-do
(`3dd5d544b9d98144bdcbd17c5a66e3e0`) and the 5 GeV-window follow-up (`3e25d544b9d981849f9cf8817f6bb379`).

## Verification (the rule: the fetched markdown must contain no `\<table` and no `\![`)

All four pages were built with `create-pages` in one call each and then fetched back and checked.

| page | `\<table` | `\![` | real `<table>` blocks | real `![` images | verdict |
|---|---|---|---|---|---|
| main | 0 | 0 | 11 (+ 1 `<table_of_contents/>`) | 4 | **pass** |
| child 1 | 0 | 0 | 8 | 2 | **pass** |
| child 2 | 0 | 0 | 10 | 5 | **pass** |
| child 3 | 0 | 0 | 1 | 1 | **pass** |

Nothing was literalised; the tables are real Notion table blocks and the images are real image
blocks. The three children appear on the main page as `<page url=...>` blocks, confirmed in the
main page's fetched content.

Two plain-text corrections were applied with `update-page` after the first fetch, both to numbers
that an exact recomputation contradicted (see "Corrections" below). `update-page` was used only for
those plain-text edits and for one property edit, never for rich content.

## The figures

**Every figure is pinned to `main` and will be blank until George commits and pushes.** The whole
folder is untracked (local HEAD `1040ee9`, GitHub `main` at `a062bd8`). No commit hash is claimed
anywhere on the pages, and both the top callout and the Provenance say so.

Base URL:
`https://raw.githubusercontent.com/GeorgeWilliam1999/LHCb-Extrapolation/main/single_network_chain_discrete_approach/`

| page | path under that base | source |
|---|---|---|
| main §3.7 | `Block_F_reweighted_loss/F4_Writeup/figures/convergence_check.png` | G2 `convergence.py`, rerun here (de-duplicated) |
| main §3.7 | `Block_F_reweighted_loss/F3_Analysis/figures_writeup/convergence_grid.png` | redrawn by `make_figures.py` |
| main §4.4 | `Block_F_reweighted_loss/F3_Analysis/figures_writeup/error_vs_p_x.png` | redrawn by `make_figures.py` |
| main §4.4 | `Block_F_reweighted_loss/F3_Analysis/figures_writeup/error_vs_p_y.png` | redrawn by `make_figures.py` |
| child 1 §2 | `Block_F_reweighted_loss/F2_Analysis/figures/error_by_component_5-30GeV.png` | F2, already free of block labels |
| child 1 §4 | `Block_F_reweighted_loss/F2_Analysis/figures/signed_errors_4-6GeV.png` | F2, already free of block labels |
| child 2 §1 | `Block_F_reweighted_loss/F3_Analysis/figures_writeup/single_step.png` | redrawn by `make_figures.py` |
| child 2 §2 | `Block_F_reweighted_loss/F3_Analysis/figures_writeup/error_vs_z.png` | redrawn by `make_figures.py` |
| child 2 §5 | `Block_F_reweighted_loss/F3_Analysis/figures_writeup/case_study_components_x0_maps.png` | redrawn by `make_figures.py` |
| child 2 §7 | `Block_F_reweighted_loss/F3_Analysis/figures/mirror_fig2_baseline.png` | F3, already free of block labels |
| child 2 §7 | `Block_F_reweighted_loss/F3_Analysis/figures/mirror_fig6_magnet_up.png` | F3, already free of block labels |
| child 3 §2 | `Block_F_reweighted_loss/F2_Analysis/figures/against_true_state.png` | F2, already free of block labels |

`make_figures.py` also wrote `error_vs_p_tx.png` and `error_vs_p_ty.png` into `figures_writeup/`;
they are not used on any page. Every figure used was inspected and carries **no block letter** in
its title, axis labels or legend.

**One figure named in the plan was not used:** `F0_Weighting/figures/weighting_preflight.png`. Its
suptitle reads "Block F pre-flight: Block E's N = 64, q = 2 network…", and regenerating it through a
title wrapper would mean re-running `check_weights.py`, which writes into F0's own `results/`. The
pre-flight is presented instead as the two tables of section 3.5, which carry the same numbers from
`preflight_shares.csv` and `check_weights.json`. `F2_Analysis/figures/blockF_vs_blockE.png` was also
not used; the plan allowed replacing it with a table, and section 4.3 of the main page is that table.

## Files in this folder

| file | what |
|---|---|
| `PLAN.md` | the specification this page was written to |
| `page.md` | the main page's markdown, as sent to `create-pages`, with the post-publication text correction applied so the file matches the live page |
| `page_child1_error_breakdown.md` | child 1's markdown, as sent |
| `page_child2_reproduced_set.md` | child 2's markdown, as sent, with its two post-publication text corrections applied |
| `page_child3_against_true_state.md` | child 3's markdown, as sent |
| `numbers.py` | every number on the pages that no existing results file held |
| `make_figures.py` | redraws the F3 figures without the block letter, into `../F3_Analysis/figures_writeup/` |
| `results/convergence_check.csv` | the plateau verdicts, from G2's `convergence.py` on these runs |
| `results/errors_near_5gev.csv`, `results/tail_near_5gev.csv` | the around-5-GeV table and the tail baseline, from G2's `errors_near_5gev.py` on these runs |
| `results/writeup_numbers.json` | the output of `numbers.py` |
| `results/figure_rebuild/` | the scratch copy `make_figures.py` lets the drawing code write into, so nothing real is overwritten |
| `figures/convergence_check.png` | written by G2's `convergence.py`; used on the main page |
| `figures/signed_errors_4-6GeV.png` | written by G2's `errors_near_5gev.py`; not used (the F2 one is) |

Nothing outside `F4_Writeup/` and `../F3_Analysis/figures_writeup/` was created or modified. No
existing analysis script was edited; `make_figures.py` imports them, exactly as F3's own runner does.

### Reproducing

```bash
PY=/data/bfys/gscriven/conda/envs/TE/bin/python; export PYTHONNOUSERSITE=1

# the plateau verdicts and the around-5-GeV tail baseline
cd ../../Block_G_low_momentum_window/G2_Analysis
$PY convergence.py       --runs ../../Block_F_reweighted_loss/F1_Training/results/full \
                         --out  ../../Block_F_reweighted_loss/F4_Writeup
$PY errors_near_5gev.py  --runs ../../Block_F_reweighted_loss/F1_Training/results/full \
                         --out  ../../Block_F_reweighted_loss/F4_Writeup

# the derived numbers and the relabelled figures
cd ../../Block_F_reweighted_loss/F4_Writeup
$PY numbers.py
$PY make_figures.py
```

**A trap worth knowing:** this folder contains `numbers.py`, which shadows the standard-library
module `numbers` that numpy imports. Both scripts here remove this folder from `sys.path` before
importing numpy. Any new script in this folder must do the same, or numpy will fail with
`AttributeError: module 'numbers' has no attribute 'Integral'`.

## Corrections made while writing

Two claims that circulate in the F2 README and the to-do worklog did not survive an exact
recomputation, and the pages carry the corrected versions. Both concern the comparison against the
Geant4-true state, and the underlying table (`against_true_state.csv`) is unchanged.

- "the network-vs-true and RK6-vs-true medians differ by 0–4 % in every band". Recomputed per band
  and per network in `numbers.py` (`true_state_ratios`): the agreement is within **3.5 %** from 2 to
  25 GeV but **8.4 %** at 25 to 200 GeV, and the five-track 1 to 2 GeV row scatters from −19 % to
  +0.4 %. The pages say that.
- "the network-vs-RK6 error is 6–20 times smaller than either". The ratio is **11.3 to 25.4** over
  the bands from 2 to 200 GeV, and 7.0 to 10.8 in the five-track 1 to 2 GeV row. The "6" came from
  that five-track row. The pages say that.

Also corrected in passing, on child 2: the exact scheme is a factor 604, 9,100 and 87,800 below the
three networks (not "600 to 80,000"), and the forward cost rises by 2.0 and then 2.8 as the step
count doubles.

The F2 README still carries the original wording; it was not edited, since it is another agent's
section and the rule is not to restructure content one did not write.

## Open items for George

1. **Push.** Until the folder is committed and pushed to `main`, all twelve figures are blank.
2. **Two placeholder links.** The main page and child 3 carry `[link to be filled by George]` in
   three places: twice for the reference-versus-truth write-up and once for the companion write-up
   on the same networks under the pooled loss. Neither page existed when this one was built
   (`../../Block_E_single_network_chain/E4_Writeup/` and `../../reference_vs_truth/` held only their
   plans), so no URL could be filled in.
3. **Ready to verify.** Provenance is located and checked; the page is Provisional and waits on
   George for Verified.
