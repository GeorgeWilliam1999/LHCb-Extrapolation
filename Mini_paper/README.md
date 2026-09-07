# LHCb extrapolation mini-paper

Overleaf-ready LaTeX source for the mini-paper on the label-free discrete-time
extrapolation programme (16 July to 6 September 2026). Built in the same mould as
the van der Pol mini-paper (`Van_Der_Pole/vdp_minipaper/`): same document class,
same packages, same figure and table conventions.

Every number in the paper traces to a committed file in this repository. The two
tables below give the map.

## Build

```sh
sh build.sh          # pdflatex, bibtex, pdflatex, pdflatex
make                 # the same thing
make zip             # rebuilds Mini_paper.zip
```

Requires `pdflatex` and `bibtex` only. No exotic packages: geometry, amsmath,
amssymb, graphicx, booktabs, caption, natbib, xcolor, hyperref.

## Overleaf

Upload `Mini_paper.zip` via **New Project -> Upload Project**. Compiles with
pdfLaTeX; bibliography via BibTeX (`natbib`, `unsrtnat`).

## Contents

- `main.tex` -- the paper (single file).
- `references.bib` -- 12 entries. Two carry `% TODO verify` comments (see below).
- `figures/` -- 13 PNGs, copied unmodified from the experiment folders.
- `build.sh`, `Makefile` -- the build.
- `Mini_paper.zip` -- the Overleaf bundle.

## Table -> source map

All paths are relative to the repository root
(`/data/bfys/gscriven/LHCb_Extrapolation_Project`). `R/` abbreviates
`Raissi_disc_time_approach/`, `D/` abbreviates `Data_generation_exploration/`.

| table | what it holds | source file(s) |
|---|---|---|
| 1 (`tab:legs`) | the four leg types and the row counts | `D/Official_xdigi/training_v2/train_official_v2.meta.json` (`leg_types`, `rows`); `D/Official_xdigi/README.md` |
| 2 (`tab:gates`) | integrity gates G1-G4 on the labels | `D/Official_xdigi/training_v2/train_official_v2.meta.json` (`gates`) |
| 3 (`tab:baseline`) | the July baseline on the frozen leg | `R/One_step_network_v2/results/summary.csv`; `R/One_step_network_v2/README.md` |
| 4 (`tab:stages`) | stage-count sweep, q in {2,4,8,16} | `R/Stage_count_sweep/results/error_vs_stages.csv`, `results/summary.csv`, `results/scheme_ceiling_same_population_q*.json`; `R/Stage_count_sweep/README.md` |
| 5 (`tab:grid`) | the 12-architecture grid plus the twin | `R/Network_size_and_seed_study/results/by_architecture.csv`, `results/summary.csv`; `R/Network_size_and_seed_study/README.md` |
| 6 (`tab:magnetup`) | reversed magnet polarity against the original | `R/Magnet_up_field/results/summary.csv`, `results/up_vs_down.csv`, `results/ceiling_summary.json`, `results/field_up_parity.json`, `results/loss_field_probe.json` |
| 7 (`tab:general`) | one network for legs A/B/C, absolute outputs, 3 widths | `R/General_leg_network/results/by_leg.csv`, `results/scheme_ceiling_same_population.csv`; `R/General_leg_network/README.md` (Verdict and "Second wave (4x200)") |
| 8 (`tab:chains-abs`) | the same networks chained, 1 to 7 legs | `R/Chained_legs/results/chain_summary.csv`; `R/Chained_legs/README.md` (Verdict 1 and "Second wave (4x200)") |
| 9 (`tab:legd-abs`) | leg D, composite against one giant step | `R/Chained_legs/results/leg_d_reproduction.csv`, `results/leg_d_ceiling_same_population.csv` |
| 10 (`tab:residual`) | residual redesign against absolute, per leg | `R/General_leg_network/results/by_leg_residual.csv`; `R/General_leg_network/README_residual.md` |
| 11 (`tab:residual-split`) | whole test split, both designs, three widths | `R/General_leg_network/results/residual_summary.csv`; `R/General_leg_network/README_residual.md` |
| 12 (`tab:stage-errors`) | error at each Gauss node and the endpoint | `R/General_leg_network/results/stage_errors_residual.csv` (and `results/stage_errors.csv` for the absolute rows) |
| 13 (`tab:tails`) | 95th percentiles by leg | `R/General_leg_network/results/by_leg_residual.csv`, column `endpoint_p95_um` |
| 14 (`tab:chains-residual`) | residual networks chained, 1 to 7 legs | `R/Chained_legs/results/chain_summary_residual.csv`, `results/selection_residual.csv` |
| 15 (`tab:legd-residual`) | leg D under the residual parameterisation | `R/Chained_legs/results/leg_d_residual.csv`, `results/leg_d_ceiling_same_population.csv` |
| 16 (`tab:verdict`) | the assembled verdict | all of the above; the columns are drawn from tables 3, 5, 6, 7, 10 and 14 |
| A.1 (figure map) | figure to repository path | this file and Appendix A of the paper |

Numbers quoted in prose but not in a table:

| claim | source |
|---|---|
| Spearman +0.965 (loss vs error), +0.995 (val vs test), seed ratios | `R/Network_size_and_seed_study/results/summary.csv`; README section "The floor is the optimiser" |
| 44 root-finder residual evaluations per leg | `R/Simple_first_pass/exact_scheme.py` run log; quoted in `R/Simple_first_pass/README.md` |
| the loss scales (655.6 mm, 360.2 mm, ...) | `R/General_leg_network/results/general_legs_meta.json`; `R/Stage_count_sweep/results/frozen_leg_q08_meta.json` |
| the O(1) scale check medians 1.09 / 1.03 / 0.95 | `R/General_leg_network/results/residual_scale_check.json` |
| the initialisation check numbers | `R/General_leg_network/results/residual_init_check.json` |
| dataset size rule, N = 8000, leg mix 1317/1060/5558 | `R/General_leg_network/results/dataset_meta.json`, `results/general_legs_meta.json` |
| chain construction counts (4563 test particles, 23500 legs) | `R/Chained_legs/results/chains_meta.json` |
| field map grid, md5s, polarity parity | `R/_shared/results/vendoring_parity.json`; `R/Magnet_up_field/results/field_up_parity.json` |
| the sample's polarity, and the correction box in §4.1 (1.73 mm vs 883 mm; 14.8 / 0.28 mm and 3973 / 169 mm by band; 100.0 % vs 0.0 % bend sign; 2 % of legs leaving the map; 15,257 particles) | `R/Magnet_tracks_dataset/results/polarity_check.json`; `R/Magnet_tracks_dataset/README.md`; figure `R/Magnet_tracks_dataset/figures/field_polarity.png` |
| restart counts and wall times, both waves | `R/General_leg_network/README_residual.md` ("The farm"); the `*_history.csv` files |
| cluster identifiers | the `README.md` / `README_residual.md` of each experiment folder |

## Figure -> source map

| figure | file | source in the repository |
|---|---|---|
| 1 | `fig_population.png` | `D/Official_xdigi/figures/v1_vs_v2_population.png` |
| 2 | `fig_baseline.png` | `R/One_step_network_v2/figures/one_step_results_v2.png` |
| 3 | `fig_stages.png` | `R/Stage_count_sweep/figures/error_vs_stages.png` |
| 4 | `fig_architecture.png` | `R/Network_size_and_seed_study/figures/floor_vs_architecture.png` |
| 5 | `fig_loss_vs_error.png` | `R/Network_size_and_seed_study/figures/loss_vs_error.png` |
| 6 | `fig_magnet_up.png` | `R/Magnet_up_field/figures/magnet_up_results.png` |
| 7 | `fig_general_legs.png` | `R/General_leg_network/figures/error_by_leg_and_momentum.png` |
| 8 | `fig_chains.png` | `R/Chained_legs/figures/error_vs_chained_legs.png` |
| 9 | `fig_residual_scale.png` | `R/General_leg_network/figures/residual_scale_check.png` |
| 10 | `fig_residual_legs.png` | `R/General_leg_network/figures/error_by_leg_and_momentum_residual.png` |
| 11 | `fig_residual_stages.png` | `R/General_leg_network/figures/stage_errors_residual.png` |
| 12 | `fig_leg_d.png` | `R/Chained_legs/figures/leg_d_reproduction.png` |
| 13 | `fig_frozen_vs_general.png` | `R/General_leg_network/figures/frozen_vs_general_residual.png` |

Every one of those PNGs is regenerated by its folder's `plot.py` (or
`plot_residual.py`, `plot_chains.py`) from committed CSV tables only.

## Bibliography: fields still to verify

Two entries carry `% TODO verify` comments in `references.bib`:

- `rohrhofer2023` -- TMLR does not use volume or page numbers; the canonical
  citation string and the OpenReview identifier were not checkable offline.
- `lhcb2024upgrade` -- volume, article number and DOI (JINST 19 (2024) P05065,
  doi:10.1088/1748-0221/19/05/P05065) are from memory and must be confirmed
  against the journal record.

`scriven2026vdp` is the van der Pol companion paper, cited as "in preparation".

## Discrepancies found while writing, and what was corrected at source

Every number in the paper was recomputed from the committed CSVs rather than
copied from the experiment READMEs. That caught six cells and two derived
figures where the READMEs disagreed with their own result files. All of them
have been corrected at source, in commits `1b7e5b6` and `66e8549`; no CSV,
figure or notebook was touched, because the result files were right all along.
This table is the full record.

### Root cause 1: the unconverged 4x200 seed was pooled

`General_leg_network/README.md` records that `w200_physics_s5` did not converge
and the protocol says unconverged runs are never pooled into a median.
`README_residual.md` nevertheless quoted medians over all ten seeds.

| cell | README said | CSV says (converged seeds, test) | file |
|---|---|---|---|
| by-leg, wave-1 4x200 physics, legs A / B / C | 234 / 1781 / 213 um | **229 / 1740 / 213 um** | `General_leg_network/results/by_leg_residual.csv` (`by_leg.csv` and `General_leg_network/README.md` already agreed) |
| by-leg, wave-1 4x200 physics, leg C seed range | [165-254] | **[165-235]** | same |
| whole split, wave-1 4x200 physics | 274 um | **273 um** (272.76) | `General_leg_network/results/residual_summary.csv` |

The all-seed medians are 234.1 / 1780.5 / 213.4 and 274.5, which is exactly what
the old cells said, so the cause is the pooling rule and not the re-scoring pass.
`README_residual.md` now states both values where it matters.

### Root cause 2: the residual data-twin cells were quoted high

The residual 4x100 twin on the two short legs was out by roughly an order of
magnitude in the by-leg table, and rounded away in the by-momentum-band table.

| cell | README said | CSV says | file |
|---|---|---|---|
| by-leg, residual 4x100 twin, leg A | 0.13 um | **0.058 um** [0.047-0.063] | `General_leg_network/results/by_leg_residual.csv` |
| by-leg, residual 4x100 twin, leg C | 0.04 um | **0.012 um** [0.0118-0.0122] | same |
| by-momentum-band, residual 4x100 twin, leg A (1-5 / 5-20 / 20-200 GeV) | 0.20 / 0.04 / 0.02 um | **0.152 / 0.034 / 0.031 um** | same |
| by-momentum-band, residual 4x100 twin, leg C (1-5 / 5-20 / 20-200 GeV) | 0.05 / 0.02 / 0.02 um | **0.044 / 0.005 / 0.003 um** | same |
| chained, residual 4x100 twin at one leg | 0.03 um | **0.01 um** (0.0090) | `Chained_legs/results/chain_summary_residual.csv` |

The leg-B twin cells were right throughout (512 um by leg; 2575 / 316 / 110 um
by band).

### The derived figures that followed from those cells

| claim | README said | recomputed from the CSVs |
|---|---|---|
| residual twin against wave-1 twin, legs A and C | 3,400x and 165x | **6,600x and 2.0 x 10^4** = 384.774 / 0.0583 and 235.242 / 0.0119, medians over the three converged twin seeds, test split, all momenta |
| the same factor restated in the tails section | 3,400x | **6,600x** |
| per-band improvement, wave-1 physics over residual physics, legs A and C | a factor 46 to 115 in every band | **a factor 46 to 200**: 46 / 70 / 80 on leg A and 200 / 154 / 116 on leg C |

The chained twin growth-ratio row (299, 11.8, 3.69, ...) was computed from the
unrounded 0.0090 and is correct as it stands; it was not changed.

### What this cost the paper

One cell: the caption of Figure 10 repeated "a factor 46 to 115 in every band"
and now reads 46 to 200. Every other number in the paper was already taken from
the CSVs and needed no change, including the whole-split 273 um and the twin
values 0.06 / 512 / 0.01 um. The paper was rebuilt after the caption fix.

### Correction, 6 September 2026: the sample is magnet-up (applied 2026-09-07)

Established by the magnet-to-magnet data set build
(`R/Magnet_tracks_dataset/check_polarity.py`, output
`results/polarity_check.json`): the official sample's conditions tag
`sim-20231017-vc-mu100` is **magnet-up**. Propagating each particle's forward
cross-magnet leg with `field.v8r1.up.bin` lands a median 1.73 mm from that
particle's own real first-SciFi state, with the bend sign right for 100.0 % of
15,257 particles; with `field.v8r1.down.bin` it lands 883 mm away, the sign
wrong for 100 %, and 2 % of legs leave the field map. Every label and score in
this paper used the down map.

No result number in the paper changed. What changed is a boxed correction and
seven sentences:

| where | before | after |
|---|---|---|
| §4.1, end of "The sample" | -- | new boxed **"Correction, 6 September 2026: the sample is magnet-up"** with the up-vs-down numbers and the four consequences; `\label{sec:sample}` added |
| abstract | "Trained on the reversed magnet polarity, for which no labelled sample exists anywhere in the project" | "Trained on the **magnet-up** polarity, for which no labelled sample exists anywhere in the project" |
| §4.2, harvest step 4 | "...using the canonical field map at magnet-down polarity." | same sentence + footnote: "The sample itself is magnet-up; see the correction box in Section 4.1. The labels are therefore field-only propagations of the right start states through the wrong polarity..." |
| §4.3, after the G2 paragraph | "...deliberately outside the surrogate's remit and stays with the existing extrapolator." | same sentence + footnote: "G2 as tabulated was measured against labels computed with the magnet-down map, so on the cross-magnet leg it is not a material-only residual and must be re-read on the magnet-up map." |
| §5.4 heading | "Training where no labels exist: the reversed magnet polarity" | "Training where no labels exist: **the magnet-up polarity**" |
| §5.4, Question paragraph | "...no training labels have ever been made for it in this project." | same sentence + footnote: "Magnet up is in fact the polarity the sample was simulated with, established after this experiment was run... the only one in this paper scored against the physically right field. Its numbers are unchanged." |
| Table 6 caption | "No labelled sample has ever been produced for the reversed polarity in this project." | "No labelled sample has ever been produced for the **magnet-up** polarity in this project, **which is nonetheless the sample's own polarity** (Section 4.1)." |
| §5.4, fiducial-asymmetry sentence | "...removes 21, 20 and 15 states from the three splits on the original polarity and none on the reversed one: ...the original polarity is the one that bends these particular soft tracks out of the map while the reversed polarity bends them back in." | "...on **magnet down** and none on **magnet up**: ... **magnet down** is the one that bends these particular soft tracks out of the map while **magnet up** bends them back in." |
| Table 16 caption | "column 3 is the same leg on the reversed magnet polarity" | "column 3 is the same leg on the **magnet-up** polarity" |
| §6.1 | "And on a field polarity for which no labels exist anywhere in the project it reaches the same floor" | "...**-- which is, in fact, the sample's own polarity (Section 4.1) --** it reaches the same floor" |
| §8 Conclusion | "it reaches the same accuracy on a field polarity for which no labelled sample exists." | "...for which no labelled sample exists, **and which is the polarity the sample itself was simulated with (Section 4.1)**." |
| Appendix A, data set and field maps | "...split by particle 258,857 / 32,389 / 32,287 with seed 20260718." | same + "The conditions tag is **magnet-up**, while every label in the set was computed with `field.v8r1.down.bin`; the test that settles it is `R/Magnet_tracks_dataset/results/polarity_check.json`..." |

The four consequences are stated in the box and nowhere softened: every method
conclusion stands (each experiment compared a network against the same equations
and the same engine); any claim that the labels are where the simulated particle
went is wrong and G2 must be re-read on the up map; the magnet-up experiment is
the sample's true polarity rather than "a polarity with no labels", with its
result (229 against 222 um) unchanged and its framing not; the magnet-to-magnet
data set and the fine reference are built on the up map.

### Noted, not changed

`README_residual.md`'s whole-split paragraph rounds three ratios loosely: the
residual arm beats the straight line by 7.6x with the physics loss (quoted as
6) and 413x with the twin (quoted as 400), and a 4x50 residual network beats a
4x200 wave-1 network by 129x (quoted as 130). These are roundings rather than
disagreements, and they are outside the corrections above.

## Length

39 pages (38 before the polarity correction box of 7 September 2026). The van der Pol mini-paper it mirrors is 35. The brief asked for
20-30; the content list in the brief (seven sections, a full theory derivation,
six results subsections each with a table and a figure, the assembled verdict,
the caveat list and the provenance appendix) does not compress below this
without dropping required material. 13 figures, 16 tables, 12 bibliography
entries. As built: no LaTeX warnings, no overfull boxes, 15 underfull hboxes,
all of them loose spacing around long typewriter paths.

## Scope note

The paper covers the programme from the data-generation restart of 16 July 2026
onward. Earlier extrapolation work is deliberately out of scope and is not
referred to anywhere in the source.

Experiment identifiers A1 to A4 appear only in Appendix A, per the house style:
no code-words in the prose.
