# Progress talk, 7 September 2026

"Physics-informed networks for track extrapolation: from van der Pol to the LHCb
magnet". Beamer, 16:9, **67 slides**, no overlays, so one frame is one page and
the slide number equals the page number throughout.

Audience: experts, no time limit. Consequences for the build: the full tables
are in the main line of the talk rather than in a backup section, each on its own
slide; the theory carries its equations on the slides; each of the four fix
papers gets its own slide (what it proves, its protocol, my implementation
check, my result); and there is a falsification slide with pass and fail criteria
stated in advance, before the direction block.

## Contents

| file | what it is |
|---|---|
| `main.tex` | the deck, single file |
| `main.pdf` | the built deck, 67 pages |
| `figures/` | 33 PNGs, copied unmodified (see the map below) |
| `notes.md` | speaker notes, one entry per slide, 2 to 4 sentences each |
| `build.sh` | `pdflatex` twice |
| `README.md` | this file: slide list, figure map, number map |

## Build

```sh
sh build.sh
```

Requires `pdflatex` only. Packages: beamer, fontenc, lmodern, amsmath, amssymb,
booktabs, graphicx, xcolor, array, url. No bibliography, so no `bibtex` pass.

Every content frame carries `[shrink=25]`, which is beamer's own mechanism: it
shrinks a frame only as far as needed, and only up to that percentage. As built
there are **zero overfull vboxes, zero underfull vboxes and zero overfull
hboxes**. Ten underfull hboxes remain; every one of them is loose interword
spacing inside a narrow table cell, and none of them is visible.

## Sources

Two written companions carry every number in this deck, and each carries its own
table-to-file and figure-to-file map.

- **LHCb**: `../../Mini_paper/main.tex` and `../../Mini_paper/README.md`
  (38 pages, 13 figures, 16 tables). Repository root
  `/data/bfys/gscriven/LHCb_Extrapolation_Project`.
- **Van der Pol**: `/data/bfys/gscriven/Van_Der_Pole/vdp_minipaper/main.tex` and
  its `README.md` (48 pages, 37 figures). Treated as **read-only** for this
  deck: figures were copied out of it and nothing in it was written to.

Abbreviations below: `R/` = `Raissi_disc_time_approach/`,
`D/` = `Data_generation_exploration/` (both under the LHCb repository root);
`V/` = `/data/bfys/gscriven/Van_Der_Pole/`.

## Slide list, with figures and number sources

| # | slide | figures | numbers come from |
|---|---|---|---|
| 1 | Title | none | none |
| 2 | Where this talk goes | none | slide counts only |
| 3 | What the extrapolator does | none | Mini_paper §1 and §3.1 (state, `qop` convention, 1.05 T at 4.7 m, 520 mm straight-line median); G2 from `D/Official_xdigi/training_v2/train_official_v2.meta.json` |
| 4 | Why a fixed cost | none | 44 root-finder evaluations: `R/Simple_first_pass/exact_scheme.py` run log, quoted in `R/Simple_first_pass/README.md`; 36 outputs = 4(q+1) at q = 8 |
| 5 | The two questions | none | Mini_paper §1.3, wording unchanged from the pre-statement |
| 6 | The two constructions | none | Raissi, Perdikaris and Karniadakis, J. Comput. Phys. 378 (2019) 686; their reported accuracies via `V/vdp_minipaper/main.tex` §3 |
| 7 | The equation of motion in z | none | `R/_shared/reference.py`; Mini_paper §3.1 including the 10 GeV micro-example |
| 8 | Collocation, and the q = 2 tableau | none | `R/_shared/irk.py`; Mini_paper §3.2 |
| 9 | The reconstruction residuals and the loss | none | `R/_shared/model.py`, function `physics_loss`; Mini_paper §3.3 |
| 10 | The scales in the loss | none | `R/General_leg_network/results/general_legs_meta.json`; `R/Stage_count_sweep/results/frozen_leg_q08_meta.json` |
| 11 | The residual parameterisation and the bending integral | none | `R/General_leg_network/residual_model.py`; scale-check medians 1.09 / 1.03 / 0.95 from `R/General_leg_network/results/residual_scale_check.json` |
| 12 | The continuous-time objective | none | `V/vdp_minipaper/main.tex` §3 (loss) and §5.7 (Jacobian and eigenvalues at the origin) |
| 13 | The splice family and the 1/T price | none | `V/vdp_minipaper/main.tex` §5.7; `V/continuous_time_network/Stability_regularization/theory.md` |
| 14 | Pseudo-time stepping, the relaxed objective | none | `V/vdp_minipaper/main.tex` §5.7; tau range from `V/.../Stability_regularization/results/axes_summary.csv`, column `tau_median` |
| 15 | Comparators and metrics | none | `R/_shared/evaluate.py`; Mini_paper §3.5 and §3.7 |
| 16 | Causal weighting (Wang, Sankaran, Perdikaris 2022) | none | `V/vdp_minipaper/main.tex` Tab. `tab:causal` and the budget-extension table, from `V/continuous_time_network/Causal_weighting/results/` |
| 17 | Fixed points (Rohrhofer et al. 2023) | none | the paper; my park counts from `V/.../Stability_regularization/results/converged_summary.csv`, column `parked` |
| 18 | The stillness regulariser (Babic et al. 2025) | none | the paper; my three regulariser arms in `converged_summary.csv` |
| 19 | Pseudo-time stepping (Wang et al. 2026), measured | none | the paper; `converged_summary.csv` and `axes_summary.csv`, rows `pseudo_unw` |
| 20 | Why van der Pol first | none | `V/README.md`; `V/vdp_minipaper/main.tex` §2.1 (period, loop extent) |
| 21 | The shared machinery and the protocol | none | `R/_shared/train.py`; Mini_paper §3.6 and §4.5 |
| 22 | The dictionary van der Pol to LHCb | none | Mini_paper §3.1 and §2.2; `V/vdp_minipaper/main.tex` §2 |
| 23 | Van der Pol data | `vdp_problem.png` | `V/vdp_minipaper/main.tex` §2.2, the four verification instruments |
| 24 | LHCb data, harvest and legs | `lhcb_population.png` | leg table and split from `D/Official_xdigi/training_v2/train_official_v2.meta.json` (`leg_types`, `rows`); split seed 20260718 |
| 25 | The label gates and the fiducial requirement | none | gates from the same `.meta.json` (`gates`); fiducial numbers from Mini_paper §3.8 and `R/One_step_network_v2/README.md` |
| 26 | The continuous-time cliff | `vdp_cliff.png`, `vdp_phase_collapse.png` | `V/vdp_minipaper/main.tex` Tab. `tab:expA`, from `V/continuous_time_network/Initial_pass/results/` |
| 27 | Causal weighting moves the cliff | `vdp_causal_compare.png` | Tab. `tab:causal`; the loss 9.7e-6 / error 0.71 run from `V/continuous_time_network/Causal_weighting/` run outputs |
| 28 | The supervised control | `vdp_control_compare.png` | `V/continuous_time_network/Supervised_control/results/` |
| 29 | **Eight objectives, the full table** | none | Tab. `tab:arms`; `V/.../Stability_regularization/results/converged_summary.csv` |
| 30 | The three failure classes | `vdp_arms_phase_T27.png` | same CSV, columns `parked`, `flow_following`, `diffuse`; loss-term observations from the `fig_arms_lossterms_T27` analysis |
| 31 | **The two resource axes, the full table** | none | Tab. `tab:axes`; `V/.../Stability_regularization/results/axes_summary.csv` |
| 32 | The two axes as pictures | `vdp_axes_density.png`, `vdp_axes_horizon.png`, `vdp_axes_long_horizon.png` | same CSV |
| 33 | The 1/T price, measured | `vdp_axes_parked_loss.png` | `axes_summary.csv`; loss × T median 0.85, range 0.84 to 0.90 over 65 parked runs |
| 34 | **The discrete-time floor, the full table** | none | `V/vdp_minipaper/main.tex` Tab. `tab:floor` and `tab:rho`; `V/discrete_time_network/results/` |
| 35 | The same floor twice | `vdp_capacity_floor.png`, `vdp_size_map_discrete.png` | `V/discrete_time_network/results/`, `V/discrete_time_network/Network_size_study/results/` |
| 36 | Size and seed on van der Pol | `vdp_seed_replication.png` | `V/discrete_time_network/Network_size_study/results/` |
| 37 | Selection on the chain | `vdp_selection.png` | same folder; far-horizon check at t = 200 |
| 38 | Routes and the data twin | none | Tab. `tab:routes` (`V/discrete_time_network/Trajectory_network/results/`) and Tab. `tab:baseline` (`V/discrete_time_network/data_trained_baseline/results/`) |
| 39 | The July baseline | `lhcb_baseline.png` | `R/One_step_network_v2/results/summary.csv` |
| 40 | **The stage sweep, the full table** | none | `R/Stage_count_sweep/results/error_vs_stages.csv`, `results/summary.csv`, `results/scheme_ceiling_same_population_q*.json` |
| 41 | The stage sweep as a picture, and the 29 µm correction | `lhcb_stages.png` | same; the correction from `R/Stage_count_sweep/README.md` |
| 42 | **The full architecture grid** | none | `R/Network_size_and_seed_study/results/by_architecture.csv` |
| 43 | The loss-error band | `lhcb_loss_vs_error.png` | `R/Network_size_and_seed_study/results/summary.csv` (Spearman +0.965 over 96 converged runs) |
| 44 | Returns collapse, and the twin is overtaken | `lhcb_architecture.png` | `results/by_architecture.csv`, `results/summary.csv` (seed ratios, validation-test rank +0.995) |
| 45 | **Reversed polarity, the full table** | none | `R/Magnet_up_field/results/summary.csv`, `results/up_vs_down.csv`, `results/ceiling_summary.json`, `results/field_up_parity.json`, `results/loss_field_probe.json` |
| 46 | Reversed polarity in pictures | `lhcb_magnet_up.png` | same |
| 47 | **One network for all legs, absolute, the full table at three widths** | none | `R/General_leg_network/results/by_leg.csv`, `results/scheme_ceiling_same_population.csv`, `results/dataset_meta.json` |
| 48 | The parameterisation hypothesis | none | Mini_paper §5.5.3; scales from `results/general_legs_meta.json` |
| 49 | **Chaining, absolute, the full table** | none | `R/Chained_legs/results/chain_summary.csv`, `results/chains_meta.json`, `results/selection.csv` |
| 50 | **Leg D, both designs, the full table** | none | `R/Chained_legs/results/leg_d_reproduction.csv`, `results/leg_d_residual.csv`, `results/leg_d_ceiling_same_population.csv` |
| 51 | The absolute-output network in pictures | `lhcb_general_legs.png`, `lhcb_chains.png`, `lhcb_leg_d.png` | as for slides 47, 49 and 50 |
| 52 | **The residual redesign, the full table plus the whole split** | none | `R/General_leg_network/results/by_leg_residual.csv`, `results/residual_summary.csv` |
| 53 | Reading the redesign across | none | same, plus `results/residual_init_check.json`; restart counts from `README_residual.md` section "The farm" |
| 54 | **Inside the step, and the tails** | none | `R/General_leg_network/results/stage_errors_residual.csv`, `results/stage_errors.csv`; p95 from column `endpoint_p95_um` of `by_leg_residual.csv` |
| 55 | The redesign in pictures | `lhcb_residual_legs.png`, `lhcb_residual_stages.png` | as for slides 52 and 54 |
| 56 | **Chaining the residual networks, the full table** | none | `R/Chained_legs/results/chain_summary_residual.csv`, `results/selection_residual.csv` |
| 57 | The cross-magnet leg across three designs | `lhcb_frozen_vs_general.png`, `lhcb_residual_scale.png` | figures from `R/General_leg_network/figures/`; medians from `results/residual_scale_check.json` |
| 58 | The verdict, and the cost statement | none | Mini_paper Tab. `tab:verdict`, drawn from tables 3, 5, 6, 7, 10 and 14 |
| 59 | Two ceilings and a floor | none | Mini_paper §3.4 (the C0 field map), §5.3, §6.2 |
| 60 | What I cannot claim yet | none | Mini_paper §6.3 (caveats) and §4.5 (single thread) |
| 61 | What would falsify the edge | none | **new for this talk**; the designs and the pass/fail criteria are mine, built on Mini_paper §7.1 and §7.4 |
| 62 | The rest of the queue | none | Mini_paper §7.1, §7.2, §7.4 |
| 63 | Theory predictions, and the thesis framing | none | Mini_paper §7.3 |
| 64 | What I would like feedback on | none | **new for this talk** |
| 65 | Reference: protocol and gates | none | Mini_paper §3.6 and §4.4 |
| 66 | Reference: provenance | none | Mini_paper Appendix A |
| 67 | Reference: where the record lives | none | Notion write-up URLs; repository URL |

Slides in **bold** are the full-table slides that were moved out of backup into
the main line for this audience.

## Figure map: deck file to original

All 33 PNGs in `figures/` are byte-identical copies. The LHCb ones came from
`Mini_paper/figures/`, which itself copies them unmodified from the experiment
folder that generated them; the van der Pol ones from `V/vdp_minipaper/figures/`,
likewise. Every one of them is regenerated by its experiment folder's plotting
script from committed CSV tables only.

| file in `figures/` | source | on slide |
|---|---|---|
| `lhcb_population.png` | `D/Official_xdigi/figures/v1_vs_v2_population.png` | 24 |
| `lhcb_baseline.png` | `R/One_step_network_v2/figures/one_step_results_v2.png` | 39 |
| `lhcb_stages.png` | `R/Stage_count_sweep/figures/error_vs_stages.png` | 41 |
| `lhcb_loss_vs_error.png` | `R/Network_size_and_seed_study/figures/loss_vs_error.png` | 43 |
| `lhcb_architecture.png` | `R/Network_size_and_seed_study/figures/floor_vs_architecture.png` | 44 |
| `lhcb_magnet_up.png` | `R/Magnet_up_field/figures/magnet_up_results.png` | 46 |
| `lhcb_general_legs.png` | `R/General_leg_network/figures/error_by_leg_and_momentum.png` | 51 |
| `lhcb_chains.png` | `R/Chained_legs/figures/error_vs_chained_legs.png` | 51 |
| `lhcb_leg_d.png` | `R/Chained_legs/figures/leg_d_reproduction.png` | 51 |
| `lhcb_residual_legs.png` | `R/General_leg_network/figures/error_by_leg_and_momentum_residual.png` | 55 |
| `lhcb_residual_stages.png` | `R/General_leg_network/figures/stage_errors_residual.png` | 55 |
| `lhcb_frozen_vs_general.png` | `R/General_leg_network/figures/frozen_vs_general_residual.png` | 57 |
| `lhcb_residual_scale.png` | `R/General_leg_network/figures/residual_scale_check.png` | 57 |
| `vdp_problem.png` | `V/vdp_minipaper/figures/fig_problem.png` | 23 |
| `vdp_cliff.png` | `V/vdp_minipaper/figures/fig_cliff.png` | 26 |
| `vdp_phase_collapse.png` | `V/vdp_minipaper/figures/fig_phase_collapse.png` | 26 |
| `vdp_causal_compare.png` | `V/vdp_minipaper/figures/fig_causal_compare.png` | 27 |
| `vdp_control_compare.png` | `V/vdp_minipaper/figures/fig_control_compare.png` | 28 |
| `vdp_arms_phase_T27.png` | `V/vdp_minipaper/figures/fig_arms_phase_T27.png` | 30 |
| `vdp_axes_density.png` | `V/vdp_minipaper/figures/fig_axes_density.png` | 32 |
| `vdp_axes_horizon.png` | `V/vdp_minipaper/figures/fig_axes_horizon.png` | 32 |
| `vdp_axes_long_horizon.png` | `V/vdp_minipaper/figures/fig_axes_long_horizon.png` | 32 |
| `vdp_axes_parked_loss.png` | `V/vdp_minipaper/figures/fig_axes_parked_loss.png` | 33 |
| `vdp_capacity_floor.png` | `V/vdp_minipaper/figures/fig_capacity_floor.png` | 35 |
| `vdp_size_map_discrete.png` | `V/vdp_minipaper/figures/fig_size_map_discrete.png` | 35 |
| `vdp_seed_replication.png` | `V/vdp_minipaper/figures/fig_seed_replication.png` | 36 |
| `vdp_selection.png` | `V/vdp_minipaper/figures/fig_selection.png` | 37 |
| `vdp_arms_summary.png` | `V/vdp_minipaper/figures/fig_arms_summary.png` | held in reserve |
| `vdp_arms_traj_T27.png` | `V/vdp_minipaper/figures/fig_arms_traj_T27.png` | held in reserve |
| `vdp_control_overlay.png` | `V/vdp_minipaper/figures/fig_control_overlay.png` | held in reserve |
| `vdp_data_baseline.png` | `V/vdp_minipaper/figures/fig_data_baseline.png` | held in reserve |
| `vdp_test_set.png` | `V/vdp_minipaper/figures/fig_test_set.png` | held in reserve |
| `vdp_head_to_head.png` | `V/vdp_minipaper/figures/fig_head_to_head.png` | held in reserve |

Twenty-seven of the 33 are placed. The six marked "held in reserve" are copied
and ready so a question can be answered with a picture rather than a sentence:
the eight-arm summary and trajectory panels (slides 29 and 30), the supervised
overlay (slide 28), the data-twin cost panel and the test-set distribution
(slide 38), and the head-to-head of the two constructions. Adding any of them is
one `\includegraphics` line.

## Numbers quoted in the deck that live in prose rather than in a CSV

Each is quoted here with its source so it can be chased if challenged.

| claim | where it lives |
|---|---|
| 44 root-finder residual evaluations per leg | `R/Simple_first_pass/README.md`, from the `exact_scheme.py` run log |
| restart counts, both waves (40 to 64 against 84 to 121; 3 of 26 against 20 of 26) | `R/General_leg_network/README_residual.md`, section "The farm", and the `*_history.csv` files |
| the 68x multithreaded slowdown | Mini_paper §4.5 |
| the harness restart-cap incident | Mini_paper §6.3 |
| Kolmogorov-Smirnov distances 0.006 to 0.069 | `D/Official_xdigi/README.md` and the figure's own panel titles |
| van der Pol reference verification (6.06 / 6.27 / 5.75; 6.4e-14; 5.6e-14) | `V/RK_Truth/README.md` and `V/vdp_minipaper/main.tex` §2.2 |
| 799 integrator steps per label | `V/vdp_minipaper/main.tex` §6.9 (799 = one step of 0.8 at h = 1e-3) |
| tau settling between 0.5 and 0.9 | `V/.../Stability_regularization/results/axes_summary.csv`, column `tau_median` |

## Deliberate scope decisions

- **Nothing before 16 July 2026.** The deck covers the programme from the
  data-generation restart onward. Earlier extrapolation work is out of scope and
  is not referred to anywhere in the source, matching the mini-paper's own scope
  note.
- **No experiment identifiers or folder code-words on any slide.** They appear
  only in this README and in the two papers' provenance appendices.
- **The `Van_Der_Pole` repository was treated as read-only.** Figures were copied
  out of it; nothing in it was written to.
- **Two commits are still local.** `333edf7` and `29e9928` carry the residual arm
  and the second wave. Slide 66 says so explicitly rather than implying that
  everything is pushed.

## Open items

- **The cost statement is an operation count, not a measurement**, and it is
  flagged as such on slides 4 and 58. Slide 61 gives the benchmark design that
  would turn it into a measurement, with the pass and fail criteria written
  before the run.
- **Slides 61 and 64 are new material** written for this talk rather than lifted
  from a paper. The criteria on slide 61 are mine and have not been reviewed by
  anyone.
- **No TODO markers are left on any slide.** Every number placed on a slide was
  taken from a committed result file or from one of the two papers, which
  themselves recompute from the CSVs. Where the two disagreed, the CSV was used,
  per the correction record in `../../Mini_paper/README.md`.
