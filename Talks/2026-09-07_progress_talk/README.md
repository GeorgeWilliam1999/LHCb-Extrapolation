# Progress talk, 7 September 2026

"Physics-informed networks for track extrapolation: from van der Pol to the LHCb
magnet". Beamer, 16:9, **74 slides**, no overlays, so one frame is one page and
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
| `main.pdf` | the built deck, 74 pages |
| `figures/` | 43 PNGs, copied unmodified (see the map below) |
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
hboxes**, and no LaTeX warnings. Two underfull hboxes remain, both on slide 22;
each is loose interword spacing inside a narrow table cell, and neither is
visible.

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
| 24 | The local simulation and reconstruction chain (added 2026-09-06) | `lhcb_event_display.png` | `D/First_Pass/README.md` and `D/First_Pass/First_reco/README.md` (Gauss v61r0p2, Boole v48r0 extended, Moore v59r4; long-track efficiency 87.4 %, purity 99.4 %, ghosts 4.8 % on five events; 100-event bulk; v1 set 166,802 rows); Notion write-up "The data, figure by figure" |
| 25 | LHCb data, harvest and legs | `lhcb_population.png` | leg table and split from `D/Official_xdigi/training_v2/train_official_v2.meta.json` (`leg_types`, `rows`); split seed 20260718 |
| 26 | Correction found last night: the sample is magnet-up (added 2026-09-07) | `lhcb_field_polarity.png` | `R/Magnet_tracks_dataset/results/polarity_check.json` and `R/Magnet_tracks_dataset/README.md` (conditions tag, both maps' md5s, the 15,257-particle propagation test, the 4,000-leg fiducial screen) |
| 27 | The label gates and the fiducial requirement | none | gates from the same `.meta.json` (`gates`); fiducial numbers from Mini_paper §3.8 and `R/One_step_network_v2/README.md` |
| 28 | The continuous-time cliff | `vdp_cliff.png`, `vdp_phase_collapse.png` | `V/vdp_minipaper/main.tex` Tab. `tab:expA`, from `V/continuous_time_network/Initial_pass/results/` |
| 29 | The classical anchor at the same step (added 2026-09-06) | `vdp_rk6_comparison.png` | `V/vdp_minipaper/main.tex` Tab. `tab:rk6`, from `V/continuous_time_network/Classical_RK_comparison/results/comparison_rel_l2.csv` |
| 30 | Causal weighting moves the cliff | `vdp_causal_compare.png` | Tab. `tab:causal`; the loss 9.7e-6 / error 0.71 run from `V/continuous_time_network/Causal_weighting/` run outputs |
| 31 | The resource axes under the original protocol (added 2026-09-06) | `vdp_size_heatmaps.png`, `vdp_success_map.png`, `vdp_density.png` | `V/vdp_minipaper/main.tex` §3.7 to §3.9; `V/continuous_time_network/Network_size_study/results/runs/` (600 json), `Collocation_study/results/summary_analysis.csv` (72 runs), `Collocation_placement_study/results/` (24 runs) |
| 32 | The supervised control | `vdp_control_compare.png` | `V/continuous_time_network/Supervised_control/results/` |
| 33 | **Eight objectives, the full table** | none | Tab. `tab:arms`; `V/.../Stability_regularization/results/converged_summary.csv` |
| 34 | The three failure classes | `vdp_arms_phase_T27.png` | same CSV, columns `parked`, `flow_following`, `diffuse`; loss-term observations from the `fig_arms_lossterms_T27` analysis |
| 35 | The eight arms in pictures (added 2026-09-06) | `vdp_arms_summary.png`, `vdp_arms_traj_T27.png`, `vdp_arms_residual_T27.png` | `V/.../Stability_regularization/results/converged_summary.csv` and the per-run `.npz` residual profiles |
| 36 | **The two resource axes, the full table** | none | Tab. `tab:axes`; `V/.../Stability_regularization/results/axes_summary.csv` |
| 37 | The two axes as pictures | `vdp_axes_density.png`, `vdp_axes_horizon.png`, `vdp_axes_long_horizon.png` | same CSV |
| 38 | The 1/T price, measured | `vdp_axes_parked_loss.png` | `axes_summary.csv`; loss × T median 0.85, range 0.84 to 0.90 over 65 parked runs |
| 39 | **The discrete-time floor, the full table** | none | `V/vdp_minipaper/main.tex` Tab. `tab:floor` and `tab:rho`; `V/discrete_time_network/results/` |
| 40 | The same floor twice | `vdp_capacity_floor.png`, `vdp_size_map_discrete.png` | `V/discrete_time_network/results/`, `V/discrete_time_network/Network_size_study/results/` |
| 41 | Size and seed on van der Pol | `vdp_seed_replication.png` | `V/discrete_time_network/Network_size_study/results/` |
| 42 | Selection on the chain | `vdp_selection.png` | same folder; far-horizon check at t = 200 |
| 43 | Routes and the data twin | none | Tab. `tab:routes` (`V/discrete_time_network/Trajectory_network/results/`) and Tab. `tab:baseline` (`V/discrete_time_network/data_trained_baseline/results/`) |
| 44 | The discrete-time column in pictures (added 2026-09-06) | `vdp_head_to_head.png`, `vdp_test_set.png`, `vdp_data_baseline.png` | as for slides 39 to 43 (`V/discrete_time_network/results/`, `Trajectory_network/results/`, `data_trained_baseline/results/`) |
| 45 | Steps 0 and 1 on LHCb: the field along legs and the exact scheme (added 2026-09-06) | `lhcb_field_along_legs.png`, `lhcb_scheme_ceiling.png` | `R/Baseline_data_exploration/results/scales.json` and `figures/field_along_legs.png`; `R/Simple_first_pass/results/scheme_scan.csv`, `scheme_error_vs_q.csv` (640/640 solves); the 23 µm test-population ceiling from `R/Stage_count_sweep/results/scheme_ceiling_same_population_q08.json` |
| 46 | The July baseline | `lhcb_baseline.png` | `R/One_step_network_v2/results/summary.csv` |
| 47 | **The stage sweep, the full table** | none | `R/Stage_count_sweep/results/error_vs_stages.csv`, `results/summary.csv`, `results/scheme_ceiling_same_population_q*.json` |
| 48 | The stage sweep as a picture, and the 29 µm correction | `lhcb_stages.png` | same; the correction from `R/Stage_count_sweep/README.md` |
| 49 | **The full architecture grid** | none | `R/Network_size_and_seed_study/results/by_architecture.csv` |
| 50 | The loss-error band | `lhcb_loss_vs_error.png` | `R/Network_size_and_seed_study/results/summary.csv` (Spearman +0.965 over 96 converged runs) |
| 51 | Returns collapse, and the twin is overtaken | `lhcb_architecture.png` | `results/by_architecture.csv`, `results/summary.csv` (seed ratios, validation-test rank +0.995) |
| 52 | **Magnet-up polarity, the full table** | none | `R/Magnet_up_field/results/summary.csv`, `results/up_vs_down.csv`, `results/ceiling_summary.json`, `results/field_up_parity.json`, `results/loss_field_probe.json` |
| 53 | Magnet-up polarity in pictures | `lhcb_magnet_up.png` | same |
| 54 | **One network for all legs, absolute, the full table at three widths** | none | `R/General_leg_network/results/by_leg.csv`, `results/scheme_ceiling_same_population.csv`, `results/dataset_meta.json` |
| 55 | The parameterisation hypothesis | none | Mini_paper §5.5.3; scales from `results/general_legs_meta.json` |
| 56 | **Chaining, absolute, the full table** | none | `R/Chained_legs/results/chain_summary.csv`, `results/chains_meta.json`, `results/selection.csv` |
| 57 | **Leg D, both designs, the full table** | none | `R/Chained_legs/results/leg_d_reproduction.csv`, `results/leg_d_residual.csv`, `results/leg_d_ceiling_same_population.csv` |
| 58 | The absolute-output network in pictures | `lhcb_general_legs.png`, `lhcb_chains.png`, `lhcb_leg_d.png` | as for slides 48, 50 and 51 |
| 59 | **The residual redesign, the full table plus the whole split** | none | `R/General_leg_network/results/by_leg_residual.csv`, `results/residual_summary.csv` |
| 60 | Reading the redesign across | none | same, plus `results/residual_init_check.json`; restart counts from `README_residual.md` section "The farm" |
| 61 | **Inside the step, and the tails** | none | `R/General_leg_network/results/stage_errors_residual.csv`, `results/stage_errors.csv`; p95 from column `endpoint_p95_um` of `by_leg_residual.csv` |
| 62 | The redesign in pictures | `lhcb_residual_legs.png`, `lhcb_residual_stages.png` | as for slides 53 and 55 |
| 63 | **Chaining the residual networks, the full table** | none | `R/Chained_legs/results/chain_summary_residual.csv`, `results/selection_residual.csv` |
| 64 | The cross-magnet leg across three designs | `lhcb_frozen_vs_general.png`, `lhcb_residual_scale.png` | figures from `R/General_leg_network/figures/`; medians from `results/residual_scale_check.json` |
| 65 | The verdict, and the cost statement | none | Mini_paper Tab. `tab:verdict`, drawn from tables 3, 5, 6, 7, 10 and 14 |
| 66 | Two ceilings and a floor | none | Mini_paper §3.4 (the C0 field map), §5.3, §6.2 |
| 67 | What I cannot claim yet | none | Mini_paper §6.3 (caveats) and §4.5 (single thread) |
| 68 | What would falsify the edge | none | **new for this talk**; the designs and the pass/fail criteria are mine, built on Mini_paper §7.1 and §7.4 |
| 69 | The rest of the queue | none | Mini_paper §7.1, §7.2, §7.4 |
| 70 | Theory predictions, and the thesis framing | none | Mini_paper §7.3 |
| 71 | What I would like feedback on | none | **new for this talk** |
| 72 | Reference: protocol and gates | none | Mini_paper §3.6 and §4.4 |
| 73 | Reference: provenance | none | Mini_paper Appendix A |
| 74 | Reference: where the record lives | none | Notion write-up URLs; repository URL |

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
| `lhcb_population.png` | `D/Official_xdigi/figures/v1_vs_v2_population.png` | 25 |
| `lhcb_baseline.png` | `R/One_step_network_v2/figures/one_step_results_v2.png` | 46 |
| `lhcb_stages.png` | `R/Stage_count_sweep/figures/error_vs_stages.png` | 48 |
| `lhcb_loss_vs_error.png` | `R/Network_size_and_seed_study/figures/loss_vs_error.png` | 50 |
| `lhcb_architecture.png` | `R/Network_size_and_seed_study/figures/floor_vs_architecture.png` | 51 |
| `lhcb_magnet_up.png` | `R/Magnet_up_field/figures/magnet_up_results.png` | 53 |
| `lhcb_general_legs.png` | `R/General_leg_network/figures/error_by_leg_and_momentum.png` | 58 |
| `lhcb_chains.png` | `R/Chained_legs/figures/error_vs_chained_legs.png` | 58 |
| `lhcb_leg_d.png` | `R/Chained_legs/figures/leg_d_reproduction.png` | 58 |
| `lhcb_residual_legs.png` | `R/General_leg_network/figures/error_by_leg_and_momentum_residual.png` | 62 |
| `lhcb_residual_stages.png` | `R/General_leg_network/figures/stage_errors_residual.png` | 62 |
| `lhcb_frozen_vs_general.png` | `R/General_leg_network/figures/frozen_vs_general_residual.png` | 64 |
| `lhcb_residual_scale.png` | `R/General_leg_network/figures/residual_scale_check.png` | 64 |
| `vdp_problem.png` | `V/vdp_minipaper/figures/fig_problem.png` | 23 |
| `vdp_cliff.png` | `V/vdp_minipaper/figures/fig_cliff.png` | 28 |
| `vdp_phase_collapse.png` | `V/vdp_minipaper/figures/fig_phase_collapse.png` | 28 |
| `vdp_causal_compare.png` | `V/vdp_minipaper/figures/fig_causal_compare.png` | 30 |
| `vdp_control_compare.png` | `V/vdp_minipaper/figures/fig_control_compare.png` | 32 |
| `vdp_arms_phase_T27.png` | `V/vdp_minipaper/figures/fig_arms_phase_T27.png` | 34 |
| `vdp_axes_density.png` | `V/vdp_minipaper/figures/fig_axes_density.png` | 37 |
| `vdp_axes_horizon.png` | `V/vdp_minipaper/figures/fig_axes_horizon.png` | 37 |
| `vdp_axes_long_horizon.png` | `V/vdp_minipaper/figures/fig_axes_long_horizon.png` | 37 |
| `vdp_axes_parked_loss.png` | `V/vdp_minipaper/figures/fig_axes_parked_loss.png` | 38 |
| `vdp_capacity_floor.png` | `V/vdp_minipaper/figures/fig_capacity_floor.png` | 40 |
| `vdp_size_map_discrete.png` | `V/vdp_minipaper/figures/fig_size_map_discrete.png` | 40 |
| `vdp_seed_replication.png` | `V/vdp_minipaper/figures/fig_seed_replication.png` | 41 |
| `vdp_selection.png` | `V/vdp_minipaper/figures/fig_selection.png` | 42 |
| `vdp_arms_summary.png` | `V/vdp_minipaper/figures/fig_arms_summary.png` | 35 |
| `vdp_arms_traj_T27.png` | `V/vdp_minipaper/figures/fig_arms_traj_T27.png` | 35 |
| `vdp_control_overlay.png` | `V/vdp_minipaper/figures/fig_control_overlay.png` | held in reserve |
| `vdp_data_baseline.png` | `V/vdp_minipaper/figures/fig_data_baseline.png` | 44 |
| `vdp_test_set.png` | `V/vdp_minipaper/figures/fig_test_set.png` | 44 |
| `vdp_head_to_head.png` | `V/vdp_minipaper/figures/fig_head_to_head.png` | 44 |
| `lhcb_event_display.png` | `D/First_Pass/figures5/event_display.png` | 24 |
| `vdp_rk6_comparison.png` | `V/vdp_minipaper/figures/fig_rk6_comparison.png` | 29 |
| `vdp_size_heatmaps.png` | `V/vdp_minipaper/figures/fig_size_heatmaps.png` | 31 |
| `vdp_success_map.png` | `V/vdp_minipaper/figures/fig_success_map.png` | 31 |
| `vdp_density.png` | `V/vdp_minipaper/figures/fig_density.png` | 31 |
| `vdp_placement.png` | `V/vdp_minipaper/figures/fig_placement.png` | held in reserve (added 2026-09-06) |
| `vdp_arms_residual_T27.png` | `V/vdp_minipaper/figures/fig_arms_residual_T27.png` | 35 |
| `lhcb_field_along_legs.png` | `R/Baseline_data_exploration/figures/field_along_legs.png` | 45 |
| `lhcb_scheme_ceiling.png` | `R/Simple_first_pass/figures/scheme_error_vs_q.png` | 45 |
| `lhcb_field_polarity.png` | `R/Magnet_tracks_dataset/figures/field_polarity.png` | 26 |

Forty-one of the 43 are placed; `vdp_control_overlay.png` and `vdp_placement.png` are held in reserve. (Revision 2026-09-06, slide numbers as of that revision: six slides added, 24, 28, 30, 34, 43 and 44, covering the local simulation chain, the classical anchor, the 600 + 72 + 24 resource-axis runs, the eight arms' residual anatomy, the discrete column's pictures, and the LHCb field and exact-scheme ceiling; every later slide number shifted accordingly. No number on any pre-existing slide changed.)

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

## Polarity correction, 7 September 2026

Established 6 September by the magnet-to-magnet data set build
(`R/Magnet_tracks_dataset/`, `results/polarity_check.json`,
`figures/field_polarity.png`): the official sample's conditions tag
`sim-20231017-vc-mu100` is **magnet-up**. Propagating each particle's forward
cross-magnet leg with `field.v8r1.up.bin` lands a median 1.73 mm from that
particle's own real first-SciFi state with the bend sign right in 100.0 % of
15,257 particles; with `field.v8r1.down.bin` it lands 883 mm away with the sign
wrong in 100 %, and 2 % of legs leave the field map. Every July and Block A
label and score - the v2 training set's `Y` column, the frozen-leg experiments,
the general-leg and residual experiments - used the down map.

What that does and does not cost is stated on the new slide 26 and nowhere
softened: every method conclusion stands, because each experiment compared a
network against the same equations and the same engine; any statement that those
labels are where the simulated particle went is wrong and G2 must be re-read on
the up map; the magnet-up experiment is the sample's true polarity rather than
"a polarity with no labels", and its numbers (same floor, 229 against 222 um)
are unchanged while its framing is not; the new magnet-to-magnet set and the
fine reference are built on the up map.

### Changes made to this deck

| where | change |
|---|---|
| new **slide 26** | "Correction found last night: the sample is magnet-up". The up-vs-down table (median miss overall and in the 1-2 and 25-200 GeV bands, bend sign, legs leaving the map), `lhcb_field_polarity.png`, and the four consequences as four lines. Every slide from 26 onward shifted by one; 73 slides became 74. |
| slide 25, harvest paragraph | "label by fourth-order Runge-Kutta at 5 mm" -> "label by fourth-order Runge-Kutta at 5 mm **with the magnet-down map (the sample is up; see the next slide)**". |
| slide 27, G2 paragraph | added "**G2 was measured with down-map labels and must be re-read on the up map (previous slide).**" |
| slide 52, title | "Training where no labels exist: the reversed magnet polarity reaches the same floor" -> "Training where no labels exist: **the magnet-up polarity, the sample's own,** reaches the same floor". |
| slide 52, table header | "magnet up (reversed) / magnet down (original)" -> "magnet up **(the sample's own)** / magnet down **(as labelled)**". |
| slide 52, right column | added "**Correction, 6 September.** Magnet up is the sample's own polarity, so this is the only experiment here scored against the physically right field. The numbers are unchanged; the framing is." |
| slide 65, verdict box | "it reaches the same floor on a polarity with no labels" -> "... on a polarity with no labels **of its own, which is in fact the sample's own polarity**". |
| slide 73, provenance | added "The conditions tag is magnet-**up**; every label was computed with the magnet-down map." |
| `notes.md` | new speaker note 26; notes 26 to 73 renumbered to 27 to 74. |
| `README.md` | slide list, figure map and cross-references renumbered; the new slide and `lhcb_field_polarity.png` added to both maps; this section. |

No result number anywhere in the deck was changed by this correction, and no
slide was reframed beyond the sentences listed above.

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
