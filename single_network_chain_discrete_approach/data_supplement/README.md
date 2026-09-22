# data_supplement — the data in pictures, drawn to scale

Seven figures for the question George's supervisors asked on 2026-09-21 and 22: what
actually happens to a track between the last Upstream Tracker plane and the first
fibre-tracker plane, which physical effect enters where, and how big each one is.

Nothing here is a sketch. The trajectories are the stored RK6 reference tracks of the
crossing set, the field profile is read from the v8r1 magnet-up map, the detector
envelopes are the z ranges of the simulated hits themselves, and every annotated number
is read from a results CSV and copied into `results/figure_facts.json`, so a figure and
the table behind it cannot drift apart.

```bash
PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python make_figures.py   # about 40 s
```

## script → output

| figure | what it shows | drawn from |
|---|---|---|
| `fig1_crossing_to_scale.png` | the 5,177.8 mm crossing with equal scale on both axes, three real tracks at 3, 12 and 45 GeV, and the field that bends them | `tracks.npz` test split; `_shared/reference.py` field map |
| `fig2_zoom_cascade.png` | four panels, each about ten times closer, at the far end: the straight line, then the truth separating, then the network separating | `tracks.npz`, `N064_q02/chain_states.npz` |
| `fig3_where_effects_enter.png` | the detector along z with the five places something happens, and a ledger of what each description of the crossing contains | hit z ranges from `Data/results/states.npz`; one real 6 GeV track |
| `fig4_error_budget.png` | every error in the problem on one logarithmic axis, per momentum band, spanning ten decades | `against_true_state.csv`, `decomposition_by_band.csv`, `comparators.csv`, `error_qdz_chain.csv`, `gates.json` |
| `fig5_charge_separation.png` | the gap split by charge: the centre flips, the width does not | `tracks.npz`, `decomposition_by_band.csv` |
| `fig6_scattering_floor.png` | the random part against momentum, with the air-only Highland curve and the material it implies | `highland_air.csv` |
| `fig7_one_step.png` | one step of 80.9 mm: the two stage states the network predicts, and an inset at the scale of its own 0.43 µm single-step error | `tracks.npz`, `error_qdz_single_step.csv` |

## the scales the figures are drawn at

| quantity | value |
|---|---|
| the crossing | 5,177.8 mm, from z = 2,648.2 to z = 7,826.0 mm |
| one step at 64 steps | 80.90 mm |
| a 3 GeV track's bend | 1,205 mm |
| straight line to the reference | 450,542 µm |
| reference to the simulated truth | 1,694 µm median, 5,219 at 2–5 GeV, 255 at 25–200 GeV |
| network to the reference | 81 µm median (max metric), 17 µm at 25–200 GeV |
| the exact collocation scheme | 0.147 µm |
| the reference integrator's own step convergence | 2.8 × 10⁻⁵ µm (28 picometres) |

## conventions

Colours are the four validated categorical slots of the house palette in their fixed
order (blue for the network, orange for the truth, aqua for the exact scheme, violet for
the integrator's floor), with the reference in ink and the null comparator in muted grey.
The palette validator needs Node, which is not installed on this host, so the published
validated values were used unchanged rather than new hues being chosen by eye. Identity
is never carried by colour alone: every series is named in a legend or labelled directly.
No block letter appears in any title, axis label, legend or caption.

Two figures cannot be true-scale and say so on their face: figure 7 states its vertical
exaggeration (about 120 times) in the axis label, and figure 4 is logarithmic because its
content spans ten decades. Everything else is drawn at equal scale on both axes, or in
the units of the axis.

## the page

**Status (2026-09-22): written up and published to Notion. Trust = Provisional.**

**Notion page:** https://app.notion.com/p/3e35d544b9d981daa65dc05e0ff09f78

Title: *The data in pictures: what happens to a track between the two planes, drawn to
scale.* Properties set: Type = Write up, Status = Done, Date = 2026-09-22, **Trust =
Provisional** (never Verified; only George sets that), Project = Track Extrapolation,
Todos = the reference-versus-truth to-do (`3e35d544…711d`) and the single-network-chain
to-do (`3dd5d544…e3e0`). The markdown sent is saved verbatim as [page.md](page.md).

Built in one `create-pages` call using the draft-then-move route (`creation_mode:
"draft"`, then `move-pages` into the write-up data source `3265d544-b9d9-8000-8b4a-000b13a4b7c6`,
then `update-page update_properties`), because a data-source-parented `create-pages` call
caps at about 7.8 kB and the page is 33 kB.

**Verification (required by `_shared/WRITEUP_RULES.md`).** The page was fetched back after
creation and the raw markdown checked:

| check | result |
|---|---|
| backslash-escaped `\<table` (literalised table) | **0 occurrences** |
| backslash-escaped `\![` (literalised image) | **0 occurrences** |
| real `<table` blocks | 4 (3 data tables + `<table_of_contents/>`) |
| real `<tr>` rows | 23 |
| real `![…](…)` images | 7, all `raw.githubusercontent.com` |
| callout blocks | 1 |
| properties | Trust = Provisional, Type = Write up, Status = Done, Date 2026-09-22, Project and both Todos linked |

Notion normalised two things cosmetically and nothing else: the three markdown links to the
companion write-up became native page mentions, and two bold spans that abut inline code or
an inline equation had their delimiters moved (`**… in **$z$`, and the bold run around
`` `…/data_supplement/` `` in the provenance). Both render correctly.

**The seven figures will not render until George commits and pushes this folder.** The whole
`data_supplement/` folder is untracked; local HEAD is b84693f and the local ref for
`origin/main` is at the same commit. All seven image URLs are pinned to `main`:

```
https://raw.githubusercontent.com/GeorgeWilliam1999/LHCb-Extrapolation/main/single_network_chain_discrete_approach/data_supplement/figures/<name>.png
```

No commit hash was invented. A callout at the top of the page says so, and so does the
Provenance property.

### two numbers that disagree with their source, recorded on the page

1. **The integrator floor on figure 4 — FIXED 2026-09-22.** The script had hard-coded
   `RK6_STEP_FLOOR = 0.0285` µm citing `Data/results/gates.json`, G3. That gate records
   `G3_step_convergence_mm_5vs1` as a median of `2.8459e-08` **mm**, that is 2.8e-5 µm or
   28 picometres (worst 0.0028483 mm = 2.8 µm, n = 200), so the drawn line was a thousand
   times too high: the gate is in millimetres and was read as micrometres. The same slip
   appears in `Data_generation_exploration/Data/README.md`, which calls the gate "28 nm";
   that line is George's to correct. The script now READS the value from
   `../reference_vs_truth/results/rk6_self_consistency.csv` instead of carrying a typed
   constant, figure 4 was regenerated with the axis extended to ten decades, and the Notion
   page records the correction. No ordering on the figure changed.
2. **The medians on figure 5, panel a.** The two dashed lines are computed inside
   `fig5_charge()` from the **test split only** (`np.median` over `test_S_post − test_truth_zpost`,
   n = 217 / 213 in the 5–10 GeV band), while `results/figure_facts.json` stores the
   **all-crossings** values from `decomposition_by_band.csv` (−1,100 / +973 µm, n = 2,216 / 2,070).
   So the annotated numbers on that panel were not the ones in `figure_facts.json`, contrary
   to the guarantee in the section above. **Fixed 2026-09-22:** `figure_facts.json` now records
   BOTH populations, each labelled (`median_dx_q*_test_split_um` for the drawn lines,
   `median_dx_q*_all_crossings_um` for the decomposition), along with
   `panel_a_is_drawn_on`. The page quotes the all-crossings values and says in one sentence
   that the panel is drawn on the test crossings alone.

Also fixed 2026-09-22: the peak-field position annotated on figure 1 ("peak 1.05 T at
z = 4,699 mm") is now stored in `figure_facts.json` as `fig1.peak_By_z_mm`, so a later
revision of the page may quote it.
