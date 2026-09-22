# reference_vs_truth — how far the field-only reference is from the Geant4 truth

**Status (2026-09-22): measured, written up, published to Notion. Trust = Provisional.**

**Notion page:**
https://app.notion.com/p/3e35d544b9d9818282fff3be2d394009

Title: *What the field-only reference shares with the Geant4 truth, and what it cannot: the
material floor, the stale start momentum, and what the networks are measured against.*
Properties set: Type = Write up, Status = Done, Date = 2026-09-22, **Trust = Provisional**
(never Verified; only George sets that), Project = Track Extrapolation, Todos = the
reference-versus-truth to-do (`3e35d544…711d`) and the single-network-chain to-do
(`3dd5d544…e3e0`).

**Verification (required by `_shared/WRITEUP_RULES.md`).** The page was fetched back after
creation and the raw markdown checked:

| check | result |
|---|---|
| backslash-escaped `\<table` (literalised table) | **0 occurrences** |
| backslash-escaped `\![` (literalised image) | **0 occurrences** |
| real `<table` blocks | 8 (7 data tables + `<table_of_contents/>`) |
| real `<tr>` / `<td>` | 60 / 296 |
| real `![…](…)` images | 3, all `raw.githubusercontent.com` |
| display equations `$$…$$` | 7 |
| callout blocks | 1 |
| diff of `page.md` against the live page | identical apart from one cosmetic bold-span normalisation by Notion |

The markdown sent is saved verbatim as [page.md](page.md).

**The figures will not render until George commits and pushes.** This whole folder
(`single_network_chain_discrete_approach/`) is untracked; local HEAD is 1040ee9 and the
GitHub `main` branch is at a062bd8. The three image URLs are pinned to `main`:

```
https://raw.githubusercontent.com/GeorgeWilliam1999/LHCb-Extrapolation/main/single_network_chain_discrete_approach/reference_vs_truth/figures/<name>.png
```

No commit hash was invented. A callout at the top of the page says so, and so does the
Provenance property.

## script → output

| script | what it does | output |
|---|---|---|
| [decompose.py](decompose.py) | reads `../Block_E_single_network_chain/E0_Track_dataset/results/tracks.npz` (all three splits concatenated, 14,482 crossings), forms `d = S_post − truth_zpost` (Geant4-true state minus the field-only RK6 reference at the particle's own first SciFi plane), splits it by charge and momentum band, divides by the bend, compares the width with Highland for air, and tabulates the three-reference numbers and the reference's own accuracy gates | `results/decomposition_by_band.csv`, `results/decomposition_by_pid.csv`, `results/highland_air.csv`, `results/three_references.csv`, `results/rk6_self_consistency.csv`, `results/decompose.log`, `figures/signed_centre_by_charge.png`, `figures/width_vs_momentum.png`, `figures/three_references.png` |

```bash
PY=/data/bfys/gscriven/conda/envs/TE/bin/python; export PYTHONNOUSERSITE=1
cd /data/bfys/gscriven/LHCb_Extrapolation_Project/single_network_chain_discrete_approach/reference_vs_truth
$PY decompose.py            # numpy + matplotlib only, about 20 s
```

Nothing under `Block_E_*`, `Block_F_*`, `Data_generation_exploration/` or `_shared/` was
edited; `decompose.py` only reads. No git state was changed.

### inputs read

| file | used for |
|---|---|
| `../Block_E_single_network_chain/E0_Track_dataset/results/tracks.npz` | every number in the decomposition tables |
| `../Block_F_reweighted_loss/F2_Analysis/results/against_true_state.csv` | the three-reference table and figure |
| `../../Data_generation_exploration/Data/results/gates.json` | gates G1, G2, G3 |
| `../Block_E_single_network_chain/E3_Analysis/results/error_qdz_chain.csv` | exact-scheme endpoint medians (**not** in `comparators.csv`, see note below) |
| `../Block_F_reweighted_loss/F3_Analysis/results/error_qdz_chain.csv` | the same, for the reweighted-loss runs |
| `../Block_E_single_network_chain/E3_Analysis/results/comparators.csv`, `../Block_F_reweighted_loss/F3_Analysis/results/comparators.csv` | straight line, material floor |

## Results

All positions in micrometres, all slopes as a dimensionless difference × 10³. The residual is
**Geant4-true minus field-only reference**, at the particle's own first SciFi plane, on all
14,482 crossings (train 11,567 + val 1,463 + test 1,452).

### The decomposition, by band and charge (`results/decomposition_by_band.csv`)

The 68 % half-width is **charge-corrected**: each charge is moved onto its own median before
pooling, so the antisymmetric systematic cannot inflate the width. The implied momentum
defect is `median(dx/bend over |bend| > 20 mm) × median P`, in MeV.

| band | n (q+ / q−) | median dx, q+ / q− | median dx/bend, q+ / q− | implied defect | 68 % half-width of dx |
|---|---|---|---|---|---|
| 1–2 GeV | 89 (63 / 26) | −14,450 / +14,832 | +8.80e-3 / +8.50e-3 | +16.0 MeV | 9,999 |
| 2–5 GeV | 5,117 (2,649 / 2,468) | −4,826 / +4,509 | +5.39e-3 / +4.97e-3 | +17.9 MeV | 4,372 |
| 5–10 GeV | 4,286 (2,216 / 2,070) | −1,100 / +973 | +2.51e-3 / +2.24e-3 | +16.4 MeV | 1,521 |
| 10–25 GeV | 3,697 (1,870 / 1,827) | −118 / +139 | +6.11e-4 / +6.91e-4 | +9.4 MeV | 647 |
| 25–50 GeV | 1,016 (496 / 520) | +24 / −20 | −2.70e-4 / −2.48e-4 | −8.1 MeV | 312 |
| 50–100 GeV | 240 (126 / 114) | +52 / −58 | −1.19e-3 / −1.29e-3 | −77.5 MeV | 167 |
| 100–200 GeV | 37 (19 / 18) | +51 / −58 | −1.73e-3 / −2.14e-3 | −240.4 MeV | 86 |
| all momenta | 14,482 (7,439 / 7,043) | −1,032 / +873 | +2.65e-3 / +2.37e-3 | +17.0 MeV | 2,262 |

Pooled over charge, max metric `max(|dx|, |dy|)`: median / p95 = 14,723 / 35,183 (1–2),
5,136 / 18,294 (2–5), 1,608 / 5,791 (5–10), 622 / 3,223 (10–25), 296 / 2,047 (25–50),
166 / 946 (50–100), 105 / 539 (100–200), **1,736 / 12,628 over all momenta**. The first four
reproduce `tracks_meta.json`'s material-floor block exactly.

### Slope residuals, both charges pooled (`results/decomposition_by_band.csv`)

| band | n | median abs dx | median abs dtx ×10³ | median abs dy | median abs dty ×10³ |
|---|---|---|---|---|---|
| 1–2 GeV | 89 | 14,723 | 6.44 | 3,763 | 1.02 |
| 2–5 GeV | 5,117 | 4,826 | 1.69 | 1,713 | 0.518 |
| 5–10 GeV | 4,286 | 1,268 | 0.356 | 818 | 0.232 |
| 10–25 GeV | 3,697 | 409 | 0.107 | 371 | 0.108 |
| 25–50 GeV | 1,016 | 178 | 0.0561 | 195 | 0.0513 |
| 50–100 GeV | 240 | 120 | 0.0339 | 72 | 0.0225 |
| 100–200 GeV | 37 | 76 | 0.0215 | 30 | 0.0123 |
| all momenta | 14,482 | 1,286 | 0.368 | 776 | 0.220 |

### Multiple scattering against Highland, air only (`results/highland_air.csv`)

Air between the planes: x/X0 = 5,177.8 mm / 304,000 mm = 0.01703, i.e. 1.703 % of a radiation
length. θ0 = (13.6e-3 / p[GeV]) √(x/X0) (1 + 0.038 ln(x/X0)) rad; RMS displacement ≈ θ0·L/√3
with L = 5,177.8 mm. The last column solves that formula for x/X0 given the measured width.

| band | median P [GeV] | air-only prediction | measured 68 % half-width | ratio | effective x/X0 |
|---|---|---|---|---|---|
| 1–2 GeV | 1.86 | 2,407 | 9,999 | 4.15 | 0.235 |
| 2–5 GeV | 3.44 | 1,304 | 4,372 | 3.35 | 0.158 |
| 5–10 GeV | 6.88 | 652 | 1,521 | 2.33 | 0.081 |
| 10–25 GeV | 14.58 | 308 | 647 | 2.10 | 0.067 |
| 25–50 GeV | 32.06 | 140 | 312 | 2.23 | 0.074 |
| 50–100 GeV | 62.50 | 72 | 167 | 2.32 | 0.080 |
| 100–200 GeV | 120.25 | 37 | 86 | 2.30 | 0.079 |

Above 5 GeV the ratio is a stable 2.1–2.3 and the effective thickness 6.7–8.1 % of a
radiation length. Below 5 GeV Highland's Gaussian-core parameterisation degrades and the
ratio climbs to 3.4 and 4.2; those rows are not read as more material.

### By particle type, 5–25 GeV, |bend| > 20 mm (`results/decomposition_by_pid.csv`)

| particle | n | median P [GeV] | median dx/bend | median abs dx | implied defect |
|---|---|---|---|---|---|
| pion | 6,139 | 9.25 | +1.65e-3 | 811 | +15.3 MeV |
| kaon | 909 | 10.44 | +1.48e-3 | 699 | +15.4 MeV |
| proton | 893 | 9.82 | +1.23e-3 | 395 | +12.1 MeV |
| muon | 42 | 7.40 | +1.11e-3 | 811 | +8.2 MeV |

### The three references, test crossings only (`results/three_references.csv`)

Median of `max(|dx|, |dy|)` at the particle's own first SciFi plane, **endpoint error**, on the
1,452 test crossings. Network columns are, in order, N = 64 / q = 2 / dz = 81 mm,
N = 128 / q = 8 / dz = 40 mm and N = 256 / q = 16 / dz = 20 mm.

| band | n | network vs reference | network vs truth | reference vs truth |
|---|---|---|---|---|
| all | 1,452 | 81 / 96 / 84 | 1,698 / 1,649 / 1,699 | 1,694 |
| 1–2 GeV | 5 | 2,214 / 2,419 / 1,577 | 13,750 / 15,091 / 17,098 | 17,034 |
| 2–5 GeV | 500 | 335 / 383 / 316 | 5,217 / 5,251 / 5,130 | 5,219 |
| 5–10 GeV | 430 | 77 / 97 / 79 | 1,686 / 1,675 / 1,688 | 1,630 |
| 10–25 GeV | 383 | 24 / 28 / 36 | 623 / 628 / 610 | 613 |
| 25–200 GeV | 134 | 17 / 20 / 23 | 276 / 270 / 274 | 255 |

Excluding the 1–2 GeV band (5 test tracks): network-vs-truth differs from reference-vs-truth
by **−2.6 % to +8.4 %**, and reference-vs-truth is **11 to 25 times** the network-vs-reference
error. Including it: −19.3 % to +8.4 %, and 7 to 25 times.

### The reference's own accuracy (`results/rk6_self_consistency.csv`)

| quantity | value | n |
|---|---|---|
| G1 re-propagation closure, median / worst | 2.3e-6 / 22.4 µm | 2,000 |
| G3 step convergence 5 mm vs 1 mm, median / worst | 2.8e-5 / 2.8 µm | 200 |
| G2 label vs next hit (short legs), median / >5 GeV / <2 GeV / p95 | 11.9 / 3.8 / 38.8 / 658.8 µm | 100,660 |
| exact scheme, N = 64, q = 2, dz = 81 mm (endpoint radial median) | 0.147 µm | 1,452 |
| exact scheme, N = 128, q = 8, dz = 40 mm | 0.0115 µm | 1,452 |
| exact scheme, N = 256, q = 16, dz = 20 mm | 0.00105 µm | 1,452 |
| straight line (endpoint radial median) | 450,542 µm | 1,452 |
| material floor, true state vs the reference track (endpoint radial median) | 1,896 µm | 1,452 |

## Reproduction of the numbers already recorded in the to-do

`decompose.py` was written to reproduce the table in the to-do
[Establish how closely the field-only reference follows the Geant4 truth…](https://app.notion.com/p/3e35d544b9d9812b804ac9e34409711d),
which was produced on 2026-09-22 by a one-off script that was never filed. Every cell
reproduces to the precision the to-do quotes, with one exception:

| cell | to-do | `decompose.py` | verdict |
|---|---|---|---|
| all n, median dx per charge, median dx/bend per charge, implied defect (2–5, 5–10, 10–25, 25–50, 50–100 GeV) | as recorded | identical to the quoted rounding | **reproduced** |
| 68 % half-width, 2–5 / 5–10 / 10–25 / 50–100 GeV | 4,370 / 1,520 / 647 / 167 | 4,372 / 1,521 / 647 / 167 | **reproduced** (rounding only) |
| 68 % half-width, 25–50 GeV | **314** | **311.6** | **0.8 % disagreement, not resolved** |
| particle-type ratios at 5–25 GeV (π, K, p, µ) | +1.7 / +1.5 / +1.2 / +1.1 e-3 | +1.65 / +1.48 / +1.23 / +1.11 e-3 | **reproduced** |
| scattering "about 2.2× the air-only Highland (0.3 mm at 15 GeV)" | 2.2× | air-only at 14.58 GeV = 308 µm ✓; ratio at 10–25 GeV = **2.10**, ratio over 5–200 GeV = 2.10–2.33 | air prediction reproduced; 2.2 is the range average, the 10–25 GeV value alone is 2.10 |
| "effective 8 % of a radiation length" | 8 % | 6.7 % at 10–25 GeV, 6.7–8.1 % over 5–200 GeV | consistent; the page quotes the range |

The 25–50 GeV half-width is the only cell that does not reproduce. The statistic used here
(each charge centred on its own median, then pooled, then the 68th percentile of the absolute
deviation) reproduces every other band exactly, so the 314 in the to-do is most likely a
different quantile convention or a transcription slip in the unfiled one-off script. Both
values are recorded; neither was chosen over the other. The page quotes **312**.

Two further statements from the F2 README were checked and **do not** reproduce exactly:

- "the network-vs-true and RK6-vs-true medians differ by 0–4 % in every band" → the measured
  range is **−2.6 % to +8.4 %** excluding the 1–2 GeV band (5 tracks), −19.3 % to +8.4 %
  including it;
- "the network-vs-RK6 error is 6–20 times smaller" → the measured range is **11 to 25 times**
  excluding that band, 7 to 25 including it.

The page uses the recomputed ranges and says in one sentence that the earlier note's numbers
are superseded. The qualitative conclusion is unchanged.

## Notes for whoever runs this next

- The exact-scheme endpoint medians are in `error_qdz_chain.csv` (column `exact_med_um`), not
  in `comparators.csv`, which carries only the straight line and the material floor. The plan
  named `comparators.csv`; `decompose.py` reads both and labels each row with its source file.
- `figures/three_references.png` is a redraw of `F2_Analysis/figures/against_true_state.png`
  from the same CSV, made so that the figure lives in this folder and its legend names the
  networks by step count, stage count and step length with no block letters.
- Step 1 of the plan on the to-do (re-dump with `MCHit::p()` and rebuild the crossing set as
  `tracks_v2.npz`) has **not** been run. Everything here is measured on the current
  `tracks.npz`, which stays untouched so that every Block E and Block F number remains
  reproducible.
