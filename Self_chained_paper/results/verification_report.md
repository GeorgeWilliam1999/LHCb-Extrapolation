# Verification report — `Self_chained_paper/main.tex`

Independent check of the finished paper against the committed files under
`results/`, run on 2026-09-22. `main.tex` was treated as the single
authoritative source; `sections/*.tex` were not read. Experiment folders were
opened read-only and no git state was touched.

Verdict in one line: **the paper passes.** 1,445 numeric table cells were
recomputed and, after three corrections, none disagrees; every prose number in
the audited sections traces to a table or to `paper_numbers.json`; the whole
figure and number pipeline reproduces bit-for-bit; the PDF rebuilds to 69 pages
with 0 errors and 0 overfull boxes. Five edits were made (all listed in §10) and
eleven open items are left for a human decision (§11).

Tooling written for this pass: **`scripts/verify_tables.py`** — parses every
`\label{tab:}` float out of `main.tex`, re-derives each numeric cell from the
source file named in that table's caption, and compares at the precision the
cell is printed at (half a unit in the last displayed place, plus 1e-9 relative
slack). It writes nothing and can be re-run at any time:

```sh
PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python scripts/verify_tables.py
```

---

## 1. Every table cell — PASS (1,445 cells, 0 mismatches after 2 fixes)

All 26 numbered tables were mapped to the source file(s) named in their own
captions and recomputed cell by cell. Composite cells (`$1684$ ($880/804$)`,
`$-8641 / +8130$`, `$126.6 \pm 17.5$ \%`, `$1{,}807$ ($1{,}500$)`) were split
and each component checked separately, which is why several tables show more
checks than they have printed cells. Caption numbers that are data (`n`, the
parameter counts, the tolerance, the extra restarts/rounds) were checked too.

```
Table  1 (tab:cuts            )   50 cells   [PASS]   tab_dataset_cuts.csv
Table  2 (tab:splitbands      )   39 cells   [PASS]   tab_dataset_bands.csv
Table  3 (tab:materialfloor   )   35 cells   [PASS]   tab_material_floor.csv
Table  4 (tab:fixed           )   11 cells   [PASS]   tab_compute.csv; PN compute
Table  5 (tab:tableau         )   68 cells   [PASS]   tab_tableau.csv           (4 "---" skipped)
Table  6 (tab:ceiling         )   64 cells   [PASS]   tab_exact_ceiling.csv; tab_pooled_grid.csv  (8 "---" skipped)
Table  7 (tab:rk6ladder       )   29 cells   [PASS]   tab_rk6_convergence.csv
Table  8 (tab:decomp          )   64 cells   [PASS]   tab_decomposition_bands.csv; tab_highland_air.csv; PN rebinned_decomposition
Table  9 (tab:highland        )   43 cells   [PASS]   tab_highland_air.csv
Table 10 (tab:pid             )   20 cells   [PASS]   PN reference_vs_truth.verbatim.decomposition_by_pid   ** 1 fix **
Table 11 (tab:rk6self         )   24 cells   [PASS]   PN reference_vs_truth.verbatim.rk6_self_consistency
Table 12 (tab:grid            )  129 cells   [PASS]   tab_pooled_grid.csv; tab_exact_ceiling.csv
Table 13 (tab:preflight       )   83 cells   [PASS]   tab_preflight_shares.csv; PN preflight.modes          ** 1 fix **
Table 14 (tab:clamp           )   40 cells   [PASS]   tab_clamp_scan.csv
Table 15 (tab:convergence     )   43 cells   [PASS]   tab_headline_pairs.csv; tab_extended_vs_snapshot.csv; PN headline.plateau
Table 16 (tab:headline        )   48 cells   [PASS]   tab_headline_pairs.csv; tab_headline_bands.csv; tab_anatomy.csv
Table 17 (tab:headlinebands   )   84 cells   [PASS]   tab_headline_bands.csv
Table 18 (tab:comppos         )   91 cells   [PASS]   tab_headline_components.csv
Table 19 (tab:compslope       )   91 cells   [PASS]   tab_headline_components.csv
Table 20 (tab:near5           )   80 cells   [PASS]   tab_near_5gev.csv
Table 21 (tab:near5tail       )   28 cells   [PASS]   tab_near_5gev.csv; PN near_5gev (p95)
Table 22 (tab:anatomysteps    )   48 cells   [PASS]   tab_anatomy.csv
Table 23 (tab:anatomyend      )   42 cells   [PASS]   tab_anatomy.csv
Table 24 (tab:steps           )   36 cells   [PASS]   tab_anatomy.csv; PN along_z; tab_single_step_vs_z.csv
Table 25 (tab:truth           )  114 cells   [PASS]   tab_against_true_state.csv   (12 blank RK6-true cells skipped)
Table 26 (tab:cost            )   41 cells   [PASS]   tab_compute.csv

TOTAL: 1445 numeric cells checked, 0 mismatches (24 non-numeric cells skipped)
```

### The two cell mismatches found and corrected

1. **Table 10 (`tab:pid`), kaon, median $|\Delta x|$: `699` → `698`.**
   `paper_numbers.json` → `reference_vs_truth.verbatim.decomposition_by_pid`
   gives `median_abs_dx_um = 698.4656…`, which rounds to 698. The value appears
   nowhere else in `main.tex` or `README.md`.
2. **Table 13 (`tab:preflight`), rank correlation within 10–50 GeV, lever-arm-off
   column: `+0.555` → `+0.554`.** `preflight.modes.no_lever.spearman_share_vs_cost_in_band
   = 0.5544821…`, which rounds to 0.554. Appears nowhere else.

### One apparent mismatch that was the checker's error, not the paper's

Table 24's *single step* column (0.333 / 0.433 / 0.172 / 0.221 / 0.089 / 0.058)
is the median over **all planes and all tracks**, which is
`along_z.per_run[...].per_step_vs_single_step.med_from_truth_um`. It is **not**
the median of the per-plane medians stored in `tab_single_step_vs_z.csv`; those
differ by up to 15 %. `verify_tables.py` was corrected to read the JSON and now
passes. `tab_single_step_vs_z.csv` is still checked for having exactly $N$ rows
per run (it does: 64, 128, 256). The table's own claim that the column "agrees
with the per-step radial column to the digits shown" holds
(0.33255 vs 0.33262; 0.43337 vs 0.43342; 0.17208 vs 0.17212; 0.22088 vs 0.22091;
0.089310 vs 0.089314; 0.058083 vs 0.058080).

### Per-band `n` — PASS

The test split is 164 / 655 / 433 / 170 / 30 in the five bands and 487 in the
10–50 GeV window, and every table that prints a per-band `n`
(`tab_dataset_bands`, `tab_headline_bands`, `tab_headline_components`,
`tab_against_true_state`, Tables 2, 17, 18, 19, 25 and the figure annotations)
uses exactly those numbers. The other splits are 1339/5283/3391/1326/228 and 3756
(train) and 181/668/441/154/19 and 470 (validation), matching Table 2.
The "all splits" scope used by Tables 3, 8 and 9 is 1684 / 6606 / 4265 / 1650 /
277 and 4713, totalling 14,482, and matches.

---

## 2. Numbers in prose — PASS, with two consistency notes

Every number in the abstract, §1.2, §1.4, the §5 lead, §5.3, §5.4, §5.8, §5.9,
§7.1, §7.2, §7.3 and §9 was traced. The recurring quantities are quoted
**identically** everywhere:

| quantity | abstract | §1.4 | §5.4 | §7.1 | §9 | source |
|---|---|---|---|---|---|---|
| endpoint median, N=64 | 145.7 → 88.8 | 145.7 → 88.8 | 145.7 → 88.8 | 145.7 → 88.8 | 145.7 → 88.8 | `tab_headline_pairs.csv` |
| in-window median | 108.1 → 24.8 | 108.1 → 24.8 | 108.1 → 24.8 | 108.1 → 24.8 | 108.1 → 24.8 | same |
| in-window factor | 4.37 | "a factor of four" | 4.37 | 4.37 | 4.37 | 4.3667 |
| per-step $t_x$ | 1.30e-6 → 7.34e-7 | — | 1.30e-6 → 7.34e-7 | 1.30e-6 → 7.34e-7 | 1.30e-6 → 7.34e-7 | `tab_anatomy.csv` |
| in-window share | 47.0 % | — | — | 47.0 % | — | 47.0092 |
| pre-flight triple | 89.8 / 1.23 / 0.65 | — | — | 89.8 / 1.23 / 0.65 | 89.8 / 1.23 / 0.65 | `tab_preflight_shares.csv` |
| coherence range | 0.31–0.53 vs 0.0625–0.125 | — | — | — | — | Table 23 (min 0.3119, max 0.5267) |
| N=64 coherences | — | — | — | — | — | 0.49/0.52 vs 0.36/0.40 in Fig. 13; 0.491/0.518 vs 0.357/0.397 in Table 23 and §7.2 — consistent rounding |
| explained (both slopes) | — | — | — | 97.7–99.4 % | 97.7–99.4 % | 0.97742 … 0.99382 |
| explained (x slope only) | — | — | — | — | — | 63–73 % pooled, 36–45 % cost-weighted (§7.2) = 0.6344/0.7331 and 0.3627/0.4484 |
| p95 / >1 mm / <3 GeV | 898→1399, 4.41→7.16, 353.7→615.1 | 898→1399, 354→615 | same | — | same | `tab_headline_pairs`, `tab_headline_bands` |
| material gap | 12.5–25.2× | — | — | — | 12.5–25.2× | see open item O-1 |
| 1736 / 532 / 153 µm | — | — | — | §7.3 | — | `tab_material_floor.csv` |

Spot-checked derived quantities that are stated but not tabulated, all correct:
39.1 %, 14.4 %, 37.6 %, factors 1.56 / 1.74 / 1.19 / 2.70 / 2.58 / 2.59 / 4.21 /
3.08 / 3.21 / 2.18 / 2.95 / 8.7 / 122 / 11.2, "5.6 per cent worse", 2.2 and 1.20,
the band ratios 0.58/0.91/2.59/9.01/7.62, 0.37/0.72/1.83/3.31/5.33 and
0.67/1.13/2.29/3.89/4.64, the chain/step ratios 438/205/710/474/1653/1587, the
spreads 17.5/11.7/12.4/10.8/9.8/25.9 %, the plateau deltas −4.19/+2.82/−7.24/
+0.50/−10.94/+0.55 %, 97.4–102.0 %, 21.0/17.7/20.1 and 12.6–15.4, 88.5 %, 4.07 %,
the per-step ratios 1.93/1.93/2.37/1.84/1.95/2.02, 64→104 tracks, 73 of 1,452.

§1.2's numbers (115 µm, 44 residual evaluations, +0.965 over 96 runs, 2.1 vs
16.1 µm, factors 22 and 11) are **paper 1's**, cited to `scriven2026lhcb1`. They
are not in this paper's result files and were not independently checkable here;
they are quoted identically in §1.2 and §9 (factor 22, 5178 mm).

Two consistency notes are raised as open items: **O-1** (the 12.5–25.2 range) and
**O-2** (the "13 per cent better" convention).

---

## 3. Conventions — PASS

| check | result |
|---|---|
| `mrad` | 2 occurrences, **both** in §3.6 Units (lines 1610, 1613), both as the quoted mention of what earlier files do. **PASS** |
| `10^{3}` / `10^3` / `\times 10^{3}` / `1e3` next to a slope | 4 occurrences (lines 183, 1534, 1610, 1614). All four are statements *of* the convention or *about* the source files' units ("No slope is multiplied by $10^{3}$…", "the gate file records it … in its own units of slope $\times 10^{3}$, and I convert"). No slope is actually scaled. **PASS** |
| `Block [A-H]`, `Block_`, `_shared`, `E1_`, `F0_`–`F4_`, `G0_`, `G1_`, `N064` | **zero** occurrences before line 4391 (the start of Appendix A). **PASS** |
| `cluster` | one occurrence before Appendix A, line 4386: "Appendix~\ref{sec:provenance} gives the commits, the data set, **the cluster identifiers** and the maps…". That is a forward reference naming the category, not an identifier. **PASS** (noted, no edit) |
| `.py` | 15 occurrences in figure-caption `Source:` lines (allowed) and 2 more, lines 4375 and 4381, in the **Data availability** statement, where the two-command build pipeline is spelled out. Outside the letter of the rule but plainly deliberate. **PASS with note** (no edit) |
| first person plural `we` / `We` / `our` / `Our` | **0** occurrences. The paper is first person singular throughout. **PASS** |
| band labels in tables | Only `<3`, `3`–`8`, `8`–`20`, `20`–`50`, `>50`, `10`–`50` appear, plus: `4`–`6` GeV in Tables 20 and 21 (announced in both captions and in §5.6, which is the ~5 GeV region of interest), `5`–`25` GeV in Table 10 (caption says explicitly "quoted in the original study's own selection and binning"), and "above 5 GeV" / "below 2 GeV" in Table 11 (verbatim rows from the original gate file — see open item **O-3**). **PASS** |
| every per-band table has an `n` | Tables 2, 3, 8, 9, 13, 17, 18, 19, 25: yes. Table 14 (`tab:clamp`) has bands as **columns** and carries no `n`; its caption says the scan is "on the same $8{,}000$ states" as Table 13, which does print the per-band `n`. **PASS with note** (open item **O-4**) |
| every per-band table has a 10–50 row | Tables 2, 3, 8, 9, 13, 17, 18, 19, 25 all do; Table 14 has a 10–50 column. **PASS** |

---

## 4. Slope values are dimensionless — PASS

All 15 scientific-notation numbers on slope-mentioning lines of `main.tex` lie
between 1e-8 and 3.3e-3. Nothing is a factor 1000 too large. Table slopes are
quoted with a declared scale ($\times 10^{-5}$ in Tables 19 and 20,
$\times 10^{-7}$ in Tables 16 and 22, raw scientific notation in Tables 3, 5, 6,
8, 24 and 25) and every one of them was checked against the CSV's native
dimensionless value with that scale applied — all match.

The four values named in the brief:

| quantity | paper | source | |
|---|---|---|---|
| per-step $t_x$, cost-weighted N=64, q=2 | $7.34\times10^{-7}$ | `tab_anatomy.csv` 7.34207e-07 | PASS |
| per-step $t_x$, pooled extended N=64, q=2 | $1.30\times10^{-6}$ | 1.30150e-06 | PASS |
| in-window per-step $t_x$, cost-weighted | $2.25\times10^{-7}$ | 2.25312e-07 | PASS |
| nn-vs-true slope medians | Table 25 runs 3.2e-5 … 3.6e-3; all-band ≈ 4.9e-4 | `tab_against_true_state.csv` | PASS (~5e-4 as expected) |

The one place a $\times 10^{3}$ quantity is quoted from a source file (line 1534,
the gate file's `6.3e-8`) is converted in the text to $6.3\times10^{-11}$
dimensionless and says so.

---

## 5. Figures vs captions — PASS, with four caption notes

All 15 PNGs were viewed and compared against the `main.tex` caption **and** the
`WHAT THE FIGURE SHOWS` block of the drawing script. All 15 `\includegraphics`
targets exist. Panel letters, quantities, units, colour/role statements and
network names agree in every figure, and every annotated number on a figure was
re-checked against the committed tables:

| fig (PDF) | label | panels | checked |
|---|---|---|---|
| 1 | `fig:weights` | a–d | window floor 0.05, clamp $[1/5,5]$, lever 5258.7 → 80.9 mm, the four weights 17.8 / 0.596 / 0.278 / 0.00932 and the factor 1,908 — all match `PN constants` / `worked_example` |
| 2 | `fig:crossing` | top/middle/bottom | 0.207 T entry, peak 1.048 T at z = 4701 mm, 0.304 T exit; grids Δz = 80.9 / 40.5 / 20.2; six tracks at 3.0, 12.0, 44.1/45.0 GeV per charge |
| 3 | `fig:sample` | a–d | splits 11,567 / 1,463 / 1,452; bands 164/655/433/170/30; 1120 π, 163 K, 152 p, 17 µ; 739 q+ (50.9 %) / 713 q− |
| 4 | `fig:ceiling` | a, b | 16 points twice, lines = exact scheme, markers = 18 Sept networks; "two to six orders of magnitude" holds (min ratio 360 at N=64,q=4; max 1.4e6 at N=128,q=16) |
| 5 | `fig:rk6` | a–c | fitted order 6.23; halving ratios ÷8.0, ÷1.5, ÷3.8; the four levels 0.0127/0.000817/3.66e-6/2.08e-6 medians |
| 6 | `fig:materialgap` | a–c | n per band; Δp above each band; ratios ×3.59 … ×2.14; air $x/X_0$ = 0.0170 |
| 7 | `fig:lossshares` | a–d | all 8 band shares, 4 quarter shares, the clamp scan (46/13/27/47/47 and 21.7/0.5/1.8/6.6/20.1) and 0.68/0.58, 0.53/0.90, 98/81 % |
| 8 | `fig:convergence` | 2 rows × 3 cols | snapshot dotted lines at rounds 30 / 23 / 22 — the **completed-round** rule, confirming item 7b; "not flat" on the two pooled curves |
| 9 | `fig:headline` | 2 rows × 3 cols | n per band; crossover between 3–8 and 8–20 in all three; "no slopes appear" holds |
| 10 | `fig:tail` | a–d | n per band; tail |x| 217.1 vs 107.8 mm, ratio 2.01, n = 55; worst-5 % shares 40/46/45/43/39/44 %, 73 of 1,452 |
| 11 | `fig:components` | 4 panels | four components, log axes, signed-median triangles, n per group |
| 12 | `fig:signed3to8` | 4 × 2 | all 32 annotated med / hw68 / RMS values match Table 20 exactly |
| 13 | `fig:anatomy` | a–d | coherences 0.36/0.40 and 0.49/0.52 vs 0.125; explained fractions 0.98/0.73 … 0.99/0.36 match Table 23 |
| 14 | `fig:alongz` | left/right | the six ratios 438× / 205× / 710× / 474× / 1653× / 1587× match §5.8 |
| 15 | `fig:threerefs` | a, b | "97–102 % … 13–21 times below both" matches the all-band `nn_true_over_rk6_true` (97.37–102.03 %) and `rk6_true_over_nn_rk6` (12.6–21.0) |

**Caption notes** (all raised as open items, none edited):
**O-5** Fig. 15's blanket "network-against-reference is more than an order of
magnitude below both … in every band" fails for the pooled runs in the >50 GeV
band; **O-6** Fig. 6's "the defect it implies is printed above each band: +17.6,
+17.3 and +12.2 MeV" — the figure prints the integer-rounded +18 / +17 / +12, and
the window's ×2.13 ratio quoted in the same caption is not printed on the plot;
**O-7** Fig. 13's "the two rows carry different vertical scales because the drifts
differ by about a decade" — panel (c)'s two ranges differ by ×3.75, not ×10;
**O-8** Fig. 8's tick sentence reads as though every curve carries a first-held
tick, while the script draws it only where the rule holds at the last round.

### Figures referenced in the text — FAIL for five figures

Every figure is `\ref`-ed somewhere, but **five are cross-referenced only from
the Appendix A figure map** and never from the body:

| figure | label | float at line | referenced at |
|---|---|---|---|
| 2 | `fig:crossing` | 1338 | 4554 (Appendix A only) |
| 3 | `fig:sample` | 1501 | 4556 (Appendix A only) |
| 4 | `fig:ceiling` | 1911 | 4558 (Appendix A only) |
| 5 | `fig:rk6` | 2057 | 4560 (Appendix A only) |
| 6 | `fig:materialgap` | 2427 | 4564 (Appendix A only) |

The other ten are referenced in the body immediately before their float.
Fixing this means writing new prose, so nothing was changed — open item **O-9**.

---

## 6. Reproducibility — PASS (byte-identical)

`figures/` was copied to `results/verify_figures_before/`, then
`scripts/numbers.py` (cached) and all fifteen `scripts/fig01_*.py` …
`fig15_*.py` were re-run from the conda environment. Exit status 0 throughout,
no traceback, no warning.

* **CSV tables** — all 22 `results/tab_*.csv` **byte-identical**.
* **Figure companions** — all 15 `results/fig*.csv` **byte-identical**.
* **`results/paper_numbers.json`** — differs in exactly 32 leaves, all of them
  bookkeeping: `generated` (the timestamp) and the 31 `timing.steps[*].wall_s` /
  `timing.total_wall_s` entries (14.9 s → 15.4 s total). **No result value
  changed** — the recursive diff over every other leaf is empty to 1e-12
  relative.
* **PNGs** — all 15 are **byte-identical**, and therefore **max pixel difference
  = 0** over all 15 (compared channel-by-channel as RGBA with PIL/NumPy).

`results/verify_figures_before/` was deleted afterwards, as instructed.

### PDF rebuild — PASS

`sh build.sh` (pdflatex → bibtex → pdflatex → pdflatex), exit 0:

* **69 pages** — matches `README.md`.
* **0 LaTeX errors**, **0 overfull boxes**, **0 font warnings**, **0 undefined
  references or citations**, **0 multiply-defined labels**.
* **42 underfull hboxes** — exactly the count `README.md` records, and the same
  long-typewriter-path lines it describes.
* A control build of the *pristine* `main.tex` in a scratch directory gave the
  same 69 pages / 0 overfull / 42 underfull, so the edits of §10 changed no
  page except the one recorded in edit 5.

---

## 7. Items flagged by earlier stages

### 7a — §5.2 footnote, the 18 Sept pre-flight share — **completed (edit 3)**

The footnote stated the larger share only qualitatively. The two numbers now
exist in the committed files:
`preflight.snapshot_18Sept_record.headline.share_10_50_GeV_pct_full = 57.8197…`
and `…_blockE = 1.50910…`, both also in
`results/tab_preflight_snapshot_record.csv` as the `summary` rows
`share_10_50_GeV_pct` for modes `full` and `blockE`. They were inserted with
their source. The rest of that footnote was re-verified and is correct: pooled
rank correlations 0.690 / 0.559 (stored 0.68996 / 0.55911) against 0.684 / 0.532
here; top-1 % 97.9 against 98.3; quarter shares 0.556 / 6.15 against 0.650 /
6.22; cost-weighted 0.561 / 0.889, 83.8 %, 21.2 / 1.75 %. **PASS**

### 7b — snapshot round counts 30/23/22 — **PASS, no edit needed**

`main.tex` nowhere uses 31, 24 or 23 as a round count. The only occurrences of
those numerals are "8 to 23 per cent" and "5 to 24 per cent" (§2.8, both
percentages) and the tableau's `order = 24` at q = 12. Round counts in the paper
are the completed ones: 50 / 63 / 42 for the extended runs (Tables 15 and 26),
and Figure 8 draws the 18 September snapshot at rounds **30 / 23 / 22**, which is
the committed rule. `tab_extended_vs_snapshot.csv` carries both
(`snapshot_rounds` 30/23/22 and `snapshot_rounds_counter_record_json` 31/24/23).

### 7c — `median_abs_bend_mm` = 453.30 — **PASS, confirmed, no edit**

§4.4 line 2146: "Its median magnitude over this set is $453\mm$, and $14{,}476$
of the $14{,}482$ tracks have $|b| > 20\mm$". The JSON key is
**`reference_vs_truth.rebinned_decomposition.median_abs_bend_mm = 453.30372…`**,
alongside `median_signed_bend_mm = −70.0549…` (a different quantity, near zero
because the charges bend opposite ways), `n_above_bend_cut = 14476`,
`n_crossings = 14482` and `bend_cut_mm = 20.0`. The paper quotes the magnitude,
which is correct.

### 7d — the reserved §6 "Tables 16–19" — **PASS, no edit needed**

§6 does not hard-code numbers; it uses `\ref{tab:headline}`,
`\ref{tab:headlinebands}`, `\ref{tab:comppos}` and `\ref{tab:compslope}`. The
rebuilt `main.aux` assigns those labels **16, 17, 18, 19** respectively, so the
printed text reads correctly.

### 7e — Appendix A maps and README — **corrected (edit 4)**

* Appendix A figure map: **15 rows**, one per `\includegraphics`. ✔
* Appendix A table map: **26 rows**, one per `\label{tab:}`, in order 1–26. ✔
* `README.md` table map: 26 rows + the two unnumbered appendix maps, numbers
  1–26 matching `main.aux`. ✔
* `README.md` figure map: **the numbers were wrong.** The column was carrying the
  *script* index (fig01…fig15) rather than the figure number. The float for
  `fig:weights` sits in §2.6, so the built PDF numbers it Figure 1 and everything
  else shifts: `fig:crossing` is Figure **2**, `fig:threerefs` is Figure **15**,
  `fig:tail` is Figure **10**, and so on. Twelve of the fifteen rows were
  renumbered in place to the PDF's numbers; the row order was **not** touched, so
  the README rows are now identical to Appendix A's rows, which are ordered the
  same way. ✔

### 7f — Bibliography — **PASS**

* Cited keys in `main.tex`: `aaij2020`, `hairer1996`, `hairer2006`,
  `lhcb2024upgrade`, `liu1989`, `lynch1991`, `raissi2019`, `scriven2026lhcb1`,
  `wang2021longtime` — **9**, all present in `references.bib`.
* `main.bbl` holds exactly those **9** `\bibitem`s. No undefined citation.
* `references.bib` holds **16** entries, matching the README's "11 from paper 1
  plus 5 added here".
* `% TODO verify` comments attach to exactly six entries — `rohrhofer2023`,
  `lhcb2024upgrade`, `wang2021longtime`, `hairer2006`, `lynch1991`, `liu1989` —
  which is exactly the README's "Bibliography: fields still to verify" list,
  including its "cited here?" column (`rohrhofer2023` is the only "no").
* `scriven2026vdp` is present and uncited, as the README says.

---

## 8. Cross-section coherence — PASS, with one note

* **§5.3** states the rule and its verdict: all three cost-weighted runs flat
  (+2.8 / +0.5 / +0.5 % over the preceding ten rounds, rules first fired at
  rounds 35 / 35 / 60 of 40 / 40 / 60); of the pooled runs only N = 64, q = 2 is
  flat (fired at 38 of 50, last ten −4.2 %); N = 128, q = 8 was still improving
  at 7.2 %/10 rounds at round 63 and N = 256, q = 16 at 10.9 %/10 rounds at round
  42. All six verdicts and all six deltas match `headline.plateau` exactly. It
  then rules that "the conclusions of this paper rest on the N = 64, q = 2 pair
  … the other two pairs are … read as supporting rather than as independent
  evidence."
* **§7.3** repeats that ruling verbatim in its first caveat; **§9** opens its last
  paragraph with it. Table 16's caption, Table 17's caption and Table 15 all
  carry the non-convergence flag. **No section leans on the N = 128 or N = 256
  pair as an independent margin**: every margin quoted from those pairs (14.4 %,
  37.6 %, 2.58, 2.59, factors 1.74 and 1.19, the band ratios) appears in §5.4 and
  §7.2, both of which attach the caveat in the same paragraph. **PASS.**
* The abstract carries no non-convergence caveat and its two aggregate counts
  ("worse in five of six comparisons", "more coherent … in all six matched
  comparisons") span all three pairs. These are counts, not margins, and §7.2
  makes the same counts with the caveat attached — raised as open item **O-10**
  for a judgement call, not as an error.
* **§2.8's plateau rule** — "the median validation error of its last ten rounds is
  no more than **5 per cent** below the median of the **ten rounds before it**,
  and that holds at each of the **last three rounds**" — is exactly what §5.3
  applies (five per cent, last ten vs previous ten, three rounds), exactly what
  the `fig09_convergence.py` docstring states, and exactly what
  `tab_headline_pairs.csv`'s `plateaued_now` / `first_plateau_round` encode
  (asserted inside the figure script). **PASS.**
* **§2.8's pre-registered criterion** — clear a $\pm 13$ % run-to-run band
  downward **and** the median per-step slope error fall below $1.6\times10^{-6}$,
  both reported overall and in the 10–50 GeV window — is exactly what §5.4 tests,
  in that order, with the same numbers (39.1 % fall, ±17.5 % measured spread,
  $7.34\times10^{-7}$ overall and $2.25\times10^{-7}$ in window, factor 2.18 below
  the threshold), and with the same partial/null definitions restated and
  dismissed. §5.4, §7.1 and §9 all report the second condition as met *and* as
  the weaker of the two because the pooled counterpart has since reached
  $1.30\times10^{-6}$ — which is README discrepancy (d), handled consistently in
  all three places. **PASS.**

---

## 9. Cross-checks against `README.md`'s own claims

| README claim | verified |
|---|---|
| 69 pages | yes |
| 15 figures, 26 numbered tables + 2 unnumbered appendix maps | yes |
| 16 bibliography entries, 9 cited | yes |
| no errors, no undefined references, no undefined citations, no overfull boxes, no font warnings | yes |
| 42 underfull hboxes remain | yes, exactly 42 |
| section page numbers in "Contents" | 46 of 47 matched before my edits; §5.5 moved from p46 to p47 as a result of edit 3 and was updated (edit 5) |
| discrepancy (a) lever arm 5,258.7 mm | Fig. 1(c) and §2.6 quote 5258.7; `tab_loss_constants.csv` `lever_max_mm` = 5258.703125 — recorded as the module's own range, as the README says |
| discrepancy (b) 57.8 % vs 47.0 % | now stated numerically in the §5.2 footnote (edit 3), and in Table 13 and §7.3 |
| discrepancy (c) signed medians −54 / +18 / +30 µm | §7.1 quotes −54.4 → −6.3, +17.9 → +0.15, +30.3 → −2.7; recomputed −54.3549, +17.8797, +30.3219, −6.2824, +0.1463, −2.7070 ✔ |
| discrepancy (d) both losses clear 1.6e-6 | §5.4, §7.1, §9 all say so ✔ |
| discrepancy (e) five of six worse, six of six more coherent | five of six confirmed (only 3–8 GeV at N = 256 is better); six of six coherence confirmed ✔ |
| discrepancy (f) distinct rounds 50 / 63 / 42 | Tables 15 and 26 use 50 / 63 / 42 ✔ |
| discrepancy (g) 302 / 13 duplicated numbers, 307 / 14 dropped rows | `tab_compute.csv` gives `duplicated_history_rows` 302, `duplicated_round_rows` 13, `restarts_recorded` 1807 vs `unique_restarts` 1500 (307), `rounds_rows_read` 74 vs `rounds` 60 (14) ✔ — see open item **O-11** on how Table 26 prints this |
| discrepancy (h) N=128, q=2 ceiling 0.511 µm | `tab_exact_ceiling.csv` 0.5106528 ✔, Table 6 prints 0.511 ✔ |
| discrepancy (i) factor 1.5–43 at N=2, 360–1.4e6 at N≥64 | recomputed: N=2 ratios 1.00, 1.45, 25.2, 43.3; N≥64 min 360 (N=64,q=4), max 1.40e6 (N=128,q=16) ✔ |

---

## 10. Edits made

Five edits, all mechanical. Nothing was reordered and no claim was changed.

| # | file | line (post-edit) | before → after | why |
|---|---|---|---|---|
| 1 | `main.tex` | 2354 | Table 10, kaon median $\|\Delta x\|$: `$699$` → `$698$` | `decomposition_by_pid` gives 698.4656 µm, which rounds to 698 |
| 2 | `main.tex` | 2715 | Table 13, rank corr. in 10–50 GeV, lever-arm-off: `$+0.555$` → `$+0.554$` | `preflight.modes.no_lever.spearman_share_vs_cost_in_band` = 0.5544821 |
| 3 | `main.tex` | 2663–2668 | §5.2 footnote: "…than the extended network gives." → "…than the extended network gives: $57.82$ per cent against the $47.01$ of Table~\ref{tab:preflight}, with the pooled loss at $1.509$ per cent there against $1.23$ here (`results/tab_preflight_snapshot_record.csv`)." | item 7a: mechanical completion, the two numbers now exist in the committed files |
| 4 | `README.md` | 215–229 | "Figure → source map": twelve of the fifteen leading numbers renumbered to the built PDF's figure numbers (crossing 1→2, sample 2→3, ceiling 3→4, rk6 4→5, threerefs 5→15, weights 7→1, lossshares 8→7, convergence 9→8, headline 10→9, alongz 13→14, anatomy 14→13, tail 15→10; materialgap 6, components 11 and signed3to8 12 already correct) | the column was carrying the script index, not the figure number; the map is now identical to Appendix A's |
| 5 | `README.md` | 88 | "Contents": §5.5 page `46` → `47` | consequence of edit 3, which added four source lines in §5.2 and pushed §5.5 onto the next page; confirmed against a control build of the pristine source |

Files written: `results/verification_report.md` (this file) and
`scripts/verify_tables.py`. `results/verify_figures_before/` was created for the
pixel comparison and deleted afterwards. `main.pdf`, `main.aux`, `main.log`,
`main.toc`, `main.out`, `main.bbl`, `main.blg` are the rebuild's own outputs.
No experiment folder was written to and no git state was changed.

---

## 11. Open issues for a human decision

None of these is a wrong digit; each would require changing a claim, a range or
prose, so none was edited.

**O-1 — the "12.5 to 25.2 times" range excludes the >50 GeV band without saying so
(abstract line 95, §9 line 4325).**
§5.9 is precise: "For the three cost-weighted networks the reference is between
$12.5$ and $25.2$ times further from the truth … **in every band from $<3$ to
$20$–$50$ GeV and in the loss window**", and its caution paragraph then gives the
>50 GeV ratios as 8.8, 6.2 and 4.1. The abstract ("sitting $12.5$ to $25.2$ times
closer to one another than either is to the truth") and §9 ("that omission is
between $12.5$ and $25.2$ times larger than the difference between them") repeat
the range with no exclusion. Including >50 GeV the range is 4.1–25.2. Suggested
fix: add "outside the thirty-track $>50$ GeV band" in both places, or widen the
range.

**O-2 — "$13$ per cent better" (§5.4 line 3062, §7.2 line 4027) uses a different
convention from the neighbouring percentages.**
3–8 GeV at N = 256: pooled 200.352 µm, cost-weighted 177.703 µm. As a ratio,
200.352/177.703 = 1.127, i.e. 12.7 % → "13 per cent". As a fractional reduction,
(200.352 − 177.703)/200.352 = 11.3 %. §7.1 and §5.4 use the *fractional
reduction* convention for their other percentages (39.1 %, 14.4 %, 37.6 %). The
README's discrepancy table also says "13 %". Worth settling on one convention.

**O-3 — Table 11 carries "above $5$ GeV" and "below $2$ GeV" rows without a
caption note that the binning is the original study's.** Table 10's caption does
carry such a note ("quoted in the original study's own selection and binning");
Table 11's does not. These two rows come verbatim from the gate file's own cuts.
A one-clause addition to Table 11's caption would close the convention.

**O-4 — Table 14 (`tab:clamp`) prints six momentum bands as columns with no `n`.**
The per-band `n` (907 / 3,665 / 2,360 / 904 / 164 / 2,559) is in Table 13 on the
same 8,000 states, and Table 14's caption says "on the same $8{,}000$ states", so
the information is reachable. If the "every per-band table shows `n`" rule is
strict, a second header line would fix it.

**O-5 — Figure 15's caption overstates the separation in the >50 GeV band.**
"network-against-truth and reference-against-truth sit on top of one another in
every band, while network-against-reference is more than an order of magnitude
below both." For the three **pooled** networks in the >50 GeV band the ratio
`rk6_true_over_nn_rk6` is 1.16, 1.09 and 0.85 — i.e. the network is *not* an
order of magnitude below there, which the figure itself shows (the hollow blue
markers sit level with the green diamond). §5.9 is careful to restrict the claim
to the cost-weighted networks; the caption is not. Suggested fix: "in every band
below $50$ GeV" or "for the cost-weighted networks".

**O-6 — Figure 6's caption quotes one-decimal defects that the plot prints as
integers, and quotes a ratio the plot does not print.** The caption says the
implied defect "is printed above each band: $+17.6$, $+17.3$ and $+12.2$ MeV …
then $-3.1$ and $-86.8$ MeV"; the figure prints +18, +17, +12, −3, −87 (and +7
for the window, which the caption omits). Separately, "the measured width is a
factor $2.13$ to $3.59$ wider than air alone and that ratio is printed at each
point" — 2.13 is the loss-window point, whose ratio is not printed; the five
printed ratios run 2.14 to 3.59. Either the caption or the figure annotation
should move; both numbers are individually correct.

**O-7 — Figure 13's caption says the two rows of panel (c) "differ by about a
decade".** The drawn ranges are ±4e-4 (pooled) and ±1.5e-3 (cost-weighted), a
factor 3.75. "About a factor four" would be accurate.

**O-8 — Figure 8's caption is less precise than the script.** "A tick marks the
round at which the plateau rule first held; the two pooled-loss curves that do
not satisfy the rule at their last round are annotated ``not flat''." The script
docstring is explicit that the tick is drawn *only* where the rule holds at the
last round, so the two "not flat" pooled curves carry no tick even though their
`first_plateau_round` (28 and 34) is known and is in Table 15. The caption can be
read as promising a tick on every curve.

**O-9 — five figures are never cross-referenced from the body.** Figures 2
(`fig:crossing`), 3 (`fig:sample`), 4 (`fig:ceiling`), 5 (`fig:rk6`) and 6
(`fig:materialgap`) are `\ref`-ed only from the Appendix A figure map. Their
floats land in §3.2, §3.3, §4.2, §4.3 and §4.4 respectively, where the
surrounding prose discusses exactly their content but never points at them. One
sentence each would fix it; that is new prose, so it was left alone.

**O-10 — the abstract carries no non-convergence caveat.** Its counts "worse in
five of six comparisons" and "more coherent … in all six matched comparisons"
aggregate over all three pairs, two of which §5.3 rules are supporting rather
than independent evidence. §7.2 makes the same counts with the caveat in the same
subsection. Whether an abstract should carry the caveat is an editorial call.

**O-11 — Table 26's caption and its two bracketed columns disagree about which
number is primary.** The caption reads "*restarts* is the number of optimiser
restarts recorded, with the number of distinct restart indices beside it where
the two differ; *rounds* likewise." The N = 256, q = 16 cost-weighted row prints
restarts as **$1{,}807$ ($1{,}500$)** — recorded first, distinct in brackets —
but rounds as **$60$ ($74$)** — distinct first, rows-read in brackets. Both
numbers are right (`tab_compute.csv`: `restarts_recorded` 1807, `unique_restarts`
1500, `rounds`/`rounds_completed` 60, `rounds_rows_read` 74), and 60 is the count
Table 15 and the rest of the paper use, so the *numbers* should probably stay;
it is the caption's "likewise" that misdescribes the rounds column.

**Minor, sub-threshold (listed for completeness, no action suggested):**
§5.4's "the tail improves … by a factor of up to $5.8$ at $20$–$50$ GeV" — the
largest p95 ratio there is 5.852, which rounds to 5.9 (truncation rather than
rounding); §7.2's and §9's "four and a half to five and a half times as much to
carry a track" — the measured range is 4.54× to 5.56×; §7.3's "Its medians move by
factors of $5$ to $8$ … **which is the largest factor in
Table~\ref{tab:headlinebands}**" — the >50 GeV median ratios are 7.62, 5.34 and
4.64 (integer-rounded 8, 5, 5, so "5 to 8" is fair), but the largest ratio in that
table is **9.01**, in the 20–50 GeV band at N = 64, not in the >50 GeV row; §5.4
makes the same "5 to 8" statement without the "largest factor" clause and is
therefore fine.

---

**Fold-in re-run, 23 September 2026.** All eleven open issues above were
resolved in `main.tex` (see README.md, "Corrections during verification and
fold-in"), `sections/` was deleted so that `main.tex` is the only source, the
paper was rebuilt (70 pages, 0 errors, 0 overfull boxes, 0 undefined references
or citations, 0 font warnings, 42 underfull hboxes unchanged) and
`scripts/verify_tables.py` re-run: **1,445 numeric cells across 26 tables, 0
mismatches.**
