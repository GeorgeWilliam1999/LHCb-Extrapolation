# Self_chained_paper/results — the paper's numbers

Everything here is written by [`../scripts/numbers.py`](../scripts/numbers.py), which imports
[`../scripts/common.py`](../scripts/common.py) for the paper's conventions, loaders and
statistics.  One command rebuilds the whole folder:

```bash
cd /data/bfys/gscriven/LHCb_Extrapolation_Project/Self_chained_paper/scripts
PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python numbers.py
```

About four minutes cold, fifteen seconds once `cache/` exists (`--no-cache` forces a rebuild).
Every experiment folder is read-only to this script; nothing outside `scripts/` and `results/`
is written.

## Conventions (fixed in `common.py`, differ from the existing analysis code)

- **Slopes are dimensionless.**  `t_x = dx/dz`, `t_y = dy/dz`.  Every slope error here is the
  raw difference (e.g. `7.34e-07`); it is never multiplied by 10³ and no key or column is
  called "mrad".  `E1_Network_grid/metrics.py` and the comparison tuples in `E3`/`F2` all
  scale slopes by 10³ and label them mrad; their slope functions are deliberately not imported.
- **Positions are in micrometres** (mm × 10³).
- **Momentum bands**: edges `[0, 3, 8, 20, 50, inf)` GeV labelled `<3`, `3-8`, `8-20`,
  `20-50`, `>50`, plus `10-50 (loss window)` reported *alongside* them (it overlaps two of
  the partition's bands on purpose).  Membership always comes from the per-split `P` array of
  `tracks.npz`, never from the stored `PBAND` column, whose edges are the old
  1/2/5/10/25/200 GeV set.
- **Splits**: train 11,567, validation 1,463, test 1,452.
- **Reference**: the field-only RK6 truth states stored in `tracks.npz` as
  `<split>_truth` (n, 257, 5) on the grid `z0 + k L/256`.  An N-step chain's step k starts on
  plane index `k * (256 // N)`; the endpoint is plane 256, `z1 = 7826.0 mm`.
- **Statistics**: median |·|, p95 of |·|, signed median, 68 % half-width `(q84 − q16)/2`,
  RMS, radial `hypot(dx, dy)`, max-metric `max(|dx|, |dy|)`.  Every key names its metric.
- **Round counting**: `rounds` everywhere in this folder is the number of **completed**
  training rounds — the number of de-duplicated rows in the run's `rounds.csv`.
  `rounds_counter_record_json` is the trainer's round *counter* at the moment the run
  stopped (`record.json["rounds"]`, identical to `progress.json["round"]`), which counts
  the round that was **in progress** and therefore has no row in `rounds.csv`.  Wherever a
  run stopped mid-round the counter is exactly one higher than the completed count: all
  twelve N ≥ 64 snapshot runs (e.g. N = 64, q = 2: 30 completed, counter 31, `in_round`
  10 of 25), while the four N = 2 runs stopped on a round boundary and the two agree.  The
  `rounds` column of `E3_Analysis/results/error_qdz_chain.csv` is copied from `record.json`
  and is therefore the counter, which is why it reads 31/24/23 where `rounds.csv` has
  30/23/22.  In the **extended** pooled runs the round that was in progress on 18 September
  was abandoned when the run was reopened, so its label is simply missing from `rounds.csv`
  (labels 31, 24 and 23 respectively) and `max(label) = completed + 1` there too.  The
  plateau rule and the validation headline are read from `rounds.csv`, so they use the
  completed count.
- **De-duplication**: the `N = 256, q = 16` reweighted run was trained by two farm jobs at
  once from restart 1,198 on, so its `history.csv` holds 302 repeated restart numbers and its
  `rounds.csv` 13 repeated round numbers.  `common.dedup` keeps the **last row written** per
  number, the rule of `Block_G_low_momentum_window/G2_Analysis/convergence.py`, and every
  reader of a `rounds.csv` or `history.csv` goes through it.

## Files

| file | what it holds |
|---|---|
| `paper_numbers.json` | every number the paper quotes, grouped by section, each group carrying a `source` string naming the files it was computed from.  Top-level keys: `conventions`, `palette`, `dataset`, `scheme`, `reference_vs_truth`, `pooled_grid_snapshot`, `pooled_extended`, `reweighted`, `headline`, `near_5gev`, `anatomy`, `along_z`, `preflight`, `constants`, `worked_example`, `against_true_state`, `compute`, `tables`, `timing`. |
| `cache/*.npz` | intermediate arrays so a rerun is cheap: `anatomy_<loss>_N<NNN>_q<qq>.npz` (per-step local errors), `single_step_<loss>_N<NNN>_q<qq>.npz`, `carried_<loss>_N<NNN>_q<qq>.npz` (each network's endpoint carried to the particle's own SciFi plane).  Delete the folder or pass `--no-cache` to rebuild. |

### The tables

| table | what it holds | sources |
|---|---|---|
| `tab_dataset_bands.csv` | per split, per band: n, fraction of the split, median momentum | `Block_E_single_network_chain/E0_Track_dataset/results/tracks.npz` |
| `tab_dataset_cuts.csv` | the cut cascade that made the track set: rows in, removed, out, particles out, and the note on each cut | `E0_Track_dataset/results/tracks_meta.json`, `E0_Track_dataset/build_tracks.log` |
| `tab_material_floor.csv` | the material floor recomputed in the paper's bands: the particle's real Geant4 state on its own first SciFi plane against the field-only RK6 reference carried there, max-metric and radial medians, p95, per component.  Two scopes: all three splits concatenated (14,482 crossings, as `reference_vs_truth/decompose.py` does) and test only | `tracks.npz` |
| `tab_tableau.csv` | the Gauss–Legendre tableau verification for q = 1, 2, 3, 4, 8, 12, 16: row sums, quadrature, collocation, symplecticity, agreement with the closed forms, and the worst residual per q | `_shared/irk.py` (`verify_tableau`, `gauss_legendre`) |
| `tab_exact_ceiling.csv` | the exact collocation scheme chained at every (N, q): the radial endpoint median from `error_qdz_chain.csv` and, where the run exists, the max-metric median/p95 and per-component medians from `exact_N*_q*.json` (slopes divided back to dimensionless).  Only N = 2 and N = 256 have their own JSON | `E2_Comparators/results/exact_*.json`, `E3_Analysis/results/error_qdz_chain.csv` |
| `tab_rk6_convergence.csv` | the RK6 step ladder on the real map: per rung, the median/p95/max of \|S(h) − S(h/2)\| and of \|S(h) − S(0.05)\| over 200 momentum-stratified legs | `Block_C_step_size_and_stages/C1_Fine_reference/results/reference_convergence.csv` |
| `tab_decomposition_bands.csv` | true-minus-reference decomposition re-binned into the paper's bands, per charge and pooled: signed and \|·\| medians of dx, dy, dtx, dty (slopes native), dx/bend, the implied momentum defect, the 68 % half-width (plain and charge-corrected), the max metric | `tracks.npz`, method from `reference_vs_truth/decompose.py` |
| `tab_highland_air.csv` | the air-only Highland prediction per band beside the measured charge-corrected 68 % half-width, their ratio and the effective x/X₀ the width implies | `tracks.npz`, `reference_vs_truth/decompose.py` |
| `tab_pooled_grid.csv` | the sixteen pooled-loss networks at the 18 September checkpoint: endpoint radial median/p95/p99, fraction beyond 1 mm, per-component medians and signed medians, the exact ceiling, the single-step median, restarts, rounds, cost per track, and the reproduction of E3's `med_um` | `E1_Network_grid/results/N*/stopped_2026-09-18/chain_states.npz`, `tracks.npz`, `E3_Analysis/results/error_qdz_chain.csv`, `error_qdz_single_step.csv` |
| `tab_extended_vs_snapshot.csv` | the three extended pooled-loss runs as the run folders now hold them, against `headline.csv`, and how many extra rounds and restarts each got beyond its snapshot | `E1_Network_grid/results/N*/`, `F2_Analysis/results/headline.csv`, `tracks.npz` |
| `tab_reweighted_runs.csv` | the three reweighted-loss networks: endpoint radial median/p95/p99, fraction beyond 1 mm, the 10–50 GeV median, per-component medians, beside `headline.csv` | `F1_Training/results/full/N*/chain_states.npz`, `F2_Analysis/results/headline.csv`, `tracks.npz` |
| `tab_headline_pairs.csv` | one row per run of the three matched pairs: radial median, p95, fraction and count beyond 1 mm, the in-band median, the validation headline with its spread, the plateau verdict and first-hold round, rounds, restarts, training wall, median wall per restart | `E1_Network_grid/results/N*/`, `F1_Training/results/full/N*/`, `tracks.npz` |
| `tab_headline_bands.csv` | the same six runs, radial median and p95 and fraction beyond 1 mm in each of the five bands plus the loss window, with n | as above |
| `tab_headline_components.csv` | the same six runs, per component (x, y in µm; tx, ty dimensionless), overall and per band: median \|·\|, p95 \|·\|, signed median, 68 % half-width, RMS | as above |
| `tab_near_5gev.csv` | the N = 64, q = 2 network under both losses in 3–8 GeV and 4–6 GeV: per-component signed statistics, the radial median, how many tracks miss by more than 1 mm, and the median starting \|x\| of those tracks against the rest | `E1_Network_grid/results/N064_q02/`, `F1_Training/results/full/N064_q02/`, `tracks.npz` |
| `tab_anatomy.csv` | the error anatomy of all six runs with dimensionless slopes: per-step \|dx\|, \|dy\|, \|dtx\|, \|dty\| medians overall and in 3–8, 8–20 and 10–50 GeV; the x and y parts at z1; the lever-arm model's prediction with both slopes, with the x slope only and with positions only; the explained fractions; coherence in tx and ty with the independent-step value | `E1_Network_grid/results/N*/`, `F1_Training/results/full/N*/`, `_shared/reference.py` (`rk6_rows`), `E1_Network_grid/chain_network.py`, `tracks.npz` |
| `tab_along_z.csv` | the chain's radial median, p95 and in-band median at every plane of the crossing, for all six runs | the runs' `chain_states.npz`, `tracks.npz` |
| `tab_single_step_vs_z.csv` | the single-step radial median plane by plane: each network applied once from the RK6 truth state on every start plane | the runs' `network.pt`/`scale.json` through `chain_network.load_network`, `tracks.npz` |
| `tab_preflight_shares.csv` | the share of the loss each band and each quarter of z carries under the five weighting modes (`blockE`, `full`, `no_lever`, `no_track`, `no_window`), on 8,000 states drawn evenly over the planes from the extended pooled N = 64, q = 2 network | `F0_Weighting/check_weights.py` (method), `F0_Weighting/weighted_loss.py`, `E1_Network_grid/results/N064_q02/`, `tracks.npz` |
| `tab_preflight_snapshot_record.csv` | the pre-flight **as F0 recorded it on 18 September 2026**, verbatim: per mode, the share of the loss in each of the *original* bands `[1, 2, 5, 10, 20, 50, 100, 200]` GeV, the quarter-of-z shares, the summary quantities (2–5 GeV, 10–50 GeV and >50 GeV shares, the two rank correlations, the top-1 % concentration) and the clamp scan.  Long format: `mode, kind, key, lo, hi, n, share_pct, states_pct`, with `kind` in `momentum` / `quarter_of_z` / `summary` / `clamp_scan`.  **This is the table the write-up's §3.5 and §5.2 footnote quote (57.82 % for the new loss in 10–50 GeV against 1.509 % for the pooled loss)**; it is measured on the 18 September checkpoint of the N = 64, q = 2 network, not on the extended one, and is therefore not the same thing as `tab_preflight_shares.csv` | `F0_Weighting/results/check_weights.json`, `F0_Weighting/results/preflight_shares.csv` |
| `tab_clamp_scan.csv` | the clamp scan (off, 2, 3, 5, 10) in the paper's bands, with the top-1 % concentration and the rank correlation of share with true cost, overall and in band | as above |
| `tab_loss_constants.csv` | per reweighted run: dz, D_ref, lev_ref, ī, the mode, the clamp, the window edges, the roll-off, the floor, and the lever-arm range | `F0_Weighting/weighted_loss.py`, `F1_Training/results/full/N*/scale.json` |
| `tab_worked_example.csv` | the write-up's §3.2 worked example recomputed from `weighted_loss.py` itself: a 3 GeV and a 20 GeV track at the first and last output planes of the N = 64, q = 2 chain, with the track bend, the window, the unclamped a_n, the lever arm and the weight on a position and on a slope residual | `F0_Weighting/weighted_loss.py`, `F1_Training/results/full/N064_q02/scale.json` |
| `tab_against_true_state.csv` | the three references re-binned into the paper's bands for all six runs: network against RK6, network against the Geant4-true SciFi state (the endpoint carried there with RK6 at 0.1 mm), and RK6 against that true state — position max-metric median and p95, slope median (dimensionless), and the two ratios | the runs' `chain_states.npz`, `tracks.npz`, `_shared/reference.py`, checked against `F2_Analysis/results/against_true_state.csv` |
| `tab_compute.csv` | per run: parameters, states per round, restarts recorded and unique, duplicated rows, rounds, median and mean wall per restart, summed history wall, recorded training wall, median iterations per restart, final loss, µs per track, and which cap ended it | the runs' `record.json`, `history.csv`, `rounds.csv` |

## Consistency checks carried in the JSON

| key | what it checks | result |
|---|---|---|
| `pooled_grid_snapshot.reproduction_check` | the recomputed radial median of all sixteen snapshot runs against `E3_Analysis/results/error_qdz_chain.csv` `med_um` | worst deviation 3.3e-14 %, all sixteen within 0.5 % |
| `pooled_extended.reproduction_check` | the three extended pooled runs against `F2_Analysis/results/headline.csv` (145.7 / 122.2 / 147.7 µm) | exact to floating point |
| `reweighted.reproduction_check` | the three reweighted runs against `headline.csv`, overall and in 10–50 GeV (88.8 / 104.6 / 92.2 and 24.8 / 27.3 / 34.6 µm) | exact to floating point |
| `anatomy.check_against_f2` | fifty-four quantities against `F2_Analysis/results/anatomy_xy_*.json` (whose slopes are stored ×10³ and are divided back) | worst deviation 3.5e-14 % |
| `against_true_state.check_against_f2_csv` | the "all" rows of the three reweighted runs against `against_true_state.csv`, positions and slopes | exact |
| `pooled_grid_snapshot.round_counting` | for all sixteen snapshot runs: completed rounds, the record's counter, E3's column, the label range and any missing label | the record always matches E3's column; the counter is completed + 1 for every N ≥ 64 run and equal to it for the four N = 2 runs |
| `preflight.reproduced_with_the_18_Sept_snapshot` | the whole pre-flight re-run on `results/N064_q02/stopped_2026-09-18/`, the checkpoint `check_weights.json` was written from | exact, which is the proof that the replayed random draw and the method match the original gate |
| `along_z.per_step_vs_single_step_check` | whether the per-step radial error of `tab_anatomy.csv` (each step from the **chain's own** state) and the single-step radial error of `tab_single_step_vs_z.csv` (each step from the **RK6 truth** state on that plane) agreeing to three digits is genuine or a bug | **genuine.**  On plane 0 the two start states are the same array and the two paths agree to 8.6e-09 – 3.8e-07 µm, which is not a start-state effect but the gap between the two *references* (RK6 at a 1 mm local step in the anatomy against the stored truth track marched at 0.1 mm) and sits at the reference's own step-size floor from Block C, C1.  On later planes the inputs genuinely differ — the accumulated chain error reaches 84–146 µm by the last plane — yet the two arrays stay correlated 0.9992–0.9999 element-wise with a median relative difference of 2.0–4.3 × 10⁻⁴.  Dividing the median drift (25–50 µm) by that gives a variation scale of 100–200 mm for the *error field*, the same order as the network's own input normalisation (`in_scale_x` = 446–452 mm), so a displacement of 0.03–0.15 mm moves the one-step error by a few parts in 10⁴ and cannot change the third digit of the median |
| `preflight.check_against_check_weights_json` | the same quantities with the **extended** network, which is what the paper uses | 0.2–8.5 % apart, as expected: the weights are identical but the residuals are not, because that run has been trained further since `check_weights.json` was written |

## Two quantities worth knowing where to find

- **`reference_vs_truth.rebinned_decomposition.median_abs_bend_mm` = 453.30 mm** — the median
  **magnitude** of the bend over the 14,482 crossings, which is the number
  `reference_vs_truth/page.md` quotes.  The *signed* median in the same block
  (`median_signed_bend_mm` = −70.05 mm) is near zero because the two charges bend opposite
  ways, and it is not the same quantity.  Per band the magnitude runs 1,243.6 mm (<3 GeV),
  639.9 (3–8), 252.0 (8–20), 109.6 (20–50), 45.5 (>50) and 182.7 mm in the loss window;
  14,476 of the 14,482 crossings clear the 20 mm cut that `decompose.py` applies before it
  forms `dx/bend`.  The bend is defined exactly as in `decompose.py`: the reference
  endpoint's x at the particle's own first SciFi plane minus a straight line through the
  start state.
- **`preflight.snapshot_18Sept_record`** — the committed 18 September pre-flight, see
  `tab_preflight_snapshot_record.csv` above.  `preflight.modes` beside it is the *paper's*
  pre-flight: the same method, the paper's band edges, and the **extended** network.  The two
  are different measurements of the same weighting and should not be mixed in one sentence.
