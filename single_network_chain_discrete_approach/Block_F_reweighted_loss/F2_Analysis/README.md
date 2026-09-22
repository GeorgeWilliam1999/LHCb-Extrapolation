# F2 — Block F against Block E, on one rule applied to both

[compare_to_blockE.py](compare_to_blockE.py) holds the judging rule, and it was written and
run **before** any Block F training, on Block E alone, so the rule could not be tuned to a
result. `results/headline.csv` has every column.

- **plateau** — a run has stopped improving once the median validation error of its last
  ten rounds is no more than 5% below the median of the ten before it, for three rounds
  running. A two-sided "within 5%" rule was tried first and thrown away: the round-to-round
  error wanders by 8–23%, so ten-round medians rarely land within 5% of each other even
  when a run has long finished. The question is whether it is still getting better.
- **headline** — the median of the last ten rounds, with the spread, not the final
  checkpoint.
- **test radial at z1** — √(Δx² + Δy²) against RK6 on the 1,452 test tracks, reported
  overall, in the 10–50 GeV band, and below 5 GeV.

## What it says about Block E (2026-09-18)

**None of Block E's sixteen runs had plateaued when training was stopped by hand.** All
eleven with twenty or more rounds were still improving, by 5–24% per ten rounds:

| run | rounds | improvement of the last 10 rounds over the previous 10 |
|---|---|---|
| N = 64, q = 2 | 30 | −18% |
| N = 64, q = 4 | 28 | −12% |
| N = 64, q = 8 | 26 | −15% |
| N = 64, q = 16 | 21 | −24% |
| N = 128, q = 2 | 30 | −12% |
| N = 128, q = 4 | 28 | −5% |
| N = 128, q = 8 | 23 | −6% |
| N = 256, q = 2 | 30 | −5% |
| N = 256, q = 4 | 28 | −9% |
| N = 256, q = 8 | 24 | −19% |
| N = 256, q = 16 | 22 | −16% |

Eleven falls out of eleven has probability 0.0005 if the wandering were pure noise, so this
is a trend and not scatter. Block E's published numbers are where its training stopped, not
where it converges — which is why the three Block E counterparts are being extended to their
own plateau alongside the Block F runs.

The test radial numbers reproduce E3's exactly (N = 64 q = 2: 166.3 µm; N = 256 q = 16:
136.8 µm), which is the check that this script measures the same thing E3 does.

## What it says on 2026-09-20 (two Block F runs finished; not yet discussed with George)

Test radial error at z1 [µm] from `chain_states.npz`, Block F against the Block E run extended to
the same plateau rule (Block E's 2026-09-18 checkpoint in brackets):

| run | all | 10–50 GeV | < 5 GeV | 95th pct | per-step slope [mrad] |
|---|---|---|---|---|---|
| Block F, N = 64, q = 2 | 88.8 | 24.8 | 373 | 1,399 | 0.00073 |
| Block E, N = 64, q = 2 | 145.7 (166.3) | 108.1 (118.0) | 242 (299) | 898 (1,154) | 0.0013 (0.0016) |
| Block F, N = 128, q = 8 | 104.6 | 27.3 | 446 | 1,611 | — |
| Block E, N = 128, q = 8 | 122.2 (158.6) | 70.5 (101.1) | 240 (310) | 928 (1,090) | — |

- **Both pre-registered success criteria hold for N = 64, q = 2:** the test radial error is 39% below
  Block E's plateau (well outside its ±17% band), and the per-step slope error is 0.00073 mrad
  against the 0.0016 threshold (E3's `error_anatomy.py` run on the Block F folder, 2026-09-20; not
  yet a committed result file).
- **The price:** below 5 GeV the error is 1.5 times worse, the 95th percentile 1.5 times worse, and the
  y endpoint error did not move (median |Δy| 55 µm against 52); all the gain is in x and in the band.
  The x-only lever-arm model of `error_anatomy.py` explains only 38% of Block F's endpoint error
  (72% for Block E), so the anatomy needs the y slope before the Block F story is told.
- Block F, N = 256, q = 16 is at round 31 and still falling 16% per ten rounds. Six Block E runs are
  also still falling (N = 64 q = 4; N = 128 q = 8; N = 256 at every q).
- `results/headline.csv` is from 2026-09-19 15:47 and predates the finishes; rerun the script when
  N = 256, q = 16 stops.

```bash
PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python compare_to_blockE.py
```

## Per component, per momentum, and the anatomy with both slopes (2026-09-20 evening)

Two more scripts, both reading the stored chain states of the finished runs and the Block E
counterpart as it stands on disk (extended):

| script | what it does | output |
|---|---|---|
| [components_and_momentum.py](components_and_momentum.py) | median \|Δx\|, \|Δy\|, \|Δtx\|, \|Δty\| at z1 with the signed median beside it, overall and in 10–50 GeV; the radial median and 95th percentile in seven momentum bands; one figure | `results/components.csv`, `results/momentum_bands.csv`, `figures/blockF_vs_blockE.png` |
| [anatomy_xy.py](anatomy_xy.py) | E3's error anatomy redone with **both** slopes: each step's error against RK6 from the state the network was given, then Σ(dx + dtx·lever) and Σ(dy + dty·lever) combined radially; per-step slope errors overall and in the band; coherence | `results/anatomy_xy_<label>.json` |

**N = 64, q = 2, Block F against Block E (extended), 1,452 test tracks.**

| | x [µm] | y [µm] | radial | radial 10–50 GeV |
|---|---|---|---|---|
| Block F, median \|error\| | **37.5** | 55.2 | **88.8** | **24.8** |
| Block E, median \|error\| | 110.7 | 52.3 | 145.7 | 108.1 |
| Block F, signed median (band) | −3.6 (−6.3) | +2.9 (+2.0) | | |
| Block E, signed median (band) | −39.4 (−54.4) | −15.5 (−22.8) | | |

The whole gain is in x and above 5 GeV. Radial median by momentum, Block F / Block E:
2–5 GeV 370 / 241; 5–10 GeV 84 / 122; 10–20 GeV 29 / 85; 20–50 GeV 16 / 145; 50–100 GeV
20 / 136; 100–200 GeV 51 / 315. Below 5 GeV Block F is 1.5 times worse, as the pre-flight
predicted when it moved 68 percentage points of the loss out of that band. Block E's
extended run also carries a signed x offset of −54 µm in the band (the median, not the
mean); Block F's is −6 µm.

**Anatomy with both slopes.** The two-slope lever-arm sum reproduces the endpoint error to
98–99% for all four runs, so the decomposition is complete (E3's x-only version explained
73% of Block E and 38% of Block F, which was the flagged gap).

| N = 64, q = 2 | per-step tx / ty [mrad] | same, 10–50 GeV | x part / y part at z1 [µm] | coherence tx / ty |
|---|---|---|---|---|
| Block F | 0.00073 / 0.00097 | 0.00023 / 0.00027 | 36 / 55 | 0.49 / 0.52 |
| Block E | 0.00130 / 0.00072 | 0.00095 / 0.00040 | 108 / 52 | 0.36 / 0.40 |

Block F's per-step x slope is 1.8 times better overall and 4 times better in the band; its
per-step y slope is 1.5 times better in the band but 1.3 times *worse* overall, because the
soft tracks that dominate the overall number were de-weighted. What remains of Block F's
error is mostly the y term, and it is more coherent along the track than Block E's (0.5
against 0.36–0.40, where 0.125 would be independent steps): smaller per-step errors that
point the same way. That is the shape of the next problem.

**N = 128, q = 8.** Block F 104.6 µm against 122.2 (−14%, only just outside Block E's ±12%
band, and that Block E run is still falling 7% per ten rounds); in the band 27.3 against
70.5 (−61%). Same pattern: x 51 against 82, y 61 against 59.

**Verdict on the pre-registered criteria, N = 64, q = 2:** both hold — 88.8 µm is 39% below
Block E's plateau, far outside its ±17% band, and the per-step x slope is 0.00073 mrad
against the 0.0016 threshold. The price, also pre-registered as "partial" territory, is
below 5 GeV (1.5 times worse) and in the 95th percentile (1,399 against 898 µm).

**Caveat on "Block E to plateau".** The keeper drops a run when the ten-round median has
stopped falling for three rounds; six Block E runs (N = 64 q = 4, N = 128 q = 8, all N = 256)
were dropped on that test and then resumed drifting down, now 3–7% per ten rounds, at
1,600–1,750 restarts. That drift is a few micrometres, an order of magnitude below the
in-band gap, but the Block E numbers are "best after 1,000–1,750 restarts", not a strict
plateau.

## Endpoint error(dz, q) per state component, 5–30 GeV (2026-09-21, George's request)

[error_tables_by_component.py](error_tables_by_component.py) → `results/error_by_component_5-30GeV.csv`,
`figures/error_by_component_5-30GeV.png`. One table per component (x, y, tx, ty), every row one
network, on the 862 test tracks with 5 ≤ p < 30 GeV. **These are endpoint errors**: the network
applied N times from the track's real state on the last UT plane, its state on the first SciFi
plane compared with the RK6 track carried there. Single-step errors (one application from an
RK6 state) are a different quantity, in `../F3_Analysis/results/error_qdz_single_step.csv`.
George (later the same day): these outputs show the networks on their own, named by N and q —
no block labels, no comparison rows. A first version carried a comparison grid; it was removed.

Median |error| (p95 in brackets):

| | N = 64, q = 2 | N = 128, q = 8 | N = 256, q = 16 (checkpoint) |
|---|---|---|---|
| x [µm] | 20.1 (219) | 31.1 (266) | **18.3** (239) |
| y [µm] | **31.5** (264) | 33.3 (407) | 36.5 (285) |
| tx [mrad] | 0.019 (0.178) | 0.022 (0.165) | **0.015** (0.162) |
| ty [mrad] | 0.023 (0.204) | **0.020** (0.188) | 0.025 (0.168) |

Signed medians are all small against the |medians| (the largest is x, −4.8 µm at N = 64 q = 2).
In this band y is 1.5× x, and ty is larger than tx for every network: what remains of the error
is in the y plane.

## Around 5 GeV, signed (2026-09-21, supervisor's question)

[errors_near_5gev.py](errors_near_5gev.py) → `results/errors_near_5gev.csv`,
`figures/signed_errors_4-6GeV.png`. Endpoint deviation per component in 3–5, 4–6, 5–7 and
10–20 GeV: median |d|, RMS, standard deviation, signed mean and median, the 68% half-width
(half the 16th–84th percentile range) and p95. RMS and standard deviation are tail-pulled;
the median and the 68% half-width are not; both are given.

At 4–6 GeV (286 test tracks), N = 64 q = 2: |Δx| median 93 µm, 68% half-width 146 µm, RMS
618 µm; |Δy| 130 / 223 / 425 µm; tx 0.078 / 0.103 / 0.284 mrad; ty 0.095 / 0.161 / 0.329 mrad.
Signed medians −27 µm (x), +28 µm (y): no offset. The RMS is six times the median because
6–7% of the band (17–20 tracks) sits beyond 1 mm radially; those start further from the beam
line (median |x₀| 190–230 mm against 114 mm for the rest), and without them the RMS in x is
180–190 µm. p95 |Δx| is 760–920 µm; 21–24% of the band is beyond 200 µm. At 10–20
GeV the same network is |Δx| 12.5 / 19 / 94 µm. The loss window is 10–50 GeV; if ~5 GeV is
the priority, `P_LO`/`P_HI` in `F0_Weighting/weighted_loss.py` is the knob.

## The networks against the Geant4-true SciFi state (2026-09-22, supervisor's question)

[against_true_state.py](against_true_state.py) → `results/against_true_state.csv`,
`figures/against_true_state.png`. Three references for a crossing: the field-only **RK6**
endpoint (what the networks are trained towards and scored against everywhere else), the
particle's **Geant4-true** state on its own first SciFi plane (the MCHit: multiple scattering
and energy loss included, no detector resolution), and the **network** endpoint. The numbers
are the ones every run's `record.json` already carries (`metrics.chain_scores`: network and
RK6 states each carried from z1 to the particle's own plane with RK6, then compared with the
true state); this script only tabulates them per momentum band. Position error is
max(|dx|, |dy|), median over the 1,452 test tracks [µm]:

| band | n | network vs RK6 (N = 64 / 128 / 256) | network vs true | RK6 vs true |
|---|---|---|---|---|
| all | 1,452 | 81 / 96 / 84 | 1,698 / 1,650 / 1,699 | 1,694 |
| 2–5 GeV | 500 | 335 / 383 / 316 | 5,217 / 5,251 / 5,130 | 5,219 |
| 5–10 GeV | 430 | 77 / 97 / 79 | 1,686 / 1,675 / 1,688 | 1,630 |
| 10–25 GeV | 383 | 24 / 28 / 36 | 623 / 628 / 610 | 614 |
| 25–200 GeV | 134 | 17 / 20 / 23 | 276 / 270 / 274 | 255 |

Against the true state the networks and RK6 are indistinguishable at the level of the
band-to-band scatter: the network-vs-true median sits within −2.6 % to +8.4 % of the RK6-vs-true
median in every band with more than five tracks, while the network-vs-RK6 error is 11 to 25
times smaller than either (both ranges recomputed from `results/against_true_state.csv` on
2026-09-22 evening in `../../reference_vs_truth/decompose.py` → `results/three_references.csv`;
the first version of this paragraph said "0–4 %" and "6–20 times", which was wrong). The gap to the true state is the material the particle crosses
(multiple scattering scales as 1/p: 5.2 mm at 2–5 GeV, 0.6 mm at 10–25 GeV), which no
field-only method — RK6 included — can reproduce, and which the networks were never asked
to learn. Reconstructed (digitised, pattern-recognised) states are not in this pipeline at
all; every "true" state is an MCHit.

**Units of the slope errors.** tx = dx/dz and ty = dy/dz are dimensionless slopes (the
MCHit's exit − entry displacement divided by its dz, `Data/harvest_states.py`); no arctan is
applied anywhere. Every "mrad" in these tables and figures is a slope difference × 10³. For
the accepted tracks (2 < η < 5, |tx| ≲ 0.27) the angle difference is Δθ ≈ Δtx/(1 + tx²),
so the numbers read as milliradians to within 7% at the edge of the acceptance and within
1% for most tracks, but the label should be "×10⁻³ (slope)". To be corrected across the
E3/F2/F3 scripts.
