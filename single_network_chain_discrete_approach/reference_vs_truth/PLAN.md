# Plan: the reference-versus-truth write-up (hand-off, 2026-09-22; George presents it 2026-09-23)

Read first: `../_shared/WRITEUP_RULES.md`; the Notion to-do "Establish how closely the field-only
reference follows the Geant4 truth, and fix the momentum used at the start plane"
(https://app.notion.com/p/3e35d544b9d9812b804ac9e34409711d): its Notes and body hold the finding,
the measured table and the five-step plan, and they are the source for this page; the F2 README
section "The networks against the Geant4-true SciFi state" and `F2_Analysis/against_true_state.py`;
`Data_generation_exploration/Data/harvest_states.py` (how a state is built from an MCHit);
`Data_generation_exploration/Official_xdigi/dump_xdigi.py` (what the dump writes; note it never
writes `MCHit::p()`); `Data_generation_exploration/Official_xdigi/training_v2/train_official_v2.meta.json`
("material_effects": "excluded by design"); `Data_generation_exploration/Data/results/gates.json`
(G2: label vs next hit 12 µm median, 3.8 µm above 5 GeV, on short legs; G3 step convergence 28 nm);
`Block_E_single_network_chain/E0_Track_dataset/README.md` and the D0 README's material-floor table;
`_shared/reference.py` (deriv, rk6_rows; RK6_STEP 0.1 mm; the field file and md5).

**Title:** "What the field-only reference shares with the Geant4 truth, and what it cannot: the
material floor, the stale start momentum, and what the networks are measured against"

**Todos relation:** ["https://app.notion.com/p/3e35d544b9d9812b804ac9e34409711d",
"https://app.notion.com/p/3dd5d544b9d98144bdcbd17c5a66e3e0"]

**Note:** the one-paragraph finding: three references (RK6 field-only, Geant4-true MCHit state,
network); network vs RK6 20–90 µm; RK6 vs truth 1.7 mm median, made of a charge-flipping
systematic (start q/p is the production momentum: 16–18 MeV deficit) and a charge-blind 1/p width
(multiple scattering in the gap, 0.65 mm at 15 GeV); neither is integration error; what is fixable
and what is the floor; the plan.

## First, file the measurement (the numbers must be reproducible before the page is written)
Write `reference_vs_truth/decompose.py` (numpy only; run with
`PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python`), reading
`Block_E_single_network_chain/E0_Track_dataset/results/tracks.npz`, all three splits concatenated
(14,482 rows), and producing `results/decomposition_by_band.csv`, `results/decomposition_by_pid.csv`,
`results/highland_air.csv`, and figures `figures/signed_centre_by_charge.png`,
`figures/width_vs_momentum.png`, `figures/three_references.png` (the last one: copy the drawing of
`Block_F_reweighted_loss/F2_Analysis/against_true_state.py`, or just reuse that figure's path).
Exactly this arithmetic (it reproduces the numbers already in the to-do; check they match):
- d = S_post − truth_zpost (true state minus RK6 carried to the particle's own plane); q = sign(S0[:,4]);
  bend = truth_zpost.x − (S0.x + S0.tx·(z_post − z0)); P from the `_P` arrays; PID from `_PID`.
- Bands (1,2),(2,5),(5,10),(10,25),(25,50),(50,100),(100,200) GeV. Per band and charge: n, median
  dx, median dy, median(dx/bend), 68 % half-width of dx about its median (np.quantile(|dx−med|,0.68)),
  RMS dx, median |dx|, median |dtx|·1e3, median |dy|, median |dty|·1e3; both charges pooled:
  median max(|dx|,|dy|), p95.
- Implied momentum deficit per band: median(dx/bend over |bend| > 20 mm) × median P, in MeV.
- By |PID| ∈ {13, 211, 321, 2212} at 5–25 GeV with |bend| > 20 mm: median(dx/bend), median |dx|.
- Highland, air only: x/X0 = 5.178 m / 304 m; θ0 = 13.6e-3/p·√(x/X0)·(1 + 0.038 ln(x/X0)) rad;
  RMS displacement ≈ θ0·L/√3 for p = 3.4, 6.9, 14.6, 32, 62 GeV (the band medians); and the
  effective x/X0 implied by the measured 68 % width at 10–25 GeV (solve Highland for x/X0).
- Also print the RK6-vs-exact-scheme and step-convergence numbers you cite, read from
  `Data_generation_exploration/Data/results/gates.json` and from
  `Block_E_single_network_chain/E3_Analysis/results/comparators.csv` /
  `Block_F_reweighted_loss/F3_Analysis/results/comparators.csv` (exact scheme rows).
Write `reference_vs_truth/README.md` mapping script → outputs, with the tables.

## Sections
### Intro
Roadmap. The three objects, defined for a reader with no background: (1) the Geant4-true state (an
MC hit: midpoint of entry and exit, slopes from the displacement, momentum attached from the
particle record = production momentum; contains scattering, energy loss, interactions up to that
hit; NO digitisation or resolution; reconstructed states are not used anywhere); (2) the field-only
RK6 reference (the equation of motion, written out, in the v8r1 up map, q/p constant, 0.1 mm
sixth-order steps, from the true last-UT state carried to z0); (3) the network endpoint (trained
label-free on the residual of that same equation; learns exactly what RK6 solves, no more). One
paragraph of history: the July decision "material effects excluded by design (master-extrapolator
responsibility)", the G2 gate on short legs (4 µm above 5 GeV) that made it look safe, and why
the magnet crossing is different (the lever arm: 0.1 mrad is 10 µm over 10 cm and 500 µm over 5 m).

### Aims
Four questions: how far is the network from RK6; how far is RK6 from the truth; what is that gap
made of (integration, field map, momentum, scattering); what can be fixed, what is a floor, and
what should the networks be scored against.

### Method
1. The data pipeline as a numbered chain: official sample → MCHits dumped → states → cut cascade
   (14,482 crossings) → RK6 labels → training states → network; at each link say which physical
   effects enter and which are absent (a table: effect | in the true state? | in RK6? | in the
   network's loss?).
2. How the comparison is made: both the network's and RK6's z1 state carried to the particle's own
   SciFi plane with RK6 (|z_post − z1| < 60 mm, max 7.5 mm in the set) and compared with the true
   state there (`metrics.chain_scores`); the max metric; the radial metric where used.
3. The decomposition: why the sign under charge flip separates a bending systematic from scattering
   (a systematic ∝ q/p flips sign; scattering is charge-blind); dx/bend as the relative over-bend
   and its reading as a momentum deficit (Δx/bend ≈ Δp/p); the Highland formula for the expected
   scattering, with x/X0 for air; what a field-scale error would look like (∝ bend, charge-blind
   in dx/bend, momentum-independent).
4. What RK6's own accuracy is and how it was established (gates.json G1/G3, comparators: exact
   scheme 0.1 µm; own floor from the Block D/E work).

### Results
- Network vs RK6 vs truth per band (the three-reference table from against_true_state.csv; the
  figure `F2_Analysis/figures/against_true_state.png` or your copy). The 0–4 % statement; the
  6–20× statement.
- The decomposition table per band and charge (your CSV): signed centre, dx/bend, implied MeV, width.
  Figures: signed_centre_by_charge.png, width_vs_momentum.png (measured 68 % width vs band with
  the Highland air-only line and the effective-x/X0 line).
- By particle type (heavier = less loss at the same p).
- The high-momentum residual (−0.1 % above 25 GeV, both charges): stated as unexplained,
  candidates named (field map version/scale; Geant4's stepper), not interpreted further.
- Slope residuals per band (the same story in tx, ty).
- What RK6 does agree with: gates G1/G3, the exact scheme, the short-leg G2 numbers, and the
  100–200 GeV band (76 µm median with the stale momentum still in).

### Conclusion and next steps
- The reference is correct for its equation; the equation omits three things: the momentum at the
  start plane (a data-pipeline defect: `MCHit::p()` was never dumped; fixable), energy loss along
  the crossing (deterministic, addable to the equation as the LHCb master extrapolator does), and
  multiple scattering (stochastic; irreducible for any deterministic reference; only its variance
  is predictable). Which of the three "what we want to model" questions the programme answers
  (field-only propagation, expected trajectory with mean loss, full distribution) and what each
  implies for the label-free construction.
- The five-step plan from the to-do, verbatim in substance, with costs; the two decisions for
  George (label with or without energy loss; what the networks are scored against).
- What does not change: every network-vs-RK6 number in the companion write-ups.

### Provenance
tracks.npz (14,482 rows, splits), `reference_vs_truth/decompose.py`, `F2_Analysis/against_true_state.py`
and the record.json files it reads (F1 runs, restarts), gates.json, comparators.csv, harvest_states.py,
dump_xdigi.py, the meta.json; working tree uncommitted, HEAD 1040ee9; figures pinned to main.

## Practicalities
This page is for tomorrow: prefer one page, complete, over a page-plus-children. Save `page.md`,
write `README.md` with the URL and the verification result. Every number from your CSVs or the
named files; none from memory.
