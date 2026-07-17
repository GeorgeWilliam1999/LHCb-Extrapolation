# Data — the extrapolation training sample, characterised

A training set for the extrapolation networks whose **input population comes
from real simulated LHCb events** and whose **labels come from the validated
field-propagation engine**. Built 2026-07-18 from the 100-event minbias sample
generated in [../First_Pass](../First_Pass) (Gauss v61r0p2, 2024 Block-7 beam,
nu = 7.6, DD4hep `run3/2024-v00.02` + `sim10/2024`).

## The argument (for the supervisor)

1. **The population is event-derived, not synthetic.** Every training state is
   the true (x, y, tx, ty) of a charged particle where it crossed a tracker
   sensor in a simulated 2024 bunch crossing (positions and local slopes read
   off the Geant4 MCHits; momentum from truth). Momentum spectra, angles,
   origins, pile-up structure and leg geometry are therefore inherited from the
   official LHCb simulation — the same software and conditions used for
   production MC. The old gen-4 corpus sampled synthetic boxes; this samples
   events. Figure `figures/population.png` overlays the training population on
   the full charged truth: the only bias is detector acceptance itself, shown
   explicitly as coverage vs eta.
2. **The legs are the geometry the extrapolator actually serves** (four types:
   vertex fetches, cross-magnet transfers both directions, plane-to-plane
   steps, downstream-track-to-vertex) — all endpoints taken from each
   particle's own event geometry. `figures/legs.png`; the cross-magnet bend
   scales as 1/p as it must.
3. **The labels are exact field-only propagation, and we can prove both
   halves.** fp64 RK4, 5 mm steps, canonical v8r1.down field (md5 in the meta),
   the identical ODE Allen integrates. Gates (`results/gates.json`,
   `figures/label_gates.png`):
   - G1 forward-then-back closure: median 2.3 nm;
   - G2 label vs the particle's ACTUAL next MCHit: median 3.8 um at p > 5 GeV,
     rising to ~40 um below 2 GeV **along the 1/p multiple-scattering line** —
     i.e. the residual is exactly the material effect that is deliberately out
     of label scope (project stance: material corrections remain with
     TrackMasterExtrapolator; the surrogate replaces only field propagation);
   - G3 step convergence 5 mm vs 1 mm: median 28 nm;
   - G4 q/p passthrough exact.
4. **Honesty flags built in.** The raw harvest includes secondary electrons and
   material-born particles (they cross sensors too). Every row carries
   `ORIGIN_R`, `ETA`, `PID` and a `CORE` flag (non-electron, origin radius
   < 10 mm, 1.8 < eta < 5.2). The track-like core has the expected pi/K/p mix
   and charge balance 0.506; the tails are kept and selectable, never hidden.
5. **No leakage.** The train/val/test split (80/10/10, seed 20260718) is by
   PARTICLE, so no particle's legs appear in two splits. Everything is
   reproducible: Gauss seeds derive from (run, event) numbers; all sampling
   seeds and provenance are in `training_v1/train_mb100_v1.meta.json`.

## Files

| File | Role |
|---|---|
| [harvest_states.py](harvest_states.py) | truth CSVs -> plane-crossing states (`results/states.npz`) |
| [make_training_set.py](make_training_set.py) | states -> legs -> RK4 labels + gates (`training_v1/`) |
| [characterise.py](characterise.py) | figures + summary CSVs (`figures/`, `results/`) |
| [characterisation.ipynb](characterisation.ipynb) | loads the results, displays the figures |
| `truth_mb100/` | truth dump of the 100 events (particles/vertices/hits CSVs) |
| `training_v1/train_mb100_v1.npz` | X[N,7], Y[N,5], LEG, SPLIT, CORE, PID, P, EVT, MCKEY |
| `training_v1/train_mb100_v1.meta.json` | full provenance: sample, field md5, engine, cuts, gates |

Row format (matches the vertex-fit corpus contract): X = (x, y, tx, ty, qop,
z0, z1) with qop = 0.299792458 q / p[GeV]; Y = state at z1, qop passthrough.
Labels computed in fp64, stored fp32.

## Numbers (v1, from 100 events)

- 400,512 charged truth particles -> 88,408 with tracker states -> 604,800
  harvested states (VP 209k / UT 106k / FT 289k).
- **166,802 training rows**: A vertex-fetch 32k · B cross-magnet 22k ·
  C plane-to-plane 101k · D downstream->PV 11k (see `results/legs_summary.csv`).
- 2,520 legs dropped (diverged/out-of-aperture: loopers — the integration
  warnings during generation are these rows being culled).
- G2 note: the ~1% tail clipped at 10 mm in the residual histogram is decays
  in flight and hard hadronic interactions between the two planes — states
  after which the particle genuinely is not where any extrapolator would put it.

## Scaling up / regenerating

The whole chain is one command per step and ~20 min end-to-end for 100 events
(Gauss dominates). For more statistics: `bash ../First_Pass/run_gauss.sh
override_bulk_events.py` with a larger `EvtMax` (or copies with different
`RunNumber` for independent samples), then dump + the three scripts here.
MagUp would need the v8r1.up field map in the label engine and a MagUp
simulation sample (conditions change) — not yet done.
