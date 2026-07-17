# First_reco — simulating the reconstruction pipeline on our own events

Goal: take the full simulated events from `../` (First_Pass: Gauss v61r0p2,
2024 conditions, DD4hep) through the **real LHCb reconstruction pipeline**,
entirely locally:

```
GaussDev5-...sim  (MC truth: MCParticles + MCHits)          [done in First_Pass]
   |
   v  Boole v48r0        digitisation: MCHits -> raw banks + truth-link tables
GaussDev5.digi
   |
   v  Moore v59r4        HLT2 light reco: VELO/UT/FT decoding -> pattern reco
reconstructed tracks     -> long tracks -> PrKalmanFilter, + PrChecker vs truth
```

Everything runs from CVMFS releases on this node — **no grid, no eos, no
kerberos**. (This is the same blocker that stalled the Moore KS0 validation of
the vertex-fit surrogate: that job's *input* lived on eos. Locally generated
.digi files remove the blocker entirely.)

## Version pairing (verified from the CVMFS manifests)

| Step | Release | Built against | Platform |
|---|---|---|---|
| Gauss | v61r0p2 | Gaudi 40.3 · Detector 3.10 · LHCb **59.0** | x86_64_v3-el9-gcc13-opt |
| Boole | v48r0 | Gaudi 40.3 · Detector 3.10 · LHCb **59.0** (exact match) | x86_64_v3-el9-gcc13-opt |
| Moore | v59r4 | Gaudi 40.4 · Detector 3.15 · LHCb **59.5** · Rec 40.4 · Allen 8.4 | x86_64_v3-el9-gcc15-opt |

Boole is the *exact* stack twin of our Gauss. Moore is five patch-levels of
LHCb ahead — legal, because the `.digi` boundary (raw banks + packed MC) is the
designed cross-release interface. Geometry/conditions are pinned in every job
to what Gauss simulated: `run3/2024-v00.02` + `sim10/2024` (Boole's *default*
would be `sim10/run3-ideal` — overridden deliberately).

## Files

| File | Role |
|---|---|
| [boole_digitise.py](boole_digitise.py) | Boole options: .sim -> .digi, Rich dropped (no Rich hits in our .sim), truth links kept |
| [run_boole.sh](run_boole.sh) | runs Boole -> `reco_output/GaussDev5.digi` + boole.log |
| [moore_hlt2_reco.py](moore_hlt2_reco.py) | Moore options: HLT2 light reco (tracking + PrKalman) + MC checking on the local .digi |
| [run_moore.sh](run_moore.sh) | runs Moore -> ntuple + histos + moore.log |

## What the .digi contains (from the DigiWriter item list, dry-run verified)

- `/Event/DAQ/RawEvent` — the raw banks (VP retina clusters, UT, FT, Calo,
  Muon): byte-identical in format to what the detector DAQ would ship. This is
  what HLT1/HLT2 decode.
- `pSim/MCParticles`, `pSim/MCVertices` — the truth, carried along.
- `Link/Raw/VP/Digits2MCHits`, `UT/Clusters2MCHits`, `FT/LiteClusters2MCHits`,
  `Ecal|Hcal/Digits2MCParticles`, `MC/TrackInfo` — the **truth-link tables**:
  which digitised signal came from which MC hit/particle. PrChecker consumes
  these to compute track-finding efficiencies, ghost rates and hit purities.

## Plan

**Phase 1 — prove the chain (this folder).**
Boole: ✅ done (5/5 events, all detectors digitised, links written).
Moore HLT2 light reco + MC checking: staged, see log. Deliverable: the
PrChecker efficiency/ghost tables and the reco ntuple on our own 5 events —
"the detector saw our simulated crossing and found its tracks."

**Phase 2 — HLT1 on the same file. ✅ DONE (2026-07-17).** Allen forward
tracking + truth checker ran in Gaudi (CPU) on the 5-event Extended .digi via
[run_hlt1_allen.sh](run_hlt1_allen.sh) + [moore_input_local.py](moore_input_local.py)
(the latter stacks in front of ANY Moore/RecoConf option file to point it at
local data). Result vs HLT2 on identical events, long-track category:
HLT1 46.9 % overall / 68.9 % (p>5 GeV) / 100 % (p>3 GeV, pT>0.5 GeV), 0 ghosts,
purity ~84 %; HLT2 87.4 % / 93.2 %, 4.8 % ghosts, purity 99.4 %. The textbook
HLT1-vs-HLT2 trade, measured on our own events: the fast trigger pass is
high-pT-biased (100 % in its design region), HLT2 is the full-quality pass.
Trigger *decisions* (hlt1_pp_default lines) could run the same way; rates are
meaningless at this sample size.

**Phase 3 — statistics and signal.** Scale minbias (Gauss is ~4 min/event/th
on this node, 28 cores; hundreds of events overnight), and/or generate a
signal sample by swapping the DecFiles event type in `../gauss_one_event.py`
(e.g. a KS-rich or B-decay event type) for vertexing-oriented studies.
Spillover remains off (not implemented in Gauss v61 2024 options).

**Phase 4 — connect to the extrapolator project.**
- The Moore KS0 comparison job for the vertex-fit surrogate
  (`hlt2_ks_nn_vs_rk.py`, staged in TE_stack, currently blocked on an eos
  kerberos ticket) can instead point `options.input_files` at locally
  generated .digi — unblocking the NN-vs-RK vertex-resolution validation
  without any CERN credential.
- The extrapolator benchmarks in TE_stack (which anchor on an eos MinBias
  .digi) gain a local, regenerable, version-matched input.
- Reco output gives real Kalman states at measurement planes matched to truth
  — validation inputs for the discrete-time (Raissi) extrapolator programme.

## Status of Phase 1 (2026-07-17): ✅ COMPLETE — full chain proven

- Boole: ✅ 5/5 events digitised.
- Moore `standalone_hlt2_light_reco` (tracking + PrKalman): ✅ 5/5 events,
  ~50 long tracks per busy event; empty elastic event handled cleanly.
- Moore **+ MC checking**: ✅ 5/5 events after the root-cause fix below.
  Headline numbers (5 local minbias events, `variants/noskip_extended/`):
  long-track efficiency **87.4 %** (93.2 % for p > 5 GeV), purity 99.4 %,
  hit efficiency 97.8 %, ghost rate 4.8 %; VELO 733 tracks at 1.8 % ghosts;
  downstream/seed/match tables all produced. Ntuple + histograms written.

### Root cause of the initial MC-checking segfault (bisected 2026-07-17)

A **Default-DigiType .digi is self-inconsistent for truth checking**: it
stores the `Link/Raw/*2MCHits` link tables but NOT the `pSim/<det>/Hits`
containers they point to. Moore's truth checkers dereference the dangling
links and segfault on the first event — regardless of event content
(reproduced with the empty event skipped), monitoring flags, or Moore version
(v59r4 and v59r0 crash identically; version skew exonerated). Bare tracking
and data monitoring never touch the links and always passed.

**Fix: `Boole().DigiType = "Extended"`** — keeps the six pSim MCHits
containers (+0.6 MB for 5 events); with them present all checkers run on all
events, including the empty elastic crossing. Applied to
[boole_digitise.py](boole_digitise.py); output becomes
`GaussDev5-Extended.digi`. Evidence: `variants/{tracking_only,moni_only,
mccheck_only,older_moore,skip_default,skip_extended,noskip_extended}/`
(each holds options, logs, and outputs). Arguably a Moore fragility worth
reporting upstream (a missing container should error, not segfault).

- Benign known warning on every PrKalmanFilter/TrackChecker instance:
  "TransportSvc is currently incompatible with DD4HEP ... disabling material
  corrections" (Rec issue #326) — expected for 2024 DD4hep reco, not an error.

## Known caveats

- **No RICH**: Gauss v61 DD4hep wrote no Rich hits, so no RICH banks, so no
  hadron PID downstream. Fine for tracking/extrapolation work; revisit Gauss
  Rich config if PID is ever needed.
- **Version skew** (.digi written by LHCb 59.0, read by 59.5): designed to
  work; watch for packing-version warnings in the Moore log.
- **First Moore run is slow** (functor compilation); subsequent runs reuse the
  cache.
- 2024 beam conditions => nu = 7.6 pile-up: busy, realistic events; event 1 of
  the 5 (index 0) is the near-empty elastic crossing and reconstructs nothing.
