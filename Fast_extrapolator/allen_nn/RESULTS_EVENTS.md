# Allen on events: the magnet crossing of PrKF with the network, against the ARKN

2026-11-27. Three runs of Allen (CPU, inside Gaudi) on the same 15,548 events, the same order, one thread each:

| run | magnet crossing UT -> T (second fit pass) | weights |
|---|---|---|
| `off` | ARKN (MR !2407 as it is) | none |
| `on_nofb` | network K = 12 2x64 everywhere, no step-error fallback (states outside the training box still go to the ARKN) | `weights/linear_nystrom_12_2x64_nofb.json` |
| `on_rule48` | network, except 48 plain Nystrom steps when abs(y) of the straight line at z = 5 m is above 700 mm or the state is outside the box; the ARKN is never called | `weights/linear_nystrom_12_2x64_rule48.json` |

One seed of the network; differences under 2x are not claimed.

## Setup

- **Allen**: branch `allen_nn_magnet_crossing` = MR !2407 (`2f8e80a52`) + `allen_patch/0001-allen-nn-magnet-crossing.patch`
  + `allen_patch/0002-corner-rule.diff`, CPU build against the nightly `lhcb-master/3237` (Gaudi v41r0, Detector, LHCb,
  Lbcom, Rec from that slot; `x86_64_v3-el9-gcc15-opt`; `allen_patch/physics/build_allen_3237.sh`), in
  `/data/bfys/gscriven/allen_physics/Allen`. The build of the night before (nightly 3236) mixed nightly LHCb with the
  released Rec v41r3, because 3236 has no Rec build, and does not run.
- **ParamFiles**: MR !2407 reads `ParametrizedKalmanFit/ParametrisedScatters.json`, only on the ParamFiles branch
  `pehoffma_HLT1_KF_ARKN` (`cf69c93`); a read-only clone is used through `PARAMFILESROOT`.
- **Events**: `2026_expected_MinBias_Sim10g_digi` (TestFileDB), file `00360851_00000002_1.digi` (5.1 GB, copied from EOS to
  `/data/bfys/gscriven/allen_samples/`), all events, MC truth.
- **Field**: the simulation (SIMCOND `upgrade/sim10-2026-v00.00-mu100`) used `field.v8r1` up, polarity +1, the map the
  network was trained on. The DD4hep tag of the TestFileDB entry (`sim10/run3-ideal`, marked "speculative") names
  `field.v6r1` and polarity -1. Both runs use condition overrides: `FieldMapFilesUp/Down` = v8r1 and `Magnet` =
  {Current 5850, Polarity 1}; the log confirms `field.v8r1.up.bin`.
- **Configuration**: `allen_patch/physics/prkf_physics.py`: the reconstruction of the sequence
  `hlt1_pp_forward_then_matching_and_downstream_with_fullkf_1300kHz_rich` (forward then matching, PrKF = `make_kalman_long`
  version 2, outlier chi2 9, two outlier iterations), no lines; Allen tracks converted to LHCb tracks
  (`ConvertAllenLongToV3Tracks_Rich`, `fromV3TrackV1Track`) and checked with Rec: `TrackIPResolutionCheckerNT` (one row a
  fitted long track: true p, fitted q/p, chi2, ndof, MC link) and `TrackResChecker` (RICH2 states against MC truth).
  `events_per_slice: 100` (one slice of all events runs out of the memory pool, and its retry fails on a raw bank).
- **Jobs**: HTCondor cluster 124135 (1 CPU, 8 GB): 30, 30 and 20 min on three different nodes (the times are not a
  speed comparison).
- **Analysis**: `python/allen_events.py RUNS off on_nofb|on_rule48` -> `results/events_*_<run>.*`, `figures/events_*_<run>.png`.
  The per-track tables (66 MB each) are not in git: `/data/bfys/gscriven/allen_physics/runs/full/events_tracks_<run>.csv`;
  the Allen outputs (ntuple.root, histos.root, run.log) are in `/data/bfys/gscriven/allen_physics/runs/full/<run>/`.

## What the test can see (read before the numbers)

In PrKF (`PrKalmanFilter.cu`, `full_fit`) the fitted state at the beamline, with its q/p, is
`x_C2B_averaged`, taken at the end of the FIRST pass (T -> UT -> VELO, line 408). The magnet crossing of that pass
(T -> UT, dz < 0) still uses the ARKN. The network replaces only the crossing of the SECOND pass (VELO -> UT -> T,
dz > 0). That crossing feeds the chi2 of the second pass, the outlier removal and the decision of fit success, and the
states after the magnet (the averaged EndT state and the RICH2 states). So the momentum resolution at the beamline can
change only through outlier removal. A real momentum test needs the T -> UT direction too (a reverse network or a
Newton inversion of the forward tool).

## Results

810,892 fitted long tracks in 15,466 events with tracks; the same tracks, events and order in all three runs. 18 fits
have a NaN q/p and chi2, the same 18 in all three runs; they are left out.

### Tracks, chi2/ndf, momentum (all fitted long tracks; dp/p of real tracks at the beamline state)

| run | tracks | real | ghosts | chi2/ndf real median / mean | chi2/ndf ghost median / mean | dp/p bias | dp/p core sigma |
|---|---|---|---|---|---|---|---|
| off (ARKN) | 810,874 | 754,917 | 55,957 | 1.2901 / 2.2209 | 7.8076 / 15.2290 | -0.048 % | 0.56 % |
| on_nofb | 810,874 | 754,917 | 55,957 | 1.2903 / 2.2210 | 7.8017 / 15.2481 | -0.048 % | 0.56 % |
| on_rule48 | 810,874 | 754,917 | 55,957 | 1.2903 / 2.2209 | 7.7992 / 15.2486 | -0.048 % | 0.56 % |

Core sigma = half the 15.9-84.1 % range of dp/p.

### Momentum resolution and bias of real tracks by true p (%)

| p (GeV) | tracks | sigma off | sigma on_nofb | sigma on_rule48 | bias off | bias on_nofb | bias on_rule48 |
|---|---|---|---|---|---|---|---|
| 2-3 | 46,719 | 0.496 | 0.496 | 0.496 | -0.0130 | -0.0130 | -0.0131 |
| 3-5 | 158,540 | 0.467 | 0.467 | 0.467 | -0.0316 | -0.0317 | -0.0317 |
| 5-8 | 168,437 | 0.444 | 0.444 | 0.444 | -0.0473 | -0.0473 | -0.0473 |
| 8-12 | 130,269 | 0.439 | 0.439 | 0.439 | -0.0508 | -0.0508 | -0.0508 |
| 12-20 | 123,613 | 0.483 | 0.483 | 0.483 | -0.0558 | -0.0557 | -0.0557 |
| 20-50 | 104,754 | 0.940 | 0.940 | 0.940 | -0.1146 | -0.1142 | -0.1143 |
| 50-200 | 20,180 | 2.847 | 2.847 | 2.847 | -1.2796 | -1.2792 | -1.2792 |

### Track by track (on - off)

| | on_nofb | on_rule48 |
|---|---|---|
| tracks with any change of q/p, chi2 or ndof | 373,038 (46.0 %) | 373,045 (46.0 %) |
| q/p changed (outlier removal) | 6,237 | 6,079 |
| ndof changed (a hit removed or kept) | 5,233 | 5,111 |
| largest relative q/p change | 6.4 % | 6.4 % |
| chi2/ndf change, real tracks: mean; 1 % / 99 % | +0.0001; -0.010 / +0.010 | -0.0000; -0.009 / +0.009 |
| chi2/ndf change, ghosts: mean; 1 % / 99 % | +0.019; -0.049 / +0.041 | +0.020; -0.046 / +0.039 |

Tracks heading to the corner (abs(y) of the straight line of the beamline state at z = 5 m above 800 mm, where the fixed
steps fail on the validation tracks): 22,053 (2.72 %), 1.43 per event, at least one in 66.6 % of the events.

| | on_nofb | on_rule48 |
|---|---|---|
| corner: chi2/ndf change mean; 0.1 % / 99.9 %; largest | -0.0016; -9.1 / +8.2; 106 | -0.0006; -5.6 / +6.1; 33 |
| corner: ndof changed | 468 | 340 |
| not corner: chi2/ndf change 0.1 % / 99.9 %; largest | -0.43 / +0.43; 1562 | -0.44 / +0.43; 1562 |

### RICH2 states against MC truth (TrackResChecker, long tracks; mean and RMS)

| state | quantity | entries | mean off | mean on_nofb | mean on_rule48 | RMS off | RMS on_nofb | RMS on_rule48 |
|---|---|---|---|---|---|---|---|---|
| RICH2 entry | tx (rad) | 572,447 | -2.300e-05 | -2.299e-05 | -2.298e-05 | 9.925e-04 | 9.925e-04 | 9.925e-04 |
| RICH2 entry | ty (rad) | 664,489 | 2.586e-06 | 2.421e-06 | 2.469e-06 | 9.143e-04 | 9.145e-04 | 9.144e-04 |
| RICH2 entry | x (mm) | 53,041 | -0.01597 | -0.01596 | -0.01596 | 0.2198 | 0.2198 | 0.2198 |
| RICH2 entry | y (mm) | 64,774 | -0.006455 | -0.006524 | -0.006365 | 0.2275 | 0.2276 | 0.2276 |

The x and y residual histograms of the checker hold only 7-8 % of the tracks (their range is narrow, the rest falls
outside); the pull histograms are empty (the converted states carry no covariance). A track-by-track difference of the
states after the magnet (ON - OFF at the first SciFi hit) is not available from these checkers.

Figures: `figures/events_momentum_*.png`, `figures/events_chi2_*.png`, `figures/events_rich2_*.png`.

## What this says

- **On these events the network changes nothing measurable.** Track count, ghost count, the chi2/ndf of real and ghost
  tracks, the momentum resolution (0.44-0.50 % below 20 GeV) and its bias, and the RICH2 residuals agree between the
  ARKN and both network runs to the digits shown. The beamline q/p is blind to the second-pass crossing by construction
  (above); the rest moves by 1e-4 in the mean chi2/ndf.
- **The fit absorbs the crossing error.** 46 % of the tracks change their chi2 at all, by 0.01 in chi2/ndf at the 1/99 %
  points. Even the corner tracks, where the network alone is wrong by up to millimetres at the first SciFi hit,
  change by -9 / +8 at the 0.1 / 99.9 % points, with mean zero; the 48-step rule narrows that to -6 / +6. In the second
  pass the state after the magnet is predicted with the covariance from VELO and UT only, so a millimetre at the
  first SciFi hit is small against the spread from the q/p uncertainty, and the SciFi hits then pull the state.
- **About 1.4 tracks per event head into the corner** (2.7 % of the fitted long tracks).
- So the case for the network in PrKF rests on speed and field reads (30 vs 55 reads a crossing, `README.md`), not on
  the physics of the fit, unless the first-pass crossing (T -> UT) is replaced too.

## Not done

- No K_S mass width: this needs the secondary vertices of Allen and their MC matching; and the beamline q/p does not
  change in this configuration anyway.
- The first-pass crossing (T -> UT) is still the ARKN.
- The number of crossings in `on_nofb` that went to the ARKN because the state was outside the training box is not
  counted by Allen (on the validation tracks it is 0).
