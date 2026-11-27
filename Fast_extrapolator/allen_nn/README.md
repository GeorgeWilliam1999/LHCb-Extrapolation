# allen_nn: the magnet crossing of the HLT1 Kalman fit (PrKF) as K Nystrom steps + one network

Goal: a drop-in for the adaptive Runge-Kutta-Nystrom (ARKN) call of the new HLT1 Kalman fit
PrKF (Allen MR !2407) on the magnet crossing, last UT hit → first SciFi hit, good enough to
consider a merge request to Allen. Way A:

    z_in (last UT hit) --Nystrom leg--> 2648.2 mm --K Nystrom steps + network--> 7826 mm --leg--> z_out (first SciFi hit)
    end = s_K + sigma_out * (q/p / sigma_qop) * NN(s0 / sigma_in)         (fastex/train.py, start "nystrom", scale "times_qop")

Everything here is new; nothing outside this folder was changed. Started 2026-11-27.

## Files

| file | what |
|---|---|
| `MR2407_NOTES.md` | P0: what MR !2407 does at the magnet (calls, planes, tolerance, Jacobian, field, cuts, polarity) |
| `include/allen_nn/rkn_common.h` | slope derivative, Nystrom step, and the step Jacobian of Allen (field held fixed), term by term |
| `include/allen_nn/arkn.h` | host copy of Allen's ARKN with its Jacobian (`propagate_base<withJacobian>`) |
| `include/allen_nn/magnet_crossing.h` | the tool: plain `__host__ __device__` functions, `propagate_magnet` = interface of `ARKN::propagate_jac` |
| `include/allen_nn/weights_json.h` | host reader of the JSON weights (no library) |
| `src/an.cpp`, `build.sh` | the standalone program (`an run / gate / time`), built per node in `build/<node>/` |
| `python/export_json.py` | a stage-2 run → `weights/<run>.json` |
| `python/gates.py` | P1 checks → `results/gates.txt` |
| `python/score.py`, `python/plots.py` | P3 → `results/*.csv`, `figures/*.png` |
| `weights/*.json` | the six linear-map networks (K = 6, 8, 12; width 32, 64; one seed) |
| `ALLEN_PROTOTYPE.md`, `allen_patch/` | P4: the Allen branch, its patch, the build script and what is missing |

## How to run

    ./build.sh                                              # g++ -O3 -march=native, no fast-math
    python python/export_json.py linear_nystrom_12_2x64     # TE env; reads the run store
    python python/gates.py 2000
    python python/score.py && python python/plots.py

## Networks (P2)

The stage-2 runs of "Make the extrapolator fast" (track set afc61a558211, linear map v8r1 up,
planes 2648.2 → 7826 mm) already hold the way-A crossing with K = 6, 8, 12 and widths 32, 64,
trained to their plateau (`/data/bfys/gscriven/fast_extrapolator_store/runs/linear_nystrom_*`).
They are exported as they are (best validation round). No new farm job was submitted: the
configurations asked for exist; only K = 4 is missing (there is a K = 4 run on the cubic map only).
One seed each: differences under 2× are not claimed.

Polarity: the generator registers the v8r1 down map, and the two v8r1 maps are exact negatives
(max |B_up + B_down| = 0). The reference on the down map with −q/p equals the up map bit for bit,
and so does the tool with `polarity = −1` (gate 4). One weight file serves both polarities.

## Checks (P1, `results/gates.txt`)

1. The host ARKN (`arkn.h`, float) gives the same states as the earlier copy
   `cpp/lhcb/allen_nystrom.h`, bit for bit, and the same accepted steps (2,000 tracks).
2. ARKN (float) vs Rec STEP (`cpp/lhcb/rec_step.h`, double): radial difference median 0.1 um,
   99 % 1.0 um (one track at 174 um: a different step choice in float); both 10.0 um from the
   reference. Transport matrix (tx0, ty0, q/p columns): median relative difference 6e-8, 99 % 2e-6.
3. The crossing in double vs the torch model of `fastex/train.py`, six networks: end state within
   2e-10 of the correction (1e-5 asked), network derivative within 1e-13 of autograd. Pass 6/6.
4. Tool float vs double: radial median 0.04 um (the largest, 118 um, is a track that falls back in
   one precision only). Polarity: down map with −q/p and polarity −1 = up map, bit for bit.

Allen side: `TestMagnetCrossingNN.cpp` (Catch2, 3 cases, 51 assertions) passes in the Allen CPU
build: the full 5×5 transport matrix of the tool equals central finite differences to 1e-6 on a
z-only field, a zero network equals plain fixed Nystrom steps, polarity symmetry is exact.

## Results (P3, `results/*.csv`, `figures/*.png`)

13,447 linear-map test tracks (track set afc61a558211, v8r1 up), 2648.2 → 7826 mm; the reference is
the track set's (6th-order RK; a dense RK4 with 4,000 steps agrees with it to 0.0004 um median,
0.25 um at most). The ParKF step runs on its own planes (2665 → 7826 mm, its own reference).
The tool has the ARKN fallback (threshold per K, `results/fallback_scan.txt`) and its transport
matrix is that of the K Nystrom steps (`network_jacobian = 0`; the network part moves a one-sigma
shift by 0.1 um at most, and with it the call costs 2.4× more, see "network in the matrix").

### End point, all 13,447 test tracks (radial um: median / 95 % / 99.9 % / largest)

| method | radial | x (um) | y (um) | tx (urad) | ty (urad) | field reads | fell back | us/call with matrix | us/call state only |
|---|---|---|---|---|---|---|---|---|---|
| tool K=6 2x32 | 3.61 / 40.4 / 222 / 2.91e+03 | 3.24 / 38.5 / 181 / 1.51e+03 | 0.442 / 6.76 / 130 / 2.68e+03 | 3.97 / 20.5 / 79.7 / 566 | 0.555 / 5.94 / 55.3 / 710 | 26.4 | 3134 | 3.58 | 2.17 |
| tool K=6 2x64 | 3.33 / 39.4 / 211 / 1.95e+03 | 3.05 / 37.7 / 154 / 707 | 0.411 / 6.39 / 114 / 1.93e+03 | 3.74 / 20.2 / 68.5 / 209 | 0.542 / 5.61 / 57.3 / 471 | 26.4 | 3134 | 4.56 | 3.12 |
| tool K=8 2x32 | 2.09 / 34.8 / 296 / 2.99e+03 | 1.84 / 32.7 / 201 / 2.8e+03 | 0.349 / 6.35 / 160 / 1.05e+03 | 3.25 / 19.7 / 80.2 / 729 | 0.527 / 7.34 / 51.3 / 396 | 25.2 | 1947 | 3.24 | 2.02 |
| tool K=8 2x64 | 1.74 / 34.3 / 235 / 1.15e+03 | 1.48 / 32.2 / 160 / 459 | 0.35 / 6.37 / 156 / 1.11e+03 | 2.24 / 17.6 / 73.8 / 134 | 0.457 / 6.85 / 52.9 / 396 | 25.2 | 1947 | 4.19 | 2.86 |
| tool K=12 2x32 | 1.47 / 28.2 / 162 / 577 | 1.23 / 26.1 / 151 / 528 | 0.319 / 5.76 / 100 / 460 | 2.17 / 16.4 / 64 / 188 | 0.448 / 7.11 / 36.5 / 97.9 | 30.1 | 1323 | 3.67 | 2.27 |
| tool K=12 2x64 | 1.4 / 27.7 / 160 / 917 | 1.14 / 25.3 / 151 / 916 | 0.326 / 5.38 / 94.2 / 183 | 2.02 / 15.7 / 64.3 / 234 | 0.438 / 6.77 / 34.5 / 60.8 | 30.1 | 1323 | 4.69 | 3.31 |
| tool K=12 2x64, no fallback | 1.4 / 29.1 / 7.54e+03 / 2.76e+04 | 1.14 / 24.4 / 5.33e+03 / 1.4e+04 | 0.332 / 9.11 / 5.84e+03 / 2.74e+04 | 2.06 / 20.8 / 2.32e+03 / 7.29e+03 | 0.466 / 11.8 / 1.73e+03 / 1.11e+04 | 25 | 0 | 3.87 | - |
| tool K=12 2x64, network in the matrix | 1.4 / 27.7 / 160 / 917 | 1.14 / 25.3 / 151 / 916 | 0.326 / 5.38 / 94.2 / 183 | 2.02 / 15.7 / 64.3 / 234 | 0.438 / 6.77 / 34.5 / 60.8 | 30.1 | 1323 | 11.4 | - |
| ARKN (Allen) | 10.1 / 38.8 / 155 / 436 | 9.89 / 38.1 / 145 / 435 | 0.313 / 4.34 / 84.2 / 183 | 3.28 / 14.4 / 55.9 / 110 | 0.255 / 3.16 / 28.7 / 60.8 | 54.7 | 0 | 5.99 | 3.29 |
| Nystrom 6 steps | 20.6 / 216 / 2.17e+04 / 4.6e+04 | 19.3 / 195 / 1.74e+04 / 4.27e+04 | 2.09 / 63 / 1.27e+04 / 4.49e+04 | 9.07 / 82.7 / 7.91e+03 / 2.92e+04 | 1.6 / 44.2 / 5.17e+03 / 1.56e+04 | 13 | 0 | 1.06 | - |
| Nystrom 8 steps | 19.4 / 141 / 2.36e+04 / 7.73e+04 | 18.9 / 128 / 2.26e+04 / 7.6e+04 | 1.21 / 37.5 / 6.89e+03 / 2.3e+04 | 8.59 / 61.5 / 6.82e+03 / 2.77e+04 | 1.07 / 26.1 / 2.5e+03 / 5.88e+03 | 17 | 0 | 1.41 | - |
| Nystrom 12 steps | 15.1 / 94.7 / 9.45e+03 / 2.5e+04 | 15 / 90.5 / 7.13e+03 / 1.87e+04 | 0.583 / 13.2 / 7.34e+03 / 2.49e+04 | 10.9 / 58.4 / 2.69e+03 / 8.96e+03 | 0.586 / 14.8 / 2.28e+03 / 9.77e+03 | 25 | 0 | 2.11 | - |
| Nystrom 32 steps | 0.33 / 5.48 / 1.04e+03 / 2.39e+03 | 0.28 / 4.34 / 562 / 1e+03 | 0.0684 / 1.92 / 822 / 2.37e+03 | 0.607 / 4.49 / 231 / 510 | 0.0558 / 1.86 / 229 / 732 | 65 | 0 | 5.49 | - |
| Rec STEP (double) | 10.1 / 38.8 / 154 / 459 | 9.91 / 38.1 / 142 / 459 | 0.312 / 4.32 / 87.9 / 184 | 3.28 / 14.4 / 55.1 / 127 | 0.254 / 3.16 / 30.8 / 66.3 | 0 | 0 | - | - |
| ParKF step (2665->7826) | 55.6 / 2.33e+03 / 8.68e+04 / 1.23e+06 | 53.7 / 2.25e+03 / 8.37e+04 / 1.22e+06 | 3.78 / 449 / 2.17e+04 / 2.05e+05 | 19.3 / 1e+03 / 4.01e+04 / 6.92e+05 | 1.74 / 140 / 6.17e+03 / 9.87e+04 | 0 | 0 | 0.411 | 0.411 |

Times: lowest of 9 passes over the 13,447 tracks, one call at a time, one core (core 40) of stbc-i3, float, g++ 11 -O3 -march=native. The node was shared (load 20-40). Rec STEP is double and not timed.

### Radial end-point error by momentum (um, median / 95 %)

| method | p < 3 GeV | 3-8 | 8-20 | > 20 |
|---|---|---|---|---|
| tool K=6 2x32 | 16.2 / 64.1 | 7.32 / 49.4 | 1.63 / 11.8 | 0.463 / 2 |
| tool K=6 2x64 | 16.2 / 64.1 | 6.8 / 47.3 | 1.58 / 12.4 | 0.504 / 1.79 |
| tool K=8 2x32 | 13.9 / 68.6 | 4.31 / 38.6 | 0.934 / 6.33 | 0.288 / 1.56 |
| tool K=8 2x64 | 13.5 / 68.6 | 3.54 / 38 | 0.726 / 6.44 | 0.228 / 1.12 |
| tool K=12 2x32 | 8.97 / 64 | 2.76 / 31.6 | 0.775 / 3.62 | 0.252 / 0.742 |
| tool K=12 2x64 | 8.49 / 64 | 2.67 / 30.8 | 0.678 / 3.55 | 0.195 / 0.713 |
| tool K=12 2x64, no fallback | 7.8 / 92.2 | 2.68 / 46.4 | 0.679 / 3.55 | 0.195 / 0.713 |
| tool K=12 2x64, network in the matrix | 8.49 / 64 | 2.67 / 30.8 | 0.678 / 3.55 | 0.195 / 0.713 |
| ARKN (Allen) | 16.2 / 64 | 14.2 / 42.1 | 9.43 / 23.5 | 2.39 / 6.93 |
| Nystrom 6 steps | 79.2 / 564 | 37 / 293 | 12.2 / 49.1 | 3.84 / 13 |
| Nystrom 8 steps | 63.7 / 529 | 32.2 / 251 | 10.3 / 29.4 | 3.1 / 8.57 |
| Nystrom 12 steps | 54.9 / 203 | 24.3 / 125 | 7.86 / 19.6 | 2.61 / 5.58 |
| Nystrom 32 steps | 1.57 / 18.3 | 0.574 / 10.6 | 0.178 / 0.65 | 0.0767 / 0.203 |
| Rec STEP (double) | 16.2 / 64.5 | 14.2 / 42.1 | 9.42 / 23.5 | 2.38 / 6.91 |
| ParKF step (2665->7826) | 1.17e+03 / 1.79e+04 | 164 / 3.39e+03 | 11.9 / 67.3 | 2.89 / 11.3 |

### Transport matrix: error of the predicted end-position shift for a one-sigma start change (um, median / 95 %), all tracks

| method | x0 50um | y0 50um | tx0 0.3mrad | ty0 0.3mrad | q/p 1% |
|---|---|---|---|---|---|
| tool K=6 2x32 | 0.176 / 1.4 | 0.836 / 7.08 | 1.42 / 13.3 | 13.8 / 121 | 1.6 / 39.2 |
| tool K=6 2x64 | 0.176 / 1.4 | 0.836 / 7.08 | 1.42 / 13.3 | 13.8 / 121 | 1.6 / 39.2 |
| tool K=8 2x32 | 0.176 / 1.4 | 0.836 / 7.08 | 1.42 / 13.3 | 13.8 / 121 | 1.58 / 38.9 |
| tool K=8 2x64 | 0.176 / 1.4 | 0.836 / 7.08 | 1.42 / 13.3 | 13.8 / 121 | 1.58 / 38.9 |
| tool K=12 2x32 | 0.176 / 1.4 | 0.836 / 7.08 | 1.42 / 13.3 | 13.8 / 121 | 1.6 / 39.2 |
| tool K=12 2x64 | 0.176 / 1.4 | 0.836 / 7.08 | 1.42 / 13.3 | 13.8 / 121 | 1.6 / 39.2 |
| tool K=12 2x64, no fallback | 0.176 / 1.4 | 0.836 / 7.08 | 1.42 / 13.3 | 13.8 / 121 | 1.6 / 39.9 |
| tool K=12 2x64, network in the matrix | 0.176 / 1.4 | 0.836 / 7.06 | 1.43 / 13.3 | 13.8 / 121 | 1.63 / 39.3 |
| ARKN (Allen) | 0.176 / 1.4 | 0.836 / 7.08 | 1.42 / 13.3 | 13.8 / 121 | 1.67 / 39.2 |
| Nystrom 6 steps | 0.176 / 1.4 | 0.836 / 7.08 | 1.43 / 13.3 | 13.8 / 121 | 1.6 / 40.3 |
| Nystrom 8 steps | 0.176 / 1.4 | 0.836 / 7.08 | 1.42 / 13.3 | 13.8 / 121 | 1.58 / 41.1 |
| Nystrom 12 steps | 0.176 / 1.4 | 0.836 / 7.08 | 1.42 / 13.3 | 13.8 / 121 | 1.6 / 39.9 |
| Nystrom 32 steps | 0.176 / 1.4 | 0.836 / 7.08 | 1.42 / 13.3 | 13.8 / 121 | 1.62 / 39.3 |
| Rec STEP (double) | 0.176 / 1.4 | 0.836 / 7.08 | 1.42 / 13.3 | 13.8 / 121 | 1.67 / 39.2 |
| ParKF step (2665->7826) | 0.175 / 1.39 | 0.842 / 7.14 | 3.28 / 37.7 | 1.42 / 19.7 | 4.81 / 115 |
| (size of the true shift, median) | 50 | 50 | 1.57e+03 | 1.56e+03 | 4.55e+03 |

Slope part (urad, median / 95 %):

| method | x0 50um | y0 50um | tx0 0.3mrad | ty0 0.3mrad | q/p 1% |
|---|---|---|---|---|---|
| tool K=12 2x64 | 0.0666 / 0.58 | 0.372 / 3.2 | 0.988 / 10.1 | 7.62 / 60.2 | 1.97 / 51.6 |
| tool K=12 2x64, network in the matrix | 0.0667 / 0.58 | 0.372 / 3.2 | 0.992 / 10.1 | 7.6 / 60.2 | 2.03 / 51.8 |
| ARKN (Allen) | 0.0666 / 0.58 | 0.372 / 3.2 | 0.989 / 10.1 | 7.62 / 60.2 | 2.02 / 51.9 |
| Nystrom 12 steps | 0.0666 / 0.58 | 0.372 / 3.2 | 0.988 / 10.1 | 7.62 / 60.2 | 1.97 / 51.8 |
| ParKF step (2665->7826) | 0.0664 / 0.579 | 0.374 / 3.21 | 1.13 / 16.2 | 0.365 / 8.73 | 2.06 / 60.2 |

### Way A: from a plane in the last UT layer (2600-2700 mm) to a plane in the first SciFi layer (7826-7900 mm)

| method | radial median | 95 % | 99.9 % | largest | field reads | fell back |
|---|---|---|---|---|---|---|
| tool K=6 2x32 | 3.82 | 43.6 | 242 | 2.97e+03 | 29.6 | 3134 |
| tool K=6 2x64 | 3.51 | 42.6 | 232 | 1.98e+03 | 29.6 | 3134 |
| tool K=8 2x32 | 2.23 | 37.9 | 293 | 3.05e+03 | 28.7 | 1947 |
| tool K=8 2x64 | 1.78 | 37.4 | 249 | 1.18e+03 | 28.7 | 1947 |
| tool K=12 2x32 | 1.54 | 30.4 | 193 | 590 | 33.8 | 1323 |
| tool K=12 2x64 | 1.47 | 30.2 | 186 | 929 | 33.8 | 1323 |
| tool K=12 2x64, no fallback | 1.47 | 29.5 | 7.61e+03 | 2.8e+04 | 29 | 0 |
| tool K=12 2x64, network in the matrix | 1.47 | 30.2 | 186 | 929 | 33.8 | 1323 |
| ARKN (Allen) | 10.9 | 41.7 | 174 | 353 | 55.1 | 0 |

Figures: `figures/accuracy_vs_time.png`, `figures/error_by_momentum.png`, `figures/transport_matrix_error.png`.

## What this says (one seed per network; differences under 2× are not claimed)

- **Typical track: the tool is 7× more accurate than the ARKN.** K = 12, width 64: radial median
  1.40 um against 10.1 um; 95 % 27.7 against 38.8 um; better in every momentum band
  (p < 3 GeV: 8.5 vs 16.2 um; 8-20 GeV: 0.68 vs 9.4 um). K = 8: 1.74 um. K = 6: 3.3 um.
- **Tail: worse than the ARKN.** 99.9 % 160 um vs 155 um (equal), but the largest error is
  917 um vs 436 um, and in x the 99.9 % is 151 vs 145 um. Without the fallback the fixed steps
  are wrong by up to 28 mm for about 0.7 % of the tracks (|y| 230-680 mm, |ty| 0.17-0.26 on z0);
  plain fixed Nystrom steps have the same tail, so it is the fixed steps, not the network. The
  fallback (step-error estimate above 0.7 mm for K = 12) sends 9.8 % of the tracks to the ARKN.
  Its threshold was chosen on the test tracks; it must be re-chosen on the validation tracks.
- **Transport matrix: as good as the ARKN's.** Every integrator here chains the step Jacobians with
  the field held fixed, as LHCb does; their errors against finite differences are the same to
  three digits (median: 0.18 / 0.84 / 1.4 / 13.8 / 1.6 um for x0 / y0 / tx0 / ty0 / q/p). The
  ty0 column (13.8 um for a 1.56 mm shift) is most likely the dB/dy term that the fixed-field
  chaining leaves out (not tested); the ParKF parametrisation, which has no such approximation, is at
  1.4 um there but worse in the other columns.
- **Time (host, one core, float): 1.3× faster than the ARKN with the matrix** (K = 12, 2x64:
  4.7 us vs 6.0 us; 2x32: 3.7 us), the same without (3.3 vs 3.3 us). 30 field reads against 55.
  On a GPU the ratio of field reads (textures) matters more; not measured.
- **A plain Nystrom with 32 fixed steps** (no network) is better still in the typical track
  (0.33 um median, 5.5 um 95 %, 5.5 us with the matrix) but has the same fixed-step tail
  (largest 2.4 mm). With the same fallback it may be the simplest candidate; not tried tonight.
- Way A (hit planes 2600-2700 → 7826-7900 mm, one leg step each side): the same picture,
  K = 12 2x64 1.47 / 30.2 um vs ARKN 10.9 / 41.7 um.

## Not done / open

- No physics test in Allen: the test sample on EOS is not readable from stbc without a CERN
  credential, and the MDF has no MC truth (see `ALLEN_PROTOTYPE.md`).
- The T → UT crossing of the first fit pass still uses the ARKN (needs a reverse network or a
  Newton inversion of the forward tool).
- K = 4 is not trained on the linear map; no farm job was submitted tonight.
- The fallback threshold is tuned on the test tracks; the training box (domain) is the largest
  absolute value of each start variable, not a real acceptance.
- Timing is on a shared login node (load 20-40 during the final timing).
