# Stage 0: results

Written by `scripts/write_results_note.py` from the tables in `results/`. Timed on stbc-i3.nikhef.nl, one thread, one pinned core.

## Summary

1. **The compiled program is correct.** 24 of 24 methods give the final states of their Python versions (largest difference 2.7e-11 mm).
2. **The copies of the two HLT2 integrators are the LHCb tools.** On the grid of the Rec test each copy prints the same 1,331 final states as the real tool. With the transport matrix, the real tools in Gaudi are 12 % and 3.5 % slower than the copies on the same core.
3. **What LHCb costs and reaches on our tracks.** The adaptive Nystrom integrator: 55.7 field reads, 3.6 us a track (5.6 us with its transport matrix), median error 10.1 um. The adaptive Cash-Karp integrator: 42.6 reads, 1.8 us, 22.2 um. The parametrised step of HLT1: no read, 0.38 us, 55.6 um in the median, but 1,170 um below 3 GeV.
4. **The unit costs.** One field read of LHCb costs 11 ns in a loop of reads and about 18 ns inside an integrator (RK4, 16 steps: 1169 ns for 64 reads). One network call in single precision with the fast tanh costs 135 ns (2 x 32), 234 ns (2 x 64), 1078 ns (2 x 128) and 3660 ns (2 x 256).
5. **The present networks are slower than LHCb.** The whole-crossing network (2 x 256) takes 3.8 us and has a median error of 79.4 um. The chained networks take 10 to 164 us.
6. **A fixed-step integrator already beats the adaptive ones.** RK4 with 32 equal steps reaches 0.296 um in 2.3 us; the adaptive Nystrom integrator of LHCb reaches 10.1 um in 3.6 us.
7. **Single precision is enough.** It changes the final position of a network by 0.17 um in the median at most.

## 1. The compiled program gives the results of Python

24 methods are checked on 1,000 tracks: 24 pass. The build without fused arithmetic differs from Python by at most 2.7e-11 mm in position. The timed build (fused arithmetic) differs from that build by at most 1.4e-12 mm for the fixed-step methods and the networks.

| Method | Map | Against Python: largest difference (mm) | Tracks with other counts | Fused arithmetic: largest difference (mm) | Tracks with other steps | Single precision: median change (um) | largest (um) |
|---|---|---|---|---|---|---|---|
| rk4:16 | cubic | 1.1e-13 | 0 | 1.4e-12 | 0 | 0.027 | 0.53 |
| rk4:64 | cubic | 2.3e-13 | 0 | 4.5e-13 | 0 | 0.042 | 0.65 |
| rk4:16 | linear | 0 | 0 | 4.5e-13 | 0 | 0.027 | 0.34 |
| rk4:64 | linear | 0 | 0 | 9.1e-13 | 0 | 0.043 | 0.6 |
| rec_cash_karp | linear | 0 | 0 | 0.038 | 1 | 0.011 | 3.6e+02 |
| rec_cash_karp | cubic | 2.7e-11 | 0 | 5.7e-09 | 0 | 0.0098 | 26 |
| allen_adaptive_nystrom | linear | 1.4e-12 | 0 | 2.7e-12 | 0 | 0.044 | 4.1 |
| allen_adaptive_nystrom | cubic | 2.7e-12 | 0 | 4.1e-12 | 0 | 0.043 | 5.8 |
| allen_fast_nystrom | linear | 0 | 0 | 4.5e-13 | 0 | 0.019 | 0.31 |
| allen_fast_nystrom | cubic | 1.1e-13 | 0 | 4.5e-13 | 0 | 0.02 | 0.31 |
| whole_crossing:dd9acf59ac88 | linear | 1.4e-12 | 0 | 9.1e-13 | 0 | 0.17 | 0.53 |
| chain:29510f525130 | cubic | 9.1e-13 | 0 | 9.1e-13 | 0 | 0.051 | 0.55 |
| chain:400182ba3233 | cubic | 2.3e-13 | 0 | 6.8e-13 | 0 | 0.041 | 0.89 |
| chain:2382b5d314a5 | cubic | 4.5e-13 | 0 | 9.1e-13 | 0 | 0.039 | 0.64 |
| chain:a8a4083032fd | cubic | 2.3e-13 | 0 | 9.1e-13 | 0 | 0.039 | 0.63 |
| chain:2e8f364b79e2 | cubic | 2.3e-13 | 0 | 1.1e-12 | 0 | 0.03 | 0.4 |
| chain:84306b16e99d | linear | 4.5e-13 | 0 | 1.1e-12 | 0 | 0.04 | 0.77 |
| chain:ffc653fadc1e | linear | 6.8e-13 | 0 | 1.1e-12 | 0 | 0.04 | 0.72 |
| chain:6c4a3b022a2f | linear | 4.5e-13 | 0 | 6.8e-13 | 0 | 0.021 | 0.32 |
| chain:29cbba959300 | linear | 6.8e-13 | 0 | 4.5e-13 | 0 | 0.03 | 0.55 |
| chain:a66c06472d3d | linear | 4.5e-13 | 0 | 4.5e-13 | 0 | 0.017 | 0.4 |
| chain:15242aa17c65 | linear | 6.8e-13 | 0 | 9.1e-13 | 0 | 0.054 | 1.1 |
| chain:6af35e016a58 | linear | 4.5e-13 | 0 | 4.5e-13 | 0 | 0.022 | 0.37 |
| chain:a0f26d05a973 | linear | 9.1e-13 | 0 | 4.5e-13 | 0 | 0.04 | 0.53 |

The adaptive Cash-Karp integrator reacts to the last digit: with fused arithmetic 1 track of 1,000 takes other steps and moves by 38 um, and with the field map held in single precision one track moves by 362 um. The fixed-step methods and the networks do not have this property.

## 2. The copies of the HLT2 integrators against the LHCb software

The test algorithm of Rec carries a grid of 1,331 states from 3000 mm to 7000 mm and prints the final states. Each copy is compared with the numbers that the real tool printed (3 decimals in mm, 4 for the slopes).

| Copy | Field map | States with the same printed numbers | Largest difference (mm) | Field reads (mean) |
|---|---|---|---|---|
| rec_cash_karp | v5r0_down | 1331 of 1331 | 0.0005 | 78.8 |
| rec_cash_karp | v5r0_up | 121 of 1331 | 3,100 | 78.8 |
| rec_cash_karp | v8r1_down | 423 of 1331 | 5.79 | 78.6 |
| rec_cash_karp | v8r1_up | 121 of 1331 | 3,100 | 78.6 |
| rec_step | v5r0_down | 1331 of 1331 | 0.000499 | 63.4 |
| rec_step | v5r0_up | 121 of 1331 | 3,100 | 63.4 |
| rec_step | v8r1_down | 443 of 1331 | 4.97 | 63.8 |
| rec_step | v8r1_up | 121 of 1331 | 3,100 | 63.8 |

On the map of the test (field.v5r0, magnet down) both copies print the numbers of the real tools for every state. The other maps do not, so the test tells the map.

Time for one state of that grid, on stbc-i2 (one core, the same for all rows):

|  | Lowest (ns) | Highest (ns) |
|---|---|---|
| Rec v41r3 in Gaudi: TrackRungeKuttaExtrapolator (adaptive Cash-Karp) with its transport matrix | 8127 | 8154 |
| Rec v41r3 in Gaudi: TrackSTEPExtrapolator (adaptive Nystrom) with its transport matrix | 9457 | 9549 |
| our program: Cash-Karp copy state only | 4548 | 5048 |
| our program: Cash-Karp copy with transport matrix | 7242 | 7627 |
| our program: adaptive Nystrom (STEP) copy state only | 6013 | 6390 |
| our program: adaptive Nystrom (STEP) copy with transport matrix | 9134 | 9544 |
| our program: RK4 with one step (4 field reads) | 136 | 227 |

The copy of the Allen adaptive integrator cannot be compared with Allen in the same way. LHCb checks Allen against TrackSTEPExtrapolator on the same grid within 0.01 mm + 1 % (positions) and 5e-5 + 1 % (slopes). Our two copies pass that same check: 0 of 1331 states are outside the limits; the median difference is 0.25 um.

## 3. The LHCb methods on our tracks

13,447 test tracks with the real momentum spectrum, field map v8r1 (magnet up, trilinear), one crossing from 2648.2 mm to 7826 mm (the parametrised step: from 2665 mm, its own first plane). The error is the distance on the last plane from our reference integrator.

| Method | Precision | Field reads (mean) | Time a track (ns), lowest | median | cold caches | Median error (um) | 95 % (um) | Tracks above 100 um (%) |
|---|---|---|---|---|---|---|---|---|
| HLT2 adaptive Nystrom (TrackSTEPExtrapolator, tolerance 5 um) | double | 55.7 | 3609 | 3647 | 10059 | 10.1 | 38.8 | 0.51 |
| HLT2 adaptive Nystrom (TrackSTEPExtrapolator, tolerance 5 um), with its transport matrix | double | 55.7 | 5635 | 5704 | 12318 | 10.1 | 38.8 | 0.51 |
| HLT2 adaptive Cash-Karp (TrackRungeKuttaExtrapolator, tolerance 1 um) | double | 42.6 | 1825 | 1872 | 6472 | 22.2 | 112 | 6.4 |
| HLT2 adaptive Cash-Karp (TrackRungeKuttaExtrapolator, tolerance 1 um), with its transport matrix | double | 42.6 | 2918 | 3001 | 7794 | 22.2 | 112 | 6.4 |
| HLT1 adaptive Nystrom (Allen, tolerance 5 um) | float | 54.7 | 3116 | 3188 | 11151 | 10.1 | 38.9 | 0.5 |
| HLT1 T-track integrator (fixed Nystrom, 500 mm, 1 read a step) | float | 11 | 743 | 753 | 3316 | 997 | 2,830 | 99 |
| HLT1 parametrised step (polynomials, no field read) | float | 0 | 381 | 408 | 3366 | 55.6 | 2,330 | 42 |

Median error (um) in the momentum bands:

| Method | below 3 GeV | 3 to 8 GeV | 8 to 20 GeV | 20 to 50 GeV | above 50 GeV |
|---|---|---|---|---|---|
| HLT2 adaptive Nystrom (TrackSTEPExtrapolator, tolerance 5 um) | 16.2 | 14.2 | 9.42 | 2.64 | 1.64 |
| HLT2 adaptive Cash-Karp (TrackRungeKuttaExtrapolator, tolerance 1 um) | 47.3 | 35 | 14.2 | 3.12 | 0.311 |
| HLT1 adaptive Nystrom (Allen, tolerance 5 um) | 16.2 | 14.2 | 9.43 | 2.64 | 1.65 |
| HLT1 T-track integrator (fixed Nystrom, 500 mm, 1 read a step) | 2,690 | 1,420 | 581 | 250 | 103 |
| HLT1 parametrised step (polynomials, no field read) | 1,170 | 164 | 11.9 | 3.32 | 1.61 |

The adaptive integrators on the 1,000 tracks that are flat in momentum (2 to 42 GeV), the sample of the first timing study:

| Method | Field reads (mean) | Time a track (ns) | Median error (um) | 95 % (um) |
|---|---|---|---|---|
| HLT2 adaptive Nystrom (TrackSTEPExtrapolator, tolerance 5 um) | 45.5 | 2865 | 4.16 | 24.4 |
| HLT2 adaptive Cash-Karp (TrackRungeKuttaExtrapolator, tolerance 1 um) | 38.4 | 1710 | 7.63 | 55.4 |
| HLT1 adaptive Nystrom (Allen, tolerance 5 um) | 44.5 | 2659 | 4.14 | 24.4 |

The parametrised step of HLT1 against references on two field maps (its polynomials were fitted to one of them); `polynomial` is the tabulated step alone, `full` adds the change of q/p and the terms in q/p that Allen applies for the material:

| Variant | Reference map | Median error (um) | 95 % (um) | Signed median of x (um) |
|---|---|---|---|---|
| HLT1 parametrised step, polynomial, polarity +1 | v8r1_up | 55.6 | 2,330 | 0.745 |
| HLT1 parametrised step, full, polarity +1 | v8r1_up | 1,600 | 10,100 | -57.2 |
| HLT1 parametrised step, polynomial, polarity +1 | v5r0_up | 433 | 2,720 | 36.1 |
| HLT1 parametrised step, full, polarity +1 | v5r0_up | 1,890 | 10,700 | -79.7 |

Check of our reference: RK4 with 4,096 steps differs from the stored reference end states by 0.00016 um (median) and 0.21 um (largest).

## 4. The cost of one field read and of one network call

| Field read | Precision of the caller | Along the tracks (ns) | Random points (ns) |
|---|---|---|---|
| trilinear, arithmetic of the present package | double | 49.7 | 50.6 |
| tricubic B-spline, present package | double | 68.8 | 103.8 |
| trilinear, the read of LHCb (Detector) | double | 14.4 | 15.6 |
| trilinear, the read of Allen | double | 15.4 | 21.4 |
| trilinear, arithmetic of the present package | float | 18.3 | 25.3 |
| tricubic B-spline, present package | float | 66.1 | 97.9 |
| trilinear, the read of LHCb (Detector) | float | 10.6 | 13.3 |
| trilinear, the read of Allen | float | 13.0 | 17.2 |

One network call (6 inputs, 8 outputs), in ns:

| Hidden layers | Multiply-adds | Single precision, fast tanh | Double precision, fast tanh | Single precision, tanh of the C library | Single precision, layers only |
|---|---|---|---|---|---|
| 2 x 8 | 176 | 67 | 83 | 324 | 35 |
| 2 x 16 | 480 | 80 | 141 | 598 | 52 |
| 2 x 32 | 1472 | 135 | 242 | 1193 | 97 |
| 2 x 64 | 4992 | 234 | 749 | 2362 | 151 |
| 2 x 128 | 18176 | 1078 | 2328 | 5324 | 913 |
| 2 x 256 | 69120 | 3660 | 9265 | 12186 | 3335 |
| 2 x 512 | 269312 | 17810 | 39365 | 35031 | 17122 |
| 3 x 8 | 240 | 98 | 114 | 456 | 45 |
| 3 x 16 | 736 | 114 | 202 | 843 | 73 |
| 3 x 32 | 2496 | 214 | 377 | 1698 | 157 |
| 3 x 64 | 9088 | 507 | 1209 | 3456 | 386 |
| 3 x 128 | 34560 | 1971 | 4162 | 7835 | 1728 |
| 3 x 256 | 134656 | 7726 | 17355 | 19512 | 7211 |
| 3 x 512 | 531456 | 32319 | 70296 | 56616 | 31377 |

## 5. Every method: time and error on the 13,447 test tracks

**RK4 with equal steps**

| Method | Precision | Field read | Network calls | Field reads | Time a track (ns) | Cold caches (ns) | Median error (um) | 95 % (um) | Worst band (um) |
|---|---|---|---|---|---|---|---|---|---|
| RK4, 4 steps, linear map | double | linear | 0 | 16 | 404 | 2414 | 200 | 2,990 | 767 |
| RK4, 4 steps, linear map | float | linear | 0 | 16 | 357 | 2074 | 200 | 2,990 | 767 |
| RK4, 4 steps, cubic map | double | cubic | 0 | 16 | 1343 | 7284 | 201 | 2,890 | 774 |
| RK4, 4 steps, cubic map | float | cubic | 0 | 16 | 1265 | 6943 | 201 | 2,890 | 774 |
| RK4, 4 steps, linear map, field read of LHCb | float | rec | 0 | 16 | 286 | 1914 | 200 | 2,990 | 767 |
| RK4, 8 steps, linear map | double | linear | 0 | 32 | 810 | 4008 | 19.5 | 145 | 67.1 |
| RK4, 8 steps, linear map | float | linear | 0 | 32 | 722 | 3456 | 19.6 | 146 | 67.3 |
| RK4, 8 steps, cubic map | double | cubic | 0 | 32 | 2655 | 11502 | 13.2 | 211 | 52.3 |
| RK4, 8 steps, cubic map | float | cubic | 0 | 32 | 2506 | 10700 | 13.2 | 211 | 52.4 |
| RK4, 8 steps, linear map, field read of LHCb | float | rec | 0 | 32 | 575 | 3326 | 19.6 | 146 | 67.4 |
| RK4, 16 steps, linear map | double | linear | 0 | 64 | 1614 | 6838 | 4.03 | 39.9 | 16.6 |
| RK4, 16 steps, linear map | float | linear | 0 | 64 | 1460 | 6042 | 3.99 | 39.8 | 16.5 |
| RK4, 16 steps, cubic map | double | cubic | 0 | 64 | 5051 | 15204 | 1.96 | 66.6 | 9.53 |
| RK4, 16 steps, cubic map | float | cubic | 0 | 64 | 4843 | 13886 | 1.94 | 66.5 | 9.53 |
| RK4, 16 steps, linear map, field read of LHCb | float | rec | 0 | 64 | 1169 | 5646 | 3.97 | 39.8 | 16.4 |
| RK4, 32 steps, linear map | double | linear | 0 | 128 | 3149 | 10534 | 0.31 | 5.46 | 1.56 |
| RK4, 32 steps, linear map | float | linear | 0 | 128 | 2874 | 9608 | 0.303 | 5.53 | 1.56 |
| RK4, 32 steps, cubic map | double | cubic | 0 | 128 | 9844 | 21800 | 0.301 | 4.82 | 1.86 |
| RK4, 32 steps, cubic map | float | cubic | 0 | 128 | 9578 | 19266 | 0.315 | 4.8 | 1.83 |
| RK4, 32 steps, linear map, field read of LHCb | float | rec | 0 | 128 | 2251 | 8986 | 0.296 | 5.56 | 1.56 |
| RK4, 64 steps, linear map | double | linear | 0 | 256 | 6077 | 13450 | 1.1 | 4.12 | 3.25 |
| RK4, 64 steps, linear map | float | linear | 0 | 256 | 5701 | 12569 | 1.06 | 4.08 | 3.16 |
| RK4, 64 steps, cubic map | double | cubic | 0 | 256 | 19462 | 31198 | 0.0244 | 0.18 | 0.118 |
| RK4, 64 steps, cubic map | float | cubic | 0 | 256 | 19046 | 28773 | 0.0767 | 0.385 | 0.163 |
| RK4, 64 steps, linear map, field read of LHCb | float | rec | 0 | 256 | 4440 | 11257 | 1.04 | 4.02 | 3.1 |

**The present trained networks**

| Method | Precision | Field read | Network calls | Field reads | Time a track (ns) | Cold caches (ns) | Median error (um) | 95 % (um) | Worst band (um) |
|---|---|---|---|---|---|---|---|---|---|
| Whole-crossing network, 2x256, linear map | double | linear | 1 | 0 | 18407 | 51842 | 79.4 | 774 | 234 |
| Whole-crossing network, 2x256, linear map | float | linear | 1 | 0 | 13597 | 32972 | 79.4 | 774 | 234 |
| Whole-crossing network, 2x256, linear map | float, fast tanh | linear | 1 | 0 | 3756 | 22718 | 79.4 | 774 | 234 |
| Chain, direct states, 64 steps, 2x128, cubic map | double | cubic | 64 | 128 | 387522 | 465890 | 7.64 | 93.7 | 31.4 |
| Chain, direct states, 64 steps, 2x128, cubic map | float | cubic | 64 | 128 | 329646 | 356877 | 7.64 | 93.6 | 31.4 |
| Chain, direct states, 64 steps, 2x128, cubic map | float, fast tanh | cubic | 64 | 128 | 79432 | 111920 | 7.62 | 93.6 | 31.4 |
| Chain, straight line, global scale, 64 steps, 2x128, cubic map | double | cubic | 64 | 128 | 431752 | 512663 | 3.55 | 36.1 | 13.3 |
| Chain, straight line, global scale, 64 steps, 2x128, cubic map | float | cubic | 64 | 128 | 372114 | 417710 | 3.54 | 36.1 | 13.4 |
| Chain, straight line, global scale, 64 steps, 2x128, cubic map | float, fast tanh | cubic | 64 | 128 | 71140 | 122694 | 3.54 | 36.1 | 13.4 |
| Chain, straight line, per-track scale (16 samples), 64 steps, 2x128, cubic map | double | cubic | 64 | 1,150 | 517605 | 596585 | 2.4 | 17.6 | 7.51 |
| Chain, straight line, per-track scale (16 samples), 64 steps, 2x128, cubic map | float | cubic | 64 | 1,150 | 447970 | 479180 | 2.41 | 17.6 | 7.51 |
| Chain, straight line, per-track scale (16 samples), 64 steps, 2x128, cubic map | float, fast tanh | cubic | 64 | 1,150 | 164249 | 210489 | 2.41 | 17.6 | 7.51 |
| Chain, straight line, per-track scale (1 sample), 64 steps, 2x128, cubic map | double | cubic | 64 | 192 | 436116 | 493477 | 2.47 | 17.7 | 7.62 |
| Chain, straight line, per-track scale (1 sample), 64 steps, 2x128, cubic map | float | cubic | 64 | 192 | 383788 | 422380 | 2.47 | 17.6 | 7.61 |
| Chain, straight line, per-track scale (1 sample), 64 steps, 2x128, cubic map | float, fast tanh | cubic | 64 | 192 | 87948 | 120440 | 2.47 | 17.6 | 7.61 |
| Physics layer (16 samples), 64 steps, 2x128, cubic map | double | cubic | 64 | 1,280 | 523061 | 605352 | 0.0184 | 0.143 | 0.0875 |
| Physics layer (16 samples), 64 steps, 2x128, cubic map | float | cubic | 64 | 1,280 | 461878 | 506371 | 0.0744 | 0.373 | 0.162 |
| Physics layer (16 samples), 64 steps, 2x128, cubic map | float, fast tanh | cubic | 64 | 1,280 | 162593 | 217817 | 0.0744 | 0.373 | 0.162 |
| Physics layer (1 sample), 64 steps, 2x128, cubic map | double | cubic | 64 | 320 | 447712 | 527280 | 0.0183 | 0.141 | 0.0861 |
| Physics layer (1 sample), 64 steps, 2x128, cubic map | float | cubic | 64 | 320 | 385744 | 415942 | 0.0743 | 0.372 | 0.162 |
| Physics layer (1 sample), 64 steps, 2x128, cubic map | float, fast tanh | cubic | 64 | 320 | 98659 | 146554 | 0.0743 | 0.372 | 0.162 |
| Physics layer (1 sample), 32 steps, 2x128, cubic map | double | cubic | 32 | 160 | 226437 | 265656 | 0.215 | 3.51 | 1.32 |
| Physics layer (1 sample), 32 steps, 2x128, cubic map | float | cubic | 32 | 160 | 197751 | 228482 | 0.219 | 3.53 | 1.33 |
| Physics layer (1 sample), 32 steps, 2x128, cubic map | float, fast tanh | cubic | 32 | 160 | 48845 | 72761 | 0.219 | 3.53 | 1.33 |
| Chain, straight line, per-track scale (1 sample), 64 steps, 2x128, linear map | double | linear | 64 | 192 | 430050 | 504182 | 2.21 | 15.5 | 5.58 |
| Chain, straight line, per-track scale (1 sample), 64 steps, 2x128, linear map | float | linear | 64 | 192 | 357460 | 406525 | 2.2 | 15.6 | 5.54 |
| Chain, straight line, per-track scale (1 sample), 64 steps, 2x128, linear map | float, fast tanh | linear | 64 | 192 | 79872 | 121156 | 2.2 | 15.6 | 5.54 |
| Chain, straight line, per-track scale (1 sample), 32 steps, 2x128, linear map | double | linear | 32 | 96 | 210912 | 253889 | 5.19 | 34.3 | 14.9 |
| Chain, straight line, per-track scale (1 sample), 32 steps, 2x128, linear map | float | linear | 32 | 96 | 191063 | 216530 | 5.17 | 34.3 | 14.8 |
| Chain, straight line, per-track scale (1 sample), 32 steps, 2x128, linear map | float, fast tanh | linear | 32 | 96 | 33450 | 65122 | 5.17 | 34.3 | 14.8 |
| Chain, straight line, per-track scale (1 sample), 16 steps, 2x128, linear map | double | linear | 16 | 48 | 114345 | 143338 | 23.9 | 138 | 78.9 |
| Chain, straight line, per-track scale (1 sample), 16 steps, 2x128, linear map | float | linear | 16 | 48 | 98673 | 116328 | 23.9 | 138 | 79 |
| Chain, straight line, per-track scale (1 sample), 16 steps, 2x128, linear map | float, fast tanh | linear | 16 | 48 | 19257 | 33548 | 23.9 | 138 | 79 |
| Chain, straight line, per-track scale (1 sample), 8 steps, 2x128, linear map | double | linear | 8 | 24 | 58146 | 80355 | 186 | 661 | 586 |
| Chain, straight line, per-track scale (1 sample), 8 steps, 2x128, linear map | float | linear | 8 | 24 | 49394 | 63719 | 186 | 661 | 586 |
| Chain, straight line, per-track scale (1 sample), 8 steps, 2x128, linear map | float, fast tanh | linear | 8 | 24 | 9628 | 22522 | 186 | 661 | 586 |
| Physics layer (1 sample), 64 steps, 2x128, linear map | double | linear | 64 | 320 | 409470 | 479010 | 0.145 | 1.31 | 0.619 |
| Physics layer (1 sample), 64 steps, 2x128, linear map | float | linear | 64 | 320 | 371177 | 411204 | 0.166 | 1.35 | 0.645 |
| Physics layer (1 sample), 64 steps, 2x128, linear map | float, fast tanh | linear | 64 | 320 | 71942 | 123676 | 0.166 | 1.35 | 0.645 |
| Physics layer (1 sample), 64 steps, 2x32, linear map | double | linear | 64 | 320 | 91300 | 104942 | 0.145 | 1.31 | 0.621 |
| Physics layer (1 sample), 64 steps, 2x32, linear map | float | linear | 64 | 320 | 90814 | 100868 | 0.166 | 1.36 | 0.661 |
| Physics layer (1 sample), 64 steps, 2x32, linear map | float, fast tanh | linear | 64 | 320 | 20895 | 32095 | 0.166 | 1.36 | 0.661 |
| Physics layer (1 sample), 16 steps, 2x128, linear map | double | linear | 16 | 80 | 114917 | 137106 | 1.76 | 55.7 | 10.6 |
| Physics layer (1 sample), 16 steps, 2x128, linear map | float | linear | 16 | 80 | 97033 | 117400 | 1.76 | 55.7 | 10.5 |
| Physics layer (1 sample), 16 steps, 2x128, linear map | float, fast tanh | linear | 16 | 80 | 19959 | 34158 | 1.76 | 55.7 | 10.5 |
