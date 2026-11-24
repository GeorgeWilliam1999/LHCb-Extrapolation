| Method | Precision | Field read | Network calls | Field reads | Time a track (ns), lowest | median | cold caches | Median error (um) | 95 % (um) | Worst band (um) | Flat sample: median (um) | 95 % (um) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| HLT2 adaptive Nystrom (TrackSTEPExtrapolator, tolerance 5 um) | double | rec | 0 | 55.7 | 3609 | 3647 | 10059 | 10.1 | 38.8 | 16.2 | 4.16 | 24.4 |
| HLT2 adaptive Nystrom (TrackSTEPExtrapolator, tolerance 5 um), with its transport matrix | double | rec | 0 | 55.7 | 5635 | 5704 | 12318 | 10.1 | 38.8 | 16.2 | 4.16 | 24.4 |
| HLT2 adaptive Nystrom (TrackSTEPExtrapolator, tolerance 5 um), flat sample only | double | rec | 0 | 45.5 | 2865 | 3124 | 8221 | 4.16 | 24.4 | 14.5 | 4.16 | 24.4 |
| HLT2 adaptive Cash-Karp (TrackRungeKuttaExtrapolator, tolerance 1 um) | double | rec | 0 | 42.6 | 1825 | 1872 | 6472 | 22.2 | 112 | 47.3 | 7.63 | 55.4 |
| HLT2 adaptive Cash-Karp (TrackRungeKuttaExtrapolator, tolerance 1 um), with its transport matrix | double | rec | 0 | 42.6 | 2918 | 3001 | 7794 | 22.2 | 112 | 47.3 | 7.63 | 55.4 |
| HLT2 adaptive Cash-Karp (TrackRungeKuttaExtrapolator, tolerance 1 um), flat sample only | double | rec | 0 | 38.4 | 1710 | 2018 | 5972 | 7.63 | 55.4 | 44.2 | 7.63 | 55.4 |
| HLT1 adaptive Nystrom (Allen, tolerance 5 um) | float | allen | 0 | 54.7 | 3116 | 3188 | 11151 | 10.1 | 38.9 | 16.2 | 4.14 | 24.4 |
| HLT1 adaptive Nystrom (Allen, tolerance 5 um), flat sample only | float | allen | 0 | 44.5 | 2659 | 2771 | 9478 | 4.14 | 24.4 | 14.9 | 4.14 | 24.4 |
| HLT1 T-track integrator (fixed Nystrom, 500 mm, 1 read a step) | float | allen | 0 | 11.0 | 743 | 753 | 3316 | 997 | 2.83e+03 | 2.69e+03 | 310 | 1.83e+03 |
| HLT1 parametrised step (polynomials, no field read) | float | linear | 0 | 0.0 | 381 | 408 | 3366 | 55.6 | 2.33e+03 | 1.17e+03 | 5.28 | 500 |
| RK4, 4 steps, linear map | double | linear | 0 | 16.0 | 404 | 427 | 2414 | 200 | 2.99e+03 | 767 | 49.7 | 876 |
| RK4, 4 steps, linear map | float | linear | 0 | 16.0 | 357 | 371 | 2074 | 200 | 2.99e+03 | 767 | 49.7 | 876 |
| RK4, 4 steps, cubic map | double | cubic | 0 | 16.0 | 1343 | 1391 | 7284 | 201 | 2.89e+03 | 774 | 49.5 | 879 |
| RK4, 4 steps, cubic map | float | cubic | 0 | 16.0 | 1265 | 1290 | 6943 | 201 | 2.89e+03 | 774 | 49.5 | 879 |
| RK4, 4 steps, linear map, field read of LHCb | float | rec | 0 | 16.0 | 286 | 302 | 1914 | 200 | 2.99e+03 | 767 | 49.7 | 876 |
| RK4, 8 steps, linear map | double | linear | 0 | 32.0 | 810 | 856 | 4008 | 19.5 | 145 | 67.1 | 5.24 | 58.7 |
| RK4, 8 steps, linear map | float | linear | 0 | 32.0 | 722 | 746 | 3456 | 19.6 | 146 | 67.3 | 5.27 | 58.9 |
| RK4, 8 steps, cubic map | double | cubic | 0 | 32.0 | 2655 | 2748 | 11502 | 13.2 | 211 | 52.3 | 3.16 | 52 |
| RK4, 8 steps, cubic map | float | cubic | 0 | 32.0 | 2506 | 2575 | 10700 | 13.2 | 211 | 52.4 | 3.17 | 52.2 |
| RK4, 8 steps, linear map, field read of LHCb | float | rec | 0 | 32.0 | 575 | 612 | 3326 | 19.6 | 146 | 67.4 | 5.27 | 58.8 |
| RK4, 16 steps, linear map | double | linear | 0 | 64.0 | 1614 | 1695 | 6838 | 4.03 | 39.9 | 16.6 | 1.06 | 12.8 |
| RK4, 16 steps, linear map | float | linear | 0 | 64.0 | 1460 | 1523 | 6042 | 3.99 | 39.8 | 16.5 | 1.04 | 12.8 |
| RK4, 16 steps, cubic map | double | cubic | 0 | 64.0 | 5051 | 5165 | 15204 | 1.96 | 66.6 | 9.53 | 0.575 | 9.69 |
| RK4, 16 steps, cubic map | float | cubic | 0 | 64.0 | 4843 | 4985 | 13886 | 1.94 | 66.5 | 9.53 | 0.572 | 9.6 |
| RK4, 16 steps, linear map, field read of LHCb | float | rec | 0 | 64.0 | 1169 | 1208 | 5646 | 3.97 | 39.8 | 16.4 | 1.04 | 12.8 |
| RK4, 32 steps, linear map | double | linear | 0 | 128.0 | 3149 | 3249 | 10534 | 0.31 | 5.46 | 1.56 | 0.101 | 1.38 |
| RK4, 32 steps, linear map | float | linear | 0 | 128.0 | 2874 | 2969 | 9608 | 0.303 | 5.53 | 1.56 | 0.0998 | 1.45 |
| RK4, 32 steps, cubic map | double | cubic | 0 | 128.0 | 9844 | 10101 | 21800 | 0.301 | 4.82 | 1.86 | 0.0688 | 1.58 |
| RK4, 32 steps, cubic map | float | cubic | 0 | 128.0 | 9578 | 9748 | 19266 | 0.315 | 4.8 | 1.83 | 0.0853 | 1.58 |
| RK4, 32 steps, linear map, field read of LHCb | float | rec | 0 | 128.0 | 2251 | 2320 | 8986 | 0.296 | 5.56 | 1.56 | 0.0967 | 1.47 |
| RK4, 64 steps, linear map | double | linear | 0 | 256.0 | 6077 | 6184 | 13450 | 1.1 | 4.12 | 3.25 | 0.352 | 2.43 |
| RK4, 64 steps, linear map | float | linear | 0 | 256.0 | 5701 | 5756 | 12569 | 1.06 | 4.08 | 3.16 | 0.346 | 2.37 |
| RK4, 64 steps, cubic map | double | cubic | 0 | 256.0 | 19462 | 19662 | 31198 | 0.0244 | 0.18 | 0.118 | 0.00478 | 0.0761 |
| RK4, 64 steps, cubic map | float | cubic | 0 | 256.0 | 19046 | 19231 | 28773 | 0.0767 | 0.385 | 0.163 | 0.0416 | 0.218 |
| RK4, 64 steps, linear map, field read of LHCb | float | rec | 0 | 256.0 | 4440 | 4528 | 11257 | 1.04 | 4.02 | 3.1 | 0.334 | 2.36 |
| Whole-crossing network, 2x256, linear map | double | linear | 1 | 0.0 | 18407 | 19001 | 51842 | 79.4 | 774 | 234 | 40.3 | 275 |
| Whole-crossing network, 2x256, linear map | float | linear | 1 | 0.0 | 13597 | 13836 | 32972 | 79.4 | 774 | 234 | 40.3 | 275 |
| Whole-crossing network, 2x256, linear map | float, fast tanh | linear | 1 | 0.0 | 3756 | 3876 | 22718 | 79.4 | 774 | 234 | 40.3 | 275 |
| Chain, direct states, 64 steps, 2x128, cubic map | double | cubic | 64 | 128.0 | 387522 | 388829 | 465890 | 7.64 | 93.7 | 31.4 | 3.82 | 43.6 |
| Chain, direct states, 64 steps, 2x128, cubic map | float | cubic | 64 | 128.0 | 329646 | 330316 | 356877 | 7.64 | 93.6 | 31.4 | 3.81 | 43.7 |
| Chain, direct states, 64 steps, 2x128, cubic map | float, fast tanh | cubic | 64 | 128.0 | 79432 | 81048 | 111920 | 7.62 | 93.6 | 31.4 | 3.8 | 43.7 |
| Chain, straight line, global scale, 64 steps, 2x128, cubic map | double | cubic | 64 | 128.0 | 431752 | 436942 | 512663 | 3.55 | 36.1 | 13.3 | 2.48 | 14.1 |
| Chain, straight line, global scale, 64 steps, 2x128, cubic map | float | cubic | 64 | 128.0 | 372114 | 374194 | 417710 | 3.54 | 36.1 | 13.4 | 2.48 | 14.1 |
| Chain, straight line, global scale, 64 steps, 2x128, cubic map | float, fast tanh | cubic | 64 | 128.0 | 71140 | 73485 | 122694 | 3.54 | 36.1 | 13.4 | 2.48 | 14.1 |
| Chain, straight line, per-track scale (16 samples), 64 steps, 2x128, cubic map | double | cubic | 64 | 1152.0 | 517605 | 519251 | 596585 | 2.4 | 17.6 | 7.51 | 1.9 | 10.5 |
| Chain, straight line, per-track scale (16 samples), 64 steps, 2x128, cubic map | float | cubic | 64 | 1152.0 | 447970 | 448370 | 479180 | 2.41 | 17.6 | 7.51 | 1.89 | 10.5 |
| Chain, straight line, per-track scale (16 samples), 64 steps, 2x128, cubic map | float, fast tanh | cubic | 64 | 1152.0 | 164249 | 165971 | 210489 | 2.41 | 17.6 | 7.51 | 1.89 | 10.5 |
| Chain, straight line, per-track scale (1 sample), 64 steps, 2x128, cubic map | double | cubic | 64 | 192.0 | 436116 | 436829 | 493477 | 2.47 | 17.7 | 7.62 | 1.87 | 10.5 |
| Chain, straight line, per-track scale (1 sample), 64 steps, 2x128, cubic map | float | cubic | 64 | 192.0 | 383788 | 385423 | 422380 | 2.47 | 17.6 | 7.61 | 1.87 | 10.5 |
| Chain, straight line, per-track scale (1 sample), 64 steps, 2x128, cubic map | float, fast tanh | cubic | 64 | 192.0 | 87948 | 88871 | 120440 | 2.47 | 17.6 | 7.61 | 1.87 | 10.5 |
| Physics layer (16 samples), 64 steps, 2x128, cubic map | double | cubic | 64 | 1280.0 | 523061 | 527269 | 605352 | 0.0184 | 0.143 | 0.0875 | 0.00369 | 0.0603 |
| Physics layer (16 samples), 64 steps, 2x128, cubic map | float | cubic | 64 | 1280.0 | 461878 | 462408 | 506371 | 0.0744 | 0.373 | 0.162 | 0.039 | 0.209 |
| Physics layer (16 samples), 64 steps, 2x128, cubic map | float, fast tanh | cubic | 64 | 1280.0 | 162593 | 163471 | 217817 | 0.0744 | 0.373 | 0.162 | 0.039 | 0.209 |
| Physics layer (1 sample), 64 steps, 2x128, cubic map | double | cubic | 64 | 320.0 | 447712 | 449648 | 527280 | 0.0183 | 0.141 | 0.0861 | 0.00369 | 0.0591 |
| Physics layer (1 sample), 64 steps, 2x128, cubic map | float | cubic | 64 | 320.0 | 385744 | 387359 | 415942 | 0.0743 | 0.372 | 0.162 | 0.039 | 0.208 |
| Physics layer (1 sample), 64 steps, 2x128, cubic map | float, fast tanh | cubic | 64 | 320.0 | 98659 | 101299 | 146554 | 0.0743 | 0.372 | 0.162 | 0.039 | 0.208 |
| Physics layer (1 sample), 32 steps, 2x128, cubic map | double | cubic | 32 | 160.0 | 226437 | 227142 | 265656 | 0.215 | 3.51 | 1.32 | 0.0507 | 1.17 |
| Physics layer (1 sample), 32 steps, 2x128, cubic map | float | cubic | 32 | 160.0 | 197751 | 198624 | 228482 | 0.219 | 3.53 | 1.33 | 0.0605 | 1.18 |
| Physics layer (1 sample), 32 steps, 2x128, cubic map | float, fast tanh | cubic | 32 | 160.0 | 48845 | 49612 | 72761 | 0.219 | 3.53 | 1.33 | 0.0605 | 1.18 |
| Chain, straight line, per-track scale (1 sample), 64 steps, 2x128, linear map | double | linear | 64 | 192.0 | 430050 | 431117 | 504182 | 2.21 | 15.5 | 5.58 | 1.71 | 8.75 |
| Chain, straight line, per-track scale (1 sample), 64 steps, 2x128, linear map | float | linear | 64 | 192.0 | 357460 | 358507 | 406525 | 2.2 | 15.6 | 5.54 | 1.71 | 8.72 |
| Chain, straight line, per-track scale (1 sample), 64 steps, 2x128, linear map | float, fast tanh | linear | 64 | 192.0 | 79872 | 82146 | 121156 | 2.2 | 15.6 | 5.54 | 1.71 | 8.72 |
| Chain, straight line, per-track scale (1 sample), 32 steps, 2x128, linear map | double | linear | 32 | 96.0 | 210912 | 211943 | 253889 | 5.19 | 34.3 | 14.9 | 2.72 | 17.7 |
| Chain, straight line, per-track scale (1 sample), 32 steps, 2x128, linear map | float | linear | 32 | 96.0 | 191063 | 191896 | 216530 | 5.17 | 34.3 | 14.8 | 2.72 | 17.7 |
| Chain, straight line, per-track scale (1 sample), 32 steps, 2x128, linear map | float, fast tanh | linear | 32 | 96.0 | 33450 | 35095 | 65122 | 5.17 | 34.3 | 14.8 | 2.72 | 17.7 |
| Chain, straight line, per-track scale (1 sample), 16 steps, 2x128, linear map | double | linear | 16 | 48.0 | 114345 | 114488 | 143338 | 23.9 | 138 | 78.9 | 10.3 | 75.3 |
| Chain, straight line, per-track scale (1 sample), 16 steps, 2x128, linear map | float | linear | 16 | 48.0 | 98673 | 99810 | 116328 | 23.9 | 138 | 79 | 10.2 | 75.3 |
| Chain, straight line, per-track scale (1 sample), 16 steps, 2x128, linear map | float, fast tanh | linear | 16 | 48.0 | 19257 | 19558 | 33548 | 23.9 | 138 | 79 | 10.2 | 75.3 |
| Chain, straight line, per-track scale (1 sample), 8 steps, 2x128, linear map | double | linear | 8 | 24.0 | 58146 | 59560 | 80355 | 186 | 661 | 586 | 56.1 | 386 |
| Chain, straight line, per-track scale (1 sample), 8 steps, 2x128, linear map | float | linear | 8 | 24.0 | 49394 | 50250 | 63719 | 186 | 661 | 586 | 56.1 | 386 |
| Chain, straight line, per-track scale (1 sample), 8 steps, 2x128, linear map | float, fast tanh | linear | 8 | 24.0 | 9628 | 9922 | 22522 | 186 | 661 | 586 | 56.1 | 386 |
| Physics layer (1 sample), 64 steps, 2x128, linear map | double | linear | 64 | 320.0 | 409470 | 410458 | 479010 | 0.145 | 1.31 | 0.619 | 0.0477 | 0.503 |
| Physics layer (1 sample), 64 steps, 2x128, linear map | float | linear | 64 | 320.0 | 371177 | 374034 | 411204 | 0.166 | 1.35 | 0.645 | 0.0617 | 0.535 |
| Physics layer (1 sample), 64 steps, 2x128, linear map | float, fast tanh | linear | 64 | 320.0 | 71942 | 74160 | 123676 | 0.166 | 1.35 | 0.645 | 0.0617 | 0.535 |
| Physics layer (1 sample), 64 steps, 2x32, linear map | double | linear | 64 | 320.0 | 91300 | 92428 | 104942 | 0.145 | 1.31 | 0.621 | 0.0477 | 0.506 |
| Physics layer (1 sample), 64 steps, 2x32, linear map | float | linear | 64 | 320.0 | 90814 | 92053 | 100868 | 0.166 | 1.36 | 0.661 | 0.0615 | 0.538 |
| Physics layer (1 sample), 64 steps, 2x32, linear map | float, fast tanh | linear | 64 | 320.0 | 20895 | 21184 | 32095 | 0.166 | 1.36 | 0.661 | 0.0615 | 0.538 |
| Physics layer (1 sample), 16 steps, 2x128, linear map | double | linear | 16 | 80.0 | 114917 | 115475 | 137106 | 1.76 | 55.7 | 10.6 | 0.506 | 14.5 |
| Physics layer (1 sample), 16 steps, 2x128, linear map | float | linear | 16 | 80.0 | 97033 | 98546 | 117400 | 1.76 | 55.7 | 10.5 | 0.498 | 14.5 |
| Physics layer (1 sample), 16 steps, 2x128, linear map | float, fast tanh | linear | 16 | 80.0 | 19959 | 20321 | 34158 | 1.76 | 55.7 | 10.5 | 0.498 | 14.5 |
