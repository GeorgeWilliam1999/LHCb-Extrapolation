# Compiled timing test (fast extrapolator), state at 2026-11-25 11:28 UTC

Test tracks: 13,447 of the cubic or the linear track set, compiled program `fx`, single precision for the networks, one core of stbc-i2. Errors are the endpoint radial error against the reference of the named map (µm). Times are the median ns per track over 15 repeats, shown in µs. Stage 0/1 numbers are final; every stage-2 network number is NOT final (runs still training or extended; test scores are from the round given, measured 2026-11-25 08:35 UTC).

## LHCb method

| name | map | state | rounds | last val. median | test median | test 95 % | worst band median | µs/track |
|---|---|---|---|---|---|---|---|---|
| HLT2 adaptive Nystrom (LHCb reference) | linear | finished |  |  | 10.1 | 38.8 | 16.2 | 5.42 |
| HLT2 adaptive Cash-Karp | linear | finished |  |  | 22.2 | 112 | 47.3 | 2.53 |
| HLT1 parametrised step | linear | finished |  |  | 55.6 | 2.33e+03 | 1.17e+03 | 0.519 |
| HLT2 adaptive Nystrom, as LHCb runs it, against the cubic reference | cubic | finished |  |  | 25.9 | 169 | 83.7 | 5.41 |

## no network

| name | map | state | rounds | last val. median | test median | test 95 % | worst band median | µs/track |
|---|---|---|---|---|---|---|---|---|
| Straight line + polynomial of degree 7 | cubic | finished |  |  | 6.69 | 622 | 49.8 | 0.615 |
| Straight line + polynomial of degree 9 | cubic | finished |  |  | 4.35 | 386 | 32.7 | 1.13 |
| Nystrom, 4 steps + polynomial of degree 7 | cubic | finished |  |  | 6.49 | 310 | 41.6 | 2.02 |
| Nystrom, 4 steps, alone | cubic | finished |  |  | 200 | 2.84e+03 | 760 | 1.31 |
| Nystrom, 8 steps + polynomial of degree 7 | cubic | finished |  |  | 3.68 | 226 | 21.9 | 3.4 |
| Nystrom, 8 steps, alone | cubic | finished |  |  | 13 | 211 | 50.7 | 2.69 |
| Nystrom, 12 steps + polynomial of degree 7 | cubic | finished |  |  | 3.33 | 57.3 | 17.6 | 4.4 |
| Nystrom, 12 steps, alone | cubic | finished |  |  | 8.51 | 72.5 | 33.3 | 3.72 |
| Straight line + polynomial of degree 7 | linear | finished |  |  | 27.5 | 512 | 115 | 0.62 |
| Nystrom, 6 steps + polynomial of degree 7 | linear | finished |  |  | 6.17 | 132 | 38.6 | 1.33 |
| Nystrom, 6 steps, alone | linear | finished |  |  | 20.6 | 216 | 79.2 | 0.675 |
| Nystrom, 8 steps + polynomial of degree 7 | linear | finished |  |  | 3.09 | 103 | 19.7 | 1.58 |
| Nystrom, 8 steps, alone | linear | finished |  |  | 19.4 | 141 | 63.7 | 0.911 |
| Nystrom, 12 steps + polynomial of degree 7 | linear | finished |  |  | 2.15 | 36.8 | 12.3 | 2.14 |
| Nystrom, 12 steps, alone | linear | finished |  |  | 15.2 | 94.7 | 54.9 | 1.45 |

## stage-2 network

| name | map | state | rounds | last val. median | test median | test 95 % | worst band median | µs/track |
|---|---|---|---|---|---|---|---|---|
| cubic_none_2x64 | cubic | training | 210 | 26.1 | 26 (round 200, provisional) | 741 | 157 | 0.527 |
| cubic_nystrom_12_2x64 | cubic | training | 44 | 1.75 |  |  |  |  |
| cubic_nystrom_4_2x32 | cubic | finished (plateau) | 43 | 5.26 |  |  |  |  |
| cubic_nystrom_4_2x64 | cubic | training | 9 | 5.24 |  |  |  |  |
| cubic_nystrom_8_2x32 | cubic | training | 11 | 2.89 |  |  |  |  |
| cubic_nystrom_8_2x64 | cubic | training | 9 | 2.75 |  |  |  |  |
| cubic_straight_line_2x128 | cubic | training | 97 | 3.65 | 3.63 (round 91, provisional) | 187 | 26.5 | 2.2 |
| cubic_straight_line_2x32 | cubic | finished (plateau) | 109 | 1.06e+04 | 1.04e+04 (round 109, provisional) | 1.03e+05 | 3.62e+04 | 0.262 |
| cubic_straight_line_2x32_seed_2 | cubic | training | 15 | 6e+03 |  |  |  |  |
| cubic_straight_line_2x64 | cubic | finished (plateau) | 196 | 4.46 | 4.5 (round 179, provisional) | 279 | 43.5 | 0.524 |
| cubic_straight_line_2x64_fixed_scale | cubic | finished (at its cap) | 200 | 20.4 | 23.2 (round 157, provisional) | 818 | 162 | 0.524 |
| cubic_straight_line_2x64_squared_loss | cubic | training | 41 | 435 | 424 (round 31, provisional) | 4.45e+03 | 1.87e+03 | 0.528 |
| cubic_straight_line_3x64 | cubic | training | 168 | 4.33 | 4.8 (round 128, provisional) | 227 | 42.4 | 0.908 |
| linear_nystrom_12_2x32 | linear | training | 11 | 1.53 |  |  |  |  |
| linear_nystrom_12_2x64 | linear | training | 9 | 1.52 |  |  |  |  |
| linear_nystrom_6_2x32 | linear | training | 11 | 3.76 |  |  |  |  |
| linear_nystrom_6_2x64 | linear | training | 9 | 3.52 |  |  |  |  |
| linear_nystrom_8_2x32 | linear | training | 12 | 2.26 |  |  |  |  |
| linear_nystrom_8_2x64 | linear | training | 10 | 1.92 |  |  |  |  |
| linear_straight_line_2x128 | linear | training | 83 | 26.2 | 25.7 (round 77, provisional) | 241 | 99.2 | 2.23 |
| linear_straight_line_2x64 | linear | training | 133 | 32.3 | 32 (round 123, provisional) | 514 | 150 | 0.534 |

Source: Fast_extrapolator/stage_2_training/results/runs.csv and polynomial_baseline.csv, /data/bfys/gscriven/fast_extrapolator_store/runs/*/state.json, condor_q.
