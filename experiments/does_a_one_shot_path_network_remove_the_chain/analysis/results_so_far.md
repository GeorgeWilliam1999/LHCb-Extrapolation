# Results so far

**Last updated:** 2026-11-06 · **State:** draft; P2 is still training. Not a results note.

Every error is the radial error at the end of the region against the sixth-order reference, on the 1,452 test tracks, unless a line says otherwise. The path runs are supervised on the exact scheme's states; the chain is not.

## The runs

| Run | Key | State | Rounds | Round scored | Validation error, best (larger of x and y, median) |
|---|---|---|---|---|---|
| P1 path, width 256 | `25934ed22604` | at its cap, no plateau; still falling (213.4 µm at round 30, 180.1 µm at round 40) | 40 | 40 (best and last) | 180.1 µm |
| P2 path, width 512 | `45093f0c80d6` | training | 35 of 40 | not scored | 197.6 µm at round 35 |
| E1 control, whole crossing | `dd9acf59ac88` | at its cap, no plateau; still falling | 40 | 40 (last; best is 39) | 70.6 µm |
| P3, P4 | | not submitted | | | |

Validation curves, larger of x and y, median, µm, by round:

| Round | 1 | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 |
|---|---|---|---|---|---|---|---|---|---|
| P1 | 1,653 | 532 | 358 | 291 | 252 | 223 | 213 | 191 | 180 |
| P2 | 1,874 | 618 | 397 | 303 | 284 | 239 | 228 | 198 | |

## The endpoint, test split

| Method | Median | p95 | Against the exact scheme, median | Field lookups per track | Report |
|---|---|---|---|---|---|
| P1 alone | 196.9 µm | 1,263 µm | 196.9 µm | 16 | `runs/25934ed22604/reports/round_0040_at_4f453087` |
| P1 + 1 sweep | 0.1455 µm | 1.23 µm | 0.0015 µm (p95 0.13 µm) | 16 + 128 | the same |
| P1 + 2 sweeps | 0.1465 µm | 1.17 µm | 1.2e-10 µm | 16 + 256 | the same |
| E1 alone (no path, no sweep possible) | 78.9 µm | 720 µm | | 0 | `runs/dd9acf59ac88/reports/round_0040_at_4f453087` |
| Euler path alone | 12.1 mm | 38.2 mm | 12.1 mm | 64 | stage 2 |
| Euler + 1 sweep | 1.01 µm | 35.6 µm | 0.99 µm | 64 + 128 | stage 2 |
| Euler + 2 sweeps | 0.1465 µm | 1.17 µm | 2e-8 µm | 64 + 256 | stage 2 |
| Exact scheme | 0.1465 µm | 1.17 µm | 0 | | store |
| Chain `25dfa17ef168` | 1.83 µm | | | 1,152 | its report |

P1 alone, by momentum band (median / p95, µm): below 3 GeV 504 / 2,295; 3 to 8 GeV 226 / 1,533; 8 to 20 GeV 142 / 551; 20 to 50 GeV 136 / 369; above 50 GeV 378 / 2,747. E1 alone: 227 / 2,426; 109 / 932; 45.3 / 300; 29.5 / 90.5; 32.2 / 65.4.

## Time per event

`evaluation/time_per_event.py`, 1 thread, login node, 13 events of 100 test tracks, P1 at round 40.

| Method | ms per event | ms per track | Median of the timed tracks | Field lookups per track |
|---|---|---|---|---|
| chain `25dfa17ef168` | 292.7 | 2.93 | 1.83 µm | 1,152 |
| P1 alone | 7.2 | 0.072 | 202 µm | 16 |
| P1 + 1 sweep | 82.8 | 0.83 | 0.144 µm | 144 |
| Euler + 1 sweep | 171.4 | 1.71 | 1.05 µm | 192 |
| Euler + 2 sweeps | 246.6 | 2.47 | 0.145 µm | 320 |
| RK4, 64 steps | 98.4 | 0.98 | 1.11 µm | 256 |

## The criteria of the README, so far

| # | Criterion | P1 | P2 | P3 | E1 (control) |
|---|---|---|---|---|---|
| 1 | best path run beats the Euler path by 100x or more | **fail**: 12.1 mm / 196.9 µm = 61x | not scored yet | not run | not a path run |
| 2 | best path run alone below 100 µm | **fail**: 196.9 µm | not scored yet | not run | (78.9 µm; the control, not a path run) |
| 3 | best path run alone below 2 µm | **fail** | not scored yet | not run | (fail) |
| 4 | path + 1 sweep at or below Euler + 1 sweep (1.01 µm), and path + 2 sweeps at the floor (0.15 µm) | **pass**: 0.1455 µm after one sweep, at the floor already; 0.1465 µm after two | not scored yet | not run | no path to sweep |
| 5 | 68 ± 3 % of tracks within one predicted sigma | n/a | n/a | not run | n/a |
| 6 | time per event at equal accuracy | measured: P1 + 1 sweep 0.83 ms per track at 0.144 µm, against Euler + 2 sweeps 2.47 ms at 0.145 µm (3.0x faster); the chain is 2.93 ms at 1.83 µm | not scored yet | not run | |

Outcome so far, read against the README's table: criterion 1 fails, so for the network alone the README reads "supervised one-shot regression over 5.2 m does not train at this size". P1 was still improving at its cap, and P2 has not ended. Criterion 4 passes for P1: one sweep from P1 reaches the exact-scheme floor (0.0015 µm from the exact scheme, median), where one sweep from Euler stops at 1.01 µm.
