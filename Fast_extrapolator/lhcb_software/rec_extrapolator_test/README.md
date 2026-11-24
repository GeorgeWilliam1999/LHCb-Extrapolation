# The extrapolator test of Rec, run in the LHCb software

Purpose: to check our compiled copies of the two HLT2 integrators against the
real tools, in their results and in their time: the adaptive Cash-Karp
integrator (`TrackRungeKuttaExtrapolator`, copy in `cpp/lhcb/rec_cash_karp.h`)
and the adaptive Runge-Kutta-Nystrom integrator (`TrackSTEPExtrapolator`, copy
in `cpp/lhcb/rec_step.h`).

The Rec release has a test of its extrapolators (Rec v41r3,
`Tr/TrackExtrapolators/tests/options/test_extrapolators.py`). The algorithm
`ExtrapolatorTester` carries a fixed grid of 1,331 states from z = 3000 mm to
z = 7000 mm (q/p from -0.0004 to +0.0004 per MeV, tx from -0.3 to 0.3, ty from
-0.25 to 0.25) and prints the final states. `options.py` is that test cut down
to one tool with its default properties; the environment variable
`FX_EXTRAPOLATOR` chooses it (`rk`, `step`, or `linear` for a straight line).

| File | What it is |
|---|---|
| `options.py` | the Gaudi options; environment variables choose the number of events, the prints and the tool |
| `run.sh <log> [core]` | runs `lb-run -c x86_64_v3-el9-gcc15-opt Rec/v41r3 gaudirun.py options.py` |
| `summarise.py <node>` | the times of the Gaudi jobs beside the times of our copy, `output/summary_<node>.csv` |
| `output/print_one_event.log`, `output/print_one_event_step.log` | one event with prints: the 1,331 final states of the real Cash-Karp tool and of the real Nystrom tool |
| `output/time_rk_*.log`, `output/time_step_*.log`, `output/time_linear_*.log` | 300 events without prints, with each integrator and with the straight-line extrapolator |
| `output/copy_on_the_grid_<node>.csv` | our copy on the same states, `fx time` |

## Results (2026-11-24, stbc-i2, AMD EPYC 7551P, core 5)

**The job.** The conditions of the test are a magnet-down simulation tag. The
log shows the field map that the software opened:
`DBASE/FieldMap/v8r1/cdf/field.v5r0.down.bin`, scale factor 1.

**Results of the tools.** Our run of the Cash-Karp tool prints the same 1,331
final states as the reference output in the Rec release
(`tests/refs/test_extrapolators.yaml`), digit for digit. Each of our two
compiled copies, with the field read of LHCb on the map `field.v5r0.down`,
prints the same 1,331 x 4 numbers as its real tool
(`stage_0_baselines/results/rec_copy_against_the_rec_reference.csv`).

**Time of the tool.** The time of one call in Gaudi is the time of the test
algorithm with the integrator minus its time with the straight-line
extrapolator, over 1,331 states. The Rec test asks for the transport matrix in
every call.

| | Time for one state (ns), lowest to highest |
|---|---|
| Cash-Karp in Gaudi (Rec v41r3), with the transport matrix | 8,127 to 8,154 (two jobs) |
| Our Cash-Karp copy, with the transport matrix | 7,242 to 7,627 |
| Our Cash-Karp copy, state only | 4,548 to 5,048 |
| Adaptive Nystrom in Gaudi (Rec v41r3), with the transport matrix | 9,457 to 9,549 (two jobs) |
| Our adaptive Nystrom copy, with the transport matrix | 9,134 to 9,544 |
| Our adaptive Nystrom copy, state only | 6,013 to 6,390 |

(`output/summary_stbc-i2.csv`.) The real Cash-Karp tool is 12 % slower than
our copy on the same node, core and states; the real Nystrom tool is 3.5 %
slower. So the time of a copy in our program stands for the time of the Rec
tool to that accuracy; the copies are a little too fast, not too slow.

The grid of the test is harder than our tracks: it has momenta down to 2.5 GeV
at slopes up to 0.3. On it the Cash-Karp integrator reads the field 79 times
a state (43 times on our test tracks) and the Nystrom integrator 63 times (56
times on our test tracks). So these times are not the time of a track of ours.

**Which tool runs where** (read from the configuration of Rec v41r3,
`Rec/RecoConf`): the track fit of HLT2 takes `TrackMasterExtrapolator`, whose
tool for long distances is `TrackRungeKuttaExtrapolator` by default
(`TrackDistanceExtraSelector`); the RICH reconstruction takes
`TrackSTEPExtrapolator`. The LHCb answer of 2026-11-24 (Maarten Van Veghel)
names the adaptive Runge-Kutta-Nystrom integrator as the reference for HLT1
and HLT2.

## Not done

- Allen has a test algorithm with the same grid (`extrapolate_states`), but
  the release has no reference output for it, and the algorithm writes its
  states to the device only. The copies of the Allen integrators and of the
  Allen parametrised step are checked against Python copies and by their
  results on our tracks, not against a run of Allen.
- The transport matrix of the copy cannot be compared with Rec: the test
  prints none. It is compared with finite differences of a fine integration
  (`stage_0_baselines/scripts/check_the_transport_matrix_of_the_copy.py`).
