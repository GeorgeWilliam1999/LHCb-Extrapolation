# Stage 0: the compiled program and the LHCb baselines

Question of this stage: what do the methods of LHCb cost and how accurate are
they on our tracks, and what do one field read and one network call cost in
compiled code? No network is trained here.

The results are in `RESULTS.md` (written from the tables) and in the notebook
`stage_0_baselines.ipynb`.

## What was built

| Part | Where |
|---|---|
| The compiled program `fx`: every method is one function, state in, state out | `../cpp/` |
| Field reads: the present package (trilinear and tricubic), LHCb's Detector, Allen | `../cpp/field/field_map.h` |
| RK4 with equal steps | `../cpp/classical/integrators.h` |
| Copies of the LHCb integrators: adaptive Cash-Karp and adaptive Nystrom (STEP) of Rec; adaptive and fixed Nystrom of Allen | `../cpp/lhcb/rec_cash_karp.h`, `rec_step.h`, `allen_nystrom.h` |
| Copy of the parametrised step of HLT1 (UT to SciFi, no field read) | `../cpp/lhcb/allen_utt.h` |
| Forward pass of the trained networks: whole-crossing network, chained stage network (direct states, straight line, physics layer) | `../cpp/networks/networks.h` |
| The export of field maps, tracks and weights to flat files; Python copies of the LHCb integrators | `../python/fastex/` |
| The Rec extrapolator test, run in the LHCb software | `../lhcb_software/rec_extrapolator_test/` |

## Set-up

| Choice | Value |
|---|---|
| Tracks | The 13,447 test tracks of the present track sets (real momentum spectrum, 1.6 to 187 GeV). Beside them the 1,000 tracks that are flat in momentum from 2 to 42 GeV |
| Crossing | 2648.2 mm to 7826 mm. The parametrised step of HLT1 starts on its own plane, 2665 mm |
| Field map | v8r1, magnet up: the map of data taking in 2025. The copies of the LHCb methods read it as their originals do (trilinear, single precision) |
| Reference | The stored reference end states of the track sets (sixth-order Runge-Kutta, steps of 0.1 mm), checked here against RK4 with 4,096 steps |
| Error | The distance on the last plane between a method and the reference, in um |
| Time | One track a call, one thread, one pinned core. A pass is one loop over all tracks; the passes of the methods take turns in a shuffled order, up to 15 times; the lowest and the median time a track are kept. 200 calls a method are also timed one by one after a sweep of a 64 MB buffer (cold caches) |
| Machine | stbc-i3, AMD EPYC 7702P, a shared login node. The core is the idlest one whose cache group is idle (`scripts/quietest_core.py`) |
| Compiler | g++ 11.5, `-O3 -march=native -fno-math-errno`; no `-ffast-math` |

## Scripts and their outputs

| Script | Output |
|---|---|
| `../python/fastex/export.py` | the flat files in the store of this experiment (`exports/`) |
| `scripts/gate_compiled_against_python.py` | `results/gate_compiled_against_python.csv` |
| `scripts/rec_copy_against_the_rec_reference.py` | `results/rec_copy_against_the_rec_reference.csv` |
| `scripts/allen_copy_against_the_step_copy.py` | `results/allen_copy_against_the_step_copy.csv` |
| `scripts/check_the_transport_matrix_of_the_copy.py` | `results/transport_matrix_of_the_copy.csv` |
| `scripts/score_the_parametrised_step.py` | `results/parametrised_step.csv` |
| `fx micro_field`, `fx micro_mlp` | `results/micro_field.csv`, `results/micro_mlp.csv` |
| `scripts/methods.py` | the list of the methods that are scored and timed |
| `scripts/score_and_time_the_methods.py accuracy` | `results/accuracy.csv` |
| `scripts/score_and_time_the_methods.py time CORE` | `results/timing.csv` |
| `scripts/score_and_time_the_methods.py table` | `results/methods.csv`, `results/methods.md` |
| `scripts/plots.py` | `figures/time_against_error.png`, `figures/cost_of_a_network_call.png`, `figures/error_by_momentum.png` |
| `scripts/write_results_note.py` | `RESULTS.md` |
| `scripts/make_notebook.py` | `stage_0_baselines.ipynb` |
| `../python/tests/` | the gates of the Python copies and of the classical integrators (pytest) |

## Limits

- The copies of the two Allen integrators and of the Allen parametrised step
  are not compared with a run of Allen. They are compared with Python copies,
  with the verified copy of the HLT2 integrator on the grid of the LHCb test,
  and by their results on our tracks.
- The transport matrix is computed for the two HLT2 integrators only. The
  networks do not have theirs yet; that is stage 3.
- Our tracks have no material and no energy loss, as in the whole project. The
  HLT1 parametrised step is scored without the terms that Allen adds for the
  material.
- The times are from a shared login node. A run on a farm node with one
  reserved core is not done yet.
