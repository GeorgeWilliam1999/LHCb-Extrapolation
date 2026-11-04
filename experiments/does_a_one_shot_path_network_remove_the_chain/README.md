# Does a one-shot path network remove the chain?

**Last updated:** 2026-11-04 · **State:** agreed; stage 1 done on branch `one_shot_path` (items 1 to 5) · **Agreed with George:** 2026-11-03 22:10, as an exploratory study

## What this directory is

An experiment of the package `rkpinn`. One network call emits the whole crossing of a track at once: every stage state and every plane state of the 64-step layout. One Newton sweep of the stage equations, applied to all stages in parallel, then corrects it. The experiment measures what the one-shot network reaches alone, what the sweep adds, and what either costs per event against the chained network.

The idea and its equations are in the assessment of 3 Nov 2026 (section 7 of "A global phase-space map from UT to SciFi"). George agreed on 2026-11-03: exploratory, not the main point.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file: the question, the design, the runs, the criteria, what is added to the package, how the runs are watched | planned |
| `configurations/` | one configuration file per run | built; names checked against the package on 2026-11-04 |
| `farm/` | what puts the runs on the farm and keeps them going; `monitor.sh` prints the state of every run on one screen | planned |
| `analysis/` | tables, figures and the results note, made after the runs end | planned |

## The question

Can one supervised network call, emitting all 128 stage states and 64 plane states of a track at once, reach the endpoint accuracy of the 64-step chain; and if not, does one parallel Newton sweep from its output reach the exact-scheme floor at a lower cost per event than the chain?

## Why it is asked

The chained network makes 64 network calls and 128 field lookups one after the other. Batching over the tracks of an event widens each call; it does not shorten the sequence. A path network makes one call per event with all tracks in one batch, and the sweep makes its 128 lookups in one gather. The Kalman study (`../does_a_kalman_like_gain_remove_the_bias/`) measured that one sweep from a forward-Euler guess reaches 0.146 µm, the exact-scheme floor. So the sweep is known to work from a crude start. What is not known is whether a network start is good enough to skip the sweep, and what the three options cost.

Definitions used below:

| Term | Meaning |
|---|---|
| path | the 64 plane states and 128 stage states of one track on the 64-step, 2-stage layout: 192 states, 768 numbers |
| path network | one network, inputs (x, y, tx, ty, q/p) on the first plane, outputs the path |
| sweep | one Newton step on the full system of stage equations G(U) = 0, all stages evaluated at once |
| exact-scheme floor | 0.147 µm: the endpoint error of the Gauss-Legendre scheme solved exactly at 64 steps and 2 stages |

## The design

| Block | Setting | Why |
|---|---|---|
| Tracks | `12a8d35c3165`: 11,567 / 1,463 / 1,452 train / validation / test | the standard set; every number is comparable with the other experiments |
| Layout | 64 steps of 80.9 mm, 2 stages, z 2648.2 to 7826.0 mm | the cell every comparison number exists for |
| Labels | the exact-scheme stage states and the reference states on the planes, already in the store (`exact_states`, `reference_states_on_planes`) | no new reference integration |
| Network | `path_network`: body of 2 tanh layers, width 256 or 512, 5 inputs, 768 outputs | the chain's body is 128 wide with 8 outputs; a 768-output head needs more width. Two widths say whether capacity limits it |
| Output form | straight line from the start state, per-track scale, as in the chain | the form that restored the old accuracy; the network learns the bend only |
| Loss | `supervised_path`: mean squared error over all 192 states and 4 components, each component divided by its spread | one number per state, nothing weighted by hand |
| Uncertainty | one run adds a diagonal covariance per plane and trains on the Gaussian likelihood | George's "Gaussian peaks": the error is encoded with the path |
| Protocol | `start_states_of_the_tracks`, all 11,567 training tracks every round | a one-shot network trains on start states; there are no drawn stage states |
| Optimiser | L-BFGS with restarts, as the chain; one run with self-scaled Broyden | the optimiser that trained every chain; Broyden was 7 to 8 times better on the chain and is checked once here |
| Stopping | validation plateau, window 10, tolerance 5 %, held 3; cap 200 rounds | the standard rule; the cap is high because a round here has no field lookups and is short |
| Control | `whole_crossing_network`, endpoint only, same width, `supervised_endpoint` loss, layout {1 step, 0 stages}, output form `direct_states` scaled by the spread of the target end states | says whether emitting the path helps the endpoint, or only serves the sweep. It differs from P1 in its output form as well: the configuration check accepts a whole-crossing network only on the layout {1, 0}, and the network binds its form with no field map, which the straight-line form refuses. Accepted in the thread on 2026-11-04 |

The loss contains no equation of motion. The runs are supervised. This is said plainly in every report: the chain is label-free, these runs are not.

## The runs

| Run | Network | Width | Loss | Optimiser | Configuration | Differs from P1 in |
|---|---|---|---|---|---|---|
| P1 | path | 256 | supervised path | L-BFGS | `path_256_lbfgs.yaml` | — |
| P2 | path | 512 | supervised path | L-BFGS | `path_512_lbfgs.yaml` | `network.width` |
| P3 | path | 256 | Gaussian path | L-BFGS | `path_256_gaussian_lbfgs.yaml` | `loss` |
| P4 | path | 256 | supervised path | self-scaled Broyden | `path_256_broyden.yaml` | `training.optimiser` |
| E1 | whole crossing, endpoint only | 256 | supervised endpoint | L-BFGS | `endpoint_256_lbfgs.yaml` | `network` (kind and output form), `steps`, `loss`, `target` (the control; by necessity) |

Each run has one seed. Differences under 2x between runs are not claimed.

Comparators computed on the login node, no training:

| Comparator | Endpoint radial median, test | Source |
|---|---|---|
| Chain, straight-line form, L-BFGS, 64/2 | 1.83 µm | `25dfa17ef168` |
| Chain, straight-line form, self-scaled Broyden | 0.53 to 0.65 µm at plateau | optimiser tranche 1 |
| Classic RK4, 64 steps | 1.08 µm | step-length study |
| Forward-Euler path, no sweep | 10.8 mm (60 test tracks; 13.7 mm on the toy dipole) | this experiment, stage 1 |
| Forward-Euler path + one sweep | 0.60 µm (60 test tracks) | this experiment, stage 1 |
| Forward-Euler path + two sweeps | 0.1489 µm (60 test tracks), the exact scheme | this experiment, stage 1 |
| Euler stages corrected step by step along the chain (not one sweep of a path) | 0.146 µm | Kalman study |
| Exact scheme | 0.147 µm | store |
| LHCb parametrised extrapolator | tens of µm, about 40 operations per track | Billoir et al. 2021 |

## The criteria, written before the runs are submitted

All errors are the endpoint radial median on the 1,452 test tracks against the sixth-order reference, as in every report.

| # | Criterion | Read against |
|---|---|---|
| 1 | The best path run beats the forward-Euler path by 100x or more | Euler endpoint error on the real field, measured first |
| 2 | The best path run alone is below 100 µm | the parametrised extrapolator; below this a no-lookup method has a use |
| 3 | The best path run alone is below 2 µm | the chain; at this level the sweep can be skipped at the chain's accuracy |
| 4 | Path + one sweep is at or below Euler + one sweep on the real field (0.60 µm on 60 test tracks) for every run; path + two sweeps reaches the exact-scheme floor (0.15 µm) | the comparator is Euler + one sweep; the floor is reached at two sweeps (reworded 2026-11-04 after the stage 1 measurement; ruled in the thread) |
| 5 | In P3, 68 ± 3 % of test tracks lie within one predicted sigma in x at the end plane | the covariance is calibrated |
| 6 | Time per event of 100 tracks, measured at equal accuracy: chain, path alone, path + sweep, Euler + sweep, RK4 | the speed claim is made only at equal accuracy |

What the outcomes mean:

| Outcome | What follows |
|---|---|
| 3 passes | a one-call, no-lookup extrapolator at chain accuracy. Timing decides whether it is a result for the paper's second part |
| 2 passes, 3 fails | the network is a faster coarse map than the parametrised extrapolator, and the sweep gives the floor. Timing of path + sweep against Euler + sweep decides whether the network earns its call |
| 1 passes, 2 fails | the network learns the bend but not to micrometres: the one-shot form is confirmed as the hard case of the step-length story. Reported; nothing further |
| 1 fails | supervised one-shot regression over 5.2 m does not train at this size. Reported with the loss curves |
| 4 fails for any run | the sweep depends on its start; the Kalman result does not generalise. Reported first; nothing else is read until it is understood |

## What is added to the package

All additions follow WORKFLOW.md section 6: a registered component, a README line, gates in `tests/`, no change to an existing default. The base is the stbc checkout at `1a9620a9` (the Kalman gain), not GitHub `184cf50e`.

| # | Component | File | What it does | Gates |
|---|---|---|---|---|
| 1 | target `reference_path` | `targets/reference_path.py` | labels of shape (n, 64, 3, 4): the exact-scheme stage states and plane states for the layout; `can_be_trained_on = True`; declares `needs_layout` | shape; equals the stored exact states; refuses a layout the store lacks |
| 1b | `exact_states --keep-stages` and `read_exact_states_with_stages` | `track_data/exact_states.py` | the exact scheme's stage states kept beside its plane states, for the training and validation splits too (the store holds the test split only) | round trip through the store |
| 2 | network `path_network` | `networks/path_network.py` | body 5 to 64 x 3 x 4 outputs; output form applied per state; `predict` fills a `PredictedTrack` with `layout` set and `end_state_was = predicted`; `whole_track` is `predict`; `has_stages = True`; a flag `trains_on_start_states = True` so the start-state protocol accepts it | shapes; same seed gives the same weights twice; a straight-line output form at zero weights gives the straight line |
| 3 | loss `supervised_path` | `losses/supervised_path.py` | mean over states and components of squared error divided by the spread per component; `needs_labels = True`; setting `uncertainty: none` or `diagonal`; with `diagonal` the network has 64 x 4 more outputs and the loss is the Gaussian negative log-likelihood | value on a known pair; the Gaussian loss at the right sigma is lower than at a wrong one |
| 4 | `training_protocols.start_states_of_the_tracks` | one line | accepts a network with `trains_on_start_states` | existing gate extended |
| 5 | `evaluation/newton_sweep.py` | new | one Newton step on G(U) = 0 from any `PredictedTrack` with a layout; reuses the analytic gain of the Kalman experiment; counts lookups and Jacobians | from the exact states the step changes nothing; from the Euler path it reaches 0.146 µm |
| 6 | `report_of_a_run --with-one-sweep` | option | the standard report, then the same tables after one sweep, with the counts | report runs on a small path run |
| 7 | `evaluation/time_per_event.py` | new | times chain, path, path + sweep, Euler + sweep and RK4 on batches of 100 tracks, CPU, with the same code paths the reports use; GPU column only if a GPU node is available | runs; prints a table |
| 8 | `equation_of_motion/toy_dipole.py` (optional) | new | the toy dipole of the assessment as a registered field, for gates that run without CVMFS | field at known points |

Drafts of items 1, 2, 3 and 5 were written in the cloud on 2026-11-03 off GitHub `184cf50e` (branch `claude/project-thread-0pnl3r`). On 2026-11-04 they were ported onto the stbc checkout, branch `one_shot_path`, with items 1b and 4 and the trainer lines, and gated (`test_reference_path.py`, `test_path_network.py`, `test_supervised_path.py`, `test_newton_sweep.py`). Items 6 and 7 are not built. Two differences from the plan: the target is built with the store as well as the layout, because a track set does not know its store; and the output form of the path network is bound with the whole region as one step, so the per-track scale is measured over the whole region. Lines the trainer needs: a target that declares `needs_layout` is built with the layout; the start-state protocol accepts a network that declares `trains_on_start_states`; `context.scale` learns `spread_of_target_states`; PredictedTrack gets an optional `extra_outputs` attribute for the Gaussian loss.

Expected size: about 600 lines of package code and 300 lines of gates. One day of work on stbc, then the 505 gates plus the new ones must pass before any run.

## The stages, end to end

| Stage | What | Where | Takes | Needs from George |
|---|---|---|---|---|
| 0 | Toy pilot: the path network and the sweep on the toy dipole, standalone scripts in `/mnt/project-files/global_map_assessment/` | cloud | about 1 hour | agreement to the design |
| 1 | Package additions 1 to 7, gates pass | stbc, a branch from `1a9620a9` | 1 day | a Remote Control session for this thread |
| 2 | Smoke run: P1 made small (width 32, 2 rounds, 500 tracks) on a scratch store, then its report with the sweep | stbc login node | 15 minutes | nothing |
| 3 | Forward-Euler path on the real field: comparator for criterion 1 | stbc login node | minutes | nothing |
| 4 | Farm: five runs, `keeper.sh` | farm, via `make_code.sh` at the commit of stage 1 | 1 to 3 days (a round is pure regression, expected under 10 minutes; 200 rounds cap) | the word "submit" in the thread; George starts the keeper himself |
| 5 | Reports with `--with-one-sweep`; timing; `analysis/` with tables, figures and results note | stbc | half a day | nothing |
| 6 | Findings discussed in the thread; then a Provisional Notion write-up with plots and equations | cloud | half a day | his reading |

## How the runs are watched

| What | How | When |
|---|---|---|
| State of every run on one screen | `farm/monitor.sh <store> <logs>`: per run its key, state, rounds done, minutes per round, last and best validation error, farm job status | at every check, and by George at any time |
| Scheduled checks | reminders set in the thread: 6 hours after submission (round time and first errors), then daily, then every 3 days; each check runs `monitor.sh` over Remote Control and updates the checklist; a message is posted only when a run ends, a job is held, or an error changes by 2x | from submission |
| Held or failed jobs | `keeper.log` and the `.err` files; a job held for a reason the keeper cannot clear is reported the same day | at every check |
| End of a run | the standard report with the sweep, posted as a table; the comparators beside it | when `state.json` says plateaued or at its cap |
| A run at its cap without a plateau | reported; not extended without George's ruling | when it happens |

## To run

See `farm/README.md` of the pilot experiment for the commands; they are the same with this folder's name. The report of a run, with the sweep, from `RK_Pinn_module`:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src /data/bfys/gscriven/conda/envs/TE/bin/python \
    -m rkpinn.run_record.report_of_a_run --store /data/bfys/gscriven/rkpinn_store \
    --run <key> --round last --split test --with-one-sweep
```

## Findings

None of the runs yet.

**Stage 1, 2026-11-04 (package, not a run).** Measured by the gate `tests/test_newton_sweep.py` and on 60 test tracks, real field, 64 steps and 2 stages, median radial endpoint error against the reference:

| Start | Alone | After one sweep | After two sweeps |
|---|---|---|---|
| exact scheme | 0.1489 µm | unchanged (largest change 9e-13 mm) | unchanged |
| forward-Euler path | 10.8 mm | 0.60 µm | 0.1489 µm |

One global sweep from the Euler path does not reach the exact-scheme floor; two do. The 0.146 µm of the Kalman study (Comparators table, "Forward-Euler path + one sweep") is from a correction made step by step along the chain, each step from Euler stages on the corrected state of the step before. It is not one sweep of a whole Euler path. Criterion 4 was reworded on 2026-11-04 to use this comparator.
