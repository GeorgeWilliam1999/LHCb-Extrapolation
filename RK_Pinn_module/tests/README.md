# Tests: the gates

**Last updated:** 2026-11-12

## What this directory is

Every gate of the package, run with pytest. A change is not finished until all of them pass.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file | built |
| `test_every_directory_is_described.py` | fails if a directory has no README, if a README does not name something in its directory, or if something marked built does not exist | built |
| `frozen_code.py` | not a gate: where the parity gates find the frozen code, the frozen tracks and the stored exact states, at the root of the project or under `Archive/`. The only file that inserts into the import path | built |
| `test_cubic_field_map.py` | the cubic maps: the linear maps unchanged; the spline passes through every grid point; equal to scipy's map_coordinates inside and beyond the edges; the torch field equal to the numpy one; a gradient continuous across a cell face where the linear one jumps; the same field as the linear map where tracks pass | built |
| `test_field_map_matches_the_frozen_code.py` | the hash of the map file; field values identical to the frozen loader on 200,000 points, both polarities; the differentiable field identical to the frozen one | built |
| `test_equation_of_motion_matches_the_frozen_code.py` | rates identical to the frozen code on 500 real states; torch rates and their gradient identical to the frozen code; torch equals numpy | built |
| `test_gauss_legendre_tableau_matches_the_frozen_code.py` | the tableau identical to the frozen one to the last bit at 16 stages and eleven other stage counts; the five identities | built |
| `test_exact_collocation_matches_the_frozen_code.py` | one step and the chain identical to the frozen solver run on this machine; the stored exact states reproduced to 1e-11 mm; every solve converged | built |
| `test_sixth_order_reference_matches_the_frozen_code.py` | end states identical to the frozen code over 40 mm and 2,589 mm; the stored tracks reproduced to the last bit on the first 33 of their 257 planes; fitted order six on a smooth field; forwards then back closes | built |
| `the_store.py` | not a gate: where the gates find the store, the settings of the project's track set and its key | built |
| `test_store_keys_and_manifest.py` | a key depends on the content and nothing else; the store overwrites nothing; the manifest lists a thing once; uncommitted changes are refused or recorded. Uses a temporary store | built |
| `test_tracks_match_the_frozen_file.py` | every cut removes what the frozen record says; 200 particles rebuilt now and the whole track set in the store are identical to the frozen file | built |
| `test_draw_of_training_states_matches_the_frozen_code.py` | the same seed draws the same states as the frozen trainer; the same number on every start plane | built |
| `test_exact_states_in_the_store.py` | the store holds the exact states of the test split at 2, 64, 128 and 256 steps and 2, 4, 8 and 16 stages; tracks solved again equal the store; the store agrees with the frozen files to 1e-11 mm | built |
| `building_blocks.py` | not a gate: the first round's states, the networks and the equations of motion the gates of the networks and the losses share | built |
| `test_predicted_track.py` | a predicted track holds what was put in it and refuses wrong shapes; the exact scheme and the reference integrator fill it, identical to the store | built |
| `test_networks.py` | the stage network and the whole-crossing network identical to the frozen one-step network; with the straight-line form, the stage network and its chain identical to the frozen `ChainNetwork` of the paper; the global scale is the median of the per-track scale; the summed end state; double precision; snapshots | built |
| `test_losses.py` | the five gates every loss must pass; the pooled loss identical to the frozen loss, value and gradient; the weights identical to the frozen weights; no loss imports a network | built |
| `test_targets.py` | every target gives the states it names, one row per track, in the order of the tracks | built |
| `small_run.py` | not a gate: a small store in a temporary folder and a small configuration, for the gates of the training | built |
| `test_configuration_and_run_key.py` | a configuration with an unknown setting, a missing setting or an unknown name is refused; the run key changes with every setting and not with the cap on the rounds | built |
| `test_default_configuration.py` | the default configurations are complete, valid and pinned to their run keys; the LHCb default is run A of the pilot | built |
| `test_training.py` | the optimiser identical to the frozen trainer; the stopping rule identical to the frozen rule; a run stopped and resumed identical to one not stopped; a second writer refused; a snapshot never overwritten; a run extended; the straight-line form trains with either scale and records it | built |
| `old_runs.py` | not a gate: where the gates of the evaluation find the six old runs and the tables of the second mini-paper | built |
| `test_evaluation_reproduces_the_paper.py` | the evaluation, run on the stored states of the old runs, gives the numbers of the paper: outputs 1 to 5, 7, 8 and 9 | built |
| `test_stage_errors_held_and_carried.py` | output 6: the contributions of all steps sum to the measured endpoint error; for the exact scheme all of it is the scheme; what is beyond first order falls with the square of the error | built |
| `test_report_of_a_run.py` | the report of a run holds the eleven outputs and says what it left out; it is of the weights of the snapshot; it is never written over another | built |
| `test_resolution.py` | output 11: the resolution is \|predicted - reference\| per component and the signed resolution is predicted - reference; its table per p band holds the median and 16th and 84th percentiles of exactly the tracks of the band; the histogram counts every track once; each single step takes the momentum of its track | built |
| `test_literature_changes.py` | the four changes of 2026-10-23: one fixed draw keeps round 1 identical and later rounds on its tracks and planes; the best round is that of the lowest validation error; the refit of the last layer has the Jacobian of autograd, lowers the loss and touches only the last layer; the self-scaled Broyden update meets the secant equation, is BFGS at tau = phi = 1, and trains a run | built |
| `test_physics_layer_and_carried_defect.py` | the three components of 2026-11-12: the physics layer (the stage network with no sweep; the stage map with one; the gradient reaches the body; refusals; a run trains), the whole-track protocol (whole tracks in order, reference states in round 1), the carried-defect loss (the defect and its carry; the pooled loss with share 0; lambda; zero at the exact scheme; the sum over the steps of a track; a run trains) | built |
| `test_bias_study_changes.py` | the two changes of 2026-10-28 (per-step bias study, Part B): residual draws keep round 1 and, with share 0, every round of `rounds_on_own_predictions`; they take states of larger residual; the z-face test; the bias term has its shapes, is zero for a residual of zero mean per bin, is the pooled loss to the last bit with share 0, and its lambda makes it the share of round 1 and is recorded; both train a run | built |
| `test_learned_gain_and_error_conditioned_network.py` | the two networks of 2026-11-03 (Kalman-like gain study, Part C): the analytic gain equals that of the Part B script to the last bit; both networks have their shapes, start as the stage network, and give no correction for a residual of zero; the distilled loss is zero at K r and is its formula; the loss on both calls is the mean of two pooled losses; each configuration is read back to its key, trains, leaves the frozen network unchanged, resumes to the same weights, and reports its counts | built |
| `test_newton_sweep.py` | one Newton sweep of the stage equations of a whole track (2026-11-04): from the exact scheme it changes nothing beyond rounding; from the forward-Euler path one sweep comes within a micrometre and two reach the exact scheme; its counts | built |
| `test_path_network.py` | the path network (2026-11-04): its shapes and the whole track it fills; the extra outputs; the same seed gives the same weights; with the last layer zero, the straight line or zero; only the first plane | built |
| `test_supervised_path.py` | the supervised path loss (2026-11-04): its value on a known pair; the Gaussian loss lowest at the right sigma; its terms; what it refuses | built |
| `test_reference_path.py` | the reference path and `exact_states --keep-stages` (2026-11-04): round trip on a small store; the labels are the stored stages and planes; what it refuses; the project's store holds the training and validation splits at 64 steps and 2 stages with their stages; a path network trains through the trainer | built |
| `test_time_and_sweeps.py` | the report with `with_one_sweep` (2026-11-04): one row alone and one after each of two sweeps, their counts, the exact scheme after two; no sweeps without it; the time per event: one row per method, the runs given or not, the lookups each costs | built |
| `test_a_run_on_the_farm.py` | a job stopped by the signal of the farm releases its lock and is resumed to the same run, to the last bit; a lock that is touched is never cleared; a lock left behind is cleared when the caller says | built |
| `test_registry_turns_names_into_components.py` | every built component can be named; unknown names and kinds are refused | built |

## The contract

A gate states what it proves in its name. A parity gate compares the package with the frozen code it was ported from, to the last bit where the arithmetic is the same.

A parity gate that cannot find its frozen file fails. It does not skip.

The frozen folders are only read. The field map is read from CVMFS, so the gates need CVMFS mounted.

Ten gates read the store of the project, at `/data/bfys/gscriven/rkpinn_store` or where the environment variable `RKPINN_STORE` points. They fail if the store does not hold the project's track set.

## To run

```bash
cd /data/bfys/gscriven/LHCb_Extrapolation_Project/RK_Pinn_module
PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python -m pytest tests -q
```

The package does not need to be installed for this: `pyproject.toml` tells pytest to find it in `src/`.

All gates take about fifteen minutes on the login node, one thread: about seven for phase 1, five for phase 2, one for phase 3, twenty seconds for phase 4 and two minutes for phase 5. Most of that is the sixth-order reference at its step of 0.1 mm over 2,589 mm, run once by the package and once by the frozen code.

## How to add to it

1. Write the gate as one test file, named for what it proves.
2. Add its row to the Contents table above and change the date.
3. Run all gates.
