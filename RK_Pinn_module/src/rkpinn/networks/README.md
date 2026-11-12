# Networks

**Last updated:** 2026-11-12

## What this directory is

The networks, how their raw outputs become states, how the end of a step is formed from its stages, and how steps are chained into a track.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file | built |
| `__init__.py` | marks the directory as part of the package; it holds no code | built |
| `stage_network.py` | takes a state and the z of the start plane; emits the stage states of one Gauss–Legendre step, and the end state too when it is set to be predicted (`StageNetwork`, `build_stage_network`) | built |
| `whole_crossing_network.py` | the supervised twin: takes the state on the first plane; emits the state on the last plane in one application (`WholeCrossingNetwork`) | built |
| `path_network.py` | takes the state on the first plane; emits every stage state and every plane state of the layout in one call, not chained, trained on the reference path; the output form is bound with the whole region as its step; with `extra_outputs` 4 also a log-sigma per state and component (`PathNetwork`, 2026-11-04) | built |
| `output_forms.py` | how raw network outputs become states. Two forms: `direct_states`, the states themselves scaled by fixed constants; `straight_line_plus_scaled_correction`, the straight line from the input plus a correction scaled per track or by fixed constants, the form of the frozen networks | built |
| `collect_and_sum.py` | the end state of a step as the input plus the step length times the weighted sum of the stage rates | built |
| `self_chain.py` | applies one stage network N times, each end state being the next input (`whole_track`) | built |
| `stage_network_of_a_run.py` | reads a trained stage network from a run in the store (settings and the weights of one round); the first call (stages, rates, residual); the root mean square of the residual and the rates over the first round's states; the scale of a correction (2026-11-03) | built |
| `stage_network_with_a_learned_gain.py` | C1 of the Kalman-like gain study: a frozen (or trained) stage network and a small network G whose 4 x 4q output matrix multiplies the stage residual; the end state is the summed one minus the correction (`StageNetworkWithALearnedGain`, 2026-11-03) | built |
| `stage_network_with_a_physics_layer.py` | the stage network with the stage map as its last layer: `sweeps_inside` applications of Y ← S0 + dz A f(Y) to the stages the body emits, the body trained through them; the end state summed from the stages after the layer; with 0 sweeps the stage network to the last bit (`StageNetworkWithAPhysicsLayer`, 2026-11-12) | built |
| `error_conditioned_stage_network.py` | C2 of the same study: the stage network is called once, a corrector reads the residual of that call (6 + 4q inputs) and gives a 4q x 4q matrix that multiplies it to correct the stages; the end state is summed from the corrected stages (`ErrorConditionedStageNetwork`, 2026-11-03) | built |

## Settings of a network

Every setting is an argument. None has a default. A configuration must state each one.

| Setting | Values | Ruled |
|---|---|---|
| `end_state` | `summed_from_the_stages`: 4q outputs, the end of the step is collected and summed. `predicted`: 4(q + 1) outputs, the last is the end state, as in the frozen code and the paper | both kept: it depends on the experiment (George, 2026-09-28) |
| `output_form` | `direct_states`: the network emits the states (George, 2026-09-28). `straight_line_plus_scaled_correction`: the network emits the correction to the straight line, as every frozen network did; added 2026-10-22 after the audit of 2026-10-19 found direct states cost a factor 6 to 25 at the stages | both kept: the form is a setting of the run |
| `scale_of_inputs` | five constants | how they are measured is a setting of the experiment (George, 2026-09-28) |
| `scale_of_outputs`, read by the output form | with `direct_states`: four constants. With `straight_line_plus_scaled_correction`: `per_track` (the scale of the frozen code, from the field along the straight line through each input), `median_of_per_track_scale_over_first_round_states` (four constants measured once when the run is made), or four constants | the same |
| `width`, `depth` | the body: `depth` layers of `width` units, each followed by the activation, then a linear layer | |
| `activation` | `tanh` | the architecture is free to vary and whether to use tanh is open (George, 2026-09-29). One activation is built; another is added to `ACTIVATIONS` in `stage_network.py` when it is named |
| `first_start_plane_mm`, `last_start_plane_mm` | the range the start plane is mapped from, onto [-1, 1] | |

## Where each file was ported from

| File | Ported from | Gate |
|---|---|---|
| `stage_network.py`, `whole_crossing_network.py` | the body, its initialisation and the order of its outputs are those of `OneStepNetwork` in `single_network_chain_discrete_approach/_shared/model.py`; the mapping of the start plane is that of `ChainNetwork` in `E1_Network_grid/chain_network.py` | `tests/test_networks.py` |
| `self_chain.py` | `carry` in `E1_Network_grid/chain_network.py` | `tests/test_networks.py` |
| `output_forms.py`: `straight_line_plus_scaled_correction` | `straight`, `field_integrals` and `residual_scale` of `ChainNetwork` in `E1_Network_grid/chain_network.py`, arithmetic unchanged; a stage network with this form, the scale `per_track` and a predicted end state is that network to the last bit, chain included | `tests/test_networks.py` |
| `output_forms.py`: `direct_states`, `collect_and_sum.py` | new | `tests/test_networks.py` |
| `stage_network_of_a_run.py`, `stage_network_with_a_learned_gain.py`, `error_conditioned_stage_network.py` | new; the gain follows the update of KalmanNet (arXiv:2107.10043) and the corrector the error-conditioned neural solvers (arXiv:2606.27354), each with the differences stated in its docstring | `tests/test_learned_gain_and_error_conditioned_network.py` |

## Settings of the two networks built on a stage network (2026-11-03)

| Setting | Values | Ruled |
|---|---|---|
| `stage_network_of_the_run` | `{key, round}` of a run of a stage network with a summed end state, the same steps, stages and tracks | the frozen run is the experiment's choice |
| `train_the_stage_network` | `false`: frozen; `true`: trained with the small network | default false (brief of 2026-11-03) |
| `inputs_of_the_gain` (gain only) | `state_and_residual`, or `state_residual_and_rates` | open |
| `width`, `depth`, `activation` | of the small network | the brief: two tanh layers of 32 to 64 |
| `scale_of_inputs` | five numbers, or `spread_of_first_round_states` | |
| `scale_of_outputs` | `same_as_the_stage_network`, `per_track`, `spread_of_first_round_states`, or four numbers: the scale of the correction | open |
| `scale_of_the_residual_and_rates` | `rms_over_first_round_states` (measured on the stage network, six significant digits), or `{residual, rates}` of 4q numbers each | |

Both report `cost_per_step()`: field lookups, field samples of the output scales, Jacobians, solves and network calls of one step. The stage network reports it too.

## The contract

A network gives `predict(states, start_planes)`, which returns a predicted track of separate steps, and `whole_track(states, layout)`, which returns a whole track. An output form is built from the `scale_of_outputs` setting by its `from_configuration`, is bound once by the network that holds it to the step length, the planes of the outputs and the field map, turns raw outputs into states with `states_from_raw(raw, input_states, start_planes)`, declares what it returns when the network's last layer is zero (a gate checks it), and says in `scale_setting()` what the network records as its scale.

A network states its settings in `settings_in_a_configuration` and is built from a configuration by `from_configuration`, with its weights drawn from the seed of the run. It says whether it has stages, in `has_stages`. It may give `cost_per_step()`, which the report writes into its cost table.

A network is in double precision whatever the default of torch is. The field map is not written into a snapshot of its weights. `settings()` gives what is needed, with the weights, to build it again.

## Places kept open

| Not built | Where it would go |
|---|---|
| a network that learns the whole chain in one output | a new file here, filling all N steps of the predicted track at once |
| a different network for each step | a new file here, holding N stage networks |

## How to add to it

1. Write the network as one file in this directory, named in plain English.
2. Register it under its name with `register` from `../registry.py`, and add the file to `COMPONENT_MODULES` there.
3. Write its gates in `tests/`.
4. Add its row to the Contents table above and change the date.
5. Write or update its card in `docs/cards/`.
6. Add or update its entry in the master index of the Notion project page.
7. Run all gates.
