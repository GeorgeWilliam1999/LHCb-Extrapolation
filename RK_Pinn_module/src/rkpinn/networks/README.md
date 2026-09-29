# Networks

**Last updated:** 2026-09-29

## What this directory is

The networks, how their raw outputs become states, how the end of a step is formed from its stages, and how steps are chained into a track.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file | built |
| `__init__.py` | marks the directory as part of the package; it holds no code | built |
| `stage_network.py` | takes a state and the z of the start plane; emits the stage states of one Gauss–Legendre step, and the end state too when it is set to be predicted (`StageNetwork`, `build_stage_network`) | built |
| `whole_crossing_network.py` | the supervised twin: takes the state on the first plane; emits the state on the last plane in one application (`WholeCrossingNetwork`) | built |
| `output_forms.py` | how raw network outputs become states. One form now, `direct_states`: the states themselves, scaled by fixed constants | built |
| `collect_and_sum.py` | the end state of a step as the input plus the step length times the weighted sum of the stage rates | built |
| `self_chain.py` | applies one stage network N times, each end state being the next input (`whole_track`) | built |

## Settings of a network

Every setting is an argument. None has a default. A configuration must state each one.

| Setting | Values | Ruled |
|---|---|---|
| `end_state` | `summed_from_the_stages`: 4q outputs, the end of the step is collected and summed. `predicted`: 4(q + 1) outputs, the last is the end state, as in the frozen code and the paper | both kept: it depends on the experiment (George, 2026-09-28) |
| `output_form` | `direct_states` | the network emits the states, not a correction to a straight line (George, 2026-09-28). The form is a setting, so another can be added |
| `scale_of_inputs` | five constants | how they are measured is a setting of the experiment (George, 2026-09-28) |
| scale of the outputs, in the output form | four constants | the same |
| `width`, `depth` | the body: `depth` layers of `width` units, each followed by the activation, then a linear layer | |
| `activation` | `tanh` | the architecture is free to vary and whether to use tanh is open (George, 2026-09-29). One activation is built; another is added to `ACTIVATIONS` in `stage_network.py` when it is named |
| `first_start_plane_mm`, `last_start_plane_mm` | the range the start plane is mapped from, onto [-1, 1] | |

## Where each file was ported from

| File | Ported from | Gate |
|---|---|---|
| `stage_network.py`, `whole_crossing_network.py` | the body, its initialisation and the order of its outputs are those of `OneStepNetwork` in `single_network_chain_discrete_approach/_shared/model.py`; the mapping of the start plane is that of `ChainNetwork` in `E1_Network_grid/chain_network.py` | `tests/test_networks.py` |
| `self_chain.py` | `carry` in `E1_Network_grid/chain_network.py` | `tests/test_networks.py` |
| `output_forms.py`, `collect_and_sum.py` | new | `tests/test_networks.py` |

Not ported: the straight line and the scale per track of `ChainNetwork`. No network of the package has them.

## The contract

A network gives `predict(states, start_planes)`, which returns a predicted track of separate steps, and `whole_track(states, layout)`, which returns a whole track. An output form declares what it returns when the network's last layer is zero, and a gate checks it.

A network states its settings in `settings_in_a_configuration` and is built from a configuration by `from_configuration`, with its weights drawn from the seed of the run. It says whether it has stages, in `has_stages`.

A network is in double precision whatever the default of torch is. The field map is not written into a snapshot of its weights. `settings()` gives what is needed, with the weights, to build it again.

## Places kept open

| Not built | Where it would go |
|---|---|
| a network that learns the whole chain in one output | a new file here, filling all N steps of the predicted track at once |
| a different network for each step | a new file here, holding N stage networks |
| the straight line plus a correction | a new output form in `output_forms.py` |

## How to add to it

1. Write the network as one file in this directory, named in plain English.
2. Register it under its name with `register` from `../registry.py`, and add the file to `COMPONENT_MODULES` there.
3. Write its gates in `tests/`.
4. Add its row to the Contents table above and change the date.
5. Write or update its card in `docs/cards/`.
6. Add or update its entry in the master index of the Notion project page.
7. Run all gates.
