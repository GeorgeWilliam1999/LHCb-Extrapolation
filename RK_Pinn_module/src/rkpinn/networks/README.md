# Networks

**Last updated:** 2026-09-28

## What this directory is

The networks, how their raw outputs become states, how the end of a step is formed from its stages, and how steps are chained into a track.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file | built |
| `stage_network.py` | takes a state and the z of the start plane; emits the stage states of one Gauss–Legendre step | planned |
| `whole_crossing_network.py` | the supervised twin: takes the state at the last UT plane; emits the state at the first SciFi plane in one application | planned |
| `output_forms.py` | how raw network outputs become states. One form now: the states themselves, scaled by fixed constants | planned |
| `collect_and_sum.py` | the end state of a step as the input plus the step length times the weighted sum of the stage rates | planned |
| `self_chain.py` | applies one stage network N times, each end state being the next input | planned |

## The contract

A network gives `predict(states, start_planes)`, which returns a predicted track. An output form declares what it returns when the network's last layer is zero, and a gate checks it.

## Places kept open

| Not built | Where it would go |
|---|---|
| a network that learns the whole chain in one output | a new file here, filling all N steps of the predicted track at once |
| a different network for each step | a new file here, holding N stage networks |
| the straight line plus a correction | a new output form in `output_forms.py` |

## How to add to it

1. Write the network as one file in this directory, named in plain English.
2. Register it under its name in `../registry.py`.
3. Write its gates in `tests/`.
4. Add its row to the Contents table above and change the date.
5. Write or update its card in `docs/cards/`.
6. Add or update its entry in the master index of the Notion project page.
7. Run all gates.
