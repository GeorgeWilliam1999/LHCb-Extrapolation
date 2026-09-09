# Block 0 — the first pass: does the scheme port to LHCb at all?

July 2026. Before asking whether a network can learn the magnet crossing, this
block established the two things that question needs: a characterised data
population, and the *exact* solution of the collocation scheme to measure any
network against. Read in order.

| Folder | Step | What it establishes |
|---|---|---|
| [`S0_Baseline_data_exploration/`](S0_Baseline_data_exploration/) | step 0 | The reference card and the data-look everything else is built on: the v8r1 field map, the equation of motion, the population of real particle states. |
| [`S1_Simple_first_pass/`](S1_Simple_first_pass/) | step 1 | The Gauss–Legendre implicit Runge–Kutta scheme solved **exactly**, with no network, so the scheme's own discretisation error is known before a network is blamed for it. |
| [`S2_One_step_network/`](S2_One_step_network/) | step 2 | The paper's technique, first attempt on LHCb: one implicit step across the magnet. |
| [`S2b_One_step_network_v2/`](S2b_One_step_network_v2/) | step 2, redone | The same experiment on official-sample data. This is the *verified baseline* the later blocks compare against, and the dataset `_shared/smoke_tests.py` gates parity on. |

## Why S2b is the one that matters downstream

`S2b_One_step_network_v2` is not a variant to be read alongside `S2` — it
supersedes it. It runs the identical experiment on the official LHCb simulated
sample rather than self-generated events, and its `results/` are what
`_shared/smoke_tests.py` checks the shared package against on every change:

- `frozen_leg_meta.json` — the dataset parity gate
- `hist_physics_seed0.csv` — the first-restart parity gate, checked bitwise

If either gate fails after a change to `_shared/`, the change altered numbers
that published results depend on.

## A correction that applies to everything in this block

The official sample (conditions tag `sim-20231017-vc-mu100`) is **magnet-up**.
Every label and score in this block used the magnet-**down** map, so the bend
sign is wrong throughout. The method conclusions stand, because a network was
always compared against the same engine that produced its labels, but the
framing "the labels are where the particle actually went" does not hold here.
[`../Block_C_step_size_and_stages/C0_Magnet_tracks_dataset/check_polarity.py`](../Block_C_step_size_and_stages/C0_Magnet_tracks_dataset/check_polarity.py)
measures it: 883 mm miss on the down map against 1.73 mm on the up map, over
15,257 particles. Block C is on the up map throughout.
