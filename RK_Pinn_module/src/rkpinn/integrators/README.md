# Integrators

**Last updated:** 2026-09-28

## What this directory is

Numerical methods that need no network: the collocation scheme the networks are trained against, solved exactly, and the fine reference everything is scored against.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file | built |
| `__init__.py` | marks the directory as part of the package; it holds no code | built |
| `gauss_legendre_tableau.py` | nodes, stage matrix and weights for any number of stages (`gauss_legendre_tableau`, `GaussLegendreTableau`), with the five identities that check them | built |
| `exact_collocation.py` | the implicit stage equations solved with a root finder, one step (`solve_one_step`) or chained (`solve_whole_track`); this is the ceiling for a network | built |
| `runge_kutta_sixth_order.py` | Butcher's seven-stage explicit method at a fixed step (`integrate_with_sixth_order`, `integrate_with_sixth_order_over_planes`); at 0.1 mm this is the reference | built |

## Where each file was ported from

| File | Ported from | Gate |
|---|---|---|
| `gauss_legendre_tableau.py` | `single_network_chain_discrete_approach/_shared/irk.py` | `tests/test_gauss_legendre_tableau_matches_the_frozen_code.py` |
| `exact_collocation.py` | `multi_network_chain_discrete_approach/Block_C_step_size_and_stages/C2_Exact_scheme_table/exact_solver.py` and `single_network_chain_discrete_approach/Block_E_single_network_chain/E2_Comparators/exact_chain.py` | `tests/test_exact_collocation_matches_the_frozen_code.py` |
| `runge_kutta_sixth_order.py` | `_shared/reference.py` (`rk6_rows` and its coefficients) | `tests/test_sixth_order_reference_matches_the_frozen_code.py` |

The arithmetic of each port is unchanged and in the same order. Only the names changed.

## What was measured when porting

- The package is identical, to the last bit, to the frozen code run on the same machine, for every file here.
- The stored exact states in `E2_Comparators/results/` were written on farm nodes. They differ from a run on the login node in the last digits, by at most 4.5e-13 mm in position and 5.6e-17 in slope on the tracks of the gate. The frozen code run on the login node differs from its own stored files by the same amounts. The gate therefore compares with the stored files to 1e-11 mm and 1e-15, and with the frozen code bit for bit.
- The stored tracks are reproduced to the last bit by `integrate_with_sixth_order_over_planes`. This was checked by hand on the first 20 test tracks on all 257 planes (2026-09-28). The gate checks the first 33 planes, to keep it short. The whole file is the gate of phase 2.

## The contract

An integrator takes an equation of motion, a state and two planes, and returns a predicted track. A tableau must pass its own checks every time it is built.

As built in phase 1:

- The predicted track is built in phase 3. Until then the integrators return arrays: `integrate_with_sixth_order` returns the end states, `solve_one_step` returns an `ExactStep` and `solve_whole_track` an `ExactTrack`, both holding the input, stage and end states. They are changed to fill the predicted track in the same change that builds it.
- The step length of the sixth-order method is an argument without a default. The reference step, 0.1 mm, is the named value `REFERENCE_STEP_LENGTH_MM`.
- The first guess of the root finder in `exact_collocation.py` is the input state carried along its own slopes. It is a starting point for the solver only, kept because the stored exact states were solved from it. It is not an output form of any network.

## How to add to it

1. Write the integrator as one file in this directory, named in plain English.
2. Register it under its name with `register` from `../registry.py`, and add the file to `COMPONENT_MODULES` there.
3. Write its gates in `tests/`.
4. Add its row to the Contents table above and change the date.
5. Write or update its card in `docs/cards/`.
6. Add or update its entry in the master index of the Notion project page.
7. Run all gates.
