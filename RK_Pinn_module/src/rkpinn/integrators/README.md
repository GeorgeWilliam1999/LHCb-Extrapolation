# Integrators

**Last updated:** 2026-09-28

## What this directory is

Numerical methods that need no network: the collocation scheme the networks are trained against, solved exactly, and the fine reference everything is scored against.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file | built |
| `gauss_legendre_tableau.py` | nodes, stage matrix and weights for any number of stages, with the five identities that check them | planned |
| `exact_collocation.py` | the implicit stage equations solved with a root finder, one step or chained; this is the ceiling for a network | planned |
| `runge_kutta_sixth_order.py` | Butcher's seven-stage explicit method at a fixed step of 0.1 mm; this is the reference | planned |

## The contract

An integrator takes an equation of motion, a state and two planes, and returns a predicted track. A tableau must pass its own checks every time it is built.

## How to add to it

1. Write the integrator as one file in this directory, named in plain English.
2. Register it under its name in `../registry.py`.
3. Write its gates in `tests/`.
4. Add its row to the Contents table above and change the date.
5. Write or update its card in `docs/cards/`.
6. Add or update its entry in the master index of the Notion project page.
7. Run all gates.
