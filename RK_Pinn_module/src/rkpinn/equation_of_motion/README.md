# Equations of motion

**Last updated:** 2026-09-28

## What this directory is

The differential equations a track obeys, and the magnetic field they need. One file per physical system.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file | built |
| `lhcb.py` | the rates of change of (x, y, slope in x, slope in y) with z in the LHCb field, in numpy and in torch; charge over momentum is conserved | planned |
| `field_map.py` | the v8r1 field map, both polarities, and its differentiable double-precision twin | planned |
| `van_der_pol.py` | the van der Pol oscillator, the second system, used to show the package is not shaped around LHCb alone | planned |

## The contract

An equation of motion gives `number_of_components` and `rates(state, z)`. The torch form must return the same numbers as the numpy form, and must be differentiable with respect to the state.

## How to add to it

1. Write the system as one file in this directory, named in plain English.
2. Register it under its name in `../registry.py`.
3. Write its gates in `tests/`.
4. Add its row to the Contents table above and change the date.
5. Write or update its card in `docs/cards/`.
6. Add or update its entry in the master index of the Notion project page.
7. Run all gates.
