# Equations of motion

**Last updated:** 2026-09-28

## What this directory is

The differential equations a track obeys, and the magnetic field they need. One file per physical system.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file | built |
| `__init__.py` | marks the directory as part of the package; it holds no code | built |
| `lhcb.py` | the rates of change of (x, y, slope in x, slope in y) with z in the LHCb field, in numpy (`LhcbEquationOfMotion`) and in torch (`DifferentiableLhcbEquationOfMotion`); charge over momentum is conserved | built |
| `field_map.py` | the v8r1 field map, both polarities (`FieldMap`), its differentiable double-precision twin (`DifferentiableFieldMap`), the hash of the map file and the corners of the map | built |
| `van_der_pol.py` | the van der Pol oscillator, the second system, used to show the package is not shaped around LHCb alone | planned |

## Where each file was ported from

| File | Ported from | Gate |
|---|---|---|
| `lhcb.py` | `single_network_chain_discrete_approach/_shared/reference.py` (`deriv`) and `_shared/model.py` (`LHCbRates`) | `tests/test_equation_of_motion_matches_the_frozen_code.py` |
| `field_map.py` | `_shared/field_v8r1.py` (`FieldV8R1`), `_shared/field_torch.py` (`FieldTorch`), `_shared/reference.py` (`field_md5`, `field_bounds`) | `tests/test_field_map_matches_the_frozen_code.py` |

The arithmetic of each port is unchanged and in the same order, so the numbers are identical to the last bit. Only the names changed.

## The contract

An equation of motion gives `number_of_components` and `rates(state, z)`. The torch form must return the same numbers as the numpy form, and must be differentiable with respect to the state.

As built for LHCb:

- The state has five entries, (x, y, slope in x, slope in y, charge over momentum). `number_of_components` is 4, the entries that change along the track. `width_of_state` is 5.
- The numpy form takes states of shape (n, 5) and returns rates of shape (n, 5), with a rate of zero for charge over momentum.
- The torch form takes the four changing entries, shape (n, q, 4), and charge over momentum separately, shape (n, 1), and returns shape (n, q, 4). This is the form the losses need.
- The field map is an argument. No equation of motion has a default polarity.

## How to add to it

1. Write the system as one file in this directory, named in plain English.
2. Register it under its name with `register` from `../registry.py`, and add the file to `COMPONENT_MODULES` there.
3. Write its gates in `tests/`.
4. Add its row to the Contents table above and change the date.
5. Write or update its card in `docs/cards/`.
6. Add or update its entry in the master index of the Notion project page.
7. Run all gates.
