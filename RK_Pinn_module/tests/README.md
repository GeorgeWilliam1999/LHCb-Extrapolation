# Tests: the gates

**Last updated:** 2026-09-28

## What this directory is

Every gate of the package, run with pytest. A change is not finished until all of them pass.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file | built |
| `test_every_directory_is_described.py` | fails if a directory has no README, if a README does not name something in its directory, or if something marked built does not exist | built |
| `frozen_code.py` | not a gate: where the parity gates find the frozen code, the frozen tracks and the stored exact states. The only file that inserts into the import path | built |
| `test_field_map_matches_the_frozen_code.py` | the hash of the map file; field values identical to the frozen loader on 200,000 points, both polarities; the differentiable field identical to the frozen one | built |
| `test_equation_of_motion_matches_the_frozen_code.py` | rates identical to the frozen code on 500 real states; torch rates and their gradient identical to the frozen code; torch equals numpy | built |
| `test_gauss_legendre_tableau_matches_the_frozen_code.py` | the tableau identical to the frozen one to the last bit at 16 stages and eleven other stage counts; the five identities | built |
| `test_exact_collocation_matches_the_frozen_code.py` | one step and the chain identical to the frozen solver run on this machine; the stored exact states reproduced to 1e-11 mm; every solve converged | built |
| `test_sixth_order_reference_matches_the_frozen_code.py` | end states identical to the frozen code over 40 mm and 2,589 mm; the stored tracks reproduced to the last bit on the first 33 of their 257 planes; fitted order six on a smooth field; forwards then back closes | built |
| `test_registry_turns_names_into_components.py` | every built component can be named; unknown names and kinds are refused | built |

## The contract

A gate states what it proves in its name. A parity gate compares the package with the frozen code it was ported from, to the last bit where the arithmetic is the same.

A parity gate that cannot find its frozen file fails. It does not skip.

The frozen folders are only read. The field map is read from CVMFS, so the gates need CVMFS mounted.

## To run

```bash
cd /data/bfys/gscriven/LHCb_Extrapolation_Project/RK_Pinn_module
PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python -m pytest tests -q
```

The package does not need to be installed for this: `pyproject.toml` tells pytest to find it in `src/`.

All gates take about seven minutes on the login node, one thread. Most of that is the sixth-order reference at its step of 0.1 mm over 2,589 mm, run once by the package and once by the frozen code.

## How to add to it

1. Write the gate as one test file, named for what it proves.
2. Add its row to the Contents table above and change the date.
3. Run all gates.
