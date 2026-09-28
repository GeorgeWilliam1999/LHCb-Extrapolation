# Tests: the gates

**Last updated:** 2026-09-28

## What this directory is

Every gate of the package, run with pytest. A change is not finished until all of them pass.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file | built |
| `test_every_directory_is_described.py` | fails if a directory has no README, if a README does not name something in its directory, or if something marked built does not exist | built |

## The contract

A gate states what it proves in its name. A parity gate compares the package with the frozen code it was ported from, to the last bit where the arithmetic is the same.

## To run

```bash
cd /data/bfys/gscriven/LHCb_Extrapolation_Project/RK_Pinn_module
PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python -m pytest tests -q
```

## How to add to it

1. Write the gate as one test file, named for what it proves.
2. Add its row to the Contents table above and change the date.
3. Run all gates.
