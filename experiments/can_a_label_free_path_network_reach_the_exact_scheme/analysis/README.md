# Analysis

**Last updated:** 2026-11-23 · **State:** A2 ended (round 40); A1 training; two studies of A1 done

Made after the runs end: the standard report of every run with and without one Newton sweep, the residual per step along the track, the training curves, the time per track, and the criteria of `../README.md` marked pass or fail.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file | built |
| `failure_of_A1/` | where and how A1 fails, against A2, the exact scheme and P1: error along the track, who fails, position against slope, where the loss and the gradient land, residual against error, one sweep. `compute.py` makes the arrays, `analyse.py` the figures and tables, `results.md` the numbers (2026-11-23) | built |
| `loss_terms_check/` | three candidate drift terms measured on A1 (round 37) and A2 (round 36) before any training: size against the loss, Spearman with the endpoint error, position and slope shares, lambda for a share of 0.5 and 0.9 (`check.py`, `results.json`); the checks before training and the cost of one loss and gradient of C1 to C3 against A1 (`time_and_check.py`) (2026-11-23) | built |
