# Evaluation

**Last updated:** 2026-09-28

## What this directory is

The standard for results. One set of conventions and one report, the same for every run and every comparator.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file | built |
| `conventions.py` | positions in micrometres, slopes dimensionless, the momentum bands, the statistics; defined here and nowhere else | planned |
| `endpoint_error.py` | error at the first SciFi plane per component, per momentum band, signed and absolute | planned |
| `error_along_the_track.py` | error plane by plane; stage errors held at every step and carried to the end of the track | planned |
| `three_references.py` | the prediction against RK6, against the exact scheme, and against the Geant4-true state | planned |
| `convergence.py` | validation error per round, and the round at which the stopping rule held | planned |
| `standard_report.py` | the ten standard outputs of `PACKAGE_PLAN.md`, section 5, as tables and figures | planned |

## The contract

The evaluation reads a predicted track and targets. It never imports a network. Figures and tables name a network by its number of steps, number of stages and step length, and say whether an error is an endpoint error or a single-step error.

## How to add to it

1. An analysis used by one experiment stays in that experiment's folder.
2. An analysis wanted for every run is added here as one file, and its output is added to the standard report.
3. Add its row to the Contents table above and change the date.
4. Update the card of the standard evaluation and its entry in the Notion master index.
5. Run all gates.
