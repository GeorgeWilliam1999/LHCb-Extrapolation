# Cards

**Last updated:** 2026-11-03

## What this directory is

One page per component. A card is written once and linked from every study that uses the component, so theory is not restated.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file | built |
| `straight_line_plus_scaled_correction.md` | the output form of the frozen networks: the straight line plus a correction scaled per track or by fixed constants; its derivation, its scale, its gates | built |
| `stage_network_with_a_learned_gain.md` | C1 of the Kalman-like gain study: the learned gain, the analytic gain it learns, its loss, its cost, the Kalman analogy | built |
| `error_conditioned_stage_network.md` | C2 of the same study: a second call that reads the residual of the first and corrects the stages; its loss, its cost, how it differs from the paper | built |

## The contract

A card has: what the component is, the full derivation with every symbol defined where it first appears, its gates, its source file, and the date. It assumes no prior knowledge.

## How to add to it

1. Write the card as one file named after the component.
2. Add its row to the Contents table above and change the date.
3. Mirror it to its entry in the Notion master index.
