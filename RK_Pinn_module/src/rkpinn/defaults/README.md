# Defaults

**Last updated:** 2026-10-09

## What this directory is

The default configurations: complete, valid configuration files that an experiment copies and then changes in what it asks about. A default is a file, not a value in code, so every run still takes every setting from its own file.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file | built |
| `__init__.py` | names the defaults, pins each one's run key, and gives the path of one by name (`path_of_the_default`) | built |
| `lhcb_64_steps_2_stages.yaml` | the LHCb default: one network of 64 steps of 80.9 mm and 2 stages, end state summed from the stages, pooled loss over the stages, run A of the pilot setting for setting, run key `bddea42ea1ef` | built |

## The contract

- A default is agreed with George before it is added. The LHCb default was agreed on 2026-10-09 (five choices: the cell, its form, what scores it, the training settings, where it lives).
- A default is never edited by an experiment. An experiment copies it into its own `configurations/` and changes one block, so that the difference between the two files is the experiment's question. `rkpinn.run_record.configuration.difference` prints it.
- The run key of each default is pinned in `__init__.py` and checked by the gate. Changing a default changes its key, and the gate fails until the pin is changed with it, so a default cannot drift unnoticed.
- What scores a run, the sixth-order reference at 0.1 mm and the exact collocation scheme, is not a setting of the configuration. It is named in the header of the file so a reader finds it, and it is fixed in `../integrators/`.

## How to add to it

1. Agree the default with George.
2. Write it as one file named by its system, steps and stages; put the agreement and what scores it in the header.
3. Add its name and run key to `THE_DEFAULTS` in `__init__.py`.
4. Add its row above and change the date.
5. Add its entry to the master index of the Notion project page, and the copy in `../../../docs/notion/project_page.md`.
6. Run all gates.
