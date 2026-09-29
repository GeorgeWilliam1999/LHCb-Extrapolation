# Training

**Last updated:** 2026-09-30

## What this directory is

The trainer and everything it needs. The trainer names no loss, no network, no target and no protocol: it takes them from the registry, by the names the configuration gives.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file | built |
| `__init__.py` | marks the directory as part of the package; it holds no code | built |
| `round_trainer.py` | trains in rounds; runs the checks before training; writes a round when it ends; resumes and extends a run (`train`, `build_run`, `checks_before_training`) | built |
| `training_protocols.py` | what is drawn each round. Two: `rounds_on_own_predictions`, the first round on reference states and later rounds on the network's own predictions; `start_states_of_the_tracks`, for the whole-crossing network | built |
| `optimiser.py` | `lbfgs_restarts`: L-BFGS restarts with the loss rescaled at the start of a round and whenever it has fallen to a set fraction | built |
| `stopping_rule.py` | `validation_plateau`: the median of the last window of rounds no more than the tolerance below the window before, at each of the last rounds held | built |
| `validation_error.py` | the three validation errors recorded after every round: the median endpoint error in x, in y, and of the larger of the two | built |
| `checkpoints.py` | the folder of a run: snapshots that are never overwritten, the records, the state, and a lock so that one job writes to a run at a time | built |

## Where each file was ported from

| File | Ported from | Gate |
|---|---|---|
| `optimiser.py` | `single_network_chain_discrete_approach/Block_E_single_network_chain/E1_Network_grid/train_network.py` (`make_opt`, `renew`, `one_restart`) | `tests/test_training.py`: weights and losses identical to the last bit after three restarts |
| `stopping_rule.py` | `Block_F_reweighted_loss/F2_Analysis/compare_to_blockE.py` (`plateaued_now`) | `tests/test_training.py`: the same verdict on 1,200 series |
| `training_protocols.py` | `rounds_on_own_predictions` from `train_network.py` (`new_round_states`); `start_states_of_the_tracks` is new | `tests/test_training.py` |
| `round_trainer.py`, `checkpoints.py`, `validation_error.py` | new; the round protocol is that of `train_network.py` | `tests/test_training.py` |

## Settings of the training

Every setting is in the configuration. None has a default.

| Setting | What it is | Value in the frozen trainer |
|---|---|---|
| `protocol.states_per_round` | the budget of states of a round | 32,000 |
| `restarts_per_round` | the most restarts in a round | none; 25 was proposed |
| `end_a_round_when_stalled` | nothing, or `gain_below` and `restarts_running`: the round ends when that many restarts running each gain less than that fraction | 0.01 and 2 |
| `optimiser.iterations_per_restart`, `history`, `tolerance_of_the_gradient`, `tolerance_of_the_change`, `line_search` | the settings of L-BFGS | 200, 120, 1e-13, 1e-16, strong Wolfe |
| `optimiser.renew_scale_when_loss_falls_to` | the fraction at which the factor on the loss is set again | 0.1 |
| `stopping_rule.quantity` | which validation error the rule is applied to | the larger of x and y |
| `stopping_rule.window_in_rounds`, `tolerance`, `rounds_held` | the plateau rule | 10, 0.05, 3 |
| `rounds_at_most` | the cap on the rounds. It is not part of the run key: raising it extends the run | 20 |

## What differs from the frozen trainer

| Frozen trainer | Here | Why |
|---|---|---|
| wrote the weights after every restart, over the last | writes a round when it ends, as a snapshot that is never overwritten | a table always matches the files it cites (WORKFLOW.md, section 3) |
| resumed inside a round | resumes from the last snapshot and repeats the interrupted round | what is on disk always describes whole rounds. At most one round of work is lost |
| ended a run when the first two restarts of a round gained under 1 %, then made a confirmation pass | ends a run on the validation error, or at its cap | convergence is never judged on the loss |
| the loss was chosen by choosing the script | the loss is named in the configuration | |
| two jobs could write to one run | one job, by a lock | |

## The contract

Convergence is judged on the validation error, never on the loss. The test split is never used for stopping.

A protocol gives `states_of_round(round, seed, tracks, layout, network, target)`. An optimiser gives `start_round(parameters)` and `one_restart(loss_now)`. A stopping rule gives `quantity` and `holds(errors)`. Each states its settings in `settings_in_a_configuration` and is built by `from_configuration`.

A run that was interrupted and resumed is identical, to the last bit, to a run that was not interrupted.

On the farm a job is stopped at its time limit and started again. The command releases the lock when it is stopped by a signal. A job that holds a lock touches it after every restart; the caller can say after how many minutes without a touch a lock counts as left behind by a job that was killed.

## To train

```bash
cd /data/bfys/gscriven/LHCb_Extrapolation_Project/RK_Pinn_module
PYTHONNOUSERSITE=1 PYTHONPATH=src /data/bfys/gscriven/conda/envs/TE/bin/python \
    -m rkpinn.training.round_trainer --store /data/bfys/gscriven/rkpinn_store \
    --configuration <file.yaml> --if-the-run-exists refuse
```

`--if-the-run-exists` is `refuse`, `resume` or `extend`. It has no default.

## How to add to it

1. Write the protocol, optimiser or stopping rule as one file in this directory, named in plain English.
2. Register it under its name with `register` from `../registry.py`, and add the file to `COMPONENT_MODULES` there.
3. Write its gates in `tests/`.
4. Add its row to the Contents table above and change the date.
5. Write or update its card in `docs/cards/`.
6. Add or update its entry in the master index of the Notion project page.
7. Run all gates.
