# Late-training oscillations: is it the learning rate?

2026-10-22 · read-only analysis · LHCb_Extrapolation_Project at commit `bdb7dcc2` (working tree has unrelated uncommitted deletions under `Mini_paper/`) · store `/data/bfys/gscriven/rkpinn_store` · env `conda/envs/TE`.
Scripts in this folder: `round_stats.py`, `noise_floor.py`, `lbfgs_arms.py`, `plots.py`. Nothing in the package, the store or the farm was changed.

**Short answer: no.** The optimiser converges cleanly in every restart of every run. The round-to-round oscillation in the training loss has the same size as the sampling noise of a 32,000-state draw. The oscillation in the validation error is much larger than the loss moves, and it hardly follows the loss at all. It comes from the per-round change of states (C) acting on a validation error that the loss constrains only weakly. The L-BFGS settings (A) and the restart scheme (B) are not the cause.

## 1. Optimiser settings and what the trainer does

| Setting | Value (both runs, all nine runs) | Where |
|---|---|---|
| Optimiser | `torch.optim.LBFGS`, full batch | `RK_Pinn_module/src/rkpinn/training/optimiser.py:57-62` |
| Learning rate `lr` | **not set → torch default 1.0** (with strong Wolfe this is only the first trial step; the line search picks the step) | `optimiser.py:58-62` |
| Line search | `strong_wolfe` | config `training.optimiser.line_search`; `optimiser.py:62` |
| History | 120 | config; `optimiser.py:60` |
| `max_iter` per restart | 200 | config; `optimiser.py:59` |
| `max_eval` | not set → torch default 1.25 × 200 = 250 | `optimiser.py:58-62` |
| `tolerance_grad` / `tolerance_change` | 1e-13 / 1e-16 | config; `optimiser.py:60-61` |
| Loss rescale | at round start the loss is scaled to 1; rescaled (with a **new** L-BFGS) when the loss falls below 0.1 × its value at the last rescale | `optimiser.py:64-84` |
| Restarts per round · states per round | 25 · 32,000 | config `training.restarts_per_round`, `training.protocol.states_per_round` |
| Stall rule inside a round | off (`end_a_round_when_stalled: null`) | config; `round_trainer.py:395-398` |

Configurations read: `rkpinn_store/runs/49e5e2450eaa/configuration.yaml:26-37` (64 steps / 2 stages, pooled loss) and `rkpinn_store/runs/aec34bbd4934/configuration.yaml:38-49` (256 / 16, cost-weighted 10-50 GeV). The optimiser blocks are identical. aec34bbd4934 was extended from a cap of 80 to 160 rounds (`state.json`) and plateaued at round 111.

The restart scheme (B):
- A new L-BFGS object is made at the **start of every round** (`optimiser.py:64-68`, called from `round_trainer.py:383`). So the curvature history is lost once per round.
- **Within** a round, every restart calls `step()` again on the **same** object, so the curvature history carries over from restart to restart. It is reset only when the 0.1 rescale fires (`optimiser.py:81-84`). In the last 30 rounds of every run the rescale fires exactly once per round, at restart 1 (`factor_renewed` = 30 in 750 late restarts). A restart therefore does not throw away curvature.
- The network after the **last** restart is kept. There is no best-of-restarts selection. The loss falls in every restart, so the last restart is also the one with the lowest loss on the draw.
- The round's model is **accepted unconditionally** (`round_trainer.py:410`). There is no rollback when the validation error gets worse. The stopping rule only reads the validation errors, using medians over 10-round windows (`stopping_rule.py:32-41`).

What is logged:
- **Per restart** (`restarts.csv`): loss before and after, gain, scale factor and whether it was renewed, iterations, function evaluations, `stopped_early`, seconds.
- **Per round** (`rounds.csv`): `loss_first` (the network's loss on the round's fresh draw, before training), `loss_last`, and three validation medians.
- The validation error is **not** logged per restart, and L-BFGS line-search failures are not logged at all (torch does not report them).

## 2. Per-run oscillation statistics (last 30 rounds)

Window = the last 30 rounds, never including round 1. Three runs have fewer than 31 rounds; for them the window is round 2 to the end (n shown). "Score" = validation endpoint error, larger of x and y, median, µm (lower is better). Δ = change from one round to the next.

| Run | key | window (n) | median score | std Δlog score | largest jump | frac. worse | std Δlog loss_last | fresh-draw jump loss_first(r+1)/loss_last(r) | round's training gain | corr(Δloss_last, Δscore) |
|---|---|---|---|---|---|---|---|---|---|---|
| 64/2 pooled | 49e5e2450eaa | 30-59 (30) | 7.92 | 0.222 | 64 % | 0.48 | 0.153 | +16 % | −17 % | 0.27 |
| 64/2 cost-w 10-50 | da3a1d00b33d | 2-24 (23) | 56.9 | 0.120 | 49 % | 0.41 | 0.064 | +10 % | −13 % | 0.47 |
| 64/2 cost-w 3-8 | f977c7b61049 | 2-27 (26) | 46.5 | 0.133 | 39 % | 0.48 | 0.093 | +14 % | −17 % | 0.08 |
| 128/8 pooled | 75200e48408c | 25-54 (30) | 7.10 | 0.158 | 49 % | 0.48 | 0.088 | +11 % | −10 % | −0.24 |
| 128/8 cost-w 10-50 | 237dc4101ad5 | 4-33 (30) | 47.3 | 0.106 | 23 % | 0.45 | 0.046 | +8 % | −8 % | 0.17 |
| 128/8 cost-w 3-8 | 521415f2c534 | 2-25 (24) | 49.1 | 0.084 | 20 % | 0.43 | 0.066 | +8 % | −10 % | 0.44 |
| 256/16 pooled | f38ec3e29819 | 13-42 (30) | 5.66 | 0.148 | 37 % | 0.45 | 0.055 | +11 % | −11 % | 0.01 |
| 256/16 cost-w 10-50 (ext.) | aec34bbd4934 | 82-111 (30) | 21.0 | 0.076 | 16 % | 0.45 | 0.024 | +7 % | −7 % | 0.27 |
| 256/16 cost-w 3-8 | 9c3c5d86784b | 17-46 (30) | 20.8 | 0.072 | 15 % | 0.45 | 0.037 | +8 % | −8 % | 0.38 |

Optimiser health in the same windows, all nine runs (6,325 late restarts):
- **0** restarts stopped early: every restart ran its full 200 iterations, using 201-224 function evaluations.
- **0** restarts in which the loss rose.
- **0** non-finite losses.
- The gain is 1.5-4.6 % at restart 1 and falls to 0.1-0.4 % by restart 25.
- All farm `.err` files for the nine runs are empty: no warnings, NaNs or tracebacks.

Two patterns stand out:
- **Steady state.** In every run, the jump in loss when the new draw arrives (+7 to +16 %) is about the same size as what the round's 25 restarts then remove (−7 to −17 %). The network re-fits each new draw, and the fit does not carry over to the next draw.
- **Score against loss.** The validation error gets worse in 41-48 % of rounds, a coin flip. Its round deltas correlate only weakly with the loss deltas (−0.24 to 0.47). In the two cost-weighted 3-8 GeV runs and in 64/2 cost-weighted 10-50 GeV, the validation error **drifts upward** for 10+ rounds while the training loss keeps falling (`per_round_score_and_loss.png`). That is a mismatch between loss and score, not instability.

## 3. Sampling-noise floor and the L-BFGS experiments

**Reproduction check.** The trainer's recorded numbers are reproduced exactly from the snapshots. For loss_last of the final round: 2.7447450829745e-08 recorded vs 2.7447450829745e-08 reproduced (49e5e2450eaa), and 9.74892e-10 vs 9.74892e-10 (aec34bbd4934). The validation errors also match exactly. So the draws below are built exactly as the trainer builds them.

Final snapshot, 12 fresh draws of 32,000 states each from the network's own chain and 12 from the reference states, no training:

| | 49e5e2450eaa (64/2 pooled) | aec34bbd4934 (256/16 cost-w) |
|---|---|---|
| loss on the round's own draw (loss_last) | 2.74e-8 | 9.75e-10 |
| loss on fresh chain draws, median [min, max] | 4.07e-8 [3.38, 4.36] | 1.04e-9 [0.99, 1.10] |
| loss on fresh reference draws, median | 3.87e-8 | 1.05e-9 |
| **noise floor**: std log(loss) over draws (chain / reference) | **0.084 / 0.117** | **0.031 / 0.042** |
| expected std of the round-to-round Δlog from draw noise alone (√2 × floor) | 0.12 / 0.17 | 0.044 / 0.060 |
| **observed** std Δlog loss_last | **0.153** | **0.024** |
| chain shift only: round R's states taken from the updated chain | 2.7457e-8 (+0.03 %) | 9.7490e-10 (+0.001 %) |
| loss of each late snapshot on **one fixed** 128,000-state set: std Δlog | **0.026** | **0.008** |
| … trend over the window | 5.82e-8 → 3.92e-8, steady | 1.18e-9 → 1.04e-9, steady |
| corr(Δlog fixed-set loss, Δlog score) | 0.17 | −0.28 |
| observed std Δlog score | 0.222 | 0.076 |

Reading:
- **The loss oscillation is draw noise.** The observed round-to-round spread of the loss sits at or below what draw noise alone gives. Measured on one fixed set, the loss falls steadily and monotonically, with 3-6× less jitter (`fixed_set_loss_vs_score.png`).
- **Of the two parts of C, only the subsample matters.** Rebuilding the states from the updated network's chain changes the loss by 0.001-0.03 %. Taking a different 32,000-state subsample changes it by 3-12 %.
- **The subsample is small.** 32,000 states are 4.3 % of the 740k-state pool (11,567 tracks × 64 start planes) and 1.1 % of the 2.96M-state pool at 256 steps. On a fresh draw the final network's loss is 6-48 % higher than on its own draw: a generalisation gap from one draw to the next.
- **The validation error does not follow the loss.** It oscillates 3-9× more than the fixed-set loss, and the two hardly correlate.

**L-BFGS experiments** (49e5e2450eaa; start from snapshot 58; 3 of the 25 restarts of round 59; 1 thread each, about 140 s per restart on the loaded login node, about 45 CPU-min in total). The full 25-restart round was too costly to repeat here, so these are reduced arms.

| Arm | Draw | lr | History | Loss after restarts 1/2/3 | Score after restarts 0/1/2/3 [µm] |
|---|---|---|---|---|---|
| base | round 59's own | 1.0 | 120 | 3.220 / 3.143 / 3.091 e-8 | 8.23 / 9.56 / 10.43 / 9.15 |
| base, repeated | round 59's own | 1.0 | 120 | identical to the bit | identical |
| lr halved | round 59's own | 0.5 | 120 | 3.247 / 3.177 / 3.125 e-8 | 8.23 / 9.73 / 9.70 / 9.69 |
| history 10 | round 59's own | 1.0 | 10 | 3.282 / 3.244 / 3.209 e-8 | 8.23 / 11.74 / 9.62 / 9.13 |
| other draw 1 | fresh | 1.0 | 120 | 3.48 / 3.39 / 3.33 e-8 | 8.23 / 11.50 / 10.73 / 10.56 |
| other draw 2 | fresh | 1.0 | 120 | 3.60 / 3.48 / 3.41 e-8 | 8.23 / 10.00 / 7.98 / 8.73 |
| (trainer, after all 25 restarts) | round 59's own | 1.0 | 120 | 2.745e-8 | 5.46 |

- The base arm reproduces `restarts.csv` round 59, restarts 1-3, exactly. Training is deterministic: there is no seed to vary, and a repeat gives the same result to the bit.
- **Spread of the score after 3 restarts:** std log = **0.033** across the three optimiser settings on the same draw, and **0.099** across three draws with the same settings. Changing the draw moves the score 3× more than changing lr or history.
- Halving lr makes the loss slightly **worse** (3.125 vs 3.091 e-8 after 3 restarts). The strong Wolfe line search already chooses the step, so a smaller lr only shrinks the first trial step.
- **Within one round on one draw**, the score swings by std Δlog 0.11 from one restart to the next, while the loss moves only 1-4 % per restart. That is the same size as the round-to-round score swing. In every arm, the first restart on the new draw made the score worse, by 16-43 %.

## 4. Verdict

| Candidate | Verdict | Evidence |
|---|---|---|
| **A** — L-BFGS settings (lr, line search, history, tolerances, max_eval) | **Not the cause** | 0 early stops, 0 restarts that raise the loss and 0 NaNs in 6,325 late restarts. The loss falls smoothly in every restart. Halving lr or cutting history to 10 changes the loss by 1-4 % and the score by std log 0.033, a third of the draw-to-draw effect (0.099). lr = 1 with strong Wolfe is the standard L-BFGS use. |
| **B** — restart scheme | **Not the cause** | The restarts reuse one L-BFGS object, so curvature is kept within a round. It is reset once per round, at restart 1. The last restart is also the lowest-loss restart. The one gap: the network is kept unconditionally and is never chosen on validation. That matters only because of the next row. |
| **C** — per-round change of states | **The source** | The loss jitter equals the 32,000-state sampling floor (observed 0.153 vs expected 0.12-0.17 for 64/2; 0.024 vs 0.044-0.060 for 256/16). On a fixed set, the loss falls steadily (jitter 0.026 / 0.008). Each round's gain equals the next draw's jump (about ±10 %). The chain-shift part is negligible (≤ 0.03 %); the subsample part dominates. |

Why the **validation error** oscillates more than the loss explains: it moves 10-43 % between restarts on the same draw while the loss moves 1-4 %. Its correlation with the fixed-set loss is 0.17 / −0.28. So the median endpoint error after the full chain lies along directions that the stage loss barely constrains. Each new draw pushes the weights along those directions, and the endpoint error goes up or down at random. In the cost-weighted 3-8 GeV runs it even drifts upward while the loss falls. The oscillation is therefore a **mismatch between loss and score, driven by the small per-round sample**, not a step-size problem. Separately, the fixed-set loss is still falling when the stopping rule fires (−1.3 % per round for 49e5e2450eaa, −0.4 % per round for aec34bbd4934). The plateau is in the validation error, not in the loss.

Levers this points to (not tested here):
1. Make the per-round sample less noisy: more states per round, or a fixed large training set reused across rounds.
2. Average the weights of the late rounds, or keep the best snapshot by validation instead of the last one. A cheap offline test: average snapshots 30-59 of 49e5e2450eaa and score the average.
3. Find out why the endpoint error is so weakly tied to the stage loss. For example, check the signed endpoint bias per round, and check which momenta the validation median is dominated by in the cost-weighted runs.

Not checked: the other seven runs' noise floors (their late statistics have the same pattern), and full 25-restart arms.

## Figures

- `per_round_score_and_loss.png`: per-round validation error and loss for all nine runs, last 30 rounds shaded.
- `spread_vs_noise_floor.png`: observed round-to-round spread vs the noise floor and vs the fixed-set loss.
- `late_restarts.png`: loss after every restart, last 6 rounds, for 49e5e2450eaa and aec34bbd4934.
- `fixed_set_loss_vs_score.png`: late snapshots on one fixed 128k set vs the validation error.
- `lbfgs_arms.png`: the six L-BFGS arms from snapshot 58.
