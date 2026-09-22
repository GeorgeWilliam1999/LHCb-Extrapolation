# E1_Network_grid — one network per (N, q), trained over the whole crossing

**Status (2026-09-20): all 16 trained; see *Running* and *Extension and stopping* below.** The grid is 16
networks: N in {2, 64, 128, 256} (dz = 2588.9, 80.9, 40.5, 20.2 mm) × q in {2, 4, 8, 16},
one farm job each. One choice is open before submission: how long a training round is
(see *Open* below).

**The network** ([chain_network.py](chain_network.py)).
- **Inputs:** the state (x, y, tx, ty, q/p), divided by its spread over the round-1 training states, plus the z of the step's start plane, mapped onto [−1, 1] (George 2026-09-16, decision 2 (a)).
- **Outputs:** the q stage states and the end state, each as the deviation from the straight line times a per-track scale. x and tx keep Block D's scale, κ|q/p|∫|B|dz along the straight line. y and ty have their own, κ|q/p|∫√(1+tx²+ty²)((1+ty²)|Bx| + |tx·ty·By| + |tx·Bz|)dz, floored at 10⁻³ of the x scale.
- **Size:** 2×128, tanh; 18,956 parameters at q = 2.
- **Loss:** the shared label-free RK-PINN loss (`_shared.model.physics_loss`, unchanged), with the stage planes set per track.

**The trainer** ([train_network.py](train_network.py)).
- **Rounds:** round 1 trains on RK6 track states at the N start planes; later rounds train on the network's own predictions for all training tracks. Each round draws 32,000 (track, plane) pairs, the same number per plane.
- **Optimiser:** L-BFGS restarts (200 iterations, history 120, strong Wolfe).
- **Rescaling:** the loss is rescaled to 1 at the start of every round and again whenever it has fallen tenfold, each time with a fresh optimiser.
- **Logging:** every restart logs its iterations, function evaluations and an early-stop flag.
- **Stopping:** a round ends when two consecutive restarts each gain under 1%, or after `--round-restarts`. Training stops when a refreshed round's first two restarts each gain under 1% (followed by a confirmation pass), or at 20 rounds or 400 restarts. These are the trainer's defaults; at N ≥ 64 the 1% rule never fired and the runs were stopped on the validation error instead (see *Extension and stopping*).
- **Resume:** a checkpoint after every restart makes runs resumable.

## script → output

| script | what it does | output |
|---|---|---|
| [chain_network.py](chain_network.py) | the network class, numpy twins of the field integrals and scales, `carry` (the network applied N times), `load_network` | — |
| [metrics.py](metrics.py) | Block D's per-component scoring, copied unchanged (`chain_scores`: at z1 against RK6, against the real SciFi state, per plane, per momentum band, q/p check) | — |
| [check_network.py](check_network.py) | gates 1–3: straight line with a zeroed last layer; torch = numpy twins; mixed start planes in one batch; finite loss and gradient; the exact collocation solution gives a machine-precision loss; the y-scale check on the RK6 tracks | `results/check_network.json`, `figures/y_scale_check.png` |
| [train_network.py](train_network.py) | one (N, q): rounds, rescaled L-BFGS, checkpoints, resume; the final record scores the validation and test chains | `results/N<NNN>_q<qq>/{scale.json, network.pt, history.csv, rounds.csv, progress.json, round_states.npz, chain_states.npz, record.json}` |
| [make_jobs.py](make_jobs.py) | the 16-line job list and submit file (slowest first); does not submit | `condor/jobs.txt`, `condor/jobs.sub` |
| [resubmit.py](resubmit.py) | resubmits unfinished networks that are not queued; `--submit` to send | `condor/jobs_round<k>.{txt,sub}` |
| [condor/wrapper.sh](condor/wrapper.sh) | runs `train_network.py` single-threaded from this folder | — |

## Gate results (2026-09-16)

1. **Model: PASS.** At N = 2 and 256, q = 2 and 16:
   - a zeroed last layer gives the straight line exactly (difference 0);
   - the torch and numpy field integrals and scales agree to 6.4e-16 (relative);
   - one batch mixing every start plane equals the per-plane outputs to 5.7e-14;
   - loss and gradient are finite.
2. **Loss: PASS.** On 40 states per case, the exact collocation solution (Block C's solver, all converged) gives a loss of 4e-35 to 1.4e-31, against 5e-6 to 0.1 for the straight line: a ratio of 1e-29 or less.
3. **y scale: PASS.** On 20,000 (track, plane) pairs per N, the median |true one-step deviation / scale| is:
   - x and tx: 0.89–1.01 (99th percentile ≤ 1.5);
   - y and ty: 0.50–0.67 (99th percentile 1.0 on the short steps, 3.9–5.0 at N = 2);
   - under Block D's single scale, y and ty would sit at 0.004: that scale is about 130 times larger than the new y scale, and about 240 times larger than the deviations it has to describe.

   The y floor is active for 7.5–10.7% of pairs. See `figures/y_scale_check.png`.
4. **Trainer smoke run: PASS** (N = 2, q = 2, 300 training tracks, 200 validation and test tracks; log `results/smoke/smoke.log`, ignored by git).
   - **Resume:** stopped after 3 restarts, resumed at restart 3, and a rerun of the finished run exits at once.
   - **No early stops:** all 150 restarts ran their full 200 iterations; the loss fell from 0.19 to 2.8e-8 with 5 rescalings, still gaining 2% per restart at the cap.
   - **Error:** validation median at z1 9,186 µm, at the exact scheme's own error at N = 2, q = 2 (9,131 µm on the same 200 validation tracks).
   - **Round refresh** (N = 64, q = 2, 3,200 states, rounds of 3 restarts; `results/smoke/smoke_rounds_N064.log`): rounds 2 and 3 trained on the network's own predictions, 50 states on each of the 64 planes; validation median at z1 went 1,589 → 1,399 → 1,099 µm over 9 restarts.
5. **Timing: done.** One restart at N = 256, q = 16 on 32,000 states: 200 iterations, 205 evaluations, 226 s (34 µs per state per evaluation, one thread, submit host at load about 50). See `results/timing/N256_q16/timing.json`.

Exact scheme at N = 2 on those 200 validation tracks (a preview of E2): 9,131 / 204 / 7.1 / 4.9 µm at q = 2 / 4 / 8 / 16.

## Running (2026-09-17)

George agreed on rounds of 25 restarts. The 16 networks were submitted as cluster 5805460 (`make_jobs.py --round-restarts 25`, 01:16).
- The first restarts on the worker nodes ran their full 200 iterations in 93–219 s.
- N = 256 and 64 at q = 16 exceeded 4 GB. They were released with 8 GB, and the job maker now defaults to 8 GB.
- A background keeper (`condor/keeper.log`) releases memory-held jobs and resubmits unfinished ones every 30 minutes.

## Extension and stopping (2026-09-18 to 2026-09-20)

- **2026-09-18, stopped by hand** after 106–159 restarts at N = 2 and 383–760 at N ≥ 64, with the loss on freshly drawn states flat for ten rounds. Every network was re-scored from the weights on disk (`finalise.py`); the E3 analysis rests on those checkpoints, which were then copied to `results/N<NNN>_q<qq>/stopped_2026-09-18/` (record, weights, rounds, history, chain states, and since 2026-09-20 the `scale.json` needed to reload them). **That stop was premature:** the validation error had not been flat. Under the plateau rule below, none of the sixteen runs had stopped improving; all eleven with twenty or more rounds were still falling 5–24% per ten rounds.
- **Extended** the same evening from the checkpoints (`--extend`, caps raised to 800 and later 1,383 restarts and 40 → 80 → 120 rounds; cluster 5809659, then 5815090 and 5816352), N = 2, q = 2 excepted (it sits at the exact scheme's own error). Extending overwrites `record.json`, `network.pt` and `chain_states.npz` in the run folder, so the run folders now hold the extended networks and rerunning the E3 scripts today gives different, better numbers than the E3 README quotes.
- **What actually decides when a run is finished** is not the trainer's 1% rule but the validation error: the plan's rule never fires at N ≥ 64 because the rescaled loss keeps falling by more than 1% per restart long after the validation error has stopped moving. A run is done when the median validation error (max(|Δx|, |Δy|) at z1, as logged in `rounds.csv`) of its last ten rounds is no more than 5% below the median of the ten before, at each of the last three rounds (`plateaued_now` in `../../Block_F_reweighted_loss/F2_Analysis/compare_to_blockE.py`). The trainer itself runs to its cap; the keeper (`condor/keeper.sh`, calling `F2_Analysis/prune_active.py` before every resubmission) stops resubmitting once the rule holds. So the validation split is used for stopping, not only for monitoring; the test split is untouched.
- **Bugs fixed on the way:** the keeper's `condor_q -af:j` query returned nothing, so it never released held jobs (2026-09-18); `resubmit.py` judged a run finished by `record.json` alone, wrong for a run being extended (2026-09-18); both blocks' `resubmit.py` matched queued jobs on `--N --q` only, so a Block F job could pass for a Block E one (2026-09-18 evening); the prune script first required the trainer's rule as well as the plateau rule, which is unattainable at N ≥ 64 (2026-09-19); the plateau test first reported the *first* round the rule held rather than whether it holds *now* (2026-09-19).
- **State on 2026-09-20:** plateaued by the rule: N = 2 at q = 4, 8, 16; N = 64 at q = 2, 8, 16; N = 128 at q = 2, 4, 16 (N = 2, q = 2 was never extended). Still falling: N = 64 q = 4; N = 128 q = 8; N = 256 at every q. N = 128, q = 16 is pruned from the keeper's list but still running to its cap. Restart counts now 106–1,758, rounds 6–71.
