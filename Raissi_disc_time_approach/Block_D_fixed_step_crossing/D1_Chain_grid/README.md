# D1_Chain_grid — one network per fixed step, chained across the magnet  (D1)

**Question.** With one architecture (two hidden layers of 128) and the
label-free discrete-time loss, how does the error of a magnet crossing depend
on the number of steps N it is cut into and the number of Gauss–Legendre
stages q per step — when the networks are trained and chained exactly as the
paper steps?

    N in {1, 4, 16, 64, 128}      dz = 5178, 1294, 324, 81, 40 mm
    q in {1, 2, ..., 20}
    one network per leg, N legs per chain, 100 chains, 4,260 networks, one seed

**The protocol is the paper's (George 2026-09-14: "exactly as the paper").**
Leg 0 trains on the real start states at z0; leg k trains on the states leg
k−1 predicted at z_k for the same particles. The chain's prediction is what the
last leg emits. Everything about the optimiser is the shared trainer's
(fp64 full-batch L-BFGS, stall-and-confirm, `--outer-cap 400`).

**Read [APPLYING_THE_NETWORKS.md](APPLYING_THE_NETWORKS.md) before using any
of these networks.** A leg is valid on its own plane, in its own chain, at its
own q, and nowhere else.

## script → output

| script | what it does | output |
|---|---|---|
| [chain_model.py](chain_model.py) | the fixed-step straight-line-residual network (no extra inputs; z0 and dz are buffers), the twin's residual data loss | — |
| [test_chain_model.py](test_chain_model.py) | the gate: torch vs numpy straight line and field integral, zeroed last layer = straight line exactly, finite losses and gradients at q = 1, 8, 20 on the 40 mm and the 5178 mm leg, the q = 0 twin | prints `ALL CHAIN MODEL CHECKS PASS` |
| [train_chain.py](train_chain.py) | one (N, q) chain: builds each leg's dataset from its predecessor's predictions (RK6 reference through the Gauss nodes for scoring), trains it through `_shared/train.py`, applies it, scores the chain at every plane and at z1 against the RK6 truth and, carried to the particle's SciFi plane, against its real state | `results/N<NNN>_q<qq>/leg<kkk>.{pt,json,_history.csv,_scale.json}`, `states.npz`, `chain.json` |
| [make_jobs.py](make_jobs.py) | the 100-line job list (long chains first) | `condor/jobs_chain.txt`, `condor/jobs_chain.sub` |
| [resubmit_chain.py](resubmit_chain.py) | resubmits unfinished chains not in the queue; they resume | `condor/jobs_chain_round<N>.*` |
| [apply_chain.py](apply_chain.py) | `load_chain(results, N, q)` → `Chain.extrapolate(S0)`; `--check` reproduces the record | — |
| [metrics.py](metrics.py) | the scoring, per component: x, y [µm], tx, ty [mrad] separately (median, p95, mean, bias) beside the summary measures pos = max(\|dx\|,\|dy\|) and slope = max(\|dtx\|,\|dty\|); q/p is asserted unchanged along every chain | — |
| [score_components.py](score_components.py) | rescores a finished chain from its `states.npz` (for records written before the per-component scoring) | rewrites the val/test blocks of `chain.json` |
| [aggregate.py](aggregate.py) | the records → the tables | `results/chain_table.csv`, `table_*.csv`, `components.csv`, `table_*_<x,y,tx,ty>_med.csv`, `per_leg.csv`, `growth.csv`, `by_p_band.csv`, `status.csv` |
| [plot.py](plot.py) | the figures, from the tables alone | `figures/*.png` |
| [analysis.ipynb](analysis.ipynb) | loads the tables and figures; computes nothing | — |

```bash
PY=/data/bfys/gscriven/conda/envs/TE/bin/python; export PYTHONNOUSERSITE=1
$PY test_chain_model.py                                  # the gate
$PY train_chain.py --N 4 --q 8 --out results             # one chain, by hand
$PY make_jobs.py && condor_submit condor/jobs_chain.sub  # the grid
$PY resubmit_chain.py --submit                           # until every chain.json exists
$PY aggregate.py && $PY plot.py
```

## Timing (2026-09-14, submit host, one thread, 2,000 training states)

| leg | q | one restart |
|---|---|---|
| 40 mm (N = 128) | 20 | 16–19 s |

so a leg that stalls and confirms in ~50 restarts is ~15 min at q = 20 and
cheaper at small q; an N = 128 chain is a day-scale job and relies on the
resume path (a leg already trained and applied is skipped; a half-trained leg
resumes from its last restart).

## Findings

(filled in when the grid has run)
