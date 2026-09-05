# `_shared` — the common foundation for the discrete-time experiments

Every experiment in `Raissi_disc_time_approach/` is one folder with its own
scripts, `results/`, `figures/` and notebook. This package holds the pieces they
all need, so that a difference between two experiments is a difference in the
experiment and never a drifted copy of the physics or the optimiser.

**The physics and the optimiser protocol are those of `../One_step_network_v2`,
unchanged.** That experiment is the verified baseline; this package only lifts
its hard-wired assumptions — one fixed leg, eight stages, the MagDown map — so
that the same code can also handle any number of stages, either field polarity,
and per-sample start planes and step lengths. When those generalisations are
switched off, the arithmetic is the baseline's, operation for operation: the
initial parameters, the physics loss, the data loss and the gradients are
bitwise identical, and one L-BFGS restart lands on the same number to the last
bit (`smoke_tests.py`).

## File → role

| file | role |
|---|---|
| [field_v8r1.py](field_v8r1.py) | the canonical v8r1 field-map loader (numpy), **vendored** from the archive at commit `1faa97e0`. Nothing imports the archive any more. |
| [vendoring_parity.py](vendoring_parity.py) | the gate on that copy: field values on 200,000 random in-map points must be *bit-identical* to the archive import, and the map md5 must be `af284c6954d2273c637a5e766b82b58e` → [results/vendoring_parity.json](results/vendoring_parity.json) |
| [irk.py](irk.py) | the Gauss-Legendre tableau for any number of stages plus its verification battery, vendored from `../Simple_first_pass/irk.py`. `python irk.py` reruns every identity check. |
| [reference.py](reference.py) | the reference card: the ODE (`deriv`), the fp64 RK4 reference (`rk4_rows`), the metric `rho`, the field paths and loaders, the frozen leg, and `load_training` for the event-derived sample |
| [field_torch.py](field_torch.py) | the differentiable fp64 torch twin of the field, so the physics loss can take gradients through **B**(x, y, z); `parity()` is its gate against the numpy loader |
| [model.py](model.py) | the one-step network, the rates, the reconstruction residuals, and the physics and data losses |
| [prepare.py](prepare.py) | the two dataset builders: the frozen leg and the general leg |
| [evaluate.py](evaluate.py) | the scoring helpers `train.py` reports with, and `chain` — the network applied leg after leg on its own output |
| [train.py](train.py) | the command-line driver: one `(mode, seed)` trained to genuine stall, confirmed, scored and recorded. Resumable. |
| [smoke_tests.py](smoke_tests.py) | the gates listed at the bottom of this file. Run before trusting anything built on this package. |
| [use_shared.py](use_shared.py) | the path helper to copy into an experiment folder |
| [condor/](condor/) | the farm wrapper, the submit template and how to use them ([condor/README.md](condor/README.md)) |

## Importing from an experiment folder

Copy `use_shared.py` into the experiment folder and import it first:

```python
import use_shared                       # noqa: F401 — puts _shared on sys.path
from _shared.prepare import frozen_leg_dataset
from _shared.evaluate import chain, score_split
```

or, without the copy, the same two lines inline:

```python
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
```

## The API

### `reference.py`

```python
V8R1_DOWN, V8R1_UP                  # the two field-map paths on CVMFS
KAPPA, C_QP, RK4_STEP, FROZEN_LEG   # Allen constants; the baseline leg
DATA_NPZ                            # the v2 (official-sample) training set

make_field(which='down') -> FieldV8R1          # 'down' | 'up', cached per process
field_path(which='down') -> str
field_md5(which='down', chunk=1<<20) -> str
field_bounds(field) -> (lo, hi)                # map corners in mm, 3-vectors each
deriv(S, z, field=None) -> (N, 5)              # dS/dz; field defaults to MagDown
rk4_rows(S0, z0, z1, step=5.0, field=None) -> (N, 5)
rho(y_ref, y_pred) -> (N,)                     # the agreed scalar relative error
load_training(split=None, leg=None, core_only=False, npz=None) -> dict
leg_indices(leg) -> list[int]                  # 'B' -> [1]; ('A','B') -> [0,1]
gauss_legendre(q, verify=True) -> (c, A, b)    # re-exported from irk.py
card(field_which='down') -> dict
```

### `field_torch.py`

```python
FieldTorch(np_field=None)                      # any FieldV8R1; None = MagDown
    forward(x, y, z) -> (Bx, By, Bz)           # Tesla, differentiable
parity(which='down', n=200_000, seed=0, tol=1e-12, verbose=True) -> dict
```

### `model.py`

```python
LHCbRates(field=None)
    forward(S4, qop, z) -> (N, q, 4)           # z is (q,) or (N, q)

OneStepNetwork(q, in_scale, out_scale, width=50, depth=4, n_extra=0)
    forward(S, extra=None) -> (N, q+1, 4)      # q stage states + the endpoint

reconstruction_residuals(model, rates, S, dz, znodes, A, b, extra=None)
physics_loss(model, rates, S, dz, znodes, A, b, extra=None) -> scalar
data_loss(model, S, ref, extra=None) -> scalar
```

`n_extra` is the number of already-normalised extra inputs the network takes;
the general-leg experiments pass 2, carrying `(z0, dz)`. `dz` may be a python
float (one frozen leg) or an `(N,)` tensor; `znodes` may be `(q,)` or `(N, q)`.

### `prepare.py`

```python
frozen_leg_dataset(q=8, field='down', n_train=2000, seed=20260718,
                   rebase_mm=60.0, fiducial=True, out_npz=None,
                   z0=None, z1=None, training_npz=None, verbose=True) -> dict

general_leg_dataset(legs=('A','B','C'), q=8, n_train=2000, field='down',
                    out_npz=None, seed=20260718, n_eval=2000,
                    fiducial=True, training_npz=None, verbose=True) -> dict
```

Both write `<out_npz>` and `<out_npz without .npz>_meta.json` (counts, scales,
how many states the fiducial cut removed). Contents:

| key | frozen | general |
|---|---|---|
| `kind`, `q`, `c`, `field`, `in_scale`, `out_scale` | ✓ | ✓ |
| `znodes` (q,), `zout` (q+1,), `z0`, `z1` | ✓ | — |
| `extra_mean`, `extra_scale`, `legs` | — | ✓ |
| `{split}_S` (N,5), `{split}_P` (N,), `{split}_ref` (N,q+1,5) | ✓ | ✓ |
| `{split}_z0` (N,), `{split}_dz` (N,) | ✓ | ✓ |
| `{split}_znodes` (N,q), `{split}_extra` (N,2), `{split}_LEG` (N,) | — | ✓ |

Both apply the fiducial requirement of the baseline: the reference trajectory
must stay inside the field map, i.e. inside the region where the ODE is defined
at all. Without it, the ~1% of states that leave the map carry about half of the
whole physics loss.

### `evaluate.py`

```python
split_arrays(data, split) -> (S, ref, z0, dz, extra, znodes)
predict(model, S, extra=None) -> (N, q+1, 4)
score_against_reference(out, S, ref, dz) -> dict
score_split(model, data, split='test') -> (dict, out)
chain(model, S0, legs, extra_mean=None, extra_scale=None) -> (N, n_legs, 5)
chain_reference(S0, legs, field='down', step=5.0) -> (N, n_legs, 5)
chain_errors(pred_states, ref_states) -> dict
```

`legs` is a sequence of `(z0, z1)` plane pairs; each entry may be a scalar pair
(the same leg for everyone) or a pair of `(N,)` arrays (a per-sample leg).
`chain` feeds each predicted endpoint forward as the next leg's start state,
carrying `qop` through unchanged.

The score dict is: `endpoint_med_um`, `endpoint_p95_um`, `stage_med_um`,
`slope_med_mrad`, `rho_mean`, `rho_median`, `straight_med_um`, `n`.

### `train.py`

```
--data <npz>            the prepared dataset
--mode physics|data     the paper's loss, or the supervised twin
--seed N
--q N                   optional; must match the dataset's q
--width 50  --depth 4
--out <dir>  --tag <str>
--field down|up         optional; defaults to the dataset's own
--outer-cap 400         safety cap on total L-BFGS restarts (both phases)
--max-iter 200          L-BFGS iterations inside one restart
--no-confirm            stop at the first stall, skip the confirmation pass
```

fp64 throughout; one thread (the environment variables are set before torch is
imported, and `torch.set_num_threads(1)` is called — multithreaded torch
spin-waits on this node and is about 68x slower). Full-batch L-BFGS with
`max_iter 200`, strong Wolfe, history 120, restarted until two consecutive
restarts each improve the loss by less than 1%. Then, unless `--no-confirm`, a
**confirmation pass**: a fresh optimiser continues the run, and it counts as
converged only if it re-stalls within two restarts with the endpoint medians
unchanged.

Outputs in `<out>/`: `<tag>.pt` (rewritten every restart), `<tag>_history.csv`
(one row per restart: mode, seed, phase, outer, loss, wall_s) and `<tag>.json`.
Because both are written every restart, **rerunning the same command resumes**
rather than starting over. The json records `mode`, `seed`, `q`, `width`,
`depth`, `restarts`, `final_loss`, `converged`, `wall_s`, the dataset and field,
and the full score dict on `train`, `val` and `test`. One `SUMMARY ...` line is
printed at the end.

## The gates

`PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python smoke_tests.py`

1. `frozen_leg_dataset(q=8)` reproduces the baseline dataset: 1979 / 2062 / 2018
   states and the same scales (measured: bitwise identical, and the 21 / 20 / 15
   states removed by the fiducial cut are the same ones).
2. The generalised model **is** the baseline model at `n_extra = 0`: identical
   parameters, physics loss, data loss and gradients, bitwise.
3. `train.py --mode physics --seed 0` reproduces the baseline's first L-BFGS
   restart bitwise (4.927638009365614e-03). The number recorded in
   `../One_step_network_v2/results/hist_physics_seed0.csv` is
   4.924321332318405e-03, 6.7e-4 relative away, **because those runs used four
   BLAS threads and everything here runs on one**: a different reduction order
   changes the last bits, and 200 L-BFGS iterations with a line search amplify
   that into the fourth digit. The original `One_step_network/model.py` run on
   one thread gives the same 4.927638009365614e-03.
4. `general_leg_dataset(('A','B','C'))` builds and a physics restart runs on it.
5. `make_field('up')` loads and the torch twin agrees with it (1.3e-15 T).
6. `chain` runs a model over three sub-legs of each sample's own leg.

The vendoring gate is separate (`vendoring_parity.py`), and the farm gate needs
a submit host (`condor/README.md`).
