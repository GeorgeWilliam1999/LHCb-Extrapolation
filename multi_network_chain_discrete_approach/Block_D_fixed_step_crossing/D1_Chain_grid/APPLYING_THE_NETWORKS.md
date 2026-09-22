# Applying the Block D networks correctly

Every network in `results/N<NNN>_q<qq>/` is a one-step extrapolator for **one
fixed leg** of the magnet crossing. Used on any other leg, in any other order,
or on a state that is not on its start plane, it returns nonsense without any
error message. This page is the contract.

## What one network is

| item | value |
|---|---|
| input | one track state on the leg's start plane z_k: `(x, y, tx, ty, q/p)` |
| units | x, y in **mm**; tx = dx/dz, ty = dy/dz (dimensionless); q/p in **Allen units**, `q/p = 0.299792458 * q / p[GeV]` (so p = 10 GeV gives 0.03) |
| output | the state's `(x, y, tx, ty)` at the q Gauss–Legendre planes inside the leg and at the leg's end plane z_k + dz; q/p is **not** an output — it is conserved, carry it through unchanged |
| precision | fp64 throughout (the networks were trained in fp64; do not cast to fp32) |
| field | the v8r1 **MagUp** map, the polarity of the sample the networks were trained on; a MagDown state is a different problem |
| population | forward tracks of the official simulated sample with 2 < η < 5, 1 < p < 200 GeV, non-electrons, whose field-only path stays inside the map. Outside that, the output is an extrapolation of the network, not of the physics |
| architecture | two hidden layers of 128 (tanh), `4(q+1)` outputs; the output is `straight line + scale × raw`, where the scale is computed from the input alone (`chain_model.py`) |

## What a chain is

For a step count N the crossing z0 = 2648.2 mm (last UT plane) to z1 = 7826.0 mm
(first SciFi plane), L = 5177.8 mm, is cut into N equal legs, dz = L / N:

    leg k : z_k = z0 + k·dz  →  z_k + dz,     k = 0 … N−1

and `results/N<NNN>_q<qq>/leg<kkk>.pt` is the network for leg k at stage count q.
To extrapolate a state from z0 to z1 you apply leg 0, take its end state, hand it
(with the original q/p) to leg 1, and so on to leg N−1. **Nothing else is valid:**

1. The input to leg 0 must be on z0 = 2648.2 mm exactly. A state on another plane
   is first transported there with a field-only integrator (`_shared.reference.rk6_rows`
   or `rk4_rows`), as D0 did with the real last-UT states.
2. The legs are applied in order 0, 1, …, N−1, each once. Leg k is never applied
   to a state on a plane other than z_k.
3. Every leg of a chain has the same q. Do not mix networks from different q
   folders — they were trained on different predecessors (see below).
4. q/p is passed through unchanged at every leg.
5. The stage outputs (the first q blocks) are internal states at the Gauss
   nodes; they can be read for diagnostics but are not the answer.

`apply_chain.py` implements exactly this and refuses malformed input:

```python
from apply_chain import load_chain
chain = load_chain("results", N=16, q=8)
states = chain.extrapolate(S0)    # S0 (n, 5) on z0; states (n, N+1, 5), one row per plane
end = states[:, -1]               # on z1
```

`python apply_chain.py --N 16 --q 8 --check` reloads the chain, runs D0's test
particles through it and must reproduce the median recorded in `chain.json`.

## Why the legs belong together (the paper's protocol)

Leg 0 was trained on the real start states. Leg k was trained on **the states leg
k−1 predicted** for the same particles — the sequential protocol of Raissi,
Perdikaris and Karniadakis (2019), section 3, in which each step's prediction is
the next step's initial data. So a later leg has learnt to take *its
predecessor's* outputs, with their error, as inputs. That is why a chain is one
object: leg 3 of the (N = 4, q = 8) chain is not interchangeable with leg 3 of
the (N = 4, q = 12) chain even though both cover the same leg.

Each leg's own record, `leg<kkk>.json`, is its error against the RK6 propagation
of *its own inputs* — how good the network is at the step it was given. The
chain's record, `chain.json`, is the error against the RK6 truth from the real
start state — what a user of the chain gets. The two differ by what each leg
inherited.

## What the files are

| file | what |
|---|---|
| `leg<kkk>.pt` | the state dict (weights, plus the constant buffers `c`, `cout`, `z0`, `dz`, `u`, `in_scale`, `out_scale`) |
| `leg<kkk>_scale.json` | the input scale the leg was trained with, its planes and q — `load_chain` rebuilds the model from this and the `.pt` |
| `leg<kkk>_history.csv` | one row per L-BFGS restart (phase, loss, wall) |
| `leg<kkk>.json` | the shared trainer's record: convergence, restarts, the leg's own-step scores on train/val/test |
| `states.npz` | the chain's predicted states on every plane for the train, val and test particles |
| `chain.json` | the chain's scores against the RK6 endpoint, the real SciFi state, the straight line — the summary measures and, under `components`, x, y, tx and ty separately with their bias; per-plane growth per component; `qop_passthrough_max_abs_change` (must be 0); the per-leg summary; cost |

Every leg was trained with seed 0 (`torch.manual_seed(0)` before construction);
`converged = true` in a leg record means the shared trainer's stall-and-confirm
protocol held; a chain with any `converged = false` leg is still a valid chain
(the paper's protocol does not stop for it) and `chain.json` says so in
`all_legs_converged`.

## Things that look right and are wrong

- Feeding a state on the last UT plane of *this* particle (2641.8 or 2656.8 mm,
  say) straight into leg 0. It must be on 2648.2 mm.
- Applying the N = 1 network twice to "go further". It goes from z0 to z1 and
  nowhere else.
- Using q/p in GeV⁻¹ rather than Allen units (a factor 0.2998).
- Casting to fp32: the residual scale on the 40 mm legs is ~1e-2 mm, and fp32
  loses it against x ~ 1e3 mm.
- Running on a MagDown state or with a MagDown field expectation.
