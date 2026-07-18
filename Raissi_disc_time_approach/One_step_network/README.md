# One_step_network — step 2: the paper's technique, first attempt on LHCb

The discrete-time construction of Raissi et al. (2019) §3.2, transplanted from
the van der Pol study to one real LHCb extrapolation: a single network that
takes a track state at the last UT plane and proposes the q stage states and
the endpoint of ONE implicit Gauss-Legendre step across the entire magnet,
trained purely by the scheme's own reconstruction equations — no labels.

## Fixed by the previous steps (nothing here is a free choice)

- **Leg**: z0 = 2648.2 mm -> z1 = 7826.0 mm (dz = 5177.8 mm), the modal plane
  pair of the real cross-magnet legs (Task C).
- **q = 8**: the exact-scheme guardrail showed the ceiling at this leg is
  ~29 um at q = 8 and plateaus there (the C0-field floor) — more stages buy
  nothing and cost outputs (Simple_first_pass).
- **Loss normalisation**: fixed per-component scales from the step-0 data look
  (frozen-leg population, label-free).
- **Protocol**: mirrored from the van der Pol implementation
  (`Van_Der_Pole/discrete_time_network/{model,training}.py`) so differences are
  the problem's, not the optimiser's: 4 hidden layers x 50 tanh, float64,
  full-batch L-BFGS (max_iter 200, strong Wolfe), up to 6 restarts, early stop
  after two <1% restarts, seeds {0, 1, 2}, 2000 training states.

## What is genuinely new vs van der Pol

- the ODE is **non-autonomous** (B varies along z) and lives in mixed units ->
  the stage nodes z0 + c_j dz are fixed constants of the leg, and inputs /
  outputs / residuals carry fixed physical scales;
- the rates need the **field, differentiably**: `field_torch.py` is a torch
  fp64 twin of the canonical trilinear v8r1 loader, gated against the numpy
  original at machine precision;
- training states are **real event states** (B-leg train split, re-based to
  the frozen planes with the fp64 engine — a <=25 mm exact transport), not a
  synthetic rectangle;
- the **data twin** (paper technique's control): identical network, seeds and
  budget, trained on RK4 stage+endpoint labels instead of the physics loss —
  the van der Pol step-6 comparison, delivered together with the headline.

## The runs

{physics, data} x seeds {0,1,2} = 6 trainings. Scored on held-out test-split
states (never trained on, split by particle): endpoint and per-stage error vs
the fp64 RK4 reference, in um and in the agreed rho metric, versus momentum,
against three baselines: the exact-scheme ceiling (29 um), the straight line,
and the data twin.

## Success criteria (honest, pre-stated)

1. The physics-only training converges (loss falls to a stable floor).
2. The measured endpoint floor is reported against the 29 um ceiling and the
   data twin — the LHCb answer to "what does the physics loss buy".
3. No expectation of beating the deployed supervised surrogate (10.7 um);
   van der Pol found physics ~2x worse than data at equal capacity.

| file | what it does |
|---|---|
| [field_torch.py](field_torch.py) | differentiable fp64 trilinear v8r1 field + parity gate vs numpy |
| [prepare_data.py](prepare_data.py) | re-base B legs to the frozen planes, scales, labels for the twin -> `results/frozen_leg_data.npz` |
| [model.py](model.py) | the one-step network, reconstruction residuals, physics + data losses |
| [training.py](training.py) | the 6-run protocol + held-out scoring -> `results/` |
| [plot_one_step.py](plot_one_step.py) | figures -> `figures/` |
| [one_step_network.ipynb](one_step_network.ipynb) | loads results, displays figures |
