# S1_Simple_first_pass — step 1: the exact scheme, no network

The Gauss-Legendre implicit Runge-Kutta scheme solved directly (root-finder,
fp64) on the LHCb equation of motion, over real legs from the event-derived
training set. Two questions, answered before any network is trained:

1. **Feasibility** — does the implicit solve converge on our C0 trilinear
   field, including one giant step across the whole magnet (the analogue of
   the paper's q = 100, dt = 0.8 Allen-Cahn step)?
2. **Ceiling** — the exact scheme's endpoint error vs the fp64 RK4 reference,
   per (q, leg type, momentum): the floor under anything a network trained on
   these same equations can achieve, and the input to choosing step 2's q.

| file | what it does |
|---|---|
| [irk.py](irk.py) | the tableau for any q + its verification battery — ported verbatim from the van der Pol study (provenance in the header); `python irk.py` reruns every identity check |
| [exact_scheme.py](exact_scheme.py) | the leg solver (stages via scipy hybr, straight-line initial guess, mixed-unit scaling) + the (q in {2,4,8,16,32}) x (leg A-D) x (stratified momentum) scan -> `results/scheme_scan.csv`, `results/scheme_error_vs_q.csv` |
| [plot_scheme.py](plot_scheme.py) | `figures/scheme_error_vs_q.png` — the ceiling curves + convergence map |
| [simple_first_pass.ipynb](simple_first_pass.ipynb) | loads the results, displays the figures |

Start states are real legs (train split, stratified in p, 32 per leg type);
the reference endpoints are recomputed in fp64. Solver details: qop held
fixed (4 unknowns per stage), residuals scaled to [mm, mm, mrad, mrad],
convergence = residual < 1e-8 in those units.
