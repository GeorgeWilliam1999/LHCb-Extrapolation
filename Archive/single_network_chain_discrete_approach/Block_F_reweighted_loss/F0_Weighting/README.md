# F0 — the weights, and the gates that justify them

[weighted_loss.py](weighted_loss.py) is the only new physics in Block F: it builds the
weight that multiplies the reconstruction residual, and nothing else in the training
changes. The form and the reasoning are in its docstring and in [../README.md](../README.md).

[check_weights.py](check_weights.py) runs four gates. `results/check_weights.json` holds
every number; `figures/weighting_preflight.png` draws the last one.

| gate | what it proves | result |
|---|---|---|
| 1 | the torch weights equal an independent numpy implementation, and the lever arm runs from one step (last plane) to L + dz(1 − c₁) (first plane) | 1.0e-15, exact |
| 2 | with mode `blockE` the weighted path reproduces `_shared.model.physics_loss` | identical to the last bit |
| 3 | the reweighting does not move the minimum: the exactly solved collocation states give machine zero under **every** mode | 6e-30 … 2e-29 of the straight line |
| 4 | the pre-flight: where the loss comes from, and whether it looks where the error is | see below |

[field_correlations.py](field_correlations.py) (added 2026-09-20) is the provenance for the three correlations quoted in the block README, which until then had no script behind them: slope error against \|B\| −0.57, cost at z1 against distance left +0.996, cost against \|B\| +0.12 (Spearman over the 64 steps of Block E's N = 64, q = 2 anatomy; `results/field_correlations.json`).

Gate 1 caught a real difference on its first run: `torch.median` returns the lower of the
two middle values and `numpy.median` averages them, which moved the clamp threshold. Both
now use the 0.5 quantile.

## The pre-flight (gate 4)

Block E's trained N = 64, q = 2 network, 8,000 (track, plane) states drawn evenly over the
planes. "Cost" is what a state's error actually incurs at z1 — its local error against RK6
taken from the same state, position plus slope times the distance left. That reference is a
diagnostic only; it never enters the loss.

| weighting | 2–5 GeV | 10–50 GeV | >50 GeV | first quarter of z | last quarter | ρ(share, cost) in 10–50 GeV |
|---|---|---|---|---|---|---|
| Block E | 97.5% | 1.5% | 0.0% | 0.6% | 6.2% | +0.559 |
| **full** | 29.2% | **57.8%** | 6.7% | **21.2%** | 1.7% | **+0.889** |
| no_lever | 24.2% | 61.7% | 6.4% | 9.7% | 21.2% | +0.552 |
| no_track | 82.4% | 11.3% | 0.3% | 6.9% | 1.0% | +0.951 |
| no_window | 71.8% | 22.2% | 3.1% | 9.4% | 0.9% | +0.881 |

Read across: Block E's loss is 97.5% about 2–5 GeV tracks and puts 0.6% of its attention on
the first quarter of the crossing, where an error costs the most. The full weighting moves
that to 57.8% in the band that matters and 21.2% in the first quarter, and inside that band
it ranks states by their true endpoint cost at +0.889 rather than +0.559.

The **overall** correlation falls (+0.690 to +0.561) and that is expected, not a defect: the
network's biggest errors today are the 2–5 GeV tracks, which this weighting deliberately
stops chasing.

### The clamp, which this gate set

`a_n` scales as |q/p|, so it spans about 200 across the sample and 40,000 once squared.
Without a clamp the 29 tracks above 100 GeV take a fifth of the loss:

| clamp | 2–5 GeV | 10–50 GeV | >50 GeV |
|---|---|---|---|
| off | 14.8% | 60.9% | 19.8% |
| 2 | 80.9% | 15.3% | 0.5% |
| 3 | 59.2% | 33.7% | 2.0% |
| **5** | **29.2%** | **57.8%** | **6.7%** |
| 10 | 15.0% | 61.8% | 18.6% |

5 is the setting: it buys nearly all of the in-band weight while keeping the >50 GeV tail
below 7%. `CLAMP = 5.0` in `weighted_loss.py`.

### One thing the weighting does not fix

The loss stays very concentrated: the top 1% of states carry 83.8% of it, against 97.9%
under Block E. That is the heavy tail of the residual distribution, not the weighting, and
no choice of weight removes it. Worth remembering if the runs behave oddly.

```bash
PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python check_weights.py
```
