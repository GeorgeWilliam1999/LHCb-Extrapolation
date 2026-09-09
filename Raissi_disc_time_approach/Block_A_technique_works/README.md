# Block A — does the technique work, and what limits it?

5–6 September 2026. Block 0 showed the scheme could be solved and a network
could be trained on one frozen magnet crossing. Block A asks the two questions
that decide whether the technique is worth pursuing: **does it work**, and
**can it be used** — that is, does it survive a network that must handle any
leg rather than one frozen crossing.

The answer split. It works; the general form did not, until the output
parameterisation was changed.

| Folder | Step | Question | Verdict |
|---|---|---|---|
| [`A1_Stage_count_sweep/`](A1_Stage_count_sweep/) | A1 | How many Gauss–Legendre stages does the crossing need? | q = 8 is the working point. Below it the network reproduces the *scheme's* own error; at q = 8 the scheme drops an order of magnitude below the network's floor and stops being the limiter; q = 16 buys nothing. |
| [`A2_Network_size_and_seed_study/`](A2_Network_size_and_seed_study/) | A2 | Is the floor set by capacity or by the optimiser? | By the optimiser. Spearman +0.965 between converged loss and endpoint error; a 4×200 network beats its own supervised twin. |
| [`A3a_General_leg_network/`](A3a_General_leg_network/) | A3a | One network for legs A, B and C, not one frozen crossing | Failed with absolute-state outputs (leg A 404 µm, leg B 2600 µm, leg C 300 µm — worse than a straight line on short legs). Fixed by the residual redesign below. |
| [`A3b_Chained_legs/`](A3b_Chained_legs/) | A3b | Does it survive being walked along a real particle path? | Errors compound roughly 2× per leg; the composite leg D is worse than one giant step. |
| [`A4_Magnet_up_field/`](A4_Magnet_up_field/) | A4 | Can it train where **no labels exist**? | Yes — 229 µm on the magnet-up map, the same floor as the labelled case. This is the one place the label-free loss buys something a supervised fit cannot have. |

## The residual redesign — the result that rescued A3a

The general-leg network failed for a reason that was not capacity. It predicted
**absolute** end states with one population-wide output scale, so a 0.1 mm leg
and a 5 m leg were asked for numbers differing by four orders of magnitude, and
the short legs were swamped.

Predicting instead the **deviation from straight-line propagation**, scaled per
leg by that leg's own bending integral (`A3a_General_leg_network/residual_model.py`,
`prepare_residual.py`, `train_residual.py`), took the whole-split test median
from ~400 µm to **2–3 µm** — legs A and C improve by 92× and 190×, and end up
2–4× *below* the straight line. Leg B improves only 2.5× (1026 µm against a
twin's 512 µm and a scheme ceiling of 46 µm), which is the shared-network cost
of a mixed leg population, not a limit of the physics loss.

Output parameterisation, not capacity, was the limiter. That lesson carries
directly into Block C, where the same straight-line-residual form is used
throughout.

## Caveats that travel with these numbers

- **Polarity.** Block A used the magnet-**down** map against an official sample
  that is magnet-**up**; see [`../Block_0_first_pass/README.md`](../Block_0_first_pass/README.md).
  The method conclusions stand, the "labels are ground truth" framing does not.
- **Threads.** The July v2 runs were four-thread; the single-thread reruns here
  differ in the fourth digit. Compare at ~1e-3 relative, never bitwise.
