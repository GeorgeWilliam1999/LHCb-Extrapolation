# Block F — reweighting the loss, and nothing else

**Question.** Block E's chains sit at 137–246 µm at the first SciFi plane while a single
step of the same network is under a micrometre, so the endpoint error is accumulation. The
anatomy (`../Block_E_single_network_chain/E3_Analysis/error_anatomy.py`) says what kind:
72% of it is slope errors made early and carried to the end, and 97% of the trained
residual comes from 2–5 GeV tracks. Neither of those is what the loss was asked for. **If
the loss is asked for the right thing instead, how much of the error goes away?**

**Status (2026-09-20): the three runs were submitted on 2026-09-18 evening (cluster 5809660).
N = 64, q = 2 and N = 128, q = 8 have finished at their 1,000-restart cap (round 40) with the
plateau rule holding; N = 256, q = 16 is at round 31 and still falling. Both pre-registered
success criteria hold for N = 64, q = 2 (see `F2_Analysis/README.md`). Not yet discussed with
George; no write-up.** Plan and worklog in the Notion to-do "Train one network per step length
and chain it to itself across the magnet".

## The one change

Block E minimises the mean squared reconstruction residual with each component divided by
its **pooled** spread. Block F measures every residual as **the displacement it would cause
at the SciFi plane, as a fraction of that track's own total bend**, with the 10–50 GeV band
weighted up:

```
loss = mean over states n, outputs j, components d of ( a_n · e_{n,j,d} / D_ref )²

  e_x  = r_x                  e_y  = r_y                  [mm]
  e_tx = r_tx · lever_{n,j}   e_ty = r_ty · lever_{n,j}   [mm]

  lever_{n,j} = z1 − (z_start + c_j·dz) + dz      the distance left to travel, plus one step
                                                  (additive, so the last plane is not weighted
                                                  to zero; dz is 1.6% of L at N = 64)
  a_n = sqrt(W(p)) · D_ref / D_n                  clamped to [1/5, 5] of its median over the
                                                  round's batch (D_ref is fixed for the run)
  D_n = κ|q/p|·Ī·L                                the track's total bend, a constant of
                                                  the TRACK, not of the step
  W(p) = 1 on 10–50 GeV, Gaussian roll-off in log p outside, floored at 0.05
```

Both factors come from the input state and the field map alone, so the loss stays
label-free. Everything else is Block E's: the same tracks and splits, the same network
(imported from Block E, not copied), the same seed, the same L-BFGS protocol, the same
rounds. `_shared/` and Block E are read-only.

### Why not weight by the field

The obvious alternative — weight up the strong-field region, where the bending happens —
is wrong, and measurably so. Along the crossing the per-step slope error correlates **−0.57**
with |B| (Spearman; it is *worst* at the low-field ends), while a step's cost at z1 correlates
**+0.996** with the distance it still has to travel and only **+0.12** with |B|
(`F0_Weighting/field_correlations.py` → `results/field_correlations.json`, over the 64 steps of
Block E's N = 64, q = 2 anatomy; Pearson −0.62 / +0.98 / −0.02). That is also why `D_n` uses the track's
bend over the whole crossing rather than the field integral of its own step: dividing by
the per-step integral would ask for constant *relative* accuracy along z and so make the
high-field middle worse in absolute terms.

## Layout

| folder | role |
|---|---|
| [F0_Weighting/](F0_Weighting/) | the weights (`weighted_loss.py`) and the four gates, including the pre-flight that set the clamp |
| [F1_Training/](F1_Training/) | Block E's round trainer with the loss swapped (`train_weighted.py`, with the diff beside it), the farm harness, the runs |
| [F2_Analysis/](F2_Analysis/) | Block F against Block E, on the same measures and the same plateau rule |

## What would count as a result

Fixed before the runs, and applied the same way to Block E's existing logs so neither side
is favoured. A run has **plateaued** once the median validation error of its last ten rounds
has stopped falling — no more than 5% below the median of the ten before it, for three
rounds running; its result is that median with the round-to-round spread beside it.

Applied to Block E, that rule says **none of its runs had plateaued** when training was
stopped by hand on 2026-09-18: all eleven with twenty or more rounds were still improving,
by 5–24% per ten rounds (eleven out of eleven falling has probability 0.0005 under pure
noise). Block E's numbers — N = 64 q = 2 at **124 µm ± 13%**, N = 128 q = 8 at **162 µm ± 9%**,
N = 256 q = 16 at **173 µm ± 18%** (validation, max metric) — are therefore where its
training stopped, not where it converges, and the Block E counterparts have to be extended
to their own plateau before the comparison means anything.

- **Success** — the N = 64, q = 2 test radial error clears the ±13% band downward **and**
  the median per-step slope error falls below its present 0.0016 mrad. Reported overall and
  in the 10–50 GeV band.
- **Partial** — in-band improves, overall does not: the window was too hard, and the clamp
  is the knob.
- **Null** — nothing clears the band. Then the residual is not the binding constraint, and
  the next lever is supervising the step Jacobian.

The ablations (`--weighting no_lever` / `no_track` / `no_window`) are **not** run up front.
They are the diagnosis if the combined result is ambiguous; the trainer takes them from the
start so they cost farm time and no new code.
