# 2026-09-23 — what the network emits: documentation relabel (Claude)

George's ruling (2026-09-23): every chained network in this programme writes its outputs as
the straight-line extrapolation of the input plus a per-track-scaled network term, and the
documentation must say so plainly. Reframing only: nothing retrained, no behaviour changed.

Checked in the source (`Block_E_single_network_chain/E1_Network_grid/chain_network.py`,
`ChainNetwork.forward`): `return self.straight(S) + self.residual_scale(S, z_start) *
self.raw(S, z_start)`, with `straight_j = (x + tx (z_j - z_start), y + ty (z_j - z_start), tx,
ty)` for each of the q stage planes and the end of the step. (`residual_scale` is the source's
identifier; it is not renamed. In this block "residual" otherwise means the reconstruction
residual r of the loss.)

Canonical wording, now in `README.md` ("The network: a correction to the straight line") and
`PLAN.md` §1:

> The network (imported unchanged from the earlier studies) emits the q Gauss–Legendre stage
> states and the endpoint state of one step, each written as the straight-line extrapolation
> of the input state plus a network-predicted correction, scaled per track:
> output_j = straight_j(S) + σ(S) ⊙ NN(S, z_start)_j. The network therefore learns the
> correction to the straight line — the magnet's bending over one step — rather than the stage
> states themselves. This departs from Raissi, Perdikaris and Karniadakis (2019), whose network
> emits the states directly. The windowed loss studied here is unchanged by this: it sees only
> the resulting stage states through the reconstruction residual. The form was introduced on 5
> September 2026 for a network serving many step lengths; it was put to George as an explicit
> choice for the fixed-step study and chosen on 14 September 2026 (question 4 of that plan);
> the single-network chain then inherited it as 'Block D's form' without re-examining it, and
> the theory did not state that the learned quantity is the correction rather than the stage
> states (noted 23 September 2026).

Edited: `README.md`, `PLAN.md`, `HANDOFF_PROMPT.md` (a dated bracketed note under the Block E
bullet; George's own text untouched), `G0_Weighting/README.md`, `G1_Training/README.md`, and
the docstrings of `G0_Weighting/windowed_loss.py` and `G0_Weighting/check_windowed_weights.py`
(gate W3 note: the stub network hands the loss the exact states, bypassing the output form).
Both .py files: `ast.dump` with docstrings stripped is identical before and after; written by
atomic replace.

Deliberately NOT edited: `G1_Training/train_windowed.py` — the three runs of cluster 5849460
are training from it and each keeper resubmission re-launches it from disk; its docstring
does not describe the output form anyway ("The network ... imported"). Nothing under
`G1_Training/results/` or `G1_Training/condor/` touched. G2/G3 READMEs and scripts do not
describe the output form, so they are unchanged. Earlier worklogs are append-only and do not
describe it either.
