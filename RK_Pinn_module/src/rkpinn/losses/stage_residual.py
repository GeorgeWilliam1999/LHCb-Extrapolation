"""The stage residual: the quantity every label-free loss is built on.

Each stage must reconstruct the input through the implicit Runge-Kutta
equations. For the training state n, the stage j and the component d,

    r[n, j, d] = S_hat[n, j, d]
                 - dz * sum over k of A[j, k] * f_d(S_hat[n, k], z_n + c_k dz)
                 - S[n, d]

where S is the input state, S_hat the stage states, A the stage matrix of the
tableau, c its nodes and f the rates of the equation of motion. The residual
is zero for every n, j and d exactly when the stage states solve the scheme.
No label appears in it. It is in the units of the state: millimetres for x and
y, no unit for the slopes.

A predicted end state has a residual of the same form, with the weights b in
place of a row of A:

    r[n, end, d] = S_hat[n, end, d] - dz * sum over k of b_k * f_d(S_hat[n, k], ...) - S[n, d]

Which terms a loss sums over is a setting of the run (George, 2026-09-28):

  "stages"                 the q stages
  "stages_and_end_state"   the q stages and the end state: q + 1 terms, as
                           equations 19 and 20 of the second mini-paper are written

How the two settings meet the two ways of forming the end state:

  end state predicted, stages_and_end_state   the paper's loss
  end state predicted, stages                 refused: the end state would be
                                              an output that nothing trains
  end state summed,    stages                 the loss of the summed form
  end state summed,    stages_and_end_state   allowed; the term of the end state
                                              is zero to rounding, because the
                                              end state is built by the formula
                                              the residual tests. It only
                                              divides the mean by q + 1
                                              instead of q

Ported from `_shared/model.py` (`reconstruction_residuals`), with the
arithmetic and its order unchanged. The frozen function divides by the scale
of the inputs before it returns; here that division belongs to the pooled loss.
Gates: tests/test_losses.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from rkpinn.predicted_track.predicted_track import END_STATE_PREDICTED, PredictedTrack

TERMS_STAGES = "stages"
TERMS_STAGES_AND_END_STATE = "stages_and_end_state"
TERMS = (TERMS_STAGES, TERMS_STAGES_AND_END_STATE)


@dataclass
class StageResidual:
    residual: torch.Tensor          # shape (m, J, 4); m = rows * steps, J = q or q + 1
    input_states: torch.Tensor      # shape (m, 5)
    start_planes_mm: torch.Tensor   # shape (m,)
    nodes_of_terms: torch.Tensor    # shape (J,): the node of each stage, and 1 for the end state
    step_length_mm: float
    terms: str


def check_terms(terms: str) -> str:
    if terms not in TERMS:
        raise ValueError("the terms of a loss are %s, not %r" % (" or ".join(TERMS), terms))
    return terms


def stage_residual(predicted_track: PredictedTrack, equation_of_motion,
                   terms: str) -> StageResidual:
    check_terms(terms)
    track = predicted_track
    if track.number_of_stages == 0:
        raise ValueError("a track with no stages has no stage residual")
    if terms == TERMS_STAGES and track.end_state_was == END_STATE_PREDICTED:
        raise ValueError(
            "the end state of this track is predicted, and a loss over the stages "
            "alone would leave it untrained. Use the terms %r, or form the end "
            "state by summing the stages" % TERMS_STAGES_AND_END_STATE)
    q = track.number_of_stages
    S = torch.as_tensor(track.input_states).reshape(-1, 5)
    stages = torch.as_tensor(track.stage_states).reshape(-1, q, 4)
    z_start = torch.as_tensor(track.start_planes_mm).reshape(-1)
    dz = track.step_length_mm
    c = torch.as_tensor(track.tableau.nodes)
    A = torch.as_tensor(track.tableau.stage_matrix)
    b = torch.as_tensor(track.tableau.weights)
    if track.stage_rates is not None:
        F = torch.as_tensor(track.stage_rates).reshape(-1, q, 4)
    else:
        F = equation_of_motion.rates(stages, S[:, 4:5], z_start[:, None] + c[None, :] * dz)
    rec_stages = stages - dz * torch.einsum("jk,nkd->njd", A, F)
    if terms == TERMS_STAGES_AND_END_STATE:
        endpoint = torch.as_tensor(track.end_states).reshape(-1, 4)
        rec_end = endpoint - dz * torch.einsum("j,njd->nd", b, F)
        rec = torch.cat([rec_stages, rec_end[:, None, :]], dim=1)
        nodes = torch.cat([c, torch.ones(1, dtype=c.dtype)])
    else:
        rec, nodes = rec_stages, c
    return StageResidual(residual=rec - S[:, None, :4], input_states=S,
                         start_planes_mm=z_start, nodes_of_terms=nodes,
                         step_length_mm=dz, terms=terms)
