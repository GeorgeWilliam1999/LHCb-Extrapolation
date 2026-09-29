"""The pooled loss: equation 19 of the second mini-paper.

    loss = mean over n, j, d of  ( r[n, j, d] / s[d] ) ** 2

where s[d] is one constant per component d, the spread of that component over
the first round's states. It asks for equal absolute accuracy in every
component, on every track, at every plane. The four constants are measured
once and do not change during a run.

This is the discrete-time loss of Raissi, Perdikaris and Karniadakis (2019)
with two changes: the division per component, without which x would dominate
the sum, and a mean in place of a sum, which only rescales it.

Ported from `_shared/model.py` (`physics_loss`). With the terms
"stages_and_end_state" and a predicted end state it gives the same number as
the frozen loss, to the last bit.
"""
from __future__ import annotations

import numpy as np
import torch

from rkpinn.losses.stage_residual import check_terms, stage_residual
from rkpinn.registry import register


def spread_of_components(states) -> np.ndarray:
    """The standard deviation of every component of the states, over the rows."""
    return np.asarray(states, dtype=np.float64).std(axis=0)


@register("loss", "pooled")
class Pooled:
    name = "pooled"
    needs_labels = False
    settings_in_a_configuration = ("terms",)

    @classmethod
    def from_configuration(cls, settings, context):
        return cls(equation_of_motion=context.equation_of_motion, terms=settings["terms"])

    def __init__(self, *, equation_of_motion, terms):
        self.equation_of_motion = equation_of_motion
        self.terms = check_terms(terms)

    def constants(self, first_round_states) -> dict:
        """The divisor: the spread of x, y, slope in x and slope in y over the
        first round's states."""
        return {"name": self.name, "terms": self.terms,
                "divisor_per_component":
                    spread_of_components(first_round_states)[:4].tolist()}

    def value(self, predicted_track, target, constants):
        residual = stage_residual(predicted_track, self.equation_of_motion,
                                  constants["terms"])
        divisor = torch.as_tensor(constants["divisor_per_component"], dtype=torch.float64)
        return ((residual.residual / divisor) ** 2).mean()
