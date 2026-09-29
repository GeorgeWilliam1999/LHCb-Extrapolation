"""The unweighted loss: the baseline with no weighting at all.

    loss = mean over n, j, d of  r[n, j, d] ** 2

The divisor is 1. The residuals of x and y are in millimetres and those of
the slopes have no unit, so x and y dominate. That is what a baseline with
none of the weightings means: the pooled loss is itself a weighting.
"""
from __future__ import annotations

from rkpinn.losses.stage_residual import check_terms, stage_residual
from rkpinn.registry import register


@register("loss", "unweighted")
class Unweighted:
    name = "unweighted"
    needs_labels = False
    settings_in_a_configuration = ("terms",)

    @classmethod
    def from_configuration(cls, settings, context):
        return cls(equation_of_motion=context.equation_of_motion, terms=settings["terms"])

    def __init__(self, *, equation_of_motion, terms):
        self.equation_of_motion = equation_of_motion
        self.terms = check_terms(terms)

    def constants(self, first_round_states=None) -> dict:
        """The loss has no constant but its terms."""
        return {"name": self.name, "terms": self.terms}

    def value(self, predicted_track, target, constants):
        residual = stage_residual(predicted_track, self.equation_of_motion,
                                  constants["terms"])
        return (residual.residual ** 2).mean()
