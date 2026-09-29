"""No target: training without labels. The loss needs only the equation of motion."""
from __future__ import annotations

from rkpinn.registry import register


@register("target", "no_target")
class NoTarget:
    name = "no_target"
    needs_labels = False

    def for_states(self, track_data, split):
        """There is nothing to give: None."""
        track_data.split(split)                 # an unknown split is still refused
        return None
