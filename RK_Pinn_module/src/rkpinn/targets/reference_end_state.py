"""The reference end state: the state of the sixth-order reference on the last
plane of the crossing. It is the label of the whole-crossing network and the
state every endpoint error is measured against. It contains the field and
nothing else."""
from __future__ import annotations

from rkpinn.registry import register


@register("target", "reference_end_state")
class ReferenceEndState:
    name = "reference_end_state"
    needs_labels = True

    def for_states(self, track_data, split):
        """Shape (n, 5), one row per track of the split, in the order of the tracks."""
        return track_data.split(split).reference_end_state
