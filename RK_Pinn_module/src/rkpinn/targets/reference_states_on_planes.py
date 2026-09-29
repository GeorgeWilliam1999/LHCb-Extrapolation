"""The reference states on the planes of a layout: the state of the sixth-order
reference on the first plane and at the end of every step. Used by the
evaluation to follow the error along the track."""
from __future__ import annotations

import numpy as np

from rkpinn.registry import register


@register("target", "reference_states_on_planes")
class ReferenceStatesOnPlanes:
    name = "reference_states_on_planes"
    needs_labels = True
    can_be_trained_on = False       # it is for the evaluation

    def __init__(self, *, number_of_steps):
        self.number_of_steps = int(number_of_steps)

    def for_states(self, track_data, split):
        """Shape (n, number_of_steps + 1, 5). The steps must fall on the planes
        of the track set."""
        starts = track_data.planes_a_step_starts_on(self.number_of_steps)
        on = np.append(starts, track_data.number_of_steps_between_planes)
        return track_data.split(split).reference_states_on_planes[:, on]
