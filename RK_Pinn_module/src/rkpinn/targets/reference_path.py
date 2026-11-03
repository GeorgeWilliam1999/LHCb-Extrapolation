"""The reference path: the exact scheme's stage states and plane states of a
layout, as labels for the path network.

For every track of a split the label has the shape (number of steps, stages
+ 1, 4): for step n the stage states S_n1 .. S_nq, then the state on the plane
the step ends on, each as (x, y, slope in x, slope in y). These are the
states one Newton sweep of the stage equations converges to, so a network
trained on them starts the sweep where it ends.

They come from the store. `exact_states` keeps the plane states of the exact
scheme; its option `--keep-stages` (added with this target) keeps the stage
states beside them under the same key. The target refuses a split or a layout
the store does not hold, and never solves anything itself.

The trainer builds a target with no arguments. This one needs the layout, so
it declares `needs_layout`, and the trainer passes the layout to a target that
declares it (round_trainer.py, one line).

Draft of 2026-11-03, written off GitHub 184cf50e for the experiment
does_a_one_shot_path_network_remove_the_chain; to be ported onto the stbc
checkout. Gates: tests/test_reference_path.py (planned).
"""
from __future__ import annotations

import numpy as np

from rkpinn.registry import register


@register("target", "reference_path")
class ReferencePath:
    name = "reference_path"
    needs_labels = True
    can_be_trained_on = True
    needs_layout = True

    def __init__(self, *, layout):
        self.layout = layout

    def for_states(self, track_data, split):
        """Shape (n, number of steps, stages + 1, 4), in the order of the
        tracks of the split."""
        from rkpinn.track_data.exact_states import read_exact_states_with_stages
        steps, stages = self.layout.number_of_steps, self.layout.number_of_stages
        planes, stage_states = read_exact_states_with_stages(
            track_data, split, steps, stages)           # (n, steps + 1, 5), (n, steps, q, 4)
        n = planes.shape[0]
        if stage_states.shape != (n, steps, stages, 4):
            raise ValueError("the stage states of the exact scheme have the shape "
                             "(n, steps, stages, 4); the store holds %r" % (stage_states.shape,))
        labels = np.empty((n, steps, stages + 1, 4))
        labels[:, :, :stages] = stage_states
        labels[:, :, stages] = planes[:, 1:, :4]
        return labels
