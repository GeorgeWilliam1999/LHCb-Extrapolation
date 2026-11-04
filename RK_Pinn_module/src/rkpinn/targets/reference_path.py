"""The reference path: the exact scheme's stage states and plane states of a
layout, as labels for the path network.

For every track of a split the label has the shape (number of steps, stages
+ 1, 4): for step n the stage states S_n1 .. S_nq, then the state on the plane
the step ends on, each as (x, y, slope in x, slope in y). These are the
states one Newton sweep of the stage equations converges to, so a network
trained on them starts the sweep where it ends.

They come from the store. `exact_states --keep-stages` keeps the stage states
of the exact scheme beside its plane states, in the same file. The plane states
are those of the exact scheme, not of the sixth-order reference: the two differ
by the floor of the scheme (0.147 um at the end of the region at 64 steps and
2 stages), and only the scheme's own states solve the stage equations, which
the sweep needs. The target refuses a split or a layout the store does not
hold, and never solves anything itself.

The trainer builds a target with no arguments. This one needs the layout and
the store, so it declares `needs_layout`, and the trainer builds a target that
declares it with both (round_trainer.py). The labels of a split are read once
and kept.

Ported onto the stbc checkout on 2026-11-04 for the experiment
does_a_one_shot_path_network_remove_the_chain. Gates: tests/test_reference_path.py.
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

    def __init__(self, *, layout, store):
        if layout.tableau is None:
            raise ValueError("the reference path needs a layout with stages")
        self.layout = layout
        self.store = store
        self._labels = {}

    def for_states(self, track_data, split):
        """Shape (n, number of steps, stages + 1, 4), in the order of the
        tracks of the split."""
        if split in self._labels:
            return self._labels[split]
        from rkpinn.track_data.exact_states import read_exact_states_with_stages
        steps, stages = self.layout.number_of_steps, self.layout.number_of_stages
        track_data.split(split)                                  # refuses an unknown split
        planes, stage_states = read_exact_states_with_stages(
            self.store, track_data.key, split, steps, stages)  # (n, steps + 1, 5), (n, steps, q, 4)
        n = track_data.split(split).number_of_tracks
        if planes.shape != (n, steps + 1, 5) or stage_states.shape != (n, steps, stages, 4):
            raise ValueError("the exact states in the store have the shapes %r and %r, not "
                             "those of %d tracks, %d steps and %d stages"
                             % (planes.shape, stage_states.shape, n, steps, stages))
        if not np.array_equal(planes[:, 0], track_data.split(split).start_state):
            raise ValueError("the exact states of the split %s do not start from the start "
                             "states of its tracks" % split)
        labels = np.empty((n, steps, stages + 1, 4))
        labels[:, :, :stages] = stage_states
        labels[:, :, stages] = planes[:, 1:, :4]
        self._labels[split] = labels
        return labels
