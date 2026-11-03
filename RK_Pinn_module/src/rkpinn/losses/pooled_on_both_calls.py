"""The pooled loss on both calls: the pooled loss of the stages a network gives
at its second call, and of those it gave at its first.

    loss = mean over n, j, d of ( r'[n, j, d] / s[d] ) ** 2   over the 2q terms

where r' is the stage residual (stage_residual.py) of the second stages for
j = 1..q and of the first stages for j = q+1..2q, and s the spread of each
component over the first round's states, as in pooled.py. It is the mean of
the two pooled losses. With a frozen first call the second half is a
constant.

The track's `first_call` holds the first stages
(networks/error_conditioned_stage_network.py). A track with no first call
gives the pooled loss itself, to the last bit: so the check of the exact
scheme, whose track has one call, is that of the pooled loss.

The loss of a first call that is not trained only adds a constant. It is kept
so that the loss is that of the brief whether the first call is trained or not.
Gates: tests/test_learned_gain_and_error_conditioned_network.py.
"""
from __future__ import annotations

import torch

from rkpinn.losses.pooled import Pooled
from rkpinn.registry import register


@register("loss", "pooled_on_both_calls")
class PooledOnBothCalls(Pooled):
    name = "pooled_on_both_calls"

    def squared_terms(self, predicted_track, target, constants):
        """Shape (m, q, 4) for a track of one call, (m, 2q, 4) for two."""
        second = super().squared_terms(predicted_track, target, constants)
        if predicted_track.first_call is None:
            return second
        first = super().squared_terms(predicted_track.first_call, target, constants)
        return torch.cat([second, first], dim=1)
