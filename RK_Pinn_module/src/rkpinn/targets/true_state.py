"""The true state: the state the simulation gives the particle on its own
plane of the fibre tracker. It contains the field, the scattering, the energy
loss and the interactions, so it is the floor that no method with the field
alone can beat.

The particle's own plane is not the last plane of the crossing, so a
prediction is carried on to it before the two are compared. The planes are
given with the states."""
from __future__ import annotations

from rkpinn.registry import register


@register("target", "true_state")
class TrueState:
    name = "true_state"
    needs_labels = True
    can_be_trained_on = False       # it is for the evaluation

    def for_states(self, track_data, split):
        """(states, planes): shapes (n, 5) and (n,), the true state on the
        particle's own plane of the fibre tracker, and the z of that plane."""
        tracks = track_data.split(split)
        return tracks.true_state_on_own_fibre_plane, tracks.own_fibre_plane_mm
