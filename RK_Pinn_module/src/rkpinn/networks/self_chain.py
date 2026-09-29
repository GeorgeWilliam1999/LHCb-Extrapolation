"""The self chain: one stage network applied to itself over a whole track.

The step length is dz = L / N. The same network is applied N times. Each end
state, with charge over momentum carried unchanged, is the input of the next
step, and the start plane advances by dz.

The result is a predicted track that holds every step: its input, its stage
states and its end state. No gradient is kept; this is how a trained network
is used and scored, and how the states of later rounds are produced.

Ported from `E1_Network_grid/chain_network.py` (`carry`, `step_outputs`). The
start plane of step k is first plane + k * dz, as there.
"""
from __future__ import annotations

import numpy as np
import torch

from rkpinn.predicted_track.predicted_track import PredictedTrack
from rkpinn.predicted_track.track_layout import TrackLayout

ROWS_AT_A_TIME = 8192


def whole_track(network, start_states, layout: TrackLayout,
                rows_at_a_time: int = ROWS_AT_A_TIME) -> PredictedTrack:
    """
    network       a stage network
    start_states  shape (n, 5), on the first plane of the layout; numpy
    layout        the planes of the track; its step length and number of
                  stages must be the network's
    """
    if layout.number_of_stages != network.number_of_stages:
        raise ValueError("the layout has %d stages and the network %d"
                         % (layout.number_of_stages, network.number_of_stages))
    if abs(layout.step_length_mm - network.step_length_mm) > 1e-9:
        raise ValueError("the layout has steps of %r mm and the network of %r mm"
                         % (layout.step_length_mm, network.step_length_mm))
    current = np.asarray(start_states, dtype=np.float64).copy()
    n, steps, q = len(current), layout.number_of_steps, layout.number_of_stages
    inputs = np.empty((n, steps, 5))
    stages = np.empty((n, steps, q, 4))
    ends = np.empty((n, steps, 4))
    planes = np.empty((n, steps))
    step_length = float(network.step_length_mm)
    with torch.no_grad():
        for k in range(steps):
            start_plane = layout.first_plane_mm + k * step_length
            inputs[:, k] = current
            planes[:, k] = start_plane
            for i in range(0, n, rows_at_a_time):
                states = torch.as_tensor(current[i:i + rows_at_a_time])
                start_planes = torch.full((len(states),), start_plane, dtype=torch.float64)
                step = network.predict(states, start_planes)
                stages[i:i + rows_at_a_time, k] = step.stage_states[:, 0].numpy()
                ends[i:i + rows_at_a_time, k] = step.end_states[:, 0].numpy()
            current = np.concatenate([ends[:, k], current[:, 4:5]], axis=1)
    return PredictedTrack(
        step_length_mm=step_length, tableau=layout.tableau, start_planes_mm=planes,
        input_states=inputs, stage_states=stages, end_states=ends,
        end_state_was=network.end_state, layout=layout,
        produced_by=network.kind + ", chained to itself")
