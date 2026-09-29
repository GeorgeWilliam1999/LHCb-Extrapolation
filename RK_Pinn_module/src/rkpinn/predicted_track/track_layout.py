"""The planes of a track: where its steps start, and where the stages of each
step lie.

A track runs from a first plane to a last plane in `number_of_steps` equal
steps. With a tableau, each step has stage planes inside it, at

    start plane of the step + node * step length.

A layout with no tableau has no stages, as for the whole-crossing network.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from rkpinn.integrators.gauss_legendre_tableau import GaussLegendreTableau


@dataclass(frozen=True)
class TrackLayout:
    first_plane_mm: float
    last_plane_mm: float
    number_of_steps: int
    tableau: Optional[GaussLegendreTableau]

    def __post_init__(self):
        if self.number_of_steps < 1:
            raise ValueError("a track has at least one step")

    @property
    def length_mm(self) -> float:
        return self.last_plane_mm - self.first_plane_mm

    @property
    def step_length_mm(self) -> float:
        return self.length_mm / self.number_of_steps

    @property
    def number_of_stages(self) -> int:
        return 0 if self.tableau is None else self.tableau.number_of_stages

    @property
    def start_planes_mm(self) -> np.ndarray:
        """z of the plane each step starts on, shape (number_of_steps,)."""
        return self.first_plane_mm + np.arange(self.number_of_steps) * self.step_length_mm

    @property
    def planes_mm(self) -> np.ndarray:
        """z of the first plane and of the end of every step, shape (number_of_steps + 1,)."""
        return (self.first_plane_mm
                + np.arange(self.number_of_steps + 1) * self.step_length_mm)

    def stage_planes_mm(self, step: int) -> np.ndarray:
        """z of the stage planes of one step, counted from 0; shape (number_of_stages,)."""
        if not 0 <= step < self.number_of_steps:
            raise IndexError("the track has the steps 0 to %d, not %d"
                             % (self.number_of_steps - 1, step))
        if self.tableau is None:
            return np.empty(0)
        return self.start_planes_mm[step] + self.tableau.nodes * self.step_length_mm
