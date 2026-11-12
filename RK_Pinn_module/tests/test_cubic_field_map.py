"""Gate: the cubic field maps (the setting `v8r1_up_cubic`, `v8r1_down_cubic`).

Proves
  * the linear maps are unchanged: loading them gives the class of before;
  * the cubic map passes through every grid point (to 1e-12 T);
  * it equals scipy.ndimage.map_coordinates (order 3, mode "nearest") on the
    same coefficients, to 1e-12 T, inside the map and beyond its edges;
  * the differentiable cubic field equals the numpy one to 1e-12 T and has a
    gradient that is continuous across a cell face, where the linear map's
    gradient jumps;
  * the equation of motion takes the cubic field through `differentiable()`;
  * where tracks pass, the two maps differ by less than 1e-3 of the peak field
    in the median and 1e-2 at the 99th percentile: the same field, smoothed.
    Near iron the spline rings (largest difference 0.13 T of 2.56 T).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.ndimage import map_coordinates

from rkpinn import registry
from rkpinn.equation_of_motion.field_map import (
    CubicFieldMap, DifferentiableCubicFieldMap, FieldMap, corners_of_field_map, load_field_map)
from rkpinn.equation_of_motion.lhcb import DifferentiableLhcbEquationOfMotion

TOLERANCE_TESLA = 1e-12


@pytest.fixture(scope="module")
def cubic():
    return load_field_map("v8r1_up_cubic")


@pytest.fixture(scope="module")
def linear():
    return load_field_map("v8r1_up")


def test_the_linear_maps_are_unchanged(linear):
    assert type(linear) is FieldMap
    assert type(load_field_map("v8r1_down")) is FieldMap
    assert type(registry.component("field_map", "v8r1_up")()) is FieldMap


def test_the_cubic_maps_are_registered(cubic):
    assert registry.component("field_map", "v8r1_up_cubic")() is cubic
    assert isinstance(registry.component("field_map", "v8r1_down_cubic")(), CubicFieldMap)


def test_the_cubic_map_passes_through_the_grid_points(cubic):
    rng = np.random.default_rng(1)
    n = cubic.number_of_points
    ix, iy, iz = (rng.integers(0, n[a], 5000) for a in range(3))
    points = [cubic.lowest_corner[a] + i / cubic.inverse_cell_size[a] for a, i in enumerate((ix, iy, iz))]
    got = cubic(*points)
    for value, grid in zip(got, (cubic.field_x, cubic.field_y, cubic.field_z)):
        assert np.abs(value - grid[iz, iy, ix] * cubic.scale_to_tesla).max() < TOLERANCE_TESLA


def test_the_cubic_map_equals_scipy(cubic):
    lowest, highest = corners_of_field_map(cubic)
    rng = np.random.default_rng(2)
    margin = 150.0     # beyond the edges too
    points = lowest - margin + rng.random((20000, 3)) * (highest - lowest + 2 * margin)
    got = cubic(points[:, 0], points[:, 1], points[:, 2])
    fractional = ((points - cubic.lowest_corner) * cubic.inverse_cell_size).T[::-1]   # z, y, x
    for value, coefficients in zip(got, cubic.coefficients):
        expected = map_coordinates(coefficients, fractional, order=3, mode="nearest", prefilter=False)
        assert np.abs(value - expected).max() < TOLERANCE_TESLA


def test_the_differentiable_cubic_map_equals_the_numpy_one(cubic):
    lowest, highest = corners_of_field_map(cubic)
    rng = np.random.default_rng(3)
    points = lowest + rng.random((20000, 3)) * (highest - lowest)
    expected = cubic(points[:, 0], points[:, 1], points[:, 2])
    differentiable = cubic.differentiable()
    assert isinstance(differentiable, DifferentiableCubicFieldMap)
    got = differentiable(*(torch.as_tensor(points[:, a]) for a in range(3)))
    for g, e in zip(got, expected):
        assert np.abs(g.numpy() - e).max() < TOLERANCE_TESLA


def _gradient_across_a_face(field_map):
    """d By / d z 1 µm below and above the face z = 5100 mm (a face of the
    grid), at x = 300.4 mm, y = 210.7 mm, inside the magnet."""
    d = field_map.differentiable()
    z = torch.tensor([5100 - 1e-3, 5100 + 1e-3], dtype=torch.float64, requires_grad=True)
    x = torch.full((2,), 300.4, dtype=torch.float64)
    y = torch.full((2,), 210.7, dtype=torch.float64)
    (g,) = torch.autograd.grad(d(x, y, z)[1].sum(), z)
    return g.numpy()


def test_the_gradient_is_continuous_across_a_face(cubic, linear):
    below, above = _gradient_across_a_face(cubic)
    assert abs(above - below) < 1e-4 * abs(above)
    below_linear, above_linear = _gradient_across_a_face(linear)
    assert abs(above_linear - below_linear) > 1e3 * abs(above - below)


def test_the_equation_of_motion_takes_the_cubic_field(cubic):
    equation = DifferentiableLhcbEquationOfMotion(cubic)
    assert isinstance(equation.field, DifferentiableCubicFieldMap)


def test_the_cubic_map_is_the_same_field_smoothed(cubic, linear):
    rng = np.random.default_rng(4)
    x = rng.uniform(-1500, 1500, 20000)
    y = rng.uniform(-1000, 1000, 20000)
    z = rng.uniform(2650, 7826, 20000)
    peak = np.abs(linear(x, y, z)[1]).max()
    for c, l in zip(cubic(x, y, z), linear(x, y, z)):
        assert np.median(np.abs(c - l)) < 1e-3 * peak
        assert np.percentile(np.abs(c - l), 99) < 1e-2 * peak
