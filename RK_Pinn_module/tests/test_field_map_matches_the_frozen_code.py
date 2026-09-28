"""Gate: the field map of the package is the field map of the frozen code.

Proves
  * the map file has the hash the frozen code recorded;
  * the numpy field gives identical values, to the last bit, to the frozen
    loader, on 200,000 random points inside the map, for both polarities;
  * the differentiable field gives identical values to the frozen
    differentiable field, and agrees with the numpy field to 1e-12 T;
  * the differentiable field has a gradient with respect to the position.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from frozen_code import frozen_module, identical
from rkpinn.equation_of_motion.field_map import (
    DifferentiableFieldMap, corners_of_field_map, hash_of_field_map_file, load_field_map)

# Recorded by the frozen gate, _shared/results/vendoring_parity.json.
HASH_OF_THE_DOWN_MAP = "af284c6954d2273c637a5e766b82b58e"

POLARITIES = ("down", "up")
NUMBER_OF_POINTS = 200_000


def random_points_inside(field_map, seed=0):
    lowest, highest = corners_of_field_map(field_map)
    generator = np.random.default_rng(seed)
    points = lowest + generator.random((NUMBER_OF_POINTS, 3)) * (highest - lowest)
    return points[:, 0], points[:, 1], points[:, 2]


def test_hash_of_the_down_map_is_the_recorded_one():
    assert hash_of_field_map_file("v8r1_down") == HASH_OF_THE_DOWN_MAP


@pytest.mark.parametrize("polarity", POLARITIES)
def test_hash_equals_the_frozen_code(polarity):
    frozen = frozen_module("reference")
    assert hash_of_field_map_file("v8r1_" + polarity) == frozen.field_md5(polarity)


@pytest.mark.parametrize("polarity", POLARITIES)
def test_corners_equal_the_frozen_code(polarity):
    frozen = frozen_module("reference")
    ours = corners_of_field_map(load_field_map("v8r1_" + polarity))
    theirs = frozen.field_bounds(frozen.make_field(polarity))
    assert identical(ours[0], theirs[0]) and identical(ours[1], theirs[1])


@pytest.mark.parametrize("polarity", POLARITIES)
def test_field_values_are_identical_to_the_frozen_loader(polarity):
    frozen = frozen_module("reference")
    ours = load_field_map("v8r1_" + polarity)
    theirs = frozen.make_field(polarity)
    x, y, z = random_points_inside(ours)
    for component_ours, component_theirs in zip(ours(x, y, z), theirs(x, y, z)):
        assert identical(component_ours, component_theirs)


@pytest.mark.parametrize("polarity", POLARITIES)
def test_differentiable_field_is_identical_to_the_frozen_one(polarity):
    frozen_reference = frozen_module("reference")
    frozen_field_torch = frozen_module("field_torch")
    ours = DifferentiableFieldMap(load_field_map("v8r1_" + polarity))
    theirs = frozen_field_torch.FieldTorch(frozen_reference.make_field(polarity))
    x, y, z = (torch.tensor(a) for a in random_points_inside(ours.field_map))
    with torch.no_grad():
        for component_ours, component_theirs in zip(ours(x, y, z), theirs(x, y, z)):
            assert identical(component_ours.numpy(), component_theirs.numpy())


@pytest.mark.parametrize("polarity", POLARITIES)
def test_differentiable_field_agrees_with_the_numpy_field(polarity):
    field_map = load_field_map("v8r1_" + polarity)
    differentiable = DifferentiableFieldMap(field_map)
    x, y, z = random_points_inside(field_map)
    in_numpy = np.stack(field_map(x, y, z), axis=1)
    with torch.no_grad():
        in_torch = torch.stack(
            differentiable(torch.tensor(x), torch.tensor(y), torch.tensor(z)), dim=1).numpy()
    assert in_torch.dtype == np.float64
    assert np.abs(in_numpy - in_torch).max() < 1e-12


def test_differentiable_field_has_a_gradient():
    differentiable = DifferentiableFieldMap(load_field_map("v8r1_up"))
    x = torch.tensor([100.0, -50.0], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([50.0, 10.0], dtype=torch.float64)
    z = torch.tensor([5000.0, 4000.0], dtype=torch.float64)
    _, field_y, _ = differentiable(x, y, z)
    field_y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert (x.grad != 0).any()
