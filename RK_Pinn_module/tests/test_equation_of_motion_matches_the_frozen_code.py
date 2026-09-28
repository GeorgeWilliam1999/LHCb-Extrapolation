"""Gate: the equation of motion of the package is that of the frozen code.

Proves
  * the numpy rates are identical, to the last bit, to the frozen `deriv` on
    500 real states at planes all along the crossing;
  * the torch rates and their gradient with respect to the state are
    identical, to the last bit, to the frozen `LHCbRates`;
  * the torch rates equal the numpy rates;
  * charge over momentum has a rate of exactly zero;
  * the constants are those of the frozen code.
"""
from __future__ import annotations

import numpy as np
import torch

from frozen_code import frozen_module, frozen_tracks, identical, real_states
from rkpinn.equation_of_motion import lhcb
from rkpinn.equation_of_motion.field_map import load_field_map

NUMBER_OF_STATES = 500
NUMBER_OF_STAGES = 4


def polarity_of_the_tracks() -> str:
    return str(frozen_tracks()["field"])


def stage_like_inputs():
    """500 real states arranged as (n, q, 4) stage states on (n, q) planes:
    each real state repeated on q planes a few millimetres apart."""
    states, planes = real_states(NUMBER_OF_STATES)
    offsets = np.linspace(0.0, 60.0, NUMBER_OF_STAGES)
    stage_states = np.repeat(states[:, None, :4], NUMBER_OF_STAGES, axis=1).copy()
    stage_states[:, :, 0] += states[:, None, 2] * offsets[None, :]
    stage_states[:, :, 1] += states[:, None, 3] * offsets[None, :]
    stage_planes = planes[:, None] + offsets[None, :]
    return stage_states, states[:, 4:5].copy(), stage_planes


def test_constants_equal_the_frozen_code():
    frozen_reference = frozen_module("reference")
    frozen_model = frozen_module("model")
    assert lhcb.KAPPA == frozen_reference.KAPPA == frozen_model.KAPPA
    assert lhcb.CHARGE_OVER_MOMENTUM_PER_INVERSE_GEV == frozen_reference.C_QP


def test_rates_are_identical_to_the_frozen_code_on_500_real_states():
    frozen = frozen_module("reference")
    polarity = polarity_of_the_tracks()
    states, planes = real_states(NUMBER_OF_STATES)
    assert states.shape == (NUMBER_OF_STATES, 5)
    ours = lhcb.LhcbEquationOfMotion(load_field_map("v8r1_" + polarity)).rates(states, planes)
    theirs = frozen.deriv(states, planes, frozen.make_field(polarity))
    assert np.abs(ours[:, 2:4]).max() > 0, "the states must lie in the field"
    assert identical(ours, theirs)


def test_rates_on_one_shared_plane_are_identical_to_the_frozen_code():
    frozen = frozen_module("reference")
    polarity = polarity_of_the_tracks()
    states, _ = real_states(NUMBER_OF_STATES)
    ours = lhcb.LhcbEquationOfMotion(load_field_map("v8r1_" + polarity)).rates(states, 5000.0)
    theirs = frozen.deriv(states, 5000.0, frozen.make_field(polarity))
    assert identical(ours, theirs)


def test_charge_over_momentum_has_a_rate_of_zero():
    states, planes = real_states(NUMBER_OF_STATES)
    equation = lhcb.LhcbEquationOfMotion(load_field_map("v8r1_" + polarity_of_the_tracks()))
    assert (equation.rates(states, planes)[:, 4] == 0.0).all()
    assert equation.number_of_components == 4


def test_torch_rates_and_gradient_are_identical_to_the_frozen_code():
    frozen_reference = frozen_module("reference")
    frozen_model = frozen_module("model")
    polarity = polarity_of_the_tracks()
    stage_states, qop, stage_planes = stage_like_inputs()

    def rates_and_gradient(rates_of):
        S4 = torch.tensor(stage_states, dtype=torch.float64, requires_grad=True)
        out = rates_of(S4, torch.tensor(qop), torch.tensor(stage_planes))
        # a weighted sum, so the gradient of every output enters
        weights = torch.linspace(0.5, 1.5, out.numel(), dtype=torch.float64).reshape(out.shape)
        (out * weights).sum().backward()
        return out.detach().numpy(), S4.grad.numpy()

    ours = lhcb.DifferentiableLhcbEquationOfMotion(load_field_map("v8r1_" + polarity))
    theirs = frozen_model.LHCbRates(frozen_reference.make_field(polarity))
    rates_ours, gradient_ours = rates_and_gradient(ours.rates)
    rates_theirs, gradient_theirs = rates_and_gradient(theirs)
    assert rates_ours.dtype == np.float64
    assert np.abs(gradient_ours[..., :2]).max() > 0, "the gradient must reach the position"
    assert identical(rates_ours, rates_theirs)
    assert identical(gradient_ours, gradient_theirs)


def test_torch_rates_with_shared_planes_are_identical_to_the_frozen_code():
    frozen_reference = frozen_module("reference")
    frozen_model = frozen_module("model")
    polarity = polarity_of_the_tracks()
    stage_states, qop, _ = stage_like_inputs()
    shared_planes = torch.tensor(np.linspace(4000.0, 4060.0, NUMBER_OF_STAGES))
    ours = lhcb.DifferentiableLhcbEquationOfMotion(load_field_map("v8r1_" + polarity))
    theirs = frozen_model.LHCbRates(frozen_reference.make_field(polarity))
    with torch.no_grad():
        a = ours.rates(torch.tensor(stage_states), torch.tensor(qop), shared_planes).numpy()
        b = theirs(torch.tensor(stage_states), torch.tensor(qop), shared_planes).numpy()
    assert identical(a, b)


def test_torch_rates_equal_the_numpy_rates():
    polarity = polarity_of_the_tracks()
    field_map = load_field_map("v8r1_" + polarity)
    stage_states, qop, stage_planes = stage_like_inputs()
    n, q = stage_planes.shape
    flat = np.concatenate(
        [stage_states.reshape(n * q, 4), np.repeat(qop, q, axis=0)], axis=1)
    in_numpy = lhcb.LhcbEquationOfMotion(field_map).rates(
        flat, stage_planes.reshape(n * q))[:, :4].reshape(n, q, 4)
    with torch.no_grad():
        in_torch = lhcb.DifferentiableLhcbEquationOfMotion(field_map).rates(
            torch.tensor(stage_states), torch.tensor(qop), torch.tensor(stage_planes)).numpy()
    scale = np.abs(in_numpy).max(axis=(0, 1))
    assert (np.abs(in_numpy - in_torch).max(axis=(0, 1)) <= 1e-14 * scale).all()
