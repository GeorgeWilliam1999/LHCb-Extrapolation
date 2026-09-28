"""Gate: the registry turns a name into a component and refuses unknown names.

Proves
  * every component built so far can be named in a configuration;
  * an unknown name and an unknown kind are refused;
  * one name cannot be given to two components;
  * the package states its version.
"""
from __future__ import annotations

import pytest

import rkpinn
from rkpinn import registry
from rkpinn.equation_of_motion.field_map import FieldMap
from rkpinn.equation_of_motion.lhcb import (
    DifferentiableLhcbEquationOfMotion, LhcbEquationOfMotion)
from rkpinn.integrators.exact_collocation import solve_one_step
from rkpinn.integrators.gauss_legendre_tableau import (
    GaussLegendreTableau, gauss_legendre_tableau)
from rkpinn.integrators.runge_kutta_sixth_order import integrate_with_sixth_order


def test_the_package_states_its_version():
    assert isinstance(rkpinn.__version__, str) and rkpinn.__version__


def test_built_components_can_be_named():
    assert registry.component("equation_of_motion", "lhcb") is LhcbEquationOfMotion
    assert (registry.component("equation_of_motion", "lhcb_differentiable")
            is DifferentiableLhcbEquationOfMotion)
    assert registry.component("tableau", "gauss_legendre") is gauss_legendre_tableau
    assert registry.component("integrator", "exact_collocation") is solve_one_step
    assert (registry.component("integrator", "runge_kutta_sixth_order")
            is integrate_with_sixth_order)
    assert registry.registered_names("field_map") == ("v8r1_down", "v8r1_up")


def test_a_configuration_can_be_resolved():
    configuration = {"equation_of_motion": {"system": "lhcb", "field_map": "v8r1_up"},
                     "steps": {"number_of_stages": 2}}
    field_map = registry.component(
        "field_map", configuration["equation_of_motion"]["field_map"])()
    assert isinstance(field_map, FieldMap)
    equation = registry.component(
        "equation_of_motion", configuration["equation_of_motion"]["system"])(field_map)
    assert equation.number_of_components == 4
    tableau = registry.component("tableau", "gauss_legendre")(
        configuration["steps"]["number_of_stages"])
    assert isinstance(tableau, GaussLegendreTableau) and tableau.number_of_stages == 2


def test_an_unknown_name_is_refused():
    with pytest.raises(registry.UnknownComponent):
        registry.component("equation_of_motion", "no_such_system")


def test_an_unknown_kind_is_refused():
    with pytest.raises(registry.UnknownComponent):
        registry.component("no_such_kind", "lhcb")
    with pytest.raises(registry.UnknownComponent):
        registry.register("no_such_kind", "anything")


def test_kinds_with_nothing_built_are_empty():
    for kind in ("network", "output_form", "loss", "target",
                 "training_protocol", "stopping_rule"):
        assert registry.registered_names(kind) == ()


def test_one_name_cannot_be_given_twice():
    with pytest.raises(ValueError):
        registry.register("tableau", "gauss_legendre")(lambda number_of_stages: None)
