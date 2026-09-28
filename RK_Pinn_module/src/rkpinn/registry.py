"""The registry: a name in a configuration file becomes a component.

A configuration file names components, for example

    equation_of_motion: {system: lhcb, field_map: v8r1_up}

and the registry turns each name into the thing it names. Adding a component
is one file and one line of registration in that file:

    from rkpinn.registry import register

    @register("equation_of_motion", "lhcb")
    class LhcbEquationOfMotion: ...

The registry refuses a kind or a name it does not know (WORKFLOW.md, section
3). It holds no component itself and imports none at the top of this file, so
a component can import `register` without a circular import.
"""
from __future__ import annotations

import importlib

# The kinds of component, as listed in WORKFLOW.md, section 6, plus the two
# that an equation of motion and an integrator are built from.
KINDS = (
    "equation_of_motion",
    "field_map",
    "tableau",
    "integrator",
    "network",
    "output_form",
    "loss",
    "target",
    "training_protocol",
    "stopping_rule",
)

# The directories of the package whose files register components. Only the
# directories that hold built components are listed; a directory is added here
# in the same change that builds its first component.
COMPONENT_MODULES = (
    "rkpinn.equation_of_motion.field_map",
    "rkpinn.equation_of_motion.lhcb",
    "rkpinn.integrators.gauss_legendre_tableau",
    "rkpinn.integrators.exact_collocation",
    "rkpinn.integrators.runge_kutta_sixth_order",
)

_components: dict[str, dict[str, object]] = {kind: {} for kind in KINDS}


class UnknownComponent(KeyError):
    """A configuration named a kind or a component the registry does not know."""


def register(kind: str, name: str):
    """Register the decorated class or function as `name` of kind `kind`."""
    if kind not in _components:
        raise UnknownComponent(
            "%r is not a kind of component; the kinds are %s" % (kind, ", ".join(KINDS)))

    def decorator(component):
        already = _components[kind].get(name)
        if already is not None and already is not component:
            raise ValueError(
                "the %s named %r is already registered as %r" % (kind, name, already))
        _components[kind][name] = component
        return component

    return decorator


def load_components() -> None:
    """Import every module that registers a component."""
    for module in COMPONENT_MODULES:
        importlib.import_module(module)


def component(kind: str, name: str):
    """The component registered as `name` of kind `kind`. Unknown names are refused."""
    load_components()
    if kind not in _components:
        raise UnknownComponent(
            "%r is not a kind of component; the kinds are %s" % (kind, ", ".join(KINDS)))
    try:
        return _components[kind][name]
    except KeyError:
        known = ", ".join(sorted(_components[kind])) or "none yet"
        raise UnknownComponent(
            "no %s is registered under the name %r; registered: %s"
            % (kind, name, known)) from None


def registered_names(kind: str) -> tuple[str, ...]:
    """The names registered for one kind, sorted."""
    load_components()
    if kind not in _components:
        raise UnknownComponent(
            "%r is not a kind of component; the kinds are %s" % (kind, ", ".join(KINDS)))
    return tuple(sorted(_components[kind]))
