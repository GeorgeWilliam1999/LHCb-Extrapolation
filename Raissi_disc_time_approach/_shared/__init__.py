"""Shared foundation for the discrete-time experiments in this folder.

Every experiment (one folder each) imports the physics, the model, the data
builders and the training driver from here, so that any difference between two
experiments comes from the experiment and not from a drifted copy of the code.

The physics and the optimiser protocol are those of `One_step_network_v2` (the
verified baseline). Nothing here changes them; the modules only generalise the
frozen-leg special case to any number of stages, either field polarity, and
per-sample start planes and step lengths.

Modules
-------
field_v8r1   vendored canonical field-map loader (numpy)
reference    the ODE, the fp64 RK4 reference, the metric, the tableau, the data
field_torch  the differentiable fp64 twin of the field, for the physics loss
model        the one-step network, the rates, the physics and data losses
prepare      the dataset builders (frozen leg, general leg)
evaluate     scoring helpers and leg-by-leg chaining
train        the command-line training driver

Import from an experiment folder with the two-line preamble in README.md, or by
copying `use_shared.py` into the experiment folder and importing it first.
"""

__all__ = [
    "field_v8r1",
    "reference",
    "field_torch",
    "model",
    "prepare",
    "evaluate",
    "train",
]
