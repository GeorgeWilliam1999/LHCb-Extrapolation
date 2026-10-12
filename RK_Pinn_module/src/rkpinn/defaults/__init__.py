"""The default configurations: complete, valid configuration files that an
experiment copies and then changes in what it asks about.

A default is a file, not a value in code. Every setting of a run still comes
from the run's own configuration file (WORKFLOW.md, section 3), so a default
hides nothing: it is a named starting point whose run key the gates pin.

    from rkpinn.defaults import path_of_the_default, THE_DEFAULTS
    read_configuration(path_of_the_default("lhcb_64_steps_2_stages"))

Gates: tests/test_default_configuration.py.
"""
from __future__ import annotations

import os

DIRECTORY = os.path.dirname(os.path.abspath(__file__))

# name -> the run key of the configuration as it stands. A change to the file
# changes its key, and the gate fails until this line is changed with it.
THE_DEFAULTS = {
    "lhcb_64_steps_2_stages": "bddea42ea1ef",
}


def path_of_the_default(name: str) -> str:
    """The path of the default configuration file with that name."""
    if name not in THE_DEFAULTS:
        raise KeyError("no default configuration is named %r; there are %s"
                       % (name, ", ".join(sorted(THE_DEFAULTS))))
    return os.path.join(DIRECTORY, name + ".yaml")
