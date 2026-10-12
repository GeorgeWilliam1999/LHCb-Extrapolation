"""Gate: the default configurations are complete, valid, and pinned.

Proves
  * every default named in THE_DEFAULTS exists as a file, and every YAML file
    in the directory is named in THE_DEFAULTS and in the README;
  * each default passes the schema and the registry, so a run can start from it;
  * each default's run key equals its pin, so a change to the file is noticed;
  * the LHCb default is, setting for setting, run A of the pilot: the same key
    as the configuration file of that run in the experiment folder, if the
    experiment folder is present.
"""
from __future__ import annotations

import os

import pytest

from rkpinn.defaults import DIRECTORY, THE_DEFAULTS, path_of_the_default
from rkpinn.run_record.configuration import read_configuration, run_key, validate

PILOT_RUN_A = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "experiments", "does_a_summed_end_state_train", "configurations",
    "64_steps_2_stages_summed_end_state.yaml")


def test_every_default_is_named_and_every_name_is_a_file():
    files = sorted(f[:-5] for f in os.listdir(DIRECTORY) if f.endswith(".yaml"))
    assert files == sorted(THE_DEFAULTS), (files, sorted(THE_DEFAULTS))
    with open(os.path.join(DIRECTORY, "README.md")) as handle:
        readme = handle.read()
    for name in THE_DEFAULTS:
        assert "`%s.yaml`" % name in readme, "%s is not in the README" % name


@pytest.mark.parametrize("name", sorted(THE_DEFAULTS))
def test_default_is_valid_and_its_key_is_pinned(name):
    configuration = read_configuration(path_of_the_default(name))
    validate(configuration)
    assert run_key(configuration) == THE_DEFAULTS[name]


def test_an_unknown_name_is_refused():
    with pytest.raises(KeyError):
        path_of_the_default("no_such_default")


@pytest.mark.skipif(not os.path.isfile(PILOT_RUN_A), reason="the pilot's folder is not here")
def test_the_lhcb_default_is_run_a_of_the_pilot():
    default = read_configuration(path_of_the_default("lhcb_64_steps_2_stages"))
    run_a = read_configuration(PILOT_RUN_A)
    assert run_key(default) == run_key(run_a) == "bddea42ea1ef"
