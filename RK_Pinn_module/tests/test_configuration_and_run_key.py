"""Gate: a configuration is checked, and the run key is computed from it.

Proves
  * a configuration with a setting the schema does not know, with a setting
    missing, or with a name the registry does not know, is refused;
  * combinations that make no sense are refused: a loss without labels on a
    network without stages, a protocol that is not the network's, a loss and
    a target that disagree on labels, a target that is for the evaluation;
  * the run key is the same for the same settings however they are written,
    changes with every setting, and does not change with the cap on the rounds;
  * a configuration written to a file is read back the same, with the same key;
  * the difference of two configurations names the settings that differ.
"""
from __future__ import annotations

import copy

import pytest

from rkpinn.run_record.configuration import (
    ConfigurationRefused, difference, read_configuration, run_key, validate,
    write_configuration)
from rkpinn.run_record.store import LENGTH_OF_A_KEY
from small_run import small_configuration, whole_crossing_configuration

KEY = "0123456789ab"


def changed(configuration, path, value):
    other = copy.deepcopy(configuration)
    block = other
    for name in path[:-1]:
        block = block[name]
    if value is REMOVED:
        del block[path[-1]]
    else:
        block[path[-1]] = value
    return other


REMOVED = object()


def test_the_configurations_of_the_gates_are_accepted():
    validate(small_configuration(KEY))
    validate(whole_crossing_configuration(KEY))


@pytest.mark.parametrize("path", (
    ("momentum",), ("network", "dropout"), ("loss", "momentum_window_gev"),
    ("training", "learning_rate"), ("training", "optimiser", "momentum"),
    ("training", "stopping_rule", "patience"), ("steps", "step_length_mm"),
    ("training", "protocol", "batch_size"),
))
def test_a_setting_that_is_not_recognised_is_refused(path):
    with pytest.raises(ConfigurationRefused, match="not recognised"):
        validate(changed(small_configuration(KEY), path, 1))


@pytest.mark.parametrize("path", (
    ("seed",), ("target",), ("network", "activation"), ("network", "end_state"),
    ("network", "scale_of_outputs"), ("loss", "terms"),
    ("training", "rounds_at_most"), ("training", "end_a_round_when_stalled"),
    ("training", "optimiser", "history"), ("training", "stopping_rule", "quantity"),
    ("training", "protocol", "states_per_round"), ("tracks", "key"),
))
def test_a_setting_that_is_missing_is_refused(path):
    with pytest.raises(ConfigurationRefused, match="lacks|must state|must hold"):
        validate(changed(small_configuration(KEY), path, REMOVED))


@pytest.mark.parametrize("path", (
    ("network", "kind"), ("loss", "name"), ("target", "name"),
    ("equation_of_motion", "system"), ("equation_of_motion", "field_map"),
    ("training", "protocol", "name"), ("training", "optimiser", "name"),
    ("training", "stopping_rule", "name"),
))
def test_a_name_the_registry_does_not_know_is_refused(path):
    with pytest.raises(ConfigurationRefused, match="registered"):
        validate(changed(small_configuration(KEY), path, "no_such_thing"))


def test_combinations_that_make_no_sense_are_refused():
    stage, crossing = small_configuration(KEY), whole_crossing_configuration(KEY)
    wrong = (
        changed(stage, ("steps", "number_of_stages"), 0),
        changed(crossing, ("steps", "number_of_steps"), 4),
        changed(stage, ("training", "protocol", "name"), "start_states_of_the_tracks"),
        changed(stage, ("target", "name"), "reference_end_state"),
        changed(crossing, ("target", "name"), "no_target"),
        changed(crossing, ("target", "name"), "true_state"),
        changed(crossing, ("loss",), {"name": "pooled", "terms": "stages"}),
        changed(stage, ("steps", "number_of_steps"), 0),
        changed(stage, ("training", "restarts_per_round"), 2.5),
        changed(stage, ("seed",), True),
    )
    for configuration in wrong:
        with pytest.raises(ConfigurationRefused):
            validate(configuration)


def test_a_stall_is_absent_or_complete():
    base = small_configuration(KEY)
    validate(changed(base, ("training", "end_a_round_when_stalled"),
                     {"gain_below": 0.01, "restarts_running": 2}))
    with pytest.raises(ConfigurationRefused):
        validate(changed(base, ("training", "end_a_round_when_stalled"),
                         {"gain_below": 0.01}))


def test_the_key_is_that_of_the_settings_however_written():
    first = small_configuration(KEY)
    key = run_key(first)
    assert len(key) == LENGTH_OF_A_KEY
    assert run_key(copy.deepcopy(first)) == key
    reordered = {name: first[name] for name in reversed(list(first))}
    assert run_key(reordered) == key
    cost = {"name": "cost_weighted", "terms": "stages",
            "momentum_window_gev": (10.0, 50.0), "roll_off": 0.6931, "floor": 0.05,
            "clamp": 5.0, "samples_of_the_field_integral": 1024, "lever_arm_is_on": True,
            "track_bend_is_on": True, "momentum_window_is_on": True,
            "reference_bend_mm": "median_of_first_round_states"}
    as_list = dict(cost, momentum_window_gev=[10.0, 50.0])
    assert run_key(small_configuration(KEY, loss=cost)) \
        == run_key(small_configuration(KEY, loss=as_list))


@pytest.mark.parametrize("path, value", (
    (("seed",), 1), (("tracks", "key"), "ba9876543210"),
    (("steps", "number_of_steps"), 8), (("steps", "number_of_stages"), 4),
    (("network", "width"), 32), (("network", "end_state"), "predicted"),
    (("network", "scale_of_inputs"), [1.0, 1.0, 1.0, 1.0, 1.0]),
    (("loss", "name"), "unweighted"),
    (("training", "restarts_per_round"), 3),
    (("training", "optimiser", "iterations_per_restart"), 6),
    (("training", "stopping_rule", "tolerance"), 0.1),
    (("training", "protocol", "states_per_round"), 800),
    (("equation_of_motion", "field_map"), "v8r1_down"),
))
def test_the_key_changes_with_every_setting(path, value):
    first = small_configuration(KEY)
    other = changed(first, path, value)
    if path == ("network", "end_state"):
        other["loss"]["terms"] = "stages_and_end_state"
    assert run_key(other) != run_key(first)


def test_the_key_does_not_change_with_the_cap_on_the_rounds():
    first = small_configuration(KEY)
    assert run_key(changed(first, ("training", "rounds_at_most"), 60)) == run_key(first)


def test_a_configuration_is_read_back_the_same(tmp_path):
    for configuration in (small_configuration(KEY), whole_crossing_configuration(KEY)):
        path = str(tmp_path / ("%s.yaml" % configuration["network"]["kind"]))
        write_configuration(configuration, path)
        again = read_configuration(path)
        assert difference(configuration, again) == []
        assert run_key(again) == run_key(configuration)


def test_the_difference_names_the_settings_that_differ():
    first = small_configuration(KEY)
    other = changed(first, ("loss", "name"), "unweighted")
    assert difference(first, other) == ["loss.name: 'pooled' -> 'unweighted'"]
    assert difference(first, first) == []
    both = changed(other, ("seed",), 4)
    assert len(difference(first, both)) == 2
