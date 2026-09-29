"""The configuration of a run: its schema, its validation and the run key.

A run takes every setting from one configuration file, so that it can be
repeated from that file alone. The file is YAML:

    equation_of_motion: {system: lhcb, field_map: v8r1_up}
    tracks:   {key: 12a8d35c3165}
    steps:    {number_of_steps: 64, number_of_stages: 2}
    network:  {kind: stage_network, output_form: direct_states, width: 128,
               depth: 2, activation: tanh, end_state: summed_from_the_stages,
               scale_of_inputs: spread_of_first_round_states,
               scale_of_outputs: spread_of_first_round_states}
    loss:     {name: pooled, terms: stages}
    target:   {name: no_target}
    training:
      protocol: {name: rounds_on_own_predictions, states_per_round: 32000}
      restarts_per_round: 25
      end_a_round_when_stalled: null
      optimiser: {name: lbfgs_restarts, iterations_per_restart: 200, ...}
      stopping_rule: {name: validation_plateau, ...}
      rounds_at_most: 60
    seed: 0

The system is named once. Its form in numpy is registered under that name,
and its differentiable form under that name followed by "_differentiable".

What is refused (WORKFLOW.md, section 3):

  * a block or a setting the schema does not know;
  * a block or a setting that is missing: no setting has a default;
  * a name the registry does not know.

The settings of a network, a loss, a protocol, an optimiser and a stopping
rule are not listed here. Each component states its own, in
`settings_in_a_configuration`, so adding a component does not change this file.

The run key is computed from the configuration, the key of the tracks
included. One setting is left out of it: `training.rounds_at_most`. It is the
cap on the rounds, and raising it extends a run; it does not make another run.
"""
from __future__ import annotations

import copy
import hashlib
import json

import yaml

from rkpinn import registry
from rkpinn.run_record.store import LENGTH_OF_A_KEY

BLOCKS = ("equation_of_motion", "tracks", "steps", "network", "loss", "target",
          "training", "seed")

SETTINGS_OF_FIXED_BLOCKS = {
    "equation_of_motion": ("system", "field_map"),
    "tracks": ("key",),
    "steps": ("number_of_steps", "number_of_stages"),
    "target": ("name",),
}

SETTINGS_OF_TRAINING = ("protocol", "restarts_per_round", "end_a_round_when_stalled",
                        "optimiser", "stopping_rule", "rounds_at_most")
SETTINGS_OF_A_STALL = ("gain_below", "restarts_running")

# block of the configuration -> (kind in the registry, the setting that names the component)
NAMED_COMPONENTS = {
    "network": ("network", "kind"),
    "loss": ("loss", "name"),
}
NAMED_COMPONENTS_OF_TRAINING = {
    "protocol": ("training_protocol", "name"),
    "optimiser": ("optimiser", "name"),
    "stopping_rule": ("stopping_rule", "name"),
}

LEFT_OUT_OF_THE_KEY = (("training", "rounds_at_most"),)


class ConfigurationRefused(ValueError):
    """The configuration has something the schema does not allow."""


def _exactly(what, given, wanted):
    if not isinstance(given, dict):
        raise ConfigurationRefused("%s must hold settings, not %r" % (what, given))
    unknown = sorted(set(given) - set(wanted))
    missing = sorted(set(wanted) - set(given))
    if unknown:
        raise ConfigurationRefused(
            "%s has settings that are not recognised: %s. Its settings are: %s"
            % (what, ", ".join(unknown), ", ".join(wanted)))
    if missing:
        raise ConfigurationRefused(
            "%s lacks settings: %s. No setting has a default" % (what, ", ".join(missing)))


def _known(kind, name):
    try:
        return registry.component(kind, name)
    except registry.UnknownComponent as refusal:
        raise ConfigurationRefused(str(refusal.args[0])) from None


def _named_component(what, block, kind, naming_setting):
    if not isinstance(block, dict) or naming_setting not in block:
        raise ConfigurationRefused("%s must state its %s" % (what, naming_setting))
    component = _known(kind, block[naming_setting])
    _exactly(what, block, (naming_setting,) + tuple(component.settings_in_a_configuration))
    return component


def _whole_number(what, value, at_least):
    if not isinstance(value, int) or isinstance(value, bool) or value < at_least:
        raise ConfigurationRefused(
            "%s must be a whole number of at least %d" % (what, at_least))


def validate(configuration: dict) -> dict:
    """The configuration, checked. Raises ConfigurationRefused."""
    _exactly("the configuration", configuration, BLOCKS)
    for block, settings in SETTINGS_OF_FIXED_BLOCKS.items():
        _exactly("the block %r" % block, configuration[block], settings)
    _known("equation_of_motion", configuration["equation_of_motion"]["system"])
    _known("field_map", configuration["equation_of_motion"]["field_map"])
    target = _known("target", configuration["target"]["name"])
    for block, (kind, naming) in NAMED_COMPONENTS.items():
        _named_component("the block %r" % block, configuration[block], kind, naming)
    training = configuration["training"]
    _exactly("the block 'training'", training, SETTINGS_OF_TRAINING)
    for setting, (kind, naming) in NAMED_COMPONENTS_OF_TRAINING.items():
        _named_component("training.%s" % setting, training[setting], kind, naming)
    if training["end_a_round_when_stalled"] is not None:
        _exactly("training.end_a_round_when_stalled",
                 training["end_a_round_when_stalled"], SETTINGS_OF_A_STALL)
    steps = configuration["steps"]
    _whole_number("steps.number_of_steps", steps["number_of_steps"], 1)
    _whole_number("steps.number_of_stages", steps["number_of_stages"], 0)
    _whole_number("training.restarts_per_round", training["restarts_per_round"], 1)
    _whole_number("training.rounds_at_most", training["rounds_at_most"], 1)
    _whole_number("the seed", configuration["seed"], 0)

    network = registry.component("network", configuration["network"]["kind"])
    loss = registry.component("loss", configuration["loss"]["name"])
    protocol = registry.component("training_protocol", training["protocol"]["name"])
    if network.has_stages and steps["number_of_stages"] < 1:
        raise ConfigurationRefused(
            "the network %r has stages, so steps.number_of_stages must be at least 1"
            % network.kind)
    if not network.has_stages and (steps["number_of_stages"] != 0
                                   or steps["number_of_steps"] != 1):
        raise ConfigurationRefused(
            "the network %r covers the crossing in one step with no stages, so steps "
            "must be {number_of_steps: 1, number_of_stages: 0}" % network.kind)
    if protocol.for_networks_with_stages != network.has_stages:
        raise ConfigurationRefused(
            "the protocol %r is not for the network %r" % (protocol.name, network.kind))
    if not target.can_be_trained_on:
        raise ConfigurationRefused(
            "the target %r is for the evaluation; a run cannot be trained on it"
            % target.name)
    if loss.needs_labels != target.needs_labels:
        raise ConfigurationRefused(
            "the loss %r %s labels and the target %r %s them"
            % (loss.name, "needs" if loss.needs_labels else "takes no",
               target.name, "gives" if target.needs_labels else "does not give"))
    if not loss.needs_labels and not network.has_stages:
        raise ConfigurationRefused(
            "the loss %r is built on the stage residual, and the network %r has no stages"
            % (loss.name, network.kind))
    return configuration


def read_configuration(path: str) -> dict:
    """The configuration in a YAML file, checked."""
    with open(path) as handle:
        return validate(yaml.safe_load(handle))


def write_configuration(configuration: dict, path: str) -> None:
    with open(path, "w") as handle:
        yaml.safe_dump(plain(configuration), handle, sort_keys=True,
                       default_flow_style=False)


def plain(value):
    """Tuples as lists and nothing but plain values: what YAML would read back."""
    return json.loads(json.dumps(value))


def content_of_the_key(configuration: dict) -> dict:
    content = plain(copy.deepcopy(configuration))
    for block, setting in LEFT_OUT_OF_THE_KEY:
        content[block].pop(setting, None)
    return content


def run_key(configuration: dict) -> str:
    """The key of the run: the start of the sha256 hash of its configuration,
    with the cap on the rounds left out."""
    text = json.dumps(content_of_the_key(validate(configuration)), sort_keys=True,
                      separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:LENGTH_OF_A_KEY]


def difference(first: dict, second: dict, _path="") -> list:
    """The settings in which two configurations differ, one line each. A
    controlled comparison differs in one block."""
    lines = []
    for name in sorted(set(first) | set(second)):
        where = _path + name
        a, b = first.get(name, "<absent>"), second.get(name, "<absent>")
        if isinstance(a, dict) and isinstance(b, dict):
            lines += difference(a, b, where + ".")
        elif plain(a) != plain(b):
            lines.append("%s: %r -> %r" % (where, a, b))
    return lines
