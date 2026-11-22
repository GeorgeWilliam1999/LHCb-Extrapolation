"""Write the configurations of this experiment from the configurations they copy.

    python make_configurations.py

Every file is a copy of an earlier configuration, changed only in the track set
(5558be31abeb), the field map (v8r1_up_cubic), the seed, and, for the second
group, the number of steps or stages. Writes the .yaml files beside this script
and ../farm/runs.txt.
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.abspath(os.path.join(HERE, "..", ".."))
OPTIMISER = "does_a_better_optimiser_improve_the_endpoint/configurations/"
STRAIGHT = "does_the_straight_line_form_restore_the_old_accuracy/configurations/"
PHYSICS = "can_the_network_alone_reach_the_exact_scheme/configurations/"
PAPER = "redo_the_papers_nine_runs_in_the_summed_form/configurations/"
SEEDS = [1, 2, 3, 4, 5]
# runs already in the store on the cubic set, seed 1, so not written again
DONE = {("straight_line_per_track_scale__lbfgs", 1): "400182ba3233",
        ("physics_layer", 1): "a8a4083032fd"}

# group 1: the headline networks, 64 steps x 2 stages, five seeds each
# (name, source, (cpus, memory MB))
HEADLINE = [
    ("direct_states__lbfgs", OPTIMISER + "direct_states__lbfgs__seed_2.yaml", (1, 12288)),
    ("direct_states__self_scaled_broyden", OPTIMISER + "direct_states__self_scaled_broyden__seed_1.yaml", (4, 16384)),
    ("straight_line_per_track_scale__lbfgs", OPTIMISER + "straight_line_per_track_scale__lbfgs__seed_2.yaml", (1, 12288)),
    ("straight_line_per_track_scale__self_scaled_broyden", OPTIMISER + "straight_line_per_track_scale__self_scaled_broyden__seed_1.yaml", (4, 16384)),
    ("straight_line_global_scale__lbfgs", STRAIGHT + "64_steps_2_stages_pooled_global_scale.yaml", (1, 12288)),
    ("physics_layer", PHYSICS + "physics_layer_64_steps.yaml", (1, 12288)),
]
# group 2: steps and stages, seed 1 only; (name, source, steps, stages, (cpus, memory))
SWEEP = [("straight_line_per_track_scale__self_scaled_broyden__%d_steps_%d_stages" % (n, q),
          OPTIMISER + "straight_line_per_track_scale__self_scaled_broyden__seed_1.yaml", n, q, (4, 16384))
         for n, q in [(16, 2), (32, 2), (128, 2), (256, 2), (64, 4), (64, 8), (64, 16)]]
SWEEP += [("physics_layer__%d_steps" % n, PHYSICS + "physics_layer_%d_steps.yaml" % n, n, 2, (1, 12288))
          for n in (16, 32)]
SWEEP += [("direct_states__lbfgs__%d_steps_%d_stages" % (n, q),
           PAPER + "%d_steps_%d_stages_pooled.yaml" % (n, q), n, q, (1, 16384))
          for n, q in [(128, 8), (256, 16)]]


def body_of(source):
    """The settings of a configuration, without its leading comment."""
    lines = open(os.path.join(EXPERIMENTS, source)).read().splitlines()
    while lines and lines[0].startswith("#"):
        lines.pop(0)
    return "\n".join(lines) + "\n"


def changed(text, seed, steps=None, stages=None):
    text, n1 = re.subn(r"(?m)^(  field_map: )v8r1_up$", r"\1v8r1_up_cubic", text)
    text, n2 = re.subn(r"(?m)^(  key: )12a8d35c3165$", r"\g<1>5558be31abeb", text)
    text, n3 = re.subn(r"(?m)^seed: \d+$", "seed: %d" % seed, text)
    assert n1 == n2 == n3 == 1, (n1, n2, n3)
    if steps is not None:
        text, n4 = re.subn(r"(?m)^(  number_of_steps: )\d+$", r"\g<1>%d" % steps, text)
        text, n5 = re.subn(r"(?m)^(  number_of_stages: )\d+$", r"\g<1>%d" % stages, text)
        assert n4 == n5 == 1
    return text


runs = []
for name, source, (cpus, memory) in HEADLINE:
    for seed in SEEDS:
        if (name, seed) in DONE:
            continue
        file = "%s__seed_%d.yaml" % (name, seed)
        header = ("# Headline network on the cubic map: %s, 64 steps x 2 stages, seed %d.\n"
                  "# Copied from %s and changed only in tracks.key (5558be31abeb),\n"
                  "# equation_of_motion.field_map (v8r1_up_cubic) and the seed. See ../README.md.\n"
                  % (name.replace("__", ", ").replace("_", " "), seed, source))
        open(os.path.join(HERE, file), "w").write(header + changed(body_of(source), seed))
        runs.append((file, cpus, memory))
for name, source, steps, stages, (cpus, memory) in SWEEP:
    file = "%s__seed_1.yaml" % name
    header = ("# Steps and stages on the cubic map: %s, %d steps x %d stages, seed 1.\n"
              "# Copied from %s and changed only in tracks.key (5558be31abeb),\n"
              "# equation_of_motion.field_map (v8r1_up_cubic), the seed and the steps and stages. See ../README.md.\n"
              % (name.split("__")[0].replace("_", " "), steps, stages, source))
    open(os.path.join(HERE, file), "w").write(header + changed(body_of(source), 1, steps, stages))
    runs.append((file, cpus, memory))
with open(os.path.join(HERE, "..", "farm", "runs.txt"), "w") as handle:
    for run in runs:
        handle.write("%s %d %d\n" % run)
print(len(runs), "configurations")
