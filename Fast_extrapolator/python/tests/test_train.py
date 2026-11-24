"""Gate of the trainer of stage 2: the correction that `export` writes gives, in
the compiled program in double precision, the end states of the network in
torch. It needs the two short runs gate_hybrid and gate_none in the store
(stage_2_training/README.md says how they are made).
"""
from __future__ import annotations

import numpy as np
import torch

from fastex import train
from fastex.compiled import run


def test_the_compiled_network_is_the_torch_one():
    for name in ("gate_hybrid", "gate_none"):
        c, constants, network, _ = train.export(name)
        states, _, _ = train.split(c["reference"], "validation")
        states = states[:500]
        if c["start"] == "nystrom":
            start = run("nystrom:%d" % c["steps"], states, train.Z0, train.Z1,
                        reader=train.READER_OF[c["reference"]], precision="double")[0][:, :4]
        else:
            start = train.start_of(states, c)
        with torch.no_grad():
            expected = start + train.correction(network, torch.tensor(states), c, constants).numpy()
        for options, limit in (("", 1e-9), (":fast_tanh=1", 1e-9)):
            compiled, _ = run("corrected:" + name + options, states, train.Z0, train.Z1,
                              reader=train.READER_OF[c["reference"]], precision="double")
            assert np.abs(compiled[:, :2] - expected[:, :2]).max() < limit * max(1.0, np.abs(expected[:, :2]).max())
