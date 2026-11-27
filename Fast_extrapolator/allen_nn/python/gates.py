"""The gates of the standalone tool (P1).

1. arkn_state (float) gives the same states as the earlier copy of the ARKN
   (cpp/lhcb/allen_nystrom.h, AllenAdaptiveNystrom), bit for bit.
2. arkn (float) against the Rec STEP integrator (cpp/lhcb/rec_step.h, double), the
   state and the 4 x 3 transport matrix: the two adaptive Nystrom integrators of LHCb.
3. The crossing in double (an gate) against the torch model of fastex/train.py: the
   end state (relative difference) and d end / d s0 against autograd.
4. The tool in float against the tool in double, and the polarity symmetry.

    python allen_nn/python/gates.py [number of tracks]      writes allen_nn/results/gates.txt
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import an_tools as an  # noqa: E402
from export_json import load  # noqa: E402

torch.set_num_threads(4)
lines = []


def say(text):
    print(text, flush=True)
    lines.append(text)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    s, ref, p = an.test_tracks()
    s, ref, p = s[:n], ref[:n], p[:n]

    # 1
    a, _, ca = an.run("arkn_state", s, an.Z0, an.Z1)
    b, _, cb = an.run("allen_adaptive_nystrom", s, an.Z0, an.Z1)
    same = np.all(a[:, :4] == b[:, :4])
    say("gate 1: arkn_state == allen_adaptive_nystrom (float), %d tracks: %s; largest difference %.3g mm; "
        "accepted steps equal: %s" % (n, same, np.abs(a - b).max(), np.all(ca[:, 1] == cb[:, 1])))

    # 2
    a, ja, ca = an.run("arkn", s, an.Z0, an.Z1)
    r, jr, _ = an.run("rec_step", s, an.Z0, an.Z1, precision="double")
    d = np.hypot(a[:, 0] - r[:, 0], a[:, 1] - r[:, 1]) * 1e3
    say("gate 2: arkn (float) vs rec_step (double): radial difference median %.3g um, 99 %% %.3g um, max %.3g um; "
        "radial error against the reference: arkn median %.2f um, rec_step %.2f um; mean field reads arkn %.1f"
        % (np.median(d), np.percentile(d, 99), d.max(),
           np.median(np.hypot(*(a[:, :2] - ref[:, :2]).T)) * 1e3,
           np.median(np.hypot(*(r[:, :2] - ref[:, :2]).T)) * 1e3, ca[:, 0].mean()))
    rel = np.abs(ja[:, :, 2:] - jr[:, :, 2:]) / (np.abs(jr[:, :, 2:]).max(axis=0, keepdims=True) + 1e-30)
    say("        transport matrix (columns tx0, ty0, q/p): median relative difference %.2g, 99 %% %.2g "
        "(relative to the largest entry of each element)" % (np.median(rel), np.percentile(rel, 99)))

    # 3
    for name in sorted(os.listdir(an.WEIGHTS)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(an.WEIGHTS, name)
        meta = json.load(open(path))
        with tempfile.TemporaryDirectory(dir=an.SCRATCH) as work:
            np.ascontiguousarray(s).tofile(os.path.join(work, "s.f64"))
            subprocess.run([an.program(), "gate", "states=" + os.path.join(work, "s.f64"), "weights=" + path,
                            "out=" + os.path.join(work, "o.f64")], check=True)
            o = np.fromfile(os.path.join(work, "o.f64")).reshape(-1, 29)
        sk, end, J = o[:, :5], o[:, 5:9], o[:, 9:].reshape(-1, 4, 5)
        c, constants, _, network, _ = load(meta["name"], meta["round"])
        x = torch.tensor(s, requires_grad=True)
        corr = network(x / torch.tensor(constants["scale_of_inputs"], dtype=torch.float64)) \
            * torch.tensor(constants["scale_of_outputs"], dtype=torch.float64) * (x[:, 4:5] / constants["scale_of_qop"])
        end_torch = torch.tensor(sk[:, :4]) + corr
        relative = np.abs(end - end_torch.detach().numpy()) / np.maximum(np.abs(corr.detach().numpy()), 1e-300)
        relative_end = np.abs(end - end_torch.detach().numpy()) / np.maximum(np.abs(end), 1e-300)
        # autograd of the correction; the Nystrom part of the matrix is the chained one of the tool
        grads = np.stack([torch.autograd.grad(corr[:, o].sum(), x, retain_graph=True)[0].numpy() for o in range(4)], 1)
        _, jn, _ = an.run("nystrom:%d" % meta["steps"], s, an.Z0, an.Z1, precision="double")
        jc = J - jn
        jc[:, :, :2] -= 0  # the Nystrom matrix of the tool has the identity in x0, y0 as nystrom: has
        rel_j = np.abs(jc - grads) / (np.abs(grads).max(axis=0, keepdims=True) + 1e-30)
        say("gate 3: %-24s end vs torch: largest |difference| / |correction| %.2g, / |end| %.2g; "
            "d correction / d s0 vs autograd: largest relative %.2g  -> %s"
            % (meta["name"], relative.max(), relative_end.max(), rel_j.max(),
               "pass" if relative.max() < 1e-5 and rel_j.max() < 1e-5 else "FAIL"))

    # 4
    w = os.path.join(an.WEIGHTS, "linear_nystrom_8_2x64.json")
    f, jf, cf = an.run("tool:" + w, s, an.Z0, an.Z1)
    g, jg, _ = an.run("tool:" + w, s, an.Z0, an.Z1, precision="double")
    d = np.hypot(f[:, 0] - g[:, 0], f[:, 1] - g[:, 1]) * 1e3
    say("gate 4: tool float vs double (8 steps, 2x64): radial median %.3g um, max %.3g um; fell back %d of %d"
        % (np.median(d), d.max(), (cf[:, 3] == 1).sum(), n))
    flipped = s.copy()
    flipped[:, 4] *= -1
    h, jh, _ = an.run("tool:" + w + ":polarity=-1", flipped, an.Z0, an.Z1, map_name="v8r1_down")
    d = np.abs(h[:, :4] - f[:, :4]).max()
    jd = np.abs(jh[:, :, :4] - jf[:, :, :4]).max(), np.abs(jh[:, :, 4] + jf[:, :, 4]).max()
    say("        polarity: tool on the down map with -q/p and polarity -1 vs the up map: largest state "
        "difference %.3g, matrix %.3g (q/p column with its sign flipped %.3g)" % (d, jd[0], jd[1]))
    r_up, _, _ = an.run("reference:2000", s[:200], an.Z0, an.Z1, precision="double")
    r_dn, _, _ = an.run("reference:2000", flipped[:200], an.Z0, an.Z1, precision="double", map_name="v8r1_down")
    say("        the reference (RK4, 2000 steps) on the down map with -q/p vs the up map: largest difference %.3g mm"
        % np.abs(r_up[:, :4] - r_dn[:, :4]).max())
    os.makedirs(os.path.join(an.FOLDER, "results"), exist_ok=True)
    with open(os.path.join(an.FOLDER, "results", "gates.txt"), "w") as handle:
        handle.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
