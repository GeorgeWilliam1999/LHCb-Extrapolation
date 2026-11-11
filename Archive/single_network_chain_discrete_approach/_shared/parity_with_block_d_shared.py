#!/usr/bin/env python
"""The gate on this copy: it must be the Block D shared package, bit for bit.

`_shared/` here started (2026-09-16) as a verbatim copy of the ten Python modules
of `../../multi_network_chain_discrete_approach/_shared/`. The copy's own smoke
tests are not usable from here (they gate against Block 0 baselines that live in
the old folder), so this script checks the copy directly:

  1. every copied module has the same sha256 as its original;
  2. the path constants that depend on where the package sits (the repository
     root, the training set, the field-map md5) resolve to the same values;
  3. the same computations through both packages give bit-identical numbers:
     the Gauss-Legendre tableau at q = 16, RK6 over 40 mm and over 2,589 mm on
     500 real start states, and the physics loss and its gradient of a seeded
     network on those states.

Once the copy is extended, (1) fails for the changed modules by design; the
record of the verbatim state is `results/parity_with_block_d_shared.json`.

Run:  PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python parity_with_block_d_shared.py
"""
import hashlib
import importlib
import importlib.util
import json
import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np   # noqa: E402
import torch         # noqa: E402

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

HERE = os.path.dirname(os.path.abspath(__file__))
OLD = os.path.abspath(os.path.join(HERE, "..", "..", "multi_network_chain_discrete_approach", "_shared"))
MODULES = ("__init__.py", "field_v8r1.py", "field_torch.py", "irk.py", "reference.py",
           "model.py", "prepare.py", "evaluate.py", "train.py", "use_shared.py")
D0 = os.path.join(OLD, "..", "Block_D_fixed_step_crossing", "D0_Crossing_dataset", "results",
                  "crossing_particles.npz")


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_old():
    spec = importlib.util.spec_from_file_location("_shared_old", os.path.join(OLD, "__init__.py"),
                                                  submodule_search_locations=[OLD])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_shared_old"] = mod
    spec.loader.exec_module(mod)
    return "_shared_old"


def main():
    out = {"old": OLD, "new": HERE}
    out["sha256_identical"] = {m: sha(os.path.join(HERE, m)) == sha(os.path.join(OLD, m)) for m in MODULES}

    sys.path.insert(0, os.path.dirname(HERE))
    pkgs = {"new": "_shared", "old": load_old()}
    ref = {k: importlib.import_module(v + ".reference") for k, v in pkgs.items()}
    mdl = {k: importlib.import_module(v + ".model") for k, v in pkgs.items()}
    assert os.path.dirname(ref["new"].__file__) == HERE and os.path.dirname(ref["old"].__file__) == OLD

    out["paths_identical"] = {
        "REPO": ref["new"].REPO == ref["old"].REPO,
        "DATA_NPZ": ref["new"].DATA_NPZ == ref["old"].DATA_NPZ,
        "field_md5_up": ref["new"].field_md5("up") == ref["old"].field_md5("up"),
    }

    S = np.load(D0)["train_S0"][:500]
    z0, L = 2648.2, 7826.0 - 2648.2
    num = {}
    tabs = {k: r.gauss_legendre(16) for k, r in ref.items()}
    num["tableau_q16"] = all(np.array_equal(a, b) for a, b in zip(tabs["new"], tabs["old"]))
    for label, dz in (("rk6_40mm", L / 128), ("rk6_2589mm", L / 2)):
        a = ref["new"].rk6_rows(S, z0, z0 + dz, field=ref["new"].make_field("up"))
        b = ref["old"].rk6_rows(S, z0, z0 + dz, field=ref["old"].make_field("up"))
        num[label] = bool(np.array_equal(a, b))

    losses, grads = {}, {}
    for k in pkgs:
        c, A, b = ref[k].gauss_legendre(4)
        dz = L / 64
        torch.manual_seed(0)
        net = mdl[k].OneStepNetwork(4, S.std(axis=0), np.append(S[:, :4].std(axis=0), S[:, 4].std()),
                                    width=128, depth=2)
        rates = mdl[k].LHCbRates(ref[k].make_field("up"))
        loss = mdl[k].physics_loss(net, rates, torch.tensor(S), dz, torch.tensor(z0 + c * dz),
                                   torch.tensor(A), torch.tensor(b))
        loss.backward()
        losses[k] = loss.item()
        grads[k] = torch.cat([p.grad.reshape(-1) for p in net.parameters()])
    num["physics_loss"] = losses["new"] == losses["old"]
    num["physics_grad"] = bool(torch.equal(grads["new"], grads["old"]))
    num["physics_loss_value"] = losses["new"]
    out["numbers_identical"] = num

    passed = (all(out["sha256_identical"].values()) and all(out["paths_identical"].values())
              and all(v for kk, v in num.items() if kk != "physics_loss_value"))
    out["PASS"] = bool(passed)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "parity_with_block_d_shared.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    print("PARITY PASS" if passed else "PARITY FAIL")
    return passed


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
