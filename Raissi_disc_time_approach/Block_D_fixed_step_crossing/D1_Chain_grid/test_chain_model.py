#!/usr/bin/env python
"""The gate on chain_model.py, run before any farm job.

1. the torch straight line equals the numpy one;
2. the torch field integral equals the numpy one (1e-12 relative);
3. with the last layer zeroed the output IS the straight line, exactly;
4. the physics loss and its gradients are finite at initialisation, at q = 1,
   8 and 20, on the shortest and the longest leg of Block D;
5. the q = 0 twin forwards to (N, 1, 4) and its loss is finite;
6. a state dict round-trips through a fresh constructor.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np, torch
import use_shared                                        # noqa: F401
from _shared.model import LHCbRates, physics_loss
from _shared.reference import FROZEN_LEG, gauss_legendre, make_field, load_training
from chain_model import (FrozenResidualNetwork, field_integral_numpy, residual_data_loss,
                         straight_line_states)

fld = make_field("up")
d = load_training(split="train", leg="B")
X = d["X"][d["X"][:, 6] > d["X"][:, 5]][:300].astype(np.float64)
S_np = X[:, :5]
S = torch.tensor(S_np)
in_scale = S_np.std(axis=0)
Z0, Z1 = FROZEN_LEG["z0"], FROZEN_LEG["z1"]
L = Z1 - Z0
ok = True
for N in (1, 128):
    dz = L / N
    for q in (1, 8, 20):
        c, A, b = gauss_legendre(q)
        torch.manual_seed(0)
        m = FrozenResidualNetwork(q, in_scale, in_scale, c=c, z0=Z0, dz=dz, field=fld)
        with torch.no_grad():
            st = m.straight(S).numpy()
            IB = m.field_integral(S).numpy()
        e1 = np.abs(st - straight_line_states(S_np, dz, c)).max()
        IBn = field_integral_numpy(S_np, Z0, dz, fld)
        e2 = np.abs(IB - IBn).max() / np.abs(IBn).max()
        z = FrozenResidualNetwork(q, in_scale, in_scale, c=c, z0=Z0, dz=dz, field=fld)
        z.load_state_dict(m.state_dict())
        with torch.no_grad():
            z.net[-1].weight.zero_(); z.net[-1].bias.zero_()
            e3 = (z(S) - torch.tensor(st)).abs().max().item()
        rates = LHCbRates(fld)
        zn = torch.tensor(Z0 + c * dz)
        loss = physics_loss(m, rates, S, dz, zn, torch.tensor(A), torch.tensor(b))
        loss.backward()
        gfin = all(torch.isfinite(p.grad).all().item() for p in m.parameters())
        sc = m.residual_scale(S)
        line = ("N=%3d q=%2d  straight %.1e  IB %.1e  zeroed %.1e  loss %.3e  grad finite %s"
                "  scale_pos med %.2e mm  n_par %d"
                % (N, q, e1, e2, e3, loss.item(), gfin, sc[:, 0, 0].median().item(),
                   sum(p.numel() for p in m.parameters())))
        print(line)
        ok &= (e1 < 1e-9 and e2 < 1e-12 and e3 == 0.0 and np.isfinite(loss.item()) and gfin)
# the twin
torch.manual_seed(0)
tw = FrozenResidualNetwork(0, in_scale, in_scale, c=np.zeros(0), z0=Z0, dz=L, field=fld)
ref = torch.tensor(np.concatenate([straight_line_states(S_np, L, np.zeros(0)),
                                   np.zeros((len(S_np), 1, 1))], axis=2))
out = tw(S)
lt = residual_data_loss(tw, S, ref)
print("twin: out shape", tuple(out.shape), "loss %.3e" % lt.item())
ok &= tuple(out.shape) == (len(S_np), 1, 4) and np.isfinite(lt.item())
print("ALL CHAIN MODEL CHECKS PASS" if ok else "CHAIN MODEL CHECKS FAIL")
raise SystemExit(0 if ok else 1)
