# _shared — machinery for the single-network chain

**What it is (2026-09-16).** A verbatim copy of the ten Python modules of
`../../multi_network_chain_discrete_approach/_shared/`:

| module | what it holds |
|---|---|
| `field_v8r1.py` | the canonical v8r1 field-map loader (numpy) |
| `field_torch.py` | its differentiable fp64 twin, for the physics loss |
| `irk.py` | the Gauss–Legendre tableau builder, verified to q = 50 |
| `reference.py` | the equation of motion, RK4 and RK6 references, constants, paths |
| `model.py` | the rates, the one-step network, the physics (RK-PINN) and data losses |
| `prepare.py` | the dataset builders and the magnet-leg row selection |
| `evaluate.py` | scoring helpers and chaining |
| `train.py` | the shared L-BFGS stall-and-confirm driver (Block D's; Block E has its own round trainer) |
| `use_shared.py` | the path helper each experiment folder copies |
| `__init__.py` | the package |

Not copied: `smoke_tests.py` and `vendoring_parity.py`, which gate the old package
against Block 0 baselines that live in the old folder, and the farm harness
(`condor/`), which Block E will write for its own trainer.

## script → output

| script | what it does | output |
|---|---|---|
| [parity_with_block_d_shared.py](parity_with_block_d_shared.py) | the gate on the copy: identical sha256 for every module; the same repository root, training-set path and field-map md5; bit-identical tableau (q = 16), RK6 over 40 mm and 2,589 mm on 500 real states, and physics loss and gradient of a seeded network through both packages | `results/parity_with_block_d_shared.json` |

**Result (2026-09-16): PASS** on every item.

Block E's extensions live in the experiment folders, not here: the network with the
start-plane input and the separate y scale is
`../Block_E_single_network_chain/E1_Network_grid/chain_network.py`, and the round
trainer is `train_network.py` beside it. The shared residual
(`model.physics_loss`) is used unchanged. Nothing in this folder has been modified
since the copy.
