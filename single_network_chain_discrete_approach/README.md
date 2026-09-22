# single_network_chain_discrete_approach — one network per step length, chained to itself

**Status (2026-09-20): Block E trained, analysed in E3 on the checkpoints of 2026-09-18, then extended to the
validation plateau (9 of 16 there, 7 still improving); Block F trained, two of its three runs finished and
plateaued, the third still training. Nothing committed yet.** The plan, its open
decisions and the worklog live in the Notion to-do
"Train one network per step length and chain it to itself across the magnet"
(https://app.notion.com/p/3dd5d544b9d98144bdcbd17c5a66e3e0).

## The approach

The crossing runs from the last UT plane, z0 = 2648.2 mm, to the first SciFi plane,
z1 = 7826.0 mm (L = 5177.8 mm). For a step length dz = L/N and a stage count q, **one
network** maps a track state at any start plane to the state dz further along. The
**same network** is used for every track and for every step: a crossing applies it N
times. Concretely, each (N, q) run has exactly one weight file (`network.pt`), and
`chain_network.carry` applies that one network N times, advancing only the start-plane
input; rebuilding the 64-step test chain by hand from the single file reproduces the
stored chain states to the bit (checked 2026-09-20). Block D, by contrast, trained a
**different** network for every step and chained those. It is trained over the whole crossing with the discrete-time Runge–Kutta
(RK-PINN) construction, and so learns the physics everywhere along the magnet.

It follows the fixed-step study in
`../multi_network_chain_discrete_approach/Block_D_fixed_step_crossing/`, which
trained a **separate** network for every step of a chain (128 networks for N = 128).
That folder was called `Raissi_disc_time_approach/` until 2026-09-16.

## Layout

| folder | what it will hold |
|---|---|
| [_shared/](_shared/) | the machinery every experiment imports: a verbatim copy of `../multi_network_chain_discrete_approach/_shared/` (checked by its parity tests), then extended with the start-plane input, the separate y scale and the round trainer |
| [Block_E_single_network_chain/](Block_E_single_network_chain/) | the first study: N = 2, 64, 128, 256 × q = 2, 4, 8, 16 (16 networks) |
| [Block_F_reweighted_loss/](Block_F_reweighted_loss/) | the same networks with ONE change: the weights inside the loss. Every residual measured as the displacement it would cause at the SciFi plane, as a fraction of that track's own bend, with the 10-50 GeV band weighted up |

Block letters continue the project's sequence (Blocks 0, A, C and D are in the old
folder), so "Block E" is unambiguous.
