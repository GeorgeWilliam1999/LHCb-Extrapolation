# Block D — fixed-step networks chained across the magnet

14 September 2026 onward. George's ruling of 2026-09-14 replaced the
dz-as-input networks of Blocks A3 and C with a faithful clone of the
discrete-time construction of Raissi, Perdikaris and Karniadakis (2019,
section 3) on the magnet crossing: **one network per fixed step**, nothing
about the step as an input, the steps chained across the magnet exactly as the
paper's scheme steps in time.

| Folder | Step | What it holds |
|---|---|---|
| [`D0_Crossing_dataset/`](D0_Crossing_dataset/) | D0 | the particles: real last-UT state transported to z0, the RK6 truth on 129 planes, the real first-SciFi state; the material floor |
| [`D1_Chain_grid/`](D1_Chain_grid/) | D1 | the grid: N in {1, 4, 16, 64, 128} steps × q in {1 … 20} stages, one network per leg trained in sequence, 4,260 networks in 100 chains; the tables and figures; **how to apply the networks** |
| [`D2_Comparators/`](D2_Comparators/) | D2 | the exact collocation scheme chained the same way (the ceiling), the one supervised twin, the comparison table |

The crossing: z0 = 2648.2 mm (last UT plane) → z1 = 7826.0 mm (first SciFi
plane), L = 5177.8 mm, field v8r1 **MagUp** (the sample's polarity). One
architecture: two hidden layers of 128, straight-line-plus-correction output
(each output is the straight line from the input plus a per-track-scaled
network correction; see below),
label-free physics loss only, one seed. The comparators: the RK6 truth at
0.1 mm (the fine Runge–Kutta), the particle's real SciFi state (data ground
truth, material included), the exact scheme at the same (N, q), the straight
line. Decisions recorded in the Notion to-do "Rebuild the magnet-crossing
extrapolator as fixed-step networks and chain them across the magnet".

**Output form (George, 2026-09-23).** The network's outputs are the q Gauss–Legendre stage states and the endpoint state of one step, each written as the straight-line extrapolation of the input state plus a network-predicted correction, scaled per track: `output_j = straight_j(S) + σ(S) ⊙ NN(S)_j`. The network therefore learns the correction to the straight line — the magnet's bending over one step — rather than the stage states themselves. This departs from Raissi, Perdikaris and Karniadakis (2019), whose network emits the states directly; the loss is unchanged and sees only the resulting stage states. The form was introduced on 5 September 2026 for a network serving many step lengths; it was put to George as an explicit choice for the fixed-step study and chosen on 14 September 2026 (question 4 of that plan); the single-network chain then inherited it as 'Block D's form' without re-examining it, and the theory did not state that the learned quantity is the correction rather than the stage states (noted 23 September 2026).
