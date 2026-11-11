# How does the stage residual become the endpoint error

**Last updated:** 2026-11-12 · **State:** planned; criteria written, no number computed yet

## What this directory is

Strand 1 of the plan "RK-PINN: plan for the next phase" (Claude Docs, 2026-11-12; George agreed on 2026-11-12 that RK-PINN theory is a goal). An analysis of existing runs. It trains nothing, submits nothing to the farm and writes nothing to the store.

## Contents

| Name | What it does | State |
|---|---|---|
| `README.md` | this file: the question, the theory, the runs, the criteria | built |
| `analysis/` | the script, the tables (`results/`), the figures (`figures/`) and the results note | planned |

## The question

The loss trains the stage residual r of each step. The score is the error at the end of the track. What is the exact chain from one to the other, and which of its links sets the size of the endpoint error?

## The theory to test

Take one step of a self-chained network from the state S_n on plane z_n, with length dz and q stages. The network emits the stages Ŷ. The exact scheme from the same S_n has the stages Y*. Write F(Y) for the rates at the stage planes, J for their Jacobian, A and b for the Gauss–Legendre tableau.

| Name | Definition | What the theory says |
|---|---|---|
| stage residual r | Ŷ − S_n − dz A F(Ŷ) | what the loss makes small |
| stage error ε | Ŷ − Y* | ε = (I − dz A⊗J)⁻¹ r to first order. Because dz A J is small, ε ≈ r: the loss measures the stage error directly |
| local defect δ | (network end state) − (exact end state), both from S_n | summed end state: δ = dz Σ b_k (F(Ŷ_k) − F(Y*_k)) ≈ dz Σ b_k J_k ε_k. In x and y this is exactly first order, because dx/dz = tx is linear. A predicted end state has no factor dz J |
| carried defect | X(S_{n+1}) − X(exact end from S_n), where X carries a state by the exact scheme from z_{n+1} to the last plane | the carried defects sum exactly to (network end − exact-chain end): the sum telescopes. The test is whether a cheap linear carry, δ_x + δ_tx (z_last − z_{n+1}) and the same in y, reproduces each carried defect |
| exact-scheme error | exact chain minus the sixth-order reference | independent of the network; 0.147 µm at 64 steps of 2 stages |

Two consequences to test:

1. **Coherence.** If the carried defects have one sign along a track, the endpoint error is about N times one carried defect; if their signs are random, about √N times. The measured lag-1 correlation of 0.85 to 0.90 predicts the first.
2. **Step length.** If the network keeps a fixed relative precision of its output, ε grows in proportion to the change over one step, that is to dz. Then δ ∝ dz², and N = L / dz coherent defects give an endpoint error ∝ dz. The step-length experiment measured single-step error ∝ dz² and end error about ∝ dz.

## The runs

All are existing runs, read at their last round, on the 1,452 test tracks.

| Purpose | Runs |
|---|---|
| main, 64 steps of 2 stages, summed | `25dfa17ef168` (straight line, per-track scale), `9cef9c160b7d` (straight line, global scale), `49e5e2450eaa` (direct states), `166cfc8cdfd6` (Broyden, straight line) |
| summed against predicted | `bddea42ea1ef` (summed), `4cc888dcc710` (predicted) |
| step length, 2 stages, straight line per-track | `a66c06472d3d` (8 steps), `6c4a3b022a2f` (16), `29cbba959300` (32), `25dfa17ef168` (64), `15242aa17c65` (128), `eeb809e9254d` (256) |

## What is computed

For each run, along its own chain on each test track, at every step: r, ε, δ, the first-order δ from ε, the first-order ε from r, the carried defect (by the exact scheme on 200 tracks; by the linear carry on all), and the exact-scheme error. Norms use millimetres for x and y and milliradians for tx and ty, the scaling of the package's root finder. Then, per run: the medians of each, the endpoint against the sum of the carried defects, and the coherence ratio |Σ carried| / √(Σ |carried|²) against √N.

## The criteria, written before any number

| # | Claim | Holds if |
|---|---|---|
| 1 | ε ≈ (I − dz A⊗J)⁻¹ r | median relative difference below 10 % in each run |
| 2 | ε ≈ r | median of \|ε − r\| / \|ε\| below 10 % in each run |
| 3 | δ ≈ dz Σ b J ε (summed runs) | median relative difference below 10 % |
| 4 | the linear carry reproduces the carried defect | on the 200 tracks: correlation of the linear and the exact carried defect above 0.95 in x and in y, and median relative difference of their sums below 10 %, in each run |
| 5 | the defects are coherent | median over tracks of \|Σ c_n\| / √(Σ c_n²), with c_n the exact carried defect in x, above √N / 2 (above 4 at 64 steps) |
| 6 | the predicted run's δ is not first order in its ε | its δ / (dz J ε) above 10 in the median, against about 1 for the summed run |
| 7 | fixed relative precision | median \|ε\| / dz varies by less than a factor 2 over 8 to 256 steps, and the median \|δ\| falls with dz with an exponent between 1.7 and 2.3 |

If a claim fails, the results note says by how much, and the theory is changed before it is written up.
