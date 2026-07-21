# One_step_network_v2 — the step-2 experiment on official-sample data

Identical experiment to `../One_step_network/` (the paper's one-step q=8 network on the
frozen magnet crossing, physics loss vs data twin, trained to genuine stall) with exactly
one change: the training population comes from the **official TestFileDB sample**
(`expected_2024_minbias_xdigi` via `Data_generation_exploration/Official_xdigi/`,
training set `train_official_v2.npz`) instead of our self-generated GaussMB100 events —
per George's 2026-07-21 directive that home-grown event generation is not to be trusted
for provenance.

Everything held fixed from v1: the frozen leg (z0 = 2648.2 -> z1 = 7826.0 mm), q = 8,
the 4x50 float64 network, both losses, L-BFGS protocol with the stall criterion
(two consecutive <1% restarts), seeds {0,1,2}, 2000-state training cap, split seed.
The v1 lesson is baked in: training goes **straight to stall** (restart cap 150, no
6-restart stage), followed by the confirmation pass.

| file | what it does |
|---|---|
| [prepare_data.py](prepare_data.py) | v1 script + `TRAINING_NPZ` override -> `results/frozen_leg_data.npz` (2000/2082/2033 by-particle) |
| [training.py](training.py) | v1 script with OUTER=150 (straight to stall); model/field imported from `../One_step_network` |
| [continue_training.py](continue_training.py) | confirmation pass: continues each run past its stall; re-stall = converged |
| [plot_one_step.py](plot_one_step.py) | figures, v2 vs the v1 converged results -> `figures/one_step_results_v2.png` |

The scripts are byte-patched copies of v1 (patch points marked `v2:` inline) so that any
result difference can only come from the training population.

## The fiducial requirement (added 2026-07-21, after the first v2 run)

The first v2 attempt converged to physics 403–595 µm vs data twin 175–213 µm — apparently
reinstating the factor-2 gap that v1 had shown to be a budget artefact. It was not a property
of the loss. `diagnose_physics_gap.py` found that **21 of 2000 training states (1%) have
reference trajectories that leave the field map** (soft 1.2–1.5 GeV tracks bending out past
|x| = 4 m, beyond both the map bounds and the LHCb acceptance), and those 21 states carried
**50% of the whole physics loss**, with the top 1% of states carrying 68–78% (v1: 29–46%).

The asymmetry is structural, and is the main portability lesson of this experiment: the
physics loss evaluates the field *at the trajectory positions it proposes*, so a trajectory
outside the map is being asked to satisfy equations built on a clamped, meaningless field —
an irreducible residual that the optimiser cannot reduce and that dominates the gradient. The
data twin never touches the field: it just fits its labels (computed with the same clamped
field) and absorbs the pathology silently. v1's smaller population happened to contain zero
such states, which is why the effect never appeared there.

`prepare_data.py` therefore applies a fiducial requirement, stated before the rerun and applied
identically to all three splits: **the reference trajectory must stay inside the field map**,
i.e. inside the region where the ODE being solved is defined at all. It removes ~1% of states
(21 train / 20 val / 15 test). The pre-cut results are preserved in `results_nofiducial/`.

With the cut, the physics runs no longer stall at a plateau (57–72 restarts before, 147–152
after) and the training loss falls from ~1.1e-4 to ~8.3e-6 — the same range as the data twin
and as v1's converged physics runs.

| file | what it does |
|---|---|
| [run_one.py](run_one.py) | trains ONE (mode, seed) to stall, checkpointing every restart — resumable, and the six run concurrently (4 threads each) on the shared node |
| [aggregate.py](aggregate.py) | scores the six checkpoints -> `results/{summary,histories}.csv`, `predictions.npz` |
| [diagnose_physics_gap.py](diagnose_physics_gap.py) | the out-of-map diagnostic behind the fiducial cut |
