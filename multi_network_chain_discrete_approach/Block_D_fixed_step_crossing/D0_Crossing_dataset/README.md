# D0_Crossing_dataset — the particles of Block D  (D0)

**Question.** What does every chain of Block D start from, what is it scored
against, and how much of a crossing is beyond any field-only method?

## script → output

| script | what it does | output |
|---|---|---|
| [build_dataset.py](build_dataset.py) | selects the forward cross-magnet particles of the MagUp sample whose last-UT and first-SciFi planes lie within 60 mm of the frozen planes, transports the real last-UT state to z0 with RK6 (0.1 mm), marches it through the 129 planes z0 + k·L/128, carries it on to the particle's own SciFi plane, and records the real state there | `results/crossing_particles.npz`, `results/crossing_particles_meta.json`, `figures/crossing_geometry.png` |

```bash
PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python build_dataset.py --workers 12   # ~5 min
```

## What is in the file

Per split (`train`, `val`, `test`; the training set's own by-particle split,
training capped at 2,000 particles by a seeded permutation, val and test not
capped):

| array | shape | what |
|---|---|---|
| `{split}_S0` | (n, 5) | the start state on z0: (x, y, tx, ty, q/p), mm, rad, Allen q/p |
| `{split}_truth` | (n, 129, 5) | the RK6 state on every plane z0 + k·L/128; chain N uses every 128/N-th |
| `{split}_truth_zpost` | (n, 5) | the same trajectory carried from z1 to the particle's SciFi plane |
| `{split}_S_post`, `{split}_z_post` | (n, 5), (n,) | the particle's REAL state on its first SciFi plane, and that plane |
| `{split}_S_pre`, `{split}_z_pre` | | the real last-UT state and plane the row came from |
| `{split}_P`, `_ETA`, `_PID`, `_EVT`, `_MCKEY`, `_PBAND` | (n,) | momentum, pseudorapidity, PDG id, event and particle keys, momentum band |
| `planes` | (129,) | the plane grid; `z0`, `z1`, `L`, `N_values`, `field`, `rk6_step_mm` scalars |

## The cut cascade (2026-09-14)

| cut | rows out |
|---|---|
| leg-B rows, both directions | 41,398 |
| pre-magnet plane is a UT plane | 38,922 |
| both directions kept | 36,534 |
| 2 < η < 5 | 35,294 |
| 1 < p < 200 GeV | 35,294 |
| non-electron | 30,514 (15,257 particles) |
| forward rows only | 15,257 |
| both planes within 60 mm of z0, z1 | 14,482 |
| fiducial (RK6 path inside the map) | 14,482 |

Splits: train 11,567 → capped 2,000; val 1,463; test 1,452. Pre-magnet planes
2593–2663 mm; post-magnet planes 7819–7833 mm.

## The material floor

The real first-SciFi state differs from the field-only RK6 truth carried to the
same plane by the material the particle crossed. No field-only method — RK6, the
exact scheme, any network trained on the equation of motion — can do better
than this against the real state:

| momentum | median [µm] | p95 [µm] | slope median [mrad] | n |
|---|---|---|---|---|
| all | 1,736 | 12,628 | 0.49 | 14,482 |
| 1–2 GeV | 14,723 | 35,183 | 6.44 | 89 |
| 2–5 GeV | 5,136 | 18,294 | 1.74 | 5,117 |
| 5–10 GeV | 1,608 | 5,791 | 0.46 | 4,286 |
| 10–25 GeV | 622 | 3,223 | 0.17 | 3,697 |
| 25–200 GeV | 251 | 1,938 | 0.07 | 1,293 |

Cost: 283 s of RK6 on 12 workers for the 14,482 crossings.
