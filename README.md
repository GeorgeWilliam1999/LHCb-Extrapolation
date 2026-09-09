# LHCb Extrapolation

Applying the physics-informed-network programme (Raissi et al. 2019, validated
on the van der Pol study) to LHCb track extrapolation — starting from the data:
generate full simulated events with the official LHCb software, run the real
reconstruction on them, and build characterised, event-derived training samples.

Canonical tracker: the **LHCb Extrpolator** project in Notion (guide at the top
of the project page; every figure documented in "The data, figure by figure").

| Folder | What it is |
|---|---|
| `Data_generation_exploration/First_Pass/` | Generate + understand simulated events locally (Gauss v61r0p2, 2024 conditions). Truth dump + event displays. |
| `Data_generation_exploration/First_Pass/First_reco/` | The reconstruction pipeline on our own events: Boole digitisation, Moore HLT2 + HLT1 (Allen), root-caused gotchas. |
| `Data_generation_exploration/Data/` | The training sample: event-derived population x gated RK4/v8r1 labels, characterisation figures, integrity gates. |
| `Raissi_disc_time_approach/` | The discrete-time technique port (gated: starts after the data-exploration phase). Three blocks, read in order — see its own [README](Raissi_disc_time_approach/README.md). |
| `Raissi_disc_time_approach/_shared/` | The machinery all three blocks import: equation of motion, tableau, field loaders, model, both losses, trainer, scorer, smoke tests, farm harness. |
| `Raissi_disc_time_approach/Block_0_first_pass/` | July 2026 — does the scheme port to LHCb at all? Data look, the exact scheme with no network, the first one-step network, and the official-sample baseline everything else is gated against. |
| `Raissi_disc_time_approach/Block_A_technique_works/` | 5–6 Sep 2026 — does the technique work, and what limits it? Stage count, network size and seed, one network for all legs, chaining, and training where no labels exist. |
| `Raissi_disc_time_approach/Block_C_step_size_and_stages/` | 7–8 Sep 2026 — the supervisors' table: error against step length and stage count over 720 trained networks, against a sixth-order reference and against real hits, plus the chained crossings. |

Bulky artifacts (.sim/.digi/.root, truth CSVs, training .npz) are gitignored —
every one is deterministically regenerable from the scripts, seeds and
provenance metadata committed here (see each folder's README).
