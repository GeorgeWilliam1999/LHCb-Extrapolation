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
| `Raissi_disc_time_approach/` | The discrete-time technique port (gated: starts after the data-exploration phase). |

Bulky artifacts (.sim/.digi/.root, truth CSVs, training .npz) are gitignored —
every one is deterministically regenerable from the scripts, seeds and
provenance metadata committed here (see each folder's README).
