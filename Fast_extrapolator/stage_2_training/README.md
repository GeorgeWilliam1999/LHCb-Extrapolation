# Stage 2: training

Status (2026-11-24): George said to continue to stage 2. The trainer, the 20
configurations of the first batch and the farm files are built and gated.
Nothing is submitted: the submission waits for his word. The design choices are
those of `DESIGN.md`.

| Part | Where |
|---|---|
| The trainer | `../python/fastex/train.py` |
| The configurations of the first batch | `scripts/first_batch.py` writes `configurations/*.yaml` |
| The farm files | `farm/` |
| The gates | `../python/tests/test_train.py`; it needs the two short runs `gate_hybrid` and `gate_none` (2 rounds of 40 iterations, width 32) in the store |

The preparation trained no network: it measures what a correction must learn
(`PREPARATION.md`).

| Script | Output |
|---|---|
| `../python/fastex/export.py splits` | every training and validation track of both track sets, in the store of this experiment |
| `scripts/what_must_a_correction_learn.py` | `results/what_a_correction_must_learn.csv` (the size of each correction), `results/polynomial_probe.csv` (what a polynomial gives of it) |
| `scripts/higher_degrees_for_the_straight_line.py` | `results/polynomial_probe_straight_line.csv` |
| `scripts/plots.py` | `figures/what_a_smooth_correction_reaches.png` |
| `../python/fastex/correction.py` | the polynomial correction: its fit, its file for the compiled program, its numpy version |
| `scripts/polynomial_baseline.py fit, gate, accuracy, time CORE, table` | the corrections in the store; `results/polynomial_baseline_fit.csv`, `_gate.csv`, `_accuracy.csv`, `_timing.csv` and `results/polynomial_baseline.csv` (test tracks, compiled) |
| `scripts/write_preparation_note.py` | `PREPARATION.md` |

The polynomial is a probe of the target and a baseline for the networks. It is
fitted on the training tracks and checked on the validation tracks. The test
tracks are used once, for the scores of the baseline in the compiled program
(`polynomial_baseline.py accuracy`); no choice is made on them.
