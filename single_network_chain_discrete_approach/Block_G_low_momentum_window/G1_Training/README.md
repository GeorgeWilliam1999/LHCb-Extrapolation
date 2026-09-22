# G1 — training with the momentum window moved

[train_windowed.py](train_windowed.py) is
`../../Block_F_reweighted_loss/F1_Training/train_weighted.py` with the loss import swapped for
[`../G0_Weighting/windowed_loss.py`](../G0_Weighting/windowed_loss.py) and `--p-lo` / `--p-hi`
added, and nothing else touched.
[results/train_windowed.diff](results/train_windowed.diff) is the diff against it, so the claim
is checkable rather than asserted: four code hunks, the rest of the diff is the docstring.

| the change | what it is |
|---|---|
| the loss import | `windowed_loss` in place of `weighted_loss`: the same weights, with the window read from the run's own constants instead of the module defaults |
| `--p-lo`, `--p-hi` | floats, GeV, defaults 10 and 50. They go into `reference_constants`, so `scale.json["weighting"]["p_lo"]` / `["p_hi"]` record the window the run was created with, and `record.json` carries it in `weighting_constants` |
| the start-up guard | a run folder whose `scale.json` window differs from the command line exits, the way a differing `--weighting` already did. A run cannot change objective half way through |

The network, the metrics, the tracks and every part of the loss except the window are
**imported**, not copied, so they cannot drift.

Runs land in `<out>/<weighting>/N<NNN>_q<qq>/`. Put the window in `--out`
(`--out results/p03-08`, zero-padded on each side) so two windows can never share a run
folder, and so the queue can tell them apart.

## The isolation gates

Two gates say the only thing that moved is the window, and a third says the window moved.
All three were run on 2026-09-21 into the session scratch area; the numbers are in
[../worklog/2026-09-21_G1_trainer_harness.md](../worklog/2026-09-21_G1_trainer_harness.md).

### Gate I1 — the unweighted training is still bit-identical

The recipe of `../../Block_F_reweighted_loss/F1_Training/README.md`, with this trainer in
place of that one. `--weighting blockE` must reproduce
`../../Block_E_single_network_chain/E1_Network_grid/train_network.py` to the last digit —
same seed, same states, same optimiser:

```bash
PY=/data/bfys/gscriven/conda/envs/TE/bin/python; export PYTHONNOUSERSITE=1
SCR=$(mktemp -d)
(cd ../../Block_E_single_network_chain/E1_Network_grid \
   && $PY train_network.py  --N 64 --q 2 --n-train 300 --n-eval 200 --stop-after 3 --out $SCR/a)
$PY train_windowed.py --N 64 --q 2 --weighting blockE --n-train 300 --n-eval 200 --stop-after 3 --out $SCR/b
diff <(cut -d, -f5,6 $SCR/a/N064_q02/history.csv) \
     <(cut -d, -f5,6 $SCR/b/blockE/N064_q02/history.csv) && echo "gate I1 passes"
```

### Gate I2 — the default window still reproduces the 10–50 GeV runs

The window defaults to 10–50 GeV, so with no `--p-lo` / `--p-hi` this trainer must reproduce
the finished runs under `../../Block_F_reweighted_loss/F1_Training/results/full/` bit for bit:
`scale.json` field for field (the input spreads and every constant of the weighting) and the
first restarts of `history.csv` (`loss_before`, `loss_after`, `n_iter`, `func_evals`) to the
last digit.

```bash
$PY train_windowed.py --N 64 --q 2 --p-lo 10 --p-hi 50 --stop-after 2 --out $SCR/i2
# then compare $SCR/i2/full/N064_q02 with
# ../../Block_F_reweighted_loss/F1_Training/results/full/N064_q02
```

About 80 s a restart at N = 64, q = 2 and about 230 s at N = 256, q = 16, so the gate is a few
minutes. Nothing is written under the reference run: the gate only reads it.

### Gate I3 — the moved window changes the objective

The same three restarts at `--p-lo 3 --p-hi 8` and at the default window, on the same 300
tracks. The round-1 losses must differ (they weight different tracks) and `scale.json` must
record `p_lo = 3`, `p_hi = 8`. The guard is checked at the same time: rerunning a 3–8 GeV run
folder with the default window exits instead of training on.

## The runs

```bash
$PY condor/make_jobs.py --p-lo 3 --p-hi 8      # the three settings, window in the path
condor_submit condor/jobs.sub                  # from this folder
cp condor/jobs.txt condor/jobs_active.txt      # what the keeper keeps alive
tmux new-window -n blockG-keeper "$PWD/condor/keeper.sh"
```

| run | dz | measured restart | expected wall to the 1,000-restart cap |
|---|---|---|---|
| N = 64, q = 2 | 80.9 mm | 78 s | ≈ 22 h |
| N = 128, q = 8 | 40.5 mm | 118 s | ≈ 33 h |
| N = 256, q = 16 | 20.2 mm | 182 s | ≈ 50 h |

The per-restart times are the medians of the finished 10–50 GeV runs on the same farm. The
N = 256, q = 16 figure is the median over its first 1,198 restarts, before a second process
started sharing the node; after that it reads 233 s, which is the contention and not the work.

Caps of 40 rounds and 1,000 restarts, 25 restarts a round — the same budget the 10–50 GeV runs
had. Jobs are resumable after every restart. `condor/keeper.sh` releases the farm's wall-time
and memory holds every half hour, drops a run out of `condor/jobs_active.txt` once
`condor/prune_active.py` says its validation error has stopped falling, and sends back anything
that left the queue unfinished.

`condor/prune_active.py` **imports** `plateaued_now` from
`../../Block_F_reweighted_loss/F2_Analysis/compare_to_blockE.py` rather than restating the
rule, so both sets of runs are judged by one function: the median validation error of the last
ten rounds no more than 5 % below the median of the ten before, three rounds running. The
trainer's own 1 % rule is not asked for as well — at N ≥ 64 with the rescaled loss it never
fires.

### The two things `condor/resubmit.py` does that its predecessor did not

On 2026-09-21 a keeper submitted a second copy of a job that was already running, and the two
processes have been writing one checkpoint since. The cause was a `condor_q` that failed and
was read as an empty queue. So:

1. **A queue that cannot be read is unknown, not empty.** A non-zero exit, an exception, or an
   error on stderr makes `queued()` return `None`, and on `None` **nothing is submitted**. The
   next pass, half an hour later, tries again.
2. **A warm `progress.json` means a live process.** A run whose `progress.json` was written in
   the last ten minutes is not resubmitted even if the queue does not show it. The trainer
   writes that file after every restart (78–182 s), so anything younger than ten minutes has a
   process in it.

The queue match is (wrapper, N, q, weighting, window) — the window is in it because two windows
are two different runs of the same (N, q) in two different folders, and the wrapper is in it
because the other blocks run the same (N, q) through their own.

```bash
$PY condor/resubmit.py --self-test   # the two rules, on a fake condor_q; writes nothing real
$PY condor/resubmit.py               # list only; --submit sends them
```
