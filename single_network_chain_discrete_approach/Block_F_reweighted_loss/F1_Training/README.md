# F1 — training with the reweighted loss

[train_weighted.py](train_weighted.py) is
`../../Block_E_single_network_chain/E1_Network_grid/train_network.py` with the loss swapped
and nothing else touched. [results/train_weighted.diff](results/train_weighted.diff) is the
diff against it, so the claim is checkable rather than asserted. The network, the metrics
and the tracks are **imported from Block E**, not copied, so they cannot drift.

`--weighting` picks the mode: `full` (the default), the ablations `no_lever`, `no_track`,
`no_window`, or `blockE` for Block E's own weights. Runs land in
`results/<weighting>/N<NNN>_q<qq>/`, so an ablation can never overwrite the real run.

## Gate F-1: the isolation gate

With `--weighting blockE` the trainer must reproduce Block E's training **bit for bit** —
same seed, same states, same optimiser, so the same losses to the last digit:

```bash
PY=/data/bfys/gscriven/conda/envs/TE/bin/python; export PYTHONNOUSERSITE=1
SCR=$(mktemp -d)
(cd ../../Block_E_single_network_chain/E1_Network_grid \
   && $PY train_network.py  --N 64 --q 2 --n-train 300 --n-eval 200 --stop-after 3 --out $SCR/blockE)
$PY train_weighted.py --N 64 --q 2 --weighting blockE --n-train 300 --n-eval 200 --stop-after 3 --out $SCR/blockF
diff <(cut -d, -f5,6 $SCR/blockE/N064_q02/history.csv) \
     <(cut -d, -f5,6 $SCR/blockF/blockE/N064_q02/history.csv) && echo "gate F-1 passes"
```

`blockE` mode calls the shared `physics_loss` itself rather than going through the weighted
path, so the reproduction is exact by construction; F0's gate 2 is what proves the weighted
path agrees with it numerically.

## The runs

```bash
$PY make_jobs.py                 # the three `full` runs, slowest first
condor_submit condor/jobs.sub
cp condor/jobs.txt condor/jobs_active.txt     # what the keeper keeps alive
tmux new-window -n blockF-keeper "$PWD/condor/keeper.sh"
```

| run | Block E's counterpart | expected wall |
|---|---|---|
| N = 64, q = 2 | 124 µm ± 13% | ≈ 20 h |
| N = 128, q = 8 | 162 µm ± 9% | ≈ 22 h |
| N = 256, q = 16 | 173 µm ± 18% | ≈ 33 h |

Each was given caps of 40 rounds and 1,000 restarts (the same number, at 25 restarts a round);
the keeper stops resubmitting a run once the plateau rule of `../F2_Analysis/compare_to_blockE.py`
holds. In the event N = 64, q = 2 and N = 128, q = 8 ran to the cap (`hit_cap = restarts` in their
records, 2026-09-19 and 2026-09-20) and the rule held at that point (last ten rounds +2.8% and
+0.5% against the ten before), so the cap ended them and the rule agreed. Jobs are resumable after every restart;
`resubmit.py` sends back anything that leaves the queue unfinished, and `condor/keeper.sh`
releases the farm's wall-time and memory holds every half hour.
