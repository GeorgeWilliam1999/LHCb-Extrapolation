# Block E — one network per step length, applied to itself across the magnet

**Question.** For a fixed step dz = L/N with N in {2, 64, 128, 256} and q in
{2, 4, 8, 16} Gauss–Legendre stages: how accurately does **one** network, applied N
times, carry a track from the last UT plane to the first SciFi plane? Where along the
crossing does its error arise? How does it compare with the exact collocation scheme
chained the same way, with the straight line, and with Block D's one-network-per-step
chains?

**Status (2026-09-20): all 16 networks trained.** They were analysed in E3 on the checkpoints
of 2026-09-18 (kept as `E1_Network_grid/results/N*/stopped_2026-09-18/`), and then extended
until the validation error stopped falling under the plateau rule of
`../Block_F_reweighted_loss/F2_Analysis/compare_to_blockE.py`; 9 of 16 have reached it, the
other 7 are still improving or on the farm. The E3 tables describe the 2026-09-18 checkpoints;
`../Block_F_reweighted_loss/F2_Analysis/results/headline.csv` tracks the extended runs. The
full plan, its open decisions and the worklog are in the Notion to-do "Train one network per
step length and chain it to itself across the magnet".

Read in order:

| folder | role |
|---|---|
| [E0_Track_dataset/](E0_Track_dataset/) | RK6 tracks across the crossing on 257 planes, the splits, the real SciFi states |
| [E1_Network_grid/](E1_Network_grid/) | the network, the round trainer, the 16 farm jobs, scoring each network |
| [E2_Comparators/](E2_Comparators/) | the exact scheme chained at N = 2 and 256, the straight line, Block D's chains |
| [E3_Analysis/](E3_Analysis/) | the tables and figures across all 16 networks, and the notebook that loads them |
