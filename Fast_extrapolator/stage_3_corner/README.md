# Stage 3: the corner of the magnet crossing

Started 2026-11-27 (George: "48-step rule + corner training"). Nothing outside this folder, `allen_nn/`
and the store folder `/data/bfys/gscriven/fast_extrapolator_store/stage_3_corner` is written.

## Why

Without the ARKN fallback, the crossing (12 Nystrom steps + network, `allen_nn`) fails for tracks whose
straight line from the state on z0 = 2648.2 mm reaches |y| > ~700 mm at z = 5000 mm: the field near the
upper pole (z 3.9-4.8 m, |B| up to 2.4 T, Bz up to 1.6 T) is too steep for 12 fixed steps
(`allen_nn/results/tail_diagnosis.txt`). On the validation tracks the tool has median 1.42 um but 99.9 %
7.1 mm and largest 27 mm. Re-placing the 12 planes gains only ~2x (`allen_nn/results/step_layouts.txt`).
The interim rule (48 plain Nystrom steps above 700 mm) gives 99.9 % 503 um, largest 846 um.

## Runs (one seed each; differences under 2x are not claimed)

| run | data | loss | stop and best round on |
|---|---|---|---|
| `corner_12_2x64`, `corner_12_2x32` | corner tracks only (|y| at 5 m > 600 mm): 9,791 real + 90,000 new | Huber of stage 2 | median on the 1,277 corner validation tracks |
| `tailw_12_2x64`, `tailw_12_2x32` | all 108,915 training tracks | Huber, weight 1 -> 20 between 500 and 900 mm of |y| at 5 m | 99 % on all validation tracks |

The corner networks are meant to replace the 48 plain steps of the rule above 700 mm (trained from
600 mm for a margin). The tail-weighted networks are meant to replace the main network everywhere.
The test tracks are scored once at the end.

New corner tracks (`python/make_corner_data.py`): a real corner training track picked at random, each
start component changed by a Gaussian of 0.05 times its spread in the corner set, kept if still in the
corner and inside the box of the training tracks; the reference is RK4 with 4,000 steps in double on the
same map and planes (equal to the track-set reference to 0.001 um median, 0.22 um at most).

## Files

| file | what |
|---|---|
| `python/make_corner_data.py` | the data (once, on a login node) -> `<store>/data/` |
| `python/train_corner.py` | the trainer (fastex.train imported, unchanged) and `--export` to the JSON of allen_nn |
| `configurations/*.yaml` | the four runs |
| `farm/jobs.sub`, `farm/wrapper.sh` | 4 cores, 8 GB, a call stops after 22 h; a resubmitted job goes on |
