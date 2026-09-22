# G0 — the loss with the momentum window moved, and the gates that justify it

[windowed_loss.py](windowed_loss.py) is the only new code in this phase. It is the reweighted
loss from
[`../../Block_F_reweighted_loss/F0_Weighting/weighted_loss.py`](../../Block_F_reweighted_loss/F0_Weighting/weighted_loss.py)
with **one** thing changed: the momentum window `W(p)` is read from the run's own constants
instead of from the module's default constants. Everything else — the lever arm, the
track-bend normalisation, the roll-off (Gaussian in log p, width ln 2), the floor (0.05), the
clamp, the modes, the loss itself — is imported from that file unchanged and never modified.
With `p_lo = 10`, `p_hi = 50` every function here returns bit for bit what the imported module
returns.

## The trap

The window lives in two places at once: the module constants `P_LO = 10.0`, `P_HI = 50.0`,
which are the *default arguments* of `band_window`, and the run's constants dict
(`const["p_lo"]`, `const["p_hi"]`, …), which is what gets written into `scale.json["weighting"]`
and so is what a finished run claims it trained with. `per_track_factor` called
`band_window(p)` with no arguments, so the torch training path always used the module
constants and never looked at the dict; the numpy twin in the gate script reads the dict.

Nothing that has already run is affected — the two agreed, because the dict was filled from the
same constants. But the whole point of this phase is to move the window, and on that path the
run folder would have said one window while the optimiser minimised another, with nothing in
the training output to show it. Three functions are therefore re-implemented here
(`reference_constants`, which now takes `p_lo`/`p_hi`/`rolloff`/`w_floor` as arguments;
`per_track_factor`, which passes them to `band_window`; and `weights`, only so that it reaches
this `per_track_factor`). The read-only module is left exactly as it is.

[check_windowed_weights.py](check_windowed_weights.py) runs four gates.
`results/check_windowed_weights.json` holds every number, `results/preflight_windows.csv` the
per-band table, `figures/window_preflight.png` draws the last gate.

```bash
PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python check_windowed_weights.py
```

Runs in about 9 s single-threaded. It writes nothing outside this folder.

## The gates

| gate | what it proves | result |
|---|---|---|
| W1 | for N = 2 and 256, q = 2 and 16, 2,000 states, every mode and **every candidate window**: these torch weights equal an independent numpy implementation | worst 1.5e-15 |
| W1 (the trap, shown) | the read-only module's `weights`, handed the same moved window, **disagrees** with the same numpy implementation | 3.07 … 3.47 relative (order unity) for every moved window; 1.0e-15 for 10–50 GeV, which is its own |
| W1 (lever arm) | the lever arm runs from one step (last plane) to L + dz(1 − c₁) (first plane) | exact |
| W2 | with the unweighted mode the weighted path reproduces the shared loss | identical to the last bit (relative difference 0) |
| W3 | the reweighting does not move the minimum: the exactly solved collocation states give machine zero under **every** mode with the 3–8 GeV window | 6e-30 … 2e-29 of the straight line |
| W4 | the pre-flight: where the loss would come from under each candidate window | see below |

## The pre-flight (W4)

The trained network **N = 64, q = 2, dz = 80.9 mm** (the finished run, 40 rounds, 1,000
restarts), 8,000 (track, plane) states drawn evenly over the 64 start planes, seed 20260918.
"Cost" is what a state's error actually incurs at the first SciFi plane — its single-step error
against a 6th-order reference taken from the same state, position plus slope times the distance
left. That reference is a diagnostic only; it never enters the loss, which stays label-free.

### Read the trimmed column, not the raw one

The residual distribution of this trained network has a very heavy tail. Under the 10–50 GeV
window **a single state out of 8,000 carries 68.8 % of the whole objective**, and under the
lower windows one state still carries 41 %. The raw shares therefore describe where that one
state happens to sit, not what the window does: raw, all six windows look identical (84.7 –
89.7 % of the loss inside 3–8 GeV). Every quantity below is therefore given twice — over all
8,000 states, and with the most extreme 1 % of states (80 of them) removed and the rest
renormalised. **The trimmed column is the one that separates the windows.** The tail itself is
a property of the residual, not of the weighting, and no choice of window removes it.

### Summary, one row per candidate window

Share of the loss, raw / with the most extreme 1 % of states removed:

| loss window | states in band | in its own band | in 3–8 GeV | > 50 GeV | > 100 GeV | top 1 % of states | largest single state | ρ(share, cost) all | ρ in its own band | ρ in 3–8 GeV |
|---|---|---|---|---|---|---|---|---|---|---|
| **3–8 GeV** | 45.9% | 85.4 / **61.0%** | 85.4 / **61.0%** | 0.05 / 0.85% | 0.05 / 0.78% | 93.8% | 41.1% | +0.949 | +0.942 | **+0.942** |
| 3–10 GeV | 53.9% | 85.7 / 63.5% | 85.1 / 59.5% | 0.06 / 0.89% | 0.05 / 0.83% | 93.6% | 41.0% | +0.937 | +0.932 | +0.942 |
| 3–20 GeV | 74.9% | 86.7 / 68.8% | 84.9 / 57.1% | 0.06 / 0.87% | 0.05 / 0.79% | 93.4% | 40.8% | +0.887 | +0.898 | +0.942 |
| 2–15 GeV | 78.1% | 99.5 / 92.6% | 84.7 / 56.5% | 0.06 / 0.84% | 0.05 / 0.79% | 93.3% | 40.8% | +0.911 | +0.921 | +0.942 |
| 4–6 GeV (narrow reference) | 19.4% | 21.9 / 25.2% | 88.2 / **66.6%** | 0.04 / 0.77% | 0.04 / 0.69% | 94.4% | 42.3% | +0.952 | +0.963 | +0.941 |
| 10–50 GeV (the window used so far) | 32.5% | 3.6 / 20.8% | 89.7 / **48.7%** | 0.13 / 0.99% | 0.09 / 0.39% | 94.4% | 68.8% | +0.620 | +0.798 | +0.757 |

Four of the six windows — 3–8, 3–10, 3–20 and 2–15 GeV — take more than half of the loss
inside their own band, raw and trimmed. The 4–6 GeV window does not (25.2 % trimmed): it is
too narrow to hold the weight it creates. All six take more than half of the loss inside 3–8
GeV raw; trimmed, five of the six do, and the 10–50 GeV window does not (48.7 %).

### Share of the loss by momentum band

Raw, then with the most extreme 1 % of states removed (in brackets). "states" is the share of
the 8,000 drawn states that fall in the band.

| momentum band | states | 3–8 GeV | 3–10 GeV | 3–20 GeV | 2–15 GeV | 4–6 GeV | 10–50 GeV |
|---|---|---|---|---|---|---|---|
| 1–2 GeV | 0.8% | 0.1 (1.4) | 0.1 (1.4) | 0.1 (1.4) | 0.1 (2.1) | 0.1 (1.0) | 0.1 (2.5) |
| 2–3 GeV | 10.9% | 13.0 (27.2) | 13.0 (27.2) | 13.0 (26.1) | 13.1 (27.4) | 10.8 (24.3) | 5.1 (18.8) |
| 3–5 GeV | 24.8% | 41.3 (41.6) | 41.2 (40.6) | 41.0 (39.0) | 41.0 (38.5) | 42.9 (45.3) | 16.4 (24.1) |
| 5–8 GeV | 21.1% | 44.1 (19.4) | 44.0 (18.9) | 43.8 (18.2) | 43.8 (17.9) | 45.3 (21.3) | 73.3 (24.5) |
| 8–10 GeV | 8.0% | 0.5 (4.0) | 0.5 (4.1) | 0.5 (3.9) | 0.5 (3.9) | 0.4 (3.4) | 1.3 (8.3) |
| 10–20 GeV | 21.0% | 0.9 (5.2) | 1.1 (6.2) | 1.3 (7.7) | 1.3 (7.5) | 0.5 (3.6) | 3.1 (14.1) |
| 20–50 GeV | 11.5% | 0.0 (0.4) | 0.0 (0.7) | 0.2 (2.8) | 0.1 (1.8) | 0.0 (0.3) | 0.5 (6.7) |
| 50–100 GeV | 1.7% | 0.0 (0.1) | 0.0 (0.1) | 0.0 (0.1) | 0.0 (0.1) | 0.0 (0.1) | 0.0 (0.6) |
| 100–200 GeV | 0.3% | 0.0 (0.8) | 0.1 (0.8) | 0.1 (0.8) | 0.1 (0.8) | 0.0 (0.7) | 0.1 (0.4) |

### Share of the loss along z, by quarter of the crossing

| loss window | first | second | third | fourth |
|---|---|---|---|---|
| 3–8 GeV | 5.9 (35.0) | 75.3 (25.9) | 17.2 (23.9) | 1.7 (15.3) |
| 3–10 GeV | 6.0 (35.1) | 75.1 (25.3) | 17.2 (24.5) | 1.7 (15.1) |
| 3–20 GeV | 6.2 (35.1) | 75.0 (25.8) | 17.1 (24.3) | 1.7 (14.8) |
| 2–15 GeV | 6.2 (35.3) | 74.9 (25.6) | 17.2 (24.3) | 1.7 (14.9) |
| 4–6 GeV | 5.4 (33.1) | 76.1 (26.7) | 17.0 (25.6) | 1.5 (14.6) |
| 10–50 GeV | 6.6 (33.1) | 85.1 (29.4) | 7.0 (24.1) | 1.3 (13.4) |

Raw, the second quarter dominates because the one extreme state starts there. Trimmed, all six
windows put about a third of the loss on the first quarter — where an error costs the most,
because it is multiplied by the longest remaining lever arm — and about 15 % on the last. The
window does not change that; the lever-arm factor does, and it is unchanged.

### The clamp, scanned for the 3–8 GeV window (report only; the clamp stays at 5)

| clamp | in 3–8 GeV | in 10–50 GeV | > 50 GeV | > 100 GeV | top 1 % | ρ(share, cost) | tracks clipped (low / high) |
|---|---|---|---|---|---|---|---|
| off | 85.4 (60.9) | 0.9 (5.5) | 0.07% | 0.06% | 93.8% | +0.949 | — |
| 2 | 81.7 (51.8) | 0.8 (4.5) | 0.01% | 0.01% | 92.7% | +0.958 | 15.1% / 1.4% |
| 3 | 85.3 (59.7) | 0.9 (5.4) | 0.02% | 0.02% | 93.6% | +0.951 | 2.4% / 0.3% |
| **5** | **85.4 (61.0)** | **0.9 (5.6)** | 0.05% | 0.05% | 93.8% | +0.949 | 0.1% / 0.0% |
| 10 | 85.4 (60.9) | 0.9 (5.5) | 0.07% | 0.06% | 93.8% | +0.949 | 0.0% / 0.0% |

The clamp was the design knob for the 10–50 GeV window, where it held the high-momentum tail
down. For a low window it is **inert**: at 5 it clips 0.1 % of tracks from below and none from
above, and turning it off changes nothing to three figures. That is the window and the
track-bend factor pulling against each other — `a_n ∝ sqrt(W(p)) · p`, so a low window shrinks
exactly the high-momentum factors the clamp used to catch. The only setting that does anything
is 2, and it moves weight *out* of 3–8 GeV (51.8 % against 61.0 % trimmed), which is the wrong
direction. 5 stays.

### The window itself, at named momenta

`W(p)` for the 3–8 GeV setting: 0.081 at 1 GeV, 0.710 at 2, 1.000 at 3, 5 and 8, 0.902 at 10,
0.174 at 20, and the floor 0.05 from 50 GeV up.

## What the pre-flight says

1. **The window is a real lever, but a modest one.** The 10–50 GeV setting already puts 48.7 %
   (trimmed) of the loss into 3–8 GeV, simply because that is where the residual is largest.
   Moving the window to 3–8 GeV buys +12 points, to 61.0 %. It is not a transformation of the
   objective.
2. **The four low windows are nearly interchangeable** on this measure: 61.0, 59.5, 57.1 and
   56.5 % trimmed in 3–8 GeV, in the order 3–8, 3–10, 3–20, 2–15. They differ more in what they
   leave on 10–50 GeV: 5.2, 6.2, 7.7 and 7.5 % trimmed, against 14.1 % for the window used so
   far. Widening the window is how the cost in the physics band is bought back.
3. **Inside 3–8 GeV every low window ranks states by their true endpoint cost equally well**
   (ρ = +0.942, against +0.757 for 10–50 GeV). The choice among them is about where the weight
   sits, not about how well it aims.
4. **The 4–6 GeV reference is the most concentrated** (66.6 % trimmed in 3–8 GeV) but holds
   only a quarter of the loss inside 4–6 GeV itself and halves the weight on 10–20 GeV. It buys
   its concentration mostly below 5 GeV (3–5 GeV takes 45.3 % against 41.6 %).
5. **The momentum tail is a non-issue here.** Above 50 GeV no low window takes as much as 1 %,
   trimmed or raw, and the clamp is doing no work.

Recommendation: **3–8 GeV**, the anchor. Caveats, the pre-flight's limits and the reading of
the pre-registration are in the worklog,
[`../worklog/2026-09-21_G0_weighting.md`](../worklog/2026-09-21_G0_weighting.md).
