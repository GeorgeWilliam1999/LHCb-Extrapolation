# Plan

The full plan is the document "Plan: a fast extrapolator at one-micrometre
accuracy": https://claude.ai/artifact/3yHeRnE6bNHWSCnmNcysct

## In short

- Target A: median radial error <= 1 um, 95 % <= 10 um, every momentum band <= 3 um.
  Target B (fallback): 5 um / 40 um / 10 um, equal to the HLT2 integrator.
- Speed: compiled, single precision, one track per call, not slower than the
  LHCb reference in the same program. The reference is the adaptive
  Runge-Kutta-Nystrom integrator (measured in stage 0: 56 field reads, 3.6 us a
  track, median error 10 um on our test tracks); the adaptive Cash-Karp integrator
  (43 reads, 1.8 us, 22 um) is the second reference.
- Levers on every family: steps 1-16, stages 2-8, field reads 0 to about 40,
  widths 16-64, compiled single precision.
- Families: 0 classical; 1 whole-crossing; 2 supervised path; 3 label-free path;
  4 chained, direct states; 5 chained, straight line; 6 physics layer; 7 gain
  variants; 8 cheap integrator plus network correction (new, needs approval).
- Stages: 0 harness and baselines; 1 no-training frontier; 2 training campaign
  (about 87 farm runs); 3 fast code; 4 verdict.

## Log

- 2026-11-24: plan written. Awaiting decisions 1-9 of the document. No farm job,
  no package edit.
- 2026-11-24, 12:44 UTC: George agreed decisions 1 to 7 in the thread: (1) accuracy
  target A, with B as fallback; (2) the task is the fixed UT-to-SciFi crossing, like
  for like with the HLT1 parametrised step; (3) cubic map for training and scoring,
  one linear-map run for each best configuration; (4) family 8 and the field inputs
  of family 1 are in; (5) extra training tracks from the reference integrator, for
  family 1 only; (6) this folder and its own run store; (7) the farm campaign of
  about 87 runs, each batch submitted after his word. His point 8 reads "I will
  find it": taken as the answer to decision 9 (which HLT1 sequence ran in data
  taking). Decision 8 (a Python environment for ONNX Runtime) is still open; it is
  needed at stage 3 only. Stage 0 started.
- 2026-11-24, 14:02 UTC: answer of Maarten Van Veghel (LHCb), passed on by George:
  the reference for HLT1 and HLT2 is an adaptive Runge-Kutta-Nystrom integrator,
  not Cash-Karp. The code agrees: TrackSTEPExtrapolator in Rec and
  AdaptiveRKNystromExtrapolator in Allen, checked against each other in a Moore
  test. It is the main LHCb reference from here on. Cash-Karp stays in the tables,
  because the Kalman fit of Rec v41r3 calls it by default. The talk that Maarten
  linked (Indico event 1621662) needs a CERN sign-in; George is asked for the PDF
  (it goes to lhcb_software/references/).
- 2026-11-24, 14:50 UTC: stage 0 is done and reported in the thread
  (stage_0_baselines/RESULTS.md, 80 methods timed on stbc-i3). In short:
  the compiled program gives the results of Python for 24 of 24 methods; the
  copies of both HLT2 integrators print the 1,331 states of the real tools and
  are within 13 % of their time; on our 13,447 test tracks (v8r1, magnet up,
  trilinear) the adaptive Nystrom integrator takes 55.7 reads and 3.6 us for a
  median error of 10.1 um, Cash-Karp 42.6 reads and 1.8 us for 22.2 um, the HLT1
  parametrised step no read and 0.38 us for 55.6 um (1.2 mm below 3 GeV); one LHCb
  field read costs 11 to 18 ns, one network call 135 ns (2 x 32) to 3.7 us
  (2 x 256); RK4 with 32 equal steps reaches 0.30 um in 2.3 us; every present
  network is slower than LHCb. So the measured fallback target B (5 / 40 / 10 um)
  is tighter than the LHCb reference in the median, not equal to it.
  Corrections of earlier statements: the planes of the HLT1 parametrised step are
  2,665 mm and 7,826 mm; the T-track integrator of Allen reads the field once a
  step (11 reads, about 1 mm), it is not a 10 um method; "LHCb reaches 4 to 10 um"
  held for the flat momentum sample only.
  Open: the PDF of the talk; a single-core timing job on the farm (needs George's
  word); the Allen copies are not compared with a run of Allen.
- 2026-11-24: which Rec tool runs where, read from the configuration of Rec v41r3
  (Rec/RecoConf): the HLT2 track fit takes TrackMasterExtrapolator, whose tool for
  long distances is TrackRungeKuttaExtrapolator (Cash-Karp) by default
  (TrackDistanceExtraSelector); the RICH reconstruction takes TrackSTEPExtrapolator
  (the adaptive Runge-Kutta-Nystrom). So both run in HLT2.
- 2026-11-24, 15:55 UTC: stage 1 is done and reported in the thread
  (stage_1_no_training/RESULTS.md; 197 methods timed on stbc-i3, 36 layouts). In short,
  on the linear map and the planes of this experiment: RK4 with 19 equal steps is
  faster and more accurate than both LHCb integrators (1.35 us; 0.85 / 14.5 / 4.8 um
  against 3.6 us and 10.1 / 38.8 / 16.2 um); 31 to 33 and 36 steps meet target A
  (2.2 us). The error on the linear map depends on where the steps fall on the grid
  of the map (kinks every 100 mm): 25 steps give 63 um; with the planes moved by 5 mm
  to 50 mm, 32 steps give 0.24 um to 6.1 um and the LHCb reference 10 um to 12 um.
  The linear and the cubic reading of the map are 22.9 um apart (median, 95 %:
  163 um); against the cubic reference the LHCb reference is at 25.9 um. On the cubic
  map the cheapest classical method at target A is the Nystrom integrator with 29
  steps, 5.4 us (a cubic read costs 66 ns, a linear one 11 ns). The exact scheme of
  a layout meets target A with 128 stage points or more on the linear map (16 x 8,
  32 x 8) and with 64 or more on the cubic map; the lowest possible time of a chained
  network with a summed end state is 2.6 us (linear) and 4.9 us (cubic), so it does
  not beat the fixed-step integrator. The Nystrom integrator (2 N + 1 reads) is not
  faster than RK4 (4 N reads) on the linear map; the cause (the chain of dependent
  operations) is not tested. Step plans help at some numbers of steps only.
  Proposed to George for stage 2 (awaiting his answer): train only a network with
  no field read and a fixed-step integrator with 4 to 10 steps plus a network
  correction; leave the chained networks out; about 20 runs for the first batch.
  The plan document has a new section "What stages 0 and 1 measured" and its wrong
  statements are corrected (revision 17).
- 2026-11-24, 16:35 UTC: preparation of stage 2, no network trained
  (stage_2_training/PREPARATION.md, DESIGN.md as a draft). A polynomial of the start
  state, fitted with a robust loss on the 108,915 training tracks and scored on the
  13,684 validation tracks, stands in for a small smooth network. Median / 95 %:
  straight line + polynomial, no field read: 4.5 / 442 um against the cubic reference
  (degree 9) and 28 / 501 um against the linear reference (degree 7); Nystrom with 8
  steps + polynomial: 3.9 / 260 um (cubic reading and reference), 3.1 / 118 um (linear
  reading and reference); with 12 steps 3.4 / 74 um and 2.2 / 40 um. An integrator
  that reads the map linearly, scored on the cubic reference, stays at 26 um: the
  difference of the two readings is not smooth. A plain least-squares fit makes the
  typical track worse, because about 1 % of the tracks (low momentum, large |y|)
  have a correction 100 times the typical one; the loss of a network must be robust.
  Reported to George as an addition to the stage 2 proposal: expect a few um, not
  1 um; use a robust loss; keep the polynomial as the baseline; train and score in
  both readings of the map.
- 2026-11-24, 17:25 UTC: the polynomial baseline as a compiled method
  (cpp/corrections/corrected_start.h, python/fastex/correction.py,
  stage_2_training/scripts/polynomial_baseline.py; method `corrected:NAME` of fx).
  Gate: 9 of 9 corrections equal the numpy version (5e-12 mm at most). Test tracks,
  single precision, stbc-i3. Linear reference: Nystrom 12 steps + polynomial of degree
  7: 25 reads, 1.43 us, 2.15 / 36.8 / 12.3 um (LHCb adaptive Nystrom in the same run:
  3.59 us, 10.1 / 38.8 / 16.2 um); 8 steps: 1.09 us, 3.1 / 103 / 19.7 um; straight line
  + polynomial: no read, 0.43 us, 27.5 / 512 / 115 um (HLT1 parametrised step: 0.38 us,
  55.6 / 2,334 / 1,175 um). Cubic reference: straight line + polynomial of degree 9:
  0.77 us, 4.35 / 386 / 32.7 um; Nystrom 12 steps (cubic reading) + polynomial:
  2.82 us, 3.33 / 57.3 / 17.6 um. The polynomial code is a first version (0.41 us for
  degree 7); it can be made faster in stage 3. Reported to George as the bar for the
  networks of stage 2. Still awaiting his answer on the stage 2 proposal.
- 2026-11-24, 20:16 UTC: George: "continue to phase 2". Taken as his yes to the reduced
  stage 2 (a network with no field read, and a fixed-step integrator plus a network
  correction); not taken as his word for a farm submission. Built: the trainer
  python/fastex/train.py (a start plus a network correction; smooth Huber loss or
  squared loss; scale fixed or times q/p; rounds of L-BFGS on all 108,915 training
  tracks; stop on the validation median; resume), the form with no start in the
  compiled program, stage_2_training/scripts/first_batch.py (20 configurations in
  configurations/), farm/jobs.sub and farm/wrapper.sh (dry run passes, NOT submitted).
  Gates: 20 pass (python/tests), with the exported network equal to torch in the
  compiled program. Two short gate runs on stbc-i2 (2 rounds of 40 iterations, width 32):
  the trainer runs and resumes; about 18 s for 40 iterations on 8 threads. The design
  choices of DESIGN.md are used as drafted; the extra-input run is replaced by an
  integrator with 12 steps on the cubic reference. Waiting for George's word to submit.
- 2026-11-24, 21:19 UTC: George: "please commit then send the runs". Committed the
  folder Fast_extrapolator as eebf9e62 on the branch that was checked out
  (label_free_path; no branch was switched, nothing pushed). Submitted the first
  batch: 20 jobs, HTCondor cluster 54668, 8 cores each (stage_2_training/farm/jobs.sub).
  Runs go to /data/bfys/gscriven/fast_extrapolator_store/runs/<name>.
- 2026-11-25, 08:20 UTC: check of the first batch. The 11 runs with an integrator
  failed at their start: the trainer called the compiled program, which is built on the
  login nodes only. Fix: the start of an integrator is computed once on a login node and
  kept in the store (`python -m fastex.train --starts`); checked that the trainer then
  runs with no compiled program. The 11 runs are sent again (farm/jobs_with_an_integrator.sub),
  under George's word of 2026-11-24 for this batch.
- 2026-11-25, 08:35 UTC: the 9 runs with no field read, test tracks, compiled, single
  precision (stage_2_training/results/runs_accuracy.csv), median / 95 % / worst band:
  cubic reference, straight line + network: 2x128 3.63 / 187 / 26.5 um (ended at 91
  rounds), 2x64 4.5 / 279 / 43.5 (running), 3x64 4.8 / 227 / 42.4 (running); controls at
  2x64: fixed scale 23.2 / 818, no start 26.0 / 741, squared loss 424 / 4,447; 2x32 did
  not train (10 mm). Linear reference: 2x128 25.7 / 241 / 99, 2x64 32.0 / 514 / 151.
  Polynomial baseline: 4.35 / 386 / 32.7 (cubic, degree 9), 27.5 / 512 / 115 (linear).
  Not timed yet: the login ticket expired and stbc-i3 cannot be reached; George is
  asked to run kinit. The 11 integrator runs are cluster 92603.
