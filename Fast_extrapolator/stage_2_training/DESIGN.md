# Stage 2, first batch: design (draft, not agreed)

Status (2026-11-24): a draft for George. Nothing here is built or trained. Each
numbered choice below is his to make; my proposal is marked.

The numbers behind the choices are in `PREPARATION.md` (no network) and in the
results notes of stages 0 and 1.

## The form of every method

    end state on the last plane = start(start state) + correction(start state)

The start is a method with no network. The correction is one call of a small
network. The start state is (x, y, tx, ty, q/p) on the first plane (2648.2 mm);
the end state is (x, y, tx, ty) on the last plane (7826 mm).

## The choices

**1. The starts.**

| Start | Field reads | Its own median error (validation tracks) |
|---|---|---|
| The straight line | 0 | 449 mm |
| The Nystrom integrator with 4, 6, 8 or 12 equal steps, reading the same map as the reference | 9, 13, 17, 25 | 203, 16, 13, 8.4 um (cubic); 202, 21, 19, 15 um (linear) |

Proposal: the straight line, and the integrator with 4 and 8 steps (cubic
reference) and with 6, 8 and 12 steps (linear reference). One control run with
no start at all (the network gives the end state itself, the form of the
present whole-crossing network).

Not in the proposal: an integrator that reads the map linearly with the cubic
reference. The difference of the two readings is not smooth, and a smooth
correction gains almost nothing (26 um stays 26 um).

Not in the first batch: the thin-magnet kick from a tabulated field integral
(the plan's "start A"). The straight line with a correction that is
proportional to q/p (choice 3) covers it.

**2. The inputs of the network.** Proposal: the five numbers of the start
state, each divided by a fixed scale (the spread over the training tracks), as
in the present networks. One extra run gives the network the four numbers of
the integrator's end state as well.

**3. The scale of the correction.** The network's four outputs are of order
one. They are multiplied by a scale to give the correction.

- (a) Four fixed numbers (the spread of the correction over the training
  tracks). This is the scale of the present whole-crossing network.
- (b) Four fixed numbers times q/p of the track, with its sign. Every
  correction is zero at q/p = 0 and grows about as |q/p| (measured exponent 1.0
  for the straight line, 1.1 to 1.5 for the integrators). With (b) the network
  learns the correction for one unit of q/p, which has the same size at 2 GeV
  and at 100 GeV.

Proposal: (b), with one control run of (a).

**4. The loss.** The error of a track is the difference between its predicted
and its reference end state, component by component.

- (a) The squared error, summed over the tracks: the present loss.
- (b) A robust loss: quadratic for an error below a threshold and linear above
  it (threshold 2 um for a position, 1e-6 for a slope).

Proposal: (b), with one control run of (a). Reason: about 1 % of the tracks
(low momentum, large |y| on the last plane) have a correction 100 times the
typical one. With the squared error they rule the fit, and the typical track
gets worse; this is measured with the polynomial in `PREPARATION.md`.

**5. The tracks.** As agreed: the 108,915 training tracks of the track set of
the reference; the 13,684 validation tracks for the stopping rule; the 13,447
test tracks for the scores only. The extra tracks from the reference integrator
(agreed for the network with no field read) wait for a later batch.

**6. The size of the network.** tanh, depth 2. Widths 32, 64 and 128 with no
field read; widths 32 and 64 with an integrator. One run of depth 3.

| Width (depth 2) | One call, compiled |
|---|---|
| 32 | 135 ns |
| 64 | 234 ns |
| 128 | 1,078 ns |

**7. The reference.** Agreed: the cubic reading, with a check on the linear
reading for the best configurations. Proposal: train and score both kinds on
both readings from the start, because the reading changes the answer: with no
field read a smooth model reaches a few um against the cubic reference and
about 28 um against the linear one.

**8. The baseline that a network must beat.** The same form with a polynomial
of the start state in place of the network (degree 7: 462 terms a component).
It needs no farm job. It is scored and timed in the compiled program like a
network.

**9. The trainer.** Proposal: the trainer of the present package (rounds of
full-batch L-BFGS in double precision, stop on the validation error, a snapshot
a round). The new forms and the robust loss are new components in
`Fast_extrapolator/python/fastex`; they are registered when a run starts, and
no file of the package is edited. Runs go to the store of this experiment.

For a run to be traceable to a commit, the folder `Fast_extrapolator` must be
committed first. That needs George's word, and the branch.

## The first batch (proposal: 20 runs)

| Group | Reference | Runs |
|---|---|---|
| No field read: straight line + correction, widths 32, 64, 128 at depth 2 and width 64 at depth 3 | cubic | 4 |
| No field read, controls at width 64: squared-error loss; fixed scale; no start | cubic | 3 |
| No field read: widths 64 and 128 | linear | 2 |
| Integrator with 4 and 8 steps + correction, widths 32 and 64 | cubic | 4 |
| Integrator with 8 steps + correction, width 64, with the integrator's end state as extra inputs | cubic | 1 |
| Integrator with 6, 8 and 12 steps + correction, widths 32 and 64 | linear | 6 |
| Total | | 20 |

More seeds follow for the best configurations only.

## What to expect

From the polynomial probe: a median error of 2 to 7 um for both kinds, and a
95 % level of 40 to 500 um. Target A (1 um, 10 um at 95 %) is not expected.
What a network can add to the polynomial is the tail: the tracks below 3 GeV
and at large |y|.

Times for one track, from the unit costs of stage 0:

| Method | Linear reading | Cubic reading |
|---|---|---|
| No field read, width 32 to 128 | 0.14 to 1.1 us | the same |
| 4 steps + width 32 | 0.45 us | 1.0 us |
| 8 steps + width 64 | 0.9 us | 1.9 us |
| 12 steps + width 64 | 1.2 us | 2.6 us |
| LHCb adaptive Runge-Kutta-Nystrom | 3.6 us | |
| LHCb adaptive Cash-Karp | 1.9 us | |
| HLT1 parametrised step | 0.38 us | |
