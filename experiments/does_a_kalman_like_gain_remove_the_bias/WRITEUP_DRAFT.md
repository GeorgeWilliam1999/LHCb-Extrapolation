# Does a Kalman-like gain remove the bias of a self-chained network? (draft)

**Date:** 2026-11-05 · **State:** draft. Parts A, B and C have results. Part C: eight farm runs, one seed each.

---

## 1. Introduction

A stage network is a small neural network that gives the stage states of one implicit Runge–Kutta step. It is applied 64 times in a chain to carry a charged track through the LHCb magnet, from z = 2648.2 mm to z = 7826 mm. Each application has a small error. The errors of consecutive steps have the same sign: their lag-1 autocorrelation along the track is 0.78 to 0.90. So they add up. This systematic error per step is the *per-step bias*.

The network gives the stages. The end state of a step is summed from the rates at the stages. So the rates at the stages are already computed, and the *stage residual* (how far the stages are from solving the stage equations) is free. This study asks whether that residual can correct the step, in the way a Kalman filter corrects a prediction with an innovation.

The study has three parts.

| Part | Question | Training |
|---|---|---|
| A | Does a model-error covariance Q for the network step make a toy track fit honest? | none |
| B | Does a linearised gain K, built from the residual and the Jacobian of the field at the stages, remove the per-step bias with no new field lookup? | none |
| C | Can a small network learn the gain, so that a step needs no Jacobian and no linear solve? | yes |

## 2. Aims

1. Measure the endpoint error of three trained networks with and without the analytic gain (Part B).
2. Measure whether the gain changes the correlation of the step errors.
3. Measure the effect of a model-error covariance on the bias and the pulls of a toy fit (Part A).
4. Count the cost of each method in field lookups, Jacobians, solves and network calls, and time each part.
5. Train two learned variants of the gain and compare them with the analytic gain at their cost (Part C).

---

## 3. Theory

### 3.1 The equation of motion

A track on a plane of constant z has the state

    S = (x, y, tx, ty, q/p)

where x and y are the position in millimetres, tx = dx/dz and ty = dy/dz are the slopes (no unit), and q/p is the charge over the momentum. With z as the independent variable the Lorentz force gives

    dx/dz  = tx
    dy/dz  = ty
    dtx/dz = κ (q/p) N (tx ty Bx − (1 + tx²) By + ty Bz)
    dty/dz = κ (q/p) N ((1 + ty²) Bx − tx ty By − tx Bz)
    d(q/p)/dz = 0

where N = sqrt(1 + tx² + ty²), (Bx, By, Bz) is the magnetic field at (x, y, z) in tesla, and κ sets the units. We write the first four rates as f(Y, z), where Y = (x, y, tx, ty). The charge over momentum is constant, so it is an input of every step and is never predicted. The field comes from a field map on a grid, with trilinear interpolation inside each cell.

A *field lookup* is one evaluation of f at one state. It reads the field map at one point.

### 3.2 The implicit Runge–Kutta step

One step goes from the state S0 on the plane z0 to the plane z0 + dz. The step length is dz = (7826 − 2648.2) / 64 = 80.903 mm. An implicit Runge–Kutta method with q stages has

- the nodes c_k (fractions of the step), k = 1 … q;
- the stage matrix A, with entries A_jk;
- the weights b_k.

The stage states Y_1 … Y_q are the solution of the *stage equations*

    G_j(Y) = Y_j − S0 − dz Σ_k A_jk f(Y_k, z0 + c_k dz) = 0,     j = 1 … q.

The end state of the step is

    S_end = S0 + dz Σ_k b_k f(Y_k, z0 + c_k dz).

We use the Gauss–Legendre tableau with q = 2. For q = 2 the nodes are c = 1/2 ∓ √3/6, the weights are b = (1/2, 1/2), and the method is of order 4. We call the solution of the stage equations, found by a root finder, the *exact scheme*. It is the best that any network of this step length and stage count can give.

### 3.3 The stage network and the summed end state

The stage network N takes the state S0 and the start plane z0, mapped linearly onto [−1, 1], and gives the q stage states:

    (Y_1 … Y_q) = N(S0, z0).

The rates at the stages are F_k = f(Y_k, z0 + c_k dz): q field lookups. The end state is *summed* from them:

    E = S0 + dz Σ_k b_k F_k.

The same network is used on every step. The end state of one step, with q/p carried, is the input of the next.

The network is trained without labels. Its loss is the *pooled loss*: the mean, over the training states n, the stages j and the components d, of

    ( r[n, j, d] / s[d] )²

where r is the stage residual (section 3.4) and s[d] is the spread of component d over the states of the first round of training.

### 3.4 The stage residual as an innovation

The stage residual of the network's stages is

    r_j = Y_j − S0 − dz Σ_k A_jk F_k,      j = 1 … q.

It has 4q numbers per step. It is zero exactly when the stages solve the stage equations. It needs only the rates F_k, which the summed end state needs anyway. So it costs nothing more.

In a Kalman filter the *innovation* is the difference between an observation and its prediction. Here the stage equations play the role of the observation: they say what the stages must satisfy. The residual r says how far the network's stages are from that. So r is the innovation of the step.

### 3.5 The linearised gain

Let J_k = ∂f/∂Y at Y_k. It is a 4 × 4 matrix (q/p is fixed). Define the 4q × 4q *stage map*

    M = I − dz (A ⊗ J),   whose block (j, k) is  δ_jk I − dz A_jk J_k.

One Newton step of the stage equations from the network's stages moves them by

    dY = −M⁻¹ r.

If f at the new stages is replaced by its linearisation F_k + J_k dY_k, the summed end state moves by

    dz Σ_k b_k J_k dY_k = −K r,

with the *gain*

    K = dz (bᵀ ⊗ J) M⁻¹,     a 4 × 4q matrix,

where the block k of bᵀ ⊗ J is b_k J_k. The corrected end state is

    S_end = E − K r.

Three facts follow.

1. K r needs the Jacobians at the stages the step already has. It needs no lookup of f at a new state. The gradient of the trilinear map comes from the same 8 cell corners as the field, so a Jacobian reuses its lookup.
2. K r is exact to first order in the stage error. If the stage error is small, the corrected end state is the end state of the exact scheme up to second order.
3. At the stages of the exact scheme r = 0, so the correction is zero.

### 3.6 The Kalman analogy

| Kalman filter | This study |
|---|---|
| predicted state | the stages and the summed end state of the network |
| observation model | the stage equations G(Y) = 0 |
| innovation | the stage residual r |
| gain | K = dz (bᵀ ⊗ J) M⁻¹ |
| update | S_end = E − K r |
| process noise covariance Q | Part A: the covariance of the network's step error |
| measurement noise covariance | none: the stage equations are exact |

The analogy is not exact. The gain here is not a ratio of covariances. It is the linearised Newton step of a deterministic equation. The term "Kalman-like" refers to the structure: a gain that maps an innovation to a correction of the state carried on.

### 3.7 The model-error covariance (Part A)

A track fit combines the extrapolator with hits. If the extrapolator has an error, the fit can carry it as a *model-error covariance* Q added to each step, as process noise in a Kalman filter. For a linear fit, Q per step is carried to the measurement plane as Σ_k T_k Q T_kᵀ, where T_k is the Jacobian of the reference track from step k to that plane. Part A tests whether such a Q removes the momentum bias and makes the pulls of the fit equal to 1.

---

## 4. The learned gain (Part C)

### 4.1 C1a: a learned gain, distilled from the analytic gain

A trained stage network is frozen. A small *gain network* G reads

- the state S0, divided by the spread of the states (5 numbers);
- the start plane z0, mapped onto [−1, 1];
- the residual r, divided by its root mean square s_r over the first round's states (4q numbers);
- optionally the rates F, divided by their root mean square (4q numbers).

G has two tanh layers and a linear layer. Its 4 × 4q outputs form a matrix W. The correction and the end state are

    D_d = s_out,d Σ_m W_dm r_m / s_r,m,
    S_end = E − D,

where s_out is the scale of the correction (section 5).

The correction is a matrix applied to r, not a free vector. So a residual of zero gives a correction of zero, whatever the weights. G can represent the analytic gain: W = diag(1/s_out) K diag(s_r). K depends on the stages through J, and the stages are a function of S0 and z0. This is the form of the KalmanNet update: a network gives the gain, and the gain multiplies the innovation.

The last layer of G starts at zero. Training then starts from the frozen network alone. r and F are detached before they enter G.

**The loss.** The label is K r, computed in training from the Jacobians at the stages. The loss is

    mean over n, d of ( (D[n, d] − (K r)[n, d]) / s[d] )².

K r comes from the equation of motion and the stages alone. No reference integrator and no reference state enter it. So the loss is label-free in the sense used here: it uses no reference label. A setting adds the pooled stage terms, for a run that trains the stage network as well.

**The cost.** Inference needs q field lookups, one call of the stage network and one call of G. It needs no Jacobian and no solve. Training needs q Jacobians and one 4q × 4q solve per step and state.

### 4.2 C2: an error-conditioned stage network

The frozen stage network is called once. A *corrector* C reads S0, z0 and r / s_r (6 + 4q inputs; 14 for q = 2). It gives a 4q × 4q matrix W. The stages are corrected and the end state is summed again:

    dY_(k,d) = s_out,d Σ_m W_(k,d),m r_m / s_r,m,
    Y' = Y + dY,
    F'_k = f(Y'_k, z0 + c_k dz),
    S_end = S0 + dz Σ_k b_k F'_k.

Y and r are detached, so no gradient flows between the two calls. With W = −diag(1/s_out) M⁻¹ diag(s_r), the correction is one Newton step.

**The loss.** The pooled loss over the stages of the second call and of the first call, as one mean. With a frozen first call the second half is a constant.

**The cost.** Inference needs 2q field lookups (q more than the stage network alone, because the rates at the corrected stages are new), one call of the stage network and one call of C. It needs no Jacobian and no solve.

### 4.3 C1b: why it is not built

C1b would train G so that the residual of the corrected stages is small, with no label at all. This cannot be written cleanly without new field lookups. The reasons:

1. G corrects the end state: 4 numbers. The residual lives on the stages: 4q numbers. The map from a stage shift to an end shift, dz (bᵀ ⊗ J), is 4 × 4q, so it has no unique inverse. A rule that lifts the end correction back to a stage shift must be chosen, and the natural rule, the Newton direction −M⁻¹ r, needs the Jacobians.
2. The residual at the shifted stages needs f at those stages: q new lookups per evaluation of the loss. Its linearisation, r + M dY, needs the Jacobians.
3. With the linearisation, the minimum over dY is dY = −M⁻¹ r. The end shift is then exactly −K r. So the linearised C1b has the same optimum as C1a, reached by a less direct loss.
4. Without the linearisation, and with a network that outputs a stage shift, the method is C2.

So C1b is C1a (linearised) or C2 (exact). Only C1a and C2 are built.

### 4.4 Cost of each method per step

| Method | Field lookups | Field samples of a per-track scale | Jacobians | Solves | Network calls |
|---|---|---|---|---|---|
| stage network, summed | 2 | 16 if per-track | 0 | 0 | 1 |
| stage network + analytic gain (B1) | 2 | as above | 2 | 1 (8 × 8) | 1 |
| C1a: stage network + learned gain | 2 | as above | 0 (2 in training) | 0 (1 in training) | 2 (one small) |
| C2: error-conditioned | 4 | as above | 0 | 0 | 2 (one small) |
| exact scheme | about 2 per Newton iteration | 0 | 2 per iteration | 1 per iteration | 0 |
| classic RK4 | 4 | 0 | 0 | 0 | 0 |

For a whole track multiply by 64. The per-track scale of the straight-line form samples the field 16 times along the straight line of each step. The counts of Part B (section 8) leave out these samples.

---

## 5. Output forms

An *output form* turns the raw outputs of a network body, of order one, into states.

| Form | State given by the network | With the last layer at zero | Scale |
|---|---|---|---|
| direct states | scale × raw output | the origin of the detector | four numbers: the spread of x, y, tx, ty over the first round's states |
| straight line plus a scaled correction | the straight line from S0 to the plane of the stage, plus scale × raw output | the straight line | *per track*: from the field integral along the straight line of the step; or *global*: the median of the per-track scale over the first round's states (four numbers) |

The per-track scale is, with I_B the integral of |B| along the straight line and I_y a bound on the integral of the y-rate,

    base_x = κ |q/p| I_B,     base_y = max(κ |q/p| I_y, 10⁻³ base_x),
    x, y:   base · dz / 2,     tx, ty: base.

The networks of Part C have one more scale: that of the correction, s_out. It is a setting:

| Setting | s_out |
|---|---|
| same as the stage network | the scale of the frozen network's output form: per track, global, or the spread of the states |
| per track | the per-track scale, computed for the correction (16 more field samples a step if the stage network has none) |
| spread of the states, or four numbers | fixed |

---

## 6. Methodology

### 6.1 Data

| Item | Value |
|---|---|
| Sample | simulated LHCb events, magnet polarity up, field map v8r1 |
| Tracks | 14,482 particles; pseudorapidity 2 to 5; momentum 1 to 200 GeV; no electrons; a hit in the Upstream Tracker |
| Region | z = 2648.2 mm to 7826 mm |
| Splits | training 11,567, validation 1,463, test 1,452 tracks |
| Reference | the sixth-order explicit Runge–Kutta method of Butcher, at a step of 0.1 mm, on 257 planes |

### 6.2 The trained networks

Three networks, all 64 steps of 80.903 mm, 2 stages, the pooled loss over the stages, the end state summed, one seed each.

| Output form | Width × depth | Test endpoint (um) |
|---|---|---|
| straight line, per-track scale | 128 × 2 | 1.83 |
| straight line, one global scale | 128 × 2 | 2.04 |
| direct states | 128 × 2 | 5.71 |

### 6.3 Training (Part C)

| Setting | Value |
|---|---|
| Rounds | round 1 on reference states on the start planes; later rounds on the states the network itself predicts along the training tracks |
| States per round | 32,000, the same number on every start plane |
| Optimiser | L-BFGS, 25 restarts a round of 200 iterations, history 120, strong Wolfe line search, the loss rescaled to 1 at the start of a round |
| Stopping | the validation endpoint error (the larger of x and y, median) stops falling: the median of the last 10 rounds is less than 5% below that of the 10 before, for 3 rounds |
| Cap | 60 rounds |
| Precision | double |
| Small network | 2 tanh layers of 64; last layer at zero at the start |

### 6.4 The score

| Item | Definition |
|---|---|
| Endpoint error | the radial distance sqrt(dx² + dy²) at z = 7826 mm between the chained prediction and the reference, on the test split |
| Statistic | the median over tracks, in micrometres |
| Momentum bands | < 3, 3–8, 8–20, 20–50, > 50 GeV |
| Signed error | the mean of dx and dy, with its standard error |
| Single-step error | one step from the reference state on each start plane, against the reference on the next plane; 92,928 steps |
| Lag-1 autocorrelation | of the signed single-step error along each track; the median over tracks |
| Floor | the exact scheme at 64 steps and 2 stages: 0.147 um |

Each network is one seed. We do not claim a difference under 2×.

### 6.5 Criteria of Part C, written before the farm runs

| # | Criterion |
|---|---|
| 1 | C1a passes on a frozen network when its test endpoint is below 0.30 um (within 2× of the analytic gain, 0.147 um). |
| 2 | C2 passes on a frozen network when its test endpoint is more than 2× below that of the frozen network alone. |
| 3 | Either passes as an extrapolator when it beats classic RK4 at 64 steps (1.08 um) with no more field lookups (256 a track). |
| 4 | Convergence is read from the validation error. A run at its cap without a plateau is reported as such. |

### 6.6 Cost accounting and timing

Counts are per step and per track. A field lookup is f at one state. A Jacobian is ∂f/∂Y at one state; it reuses the lookup of f at that state. A network call is one application of a network body to one state.

The times of section 8.4 are from the CPU of a shared login node, in double precision, with torch. The node had a load of about 240. So only the ratios are of use. The network is not a fused kernel. The Jacobian comes from automatic differentiation, not from a closed-form gradient of the trilinear map. In a production extrapolator, a closed-form gradient, a fused network kernel or single precision would each change the ratios.

---

## 7. Results: Part B, the analytic gain

### 7.1 Endpoint

Test endpoint radial median (um), per momentum band.

| Stages from | Correction | Field lookups / step | Jacobians / step | Network calls / step | Endpoint | < 3 | 3–8 | 8–20 | 20–50 | > 50 |
|---|---|---|---|---|---|---|---|---|---|---|
| straight line, per-track scale | none | 2 | 0 | 1 | 1.83 | 4.94 | 2.10 | 1.44 | 1.21 | 1.89 |
| | gain, chained | 2 | 2 | 1 | 0.147 | 0.66 | 0.249 | 0.0852 | 0.0353 | 0.0115 |
| straight line, one global scale | none | 2 | 0 | 1 | 2.04 | 7.42 | 3.21 | 1.31 | 1.30 | 0.654 |
| | gain, chained | 2 | 2 | 1 | 0.147 | 0.66 | 0.249 | 0.0852 | 0.0353 | 0.0115 |
| direct states | none | 2 | 0 | 1 | 5.71 | 21.7 | 8.74 | 3.54 | 2.96 | 3.24 |
| | gain, chained | 2 | 2 | 1 | 0.147 | 0.666 | 0.249 | 0.0852 | 0.0353 | 0.0115 |
| straight line, no network | none | 2 | 0 | 0 | 7020 | 22369 | 10171 | 3690 | 1702 | 645 |
| | gain, chained | 2 | 2 | 0 | 2.36 | 53.9 | 6.88 | 0.307 | 0.0182 | 0.0111 |
| explicit Euler, no network | none | 3 | 0 | 0 | 5.45 | 22.8 | 8.43 | 2.91 | 1.29 | 0.486 |
| | gain, chained | 3 | 2 | 0 | 0.146 | 0.658 | 0.248 | 0.0852 | 0.0353 | 0.0115 |
| exact scheme | — | — | — | 0 | 0.147 | 0.66 | 0.25 | 0.085 | 0.035 | 0.012 |
| classic RK4 | — | 4 | 0 | 0 | 1.08 | | | | | |

- With the gain, every network gives the exact scheme in every band.
- From the straight line with no network, the gain gives 2.36 um. At the same lookups the network is worth 16× at the endpoint.
- From explicit Euler stages, which need 1 more lookup a step, the gain gives the exact scheme with no network.

### 7.2 Is the correction linear?

The corrections of the plain chain were also carried to the end with the Jacobians of the reference track and added to the plain endpoint. This gives the same endpoint as the chained correction within 0.001 um for every network and for the Euler start. So the correction is linear in the step error and does not need to be fed into the next step. Adding the corrections without the carry is worse (0.15 to 0.20 um for the networks), because a slope correction moves the position downstream.

### 7.3 Single steps

| Stages from | Correction | Radial median (um) | Radial rms (um) | rms tx (urad) | rms ty (urad) | Lag-1 x | Lag-1 y |
|---|---|---|---|---|---|---|---|
| exact scheme | — | 0.000718 | 0.0219 | 0.295 | 0.247 | −0.22 | −0.18 |
| straight line, per-track scale | none | 0.0721 | 0.893 | 0.300 | 0.249 | +0.85 | +0.90 |
| | gain | 0.000718 | 0.0219 | 0.295 | 0.247 | −0.22 | −0.18 |
| straight line, one global scale | none | 0.117 | 1.04 | 0.303 | 0.246 | +0.78 | +0.87 |
| | gain | 0.000718 | 0.0219 | 0.295 | 0.247 | −0.22 | −0.18 |
| direct states | none | 0.486 | 2.80 | 0.344 | 0.269 | +0.87 | +0.89 |
| | gain | 0.000719 | 0.0219 | 0.295 | 0.247 | −0.22 | −0.18 |

- The gain makes the single step that of the exact scheme. The position error falls by 100× to 680× in the median.
- The lag-1 autocorrelation falls from +0.78 to +0.90 to −0.22 and −0.18, the values of the exact scheme. The per-step bias is gone.
- The slope error hardly changes (0.30 → 0.295 urad). It is the error of the exact scheme itself, at the cell faces of the field map. No correction of the stage equations can remove it.

## 8. Results: Part A, the model-error covariance, and timing

### 8.1 The toy fit

The fit finds (x, y, tx, ty, q/p) on z = 2648.2 mm from 4 hits in the Upstream Tracker (50 um each) and 12 hits in the fibre tracker (75 um each). It has no multiple scattering and no energy loss, so its resolution is that of the hits alone. The models differ only from 2648.2 mm to 7826 mm. Each track has 200 noise draws.

| Setting of Q | What it is |
|---|---|
| no Q | the hit covariance only |
| Q per step | the covariance of the network's signed single-step error per momentum band, carried to 7826 mm and added to the fibre-tracker hits |
| Q at the end | the covariance of the chained endpoint error per band |

### 8.2 Results

| Network | Q | Band | Bias / resolution of dp/p | Pull tx | Pull q/p |
|---|---|---|---|---|---|
| straight line, per-track scale | none | all | −0.0007 | 1.000 | 0.999 |
| direct states | none | < 3 GeV | −0.0036 | 1.365 | 1.123 |
| direct states | per step | < 3 GeV | −0.0034 | 1.259 | 1.096 |
| direct states | at the end | < 3 GeV | −0.0033 | 1.032 | 1.068 |
| exact scheme | none | < 3 GeV | −0.0063 | 1.002 | 1.002 |

- The momentum bias is at most 1.1% of the resolution, with or without Q. Q changes the weights of the fit, not the model error.
- Only direct states below 3 GeV has wide pulls. Q per step narrows them a little; Q at the end narrows them to near 1.
- Q per step is 1.1× to 2.6× too small in sigma at the endpoint, because the step errors are correlated along the track (section 7.3) and a per-step Q treats them as independent.

### 8.3 The size of Q

| Network | Band | Per-step sigma x (um) | Per-step sigma tx (urad) | Per-step Q carried to 7826 mm: sigma x (um) | Measured endpoint sigma x (um) | Ratio |
|---|---|---|---|---|---|---|
| straight line, per-track scale | < 3 GeV | 1.67 | 0.726 | 22.4 | 26.4 | 1.2 |
| straight line, per-track scale | > 50 GeV | 0.223 | 0.011 | 1.80 | 4.67 | 2.6 |
| direct states | < 3 GeV | 6.08 | 0.952 | 62.0 | 146 | 2.4 |
| direct states | > 50 GeV | 0.256 | 0.011 | 2.06 | 2.25 | 1.1 |

### 8.4 Timing

Time of each part per step and track, one thread (see section 6.6 for the caveats).

| Part | ns | In units of 2 field lookups |
|---|---|---|
| 2 field lookups | 887 | 1.0 |
| 2 field lookups + 2 Jacobians | 3,267 | 3.7 |
| 8 × 8 solve and K r | 1,657 | 1.9 |
| Euler stages (1 lookup) | 493 | 0.56 |
| 1 call of the stage network (128 × 2) | 7,379–7,931 | 8.3–8.9 |

| Method | ns per step | Endpoint (um) |
|---|---|---|
| network, plain | 8,270–8,820 | 1.83–5.71 |
| network + analytic gain | 12,300–12,860 | 0.147 |
| network + one fixed-point sweep | 9,150–9,710 | 0.145–0.147 |
| straight line + gain | 4,920 | 2.36 |
| Euler + gain | 5,420 | 0.146 |
| Euler + one fixed-point sweep | 2,270 | 0.148 |
| classic RK4 (estimate) | 1,770 | 1.08 |

- On this code one network call costs as much as 8 to 9 pairs of field lookups. So every method with a network is slower than every method without one.
- The Jacobians and the solve cost more than one fixed-point sweep. So the analytic gain is slower than a sweep, although it needs fewer lookups.
- A learned gain (C1a) removes the Jacobians and the solve (4.6 lookup pairs) and adds one call of a small network. Its body is 14 → 64 → 64 → 32, against 6 → 128 → 128 → 8 for the stage network.

---

## 9. Results: Part C, the learned gain

### 9.1 Smoke run

The smoke run trains C1a and C2 on the frozen per-track straight-line network for 3 rounds of 4,000 states (5 restarts of 100 iterations). It shows that the loss falls and that the report runs. It is far from a plateau, so it says nothing about what the methods reach.

| Model | Test endpoint (um) | < 3 | 3–8 | 8–20 | 20–50 | > 50 | Field lookups / step |
|---|---|---|---|---|---|---|---|
| frozen network alone | 1.83 | 4.94 | 2.10 | 1.44 | 1.21 | 1.89 | 2 |
| C1a, 3 smoke rounds | 1.45 | 3.69 | 1.54 | 1.19 | 1.12 | 1.83 | 2 |
| C2, 3 smoke rounds | 1.31 | 4.12 | 1.30 | 1.11 | 1.05 | 1.80 | 4 |
| analytic gain | 0.147 | 0.66 | 0.249 | 0.0852 | 0.0353 | 0.0115 | 2 (+ 2 Jacobians) |

### 9.2 Farm runs

Eight runs, one seed each: C1a and C2 on each of the three frozen networks, and on direct states with a per-track scale of the correction. The training is that of section 6.3. Every run reached the plateau of the validation rule; none reached the cap. Six plateaued at round 22, the earliest round the rule allows.

Test endpoint, radial median (um):

| Variant | Frozen network | Scale of the correction | Endpoint | < 3 | 3–8 | 8–20 | 20–50 | > 50 | Field lookups / step |
|---|---|---|---|---|---|---|---|---|---|
| frozen alone | straight line, per-track | — | 1.83 | 4.94 | 2.10 | 1.44 | 1.21 | 1.89 | 2 + 16 samples |
| C1a | straight line, per-track | per track | 1.43 | 3.73 | 1.54 | 1.20 | 1.08 | 1.83 | 2 + 16 samples |
| C2 | straight line, per-track | per track | 1.02 | 1.87 | 0.972 | 0.896 | 0.979 | 1.76 | 4 + 16 samples |
| frozen alone | straight line, global | — | 2.04 | 7.42 | 3.21 | 1.31 | 1.30 | 0.654 | 2 |
| C1a | straight line, global | four numbers | 0.341 | 1.32 | 0.511 | 0.247 | 0.187 | 0.168 | 2 |
| C2 | straight line, global | four numbers | 0.148 | 0.665 | 0.246 | 0.0868 | 0.0361 | 0.0112 | 4 |
| frozen alone | direct states | — | 5.71 | 21.7 | 8.75 | 3.54 | 2.96 | 3.24 | 2 |
| C1a | direct states | state spread | 0.807 | 4.14 | 1.10 | 0.559 | 0.454 | 0.455 | 2 |
| C2 | direct states | state spread | 0.145 | 0.643 | 0.252 | 0.0862 | 0.0340 | 0.0120 | 4 |
| C1a | direct states | per track | 4.17 | 11.9 | 5.11 | 3.13 | 2.78 | 3.17 | 2 + 16 samples |
| C2 | direct states | per track | 1.27 | 1.99 | 1.19 | 1.05 | 1.44 | 2.27 | 4 + 16 samples |
| exact scheme | — | — | 0.147 | 0.66 | 0.25 | 0.085 | 0.035 | 0.012 | about 2 per iteration |
| analytic gain | any | — | 0.147 | 0.66 | 0.249 | 0.0852 | 0.0353 | 0.0115 | 2, + 2 Jacobians and a solve |
| classic RK4 | — | — | 1.08 | | | | | | 4 |

Single steps (92,928 test steps):

| Variant | Frozen network | Scale of the correction | Radial median (um) | rms tx (urad) | Lag-1 x | Lag-1 y |
|---|---|---|---|---|---|---|
| exact scheme | — | — | 0.000718 | 0.295 | −0.22 | −0.18 |
| C1a | straight line, global | four numbers | 0.00829 | 0.311 | −0.22 | +0.91 |
| C2 | straight line, global | four numbers | 0.000825 | 0.295 | −0.22 | +0.12 |
| C1a | direct states | state spread | 0.00136 | 0.356 | −0.21 | +0.53 |
| C2 | direct states | state spread | 0.00136 | 0.295 | −0.21 | +0.53 |
| C1a | straight line, per-track | per track | 0.0518 | 0.314 | +0.82 | +0.90 |
| C2 | straight line, per-track | per track | 0.0347 | 0.319 | +0.75 | +0.89 |
| C1a | direct states | per track | 0.286 | 0.363 | +0.82 | +0.88 |
| C2 | direct states | per track | 0.0522 | 0.303 | +0.80 | +0.88 |

Against the criteria of section 6.5:

| # | Criterion | Result |
|---|---|---|
| 1 | C1a below 0.30 um | fails on every network; the best is 0.341 um |
| 2 | C2 more than 2× below its frozen network | passes on three of four: 14× (global), 39× (direct states), 4.5× (direct states, per-track scale); fails on the per-track straight line (1.8×) |
| 3 | beats RK4 at no more than 256 lookups a track | C2 passes on the global straight line and on direct states (0.148 and 0.145 um at 256); C1a passes on the same two (0.341 and 0.807 um at 128) |
| 4 | plateau reached | every run |

### 9.3 What Part C shows

1. **C2 reaches the exact scheme.** On a network with a fixed output scale, the error-conditioned network gives 0.148 um and 0.145 um, against 0.147 um for the exact scheme, in every band. It needs 4 field lookups a step and no Jacobian. It reached this by round 2 of training.
2. **C1a comes close but does not reach it.** The learned gain gives 0.341 um and 0.807 um at 2 field lookups a step. It removes the lag-1 correlation in x but not in y, and it leaves a slope error above that of the exact scheme.
3. **A per-track scale of the correction fails.** All four runs with it keep a lag-1 of +0.75 to +0.82 in x. For C2 the cause is measured (point 4). For C1a it is not.
4. **C2 learns one fixed-point sweep.** Where C2 succeeds, its correction is within 2% to 17% (median) of dY = −r, one fixed-point sweep of the stage equations. The sweep itself, Y' = S0 + dz A F with no second network, gives 0.145 to 0.147 um from every frozen network, at the same 4 lookups a step. So C2 adds nothing to a sweep. With a fixed scale the matrix C2 must give is constant over the states (dz A J is small, so M ≈ I); with the per-track scale it varies by 1.75 to 2.73 decades, which is why that form fails.
5. **Cost in time.** On the code of this package one network call costs as much as 8 to 9 pairs of field lookups (section 8.4). C2 costs two network calls and 4 lookups a step. RK4 costs 4 lookups. So at equal lookups C2 is 7× more accurate than RK4, but on this code it is slower.

### 9.4 Questions left open

1. Why does C1a fail with a per-track scale? The spread of its target does not explain it.
2. Does C1a reach the exact scheme in y with the rates as an input, or with a longer training?
3. Is the result of C2 the same for other seeds?
4. What does C2 cost in time in a fused, single-precision kernel?

## 10. Related work

- **KalmanNet** (Revach, Shlezinger, Ni, López Escoriza, van Sloun and Eldar, arXiv:2107.10043) keeps the structure of the Kalman filter and replaces the computation of the gain by a recurrent network that reads features of the innovation. The learned gain of C1a uses the same structure: a network gives a gain matrix that multiplies the innovation. KalmanNet is trained on the true states; C1a is trained on the linearised gain, with no reference state.
- **Unsupervised KalmanNet** (Revach, Shlezinger, Locher, Ni, van Sloun and Eldar, arXiv:2110.09005) trains KalmanNet without true states, from the prediction error of the next observation. C1a is label-free in a similar sense: its target is computed from the equation of motion alone.
- **Error-conditioned neural solvers** (Jiang, Wang, Chen, Kwak, Park and others, arXiv:2606.27354) give a first estimate of a PDE solution, then let a corrector network read the PDE residual of the estimate as an input and correct it, a few times, with the input of each correction detached. They argue that minimising the residual is an unreliable proxy for accuracy in ill-conditioned problems, and train on labels. C2 takes the residual as an input in the same way, but trains on the residual and applies the corrector once.
- **Runge–Kutta physics-informed networks** (Stiasny, Chevalier and Chatzivasileiadis, arXiv:2106.15987) train a network to give the stages of an implicit Runge–Kutta step of a dynamical system, with a loss on the stage equations, so that one network call replaces the iterations of the implicit solve. The stage network here follows that idea for the equation of motion in the LHCb field.
- **Phase-space flows with time-discrete implicit Runge–Kutta physics-informed networks** (Fernández Corral, Mendoza, Iske, Yachmenev and Küpper, arXiv:2409.16826) give the network the phase-space state as input, so that one trained network propagates every trajectory one step and is applied again for later steps. They use Gauss–Legendre tableaux with up to 499 stages for particles in a central force and in a periodic electric field. The self-chained stage network here applies the same idea, with the start plane as a further input because the LHCb field is not periodic along the track.

---

## 11. What is not done

- C1b is not built (section 4.3).
- Part C has one seed per run.
- The cause of the failure of C1a with a per-track scale of the correction (section 9.3) is not measured.
- The timing is on a loaded CPU node in this package's code, not in the target code.
- The hit layout and resolutions of the toy fit were not re-checked against the LHCb Tracker Upgrade TDR.
- Each network is one seed.
