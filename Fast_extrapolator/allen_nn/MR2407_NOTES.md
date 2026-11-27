# Allen MR !2407 (PrKF, the Kalman fit with the adaptive RKN): what it does at the magnet

Read on 2026-11-27 from `refs/merge-requests/2407/head` = commit `2f8e80a52`
("Modify Dumpers to point to new sequences", Penelope Hoffmann, 2026-11-18), in
`/tmp/gscriven/lhcb_read/Allen_mr2407` (cloned over https; the ssh route to
gitlab.cern.ch hung at the key, see the end).

## State of the MR

- Title "Draft: Kalman filter using adaptive RKN extrapolator", **open, draft**,
  not merged. Source `pehoffma_adaptive_RKN`, target `master`. Last update
  2026-11-18. 40 commits above the merge base `8dfb00d7`; master has moved on
  (last master commit read: 2026-11-25).
- 43 files, +2,555 / -656. The new fit is in
  - `device/kalman/ParKalman/src/PrKalmanFilter.cu` (672 lines, the fit `PrKalmanFilter::full_fit`,
    the kernels `full_kalman_filter` and `refit_outliers_full_fit`)
  - `device/kalman/ParKalman/include/PrKalmanMethods.cuh` (475, predict/update/extrapolate)
  - `device/kalman/ParKalman/include/PrKalmanFilter.cuh` (121, the algorithm)
  - `device/kalman/ParKalman/include/KalmanSharedMethods.cuh` (659, shared with ParKF; moved out of ParKalmanMethods.cuh)
  - `device/kalman/ParKalman/include/KalmanScatteringConstants.cuh`, `src/KalmanScatteringConstants.cu` (noise parameters from a PARAM file)
  - `device/kalman/ParKalman/include/RungeKuttaExtrapolator.cuh` (ARKN: `propagate_jac` and a new `propagate` without Jacobian)
  - `device/event_model/kalman/include/ExtrapolatorCommon.cuh` (State, slope derivative, the RKN Jacobian)
  - `ParKalmanDefinitions.cuh` renamed `KalmanDefinitions.cuh` (namespace `KalmanFilter`)
  - sequences `hlt1_pp_forward_then_matching_and_downstream_with_fullkf_1300kHz_rich[_from_cones].py`,
    `AllenConf/secondary_vertex_reconstruction.py` (`make_full_kalman_long`, outlier chi2 9, 2 outlier iterations),
    Dumpers test scripts `Allen_hlt1_pp_run4_{data,mc,rich_data}.sh`.

## Where PrKF calls the ARKN

Every transport in PrKF is `Extrapolators::AdaptiveRKNystromExtrapolator` (alias `ARKN`):

| function (PrKalmanMethods.cuh) | call | Jacobian | used for |
|---|---|---|---|
| `PredictState` (called by `PredictStateV/UT/UTT_T`) | `ARKN::propagate_jac(x_ref → x_state, F, zIn, zTo)` | yes | every hit-to-hit prediction |
| `ExtrapolateEndT` | `propagate`, VELO seed → z = 9410 | no | seed of the fit, a full VELO→T crossing |
| `ExtrapolateLastMeasurement` | `propagate`, 9410 → last SciFi hit | no | SciFi seed |
| `ExtrapolateUT` | `propagate`, first VELO hit z → last UT hit | no | reference vector at the last UT hit |
| `ExtrapolateRICH` | `propagate` | no | RICH1/RICH2 states |

The prediction is linearised about a reference: `x_state = ARKN(x_ref)`, then
`x = F x + (x_state − F x_ref)` and `C = F C Fᵀ` (+ Q before or after, by direction).
`x_ref` is the reference state, not the filtered state, so the extrapolator is
called once per prediction, on the reference.

### Order of the fit (`full_fit`, PrKalmanFilter.cu)

1. Seed from the VELO; `ExtrapolateEndT` (VELO → 9410 mm) and
   `ExtrapolateLastMeasurement` (→ last SciFi hit).
2. **First pass, backward in z** (T → UT → VELO): SciFi hits from the last to the first
   (`PredictStateUTT_T`), then **line 246: `PredictStateUT(..., dev_scatter_UTT_pars, ..., hit_counter = n_ut_hits − 1)`:
   first SciFi hit → last UT hit, dz < 0. This is the magnet crossing, one `propagate_jac` call.**
   Then the reference is reset from the VELO side (`ExtrapolateUT`), UT hits, `PredictStateV` VELO hits.
3. **Second pass, forward in z** (VELO → UT → T), with the RICH1 states between VELO and UT:
   **line 382: `PredictStateUTT_T(..., dev_scatter_UTT_pars, ..., first SciFi hit)`: last UT hit →
   first SciFi hit, dz > 0. The magnet crossing again, one `propagate_jac` call.** The comment says
   "Magnet extrapolation of both x and the reference vector, as no reference vector available behind the magnet".
   Then SciFi hits, the averaged EndT state and RICH2.
4. The outlier kernel `refit_outliers_full_fit` runs `full_fit` again (at most
   `max_outlier_iterations` = 2 more fits) for tracks with a hit above chi2 9.

So the magnet crossing is **one call** each way, from `zAtYEq0()` of the last UT hit
(about 2,600–2,700 mm) to `z0()` of the first SciFi hit (about 7,830–7,900 mm), and
back. A per-track fit has two magnet crossings with Jacobian (one each direction)
plus up to two more per outlier refit, plus the VELO→9410 seed crossing without Jacobian.

The UT→T direction (dz > 0) is the direction of our networks. The T→UT
direction (dz < 0) needs either the ARKN, an inverse network, or a Newton
inversion of the forward tool (2–3 forward calls with Jacobian).

## ARKN settings (RungeKuttaExtrapolator.cuh, unchanged by the MR except the Jacobian switch)

- tolerance 0.005 mm (`inv_tolerance = 1/0.005`), `sigma = 5.5` (a step is accepted when
  err/tol ≤ 5.5), step 10 mm to 1,000 mm, step scale 0.125 to 4 (shrink `(err/tol)^-1/4`,
  grow `(err/tol)^-1/5`), at most 1,000 RK step attempts, stop when |z − z_out| ≤ 1e-3 mm.
- The step: Nystrom, three field reads (start, middle, end); the end field is reused as the
  start of the next step; a rejected step reuses k0. Error estimate `(k0 − k1 − k2 + k3) dz²`.
- Sanity checks set `fit_success = false`: |tx| or |ty| > 10, |qop·By| > 1/1000 (in the units
  of the integrator), or the last of the 1,000 attempts failed. A NaN q/p on entry also fails.
- Units: on entry `qop · c_light · eplus` (c_light = 299.792458 in float), q/p of LHCb in 1/MeV;
  the field is the DeMagnet grid in Gaudi units (1 T = 1e-3). Same as our `RecField` reader.

## The Jacobian (ExtrapolatorCommon.cuh, `evaluateJacobianRKN`)

- Built step by step, "like the HLT2 RKN as ODE solution": per accepted step, before the
  state update, the four stages are differentiated with respect to (tx, ty, q/p) **with the
  field held fixed** at the three field values already read (no dB/dx term), and the 2×3
  position and slope blocks are added to the running 5×5 (which starts at the identity).
  ∂x/∂x = ∂y/∂y = 1 and no column for x, y beyond the identity.
- The q/p column is multiplied by c_light at the end (back to LHCb units).
- `propagate` (no Jacobian) is a template switch added by this MR (`withJacobian`).

## Field access

- `MagneticField::Magfield::fieldVectorLinearInterpolation(float3)`; float everywhere.
- CUDA build: **texture memory by default** (`MAGFIELD_USE_TEXTURE` ON, CMakeLists.txt:147),
  `tex3D` with `cudaFilterModeLinear`, border address mode (zero outside). Hardware trilinear
  filtering keeps the fractional weights in 9-bit fixed point (1/256 of a cell = 0.4 mm on a
  100 mm grid), so the GPU field is not exactly the trilinear field.
- CPU and HIP builds: `MAGFIELD_USE_TEXTURE` is forced OFF; plain float trilinear on three
  arrays, zero outside the grid (index check `ix >= Nx − 1` etc.). This is the read that our
  `AllenField` copies.
- The grid comes from `DeMagnet::fieldGrid()` (81 × 81 × 146, 100 mm), so the polarity and
  the scale of the conditions are already in the map.

## Momentum and slope cuts

- None inside PrKF beyond the sanity checks above (|t| > 10, curvature, NaN q/p) and the
  requirement in a comment "We require (so far) at least two UT hits". `ResetInitCov` sets
  σ(q/p) = 0.3 |q/p|. Energy loss is applied only if |p| stays above 10 MeV.
- The tracks come from forward / matching long-track reconstruction; their p cuts are set
  upstream (not read here).

## Magnet polarity

- Not handled in PrKF or the ARKN: the field map of the run (DeMagnet) carries the sign.
  (`ParKF` and the downstream KF take a `magnet_polarity` argument for their parametrisations;
  PrKF does not.) A network trained for one polarity needs either a second weight file or the
  symmetry B → −B ⇔ q/p → −q/p (valid when the two maps are exact negatives).

## Consequences for the tool

1. The tool replaces `ARKN::propagate_jac` in `PredictState` when `zIn` is the last UT hit and
   `zTo` is the first SciFi hit (the call at PrKalmanFilter.cu:382, `dev_scatter_UTT_pars`).
   Way A: Nystrom leg zIn → 2648.2, network crossing 2648.2 → 7826, Nystrom leg 7826 → zTo.
   The Jacobian is the product leg · crossing · leg.
2. The first-pass crossing (line 246, T → UT) is the reverse direction; it stays on the ARKN
   in the prototype (or a Newton inversion later).
3. The switch must also cover `fit_success`: the tool should fail on the same sanity checks.
4. Our networks are trained on the trilinear map of the CPU build. On a GPU with textures the
   ARKN reads a slightly different field; this matters for a comparison on GPU only.

## Access

- `git clone https://gitlab.cern.ch/lhcb/Allen.git` and `git fetch origin refs/merge-requests/2407/head` work
  anonymously from stbc-i3. The ssh route (port 7999, `/user/gscriven/.ssh/id_ed25519`) stopped at the
  publickey step (the key needs a passphrase or an agent); not needed for reading. Nothing was pushed to CERN.
