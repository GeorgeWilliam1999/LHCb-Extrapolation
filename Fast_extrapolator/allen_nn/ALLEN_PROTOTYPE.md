# P4: the prototype in Allen (local branch, not pushed to CERN)

Local branch `allen_nn_magnet_crossing` of the Allen clone
`/tmp/gscriven/lhcb_read/Allen_mr2407` (worktree `/tmp/gscriven/allen_build/Allen`), on top of
MR !2407 (`2f8e80a52`). The patch is `allen_nn/allen_patch/0001-allen-nn-magnet-crossing.patch`
(`git am` on `refs/merge-requests/2407/head`). Nothing was pushed to gitlab.cern.ch.

## What the patch does

- `device/kalman/ParKalman/include/allen_nn/{rkn_common.h,magnet_crossing.h}`: the tool (copies of
  `Fast_extrapolator/allen_nn/include/allen_nn/`), plain `__host__ __device__` functions.
- `device/kalman/ParKalman/include/MagnetCrossingNN.cuh`: the field adapter
  (`Magfield::fieldVectorLinearInterpolation`) and `MagnetCrossingNN::propagate_jac`, the interface of
  `ARKN::propagate_jac` (Vector5, `Extrapolators::State::Jacobian`, `fit_success`).
- `PrKalmanMethods.cuh`: `PredictState` and `PredictStateUTT_T` take an optional network pointer and
  polarity. When the pointer is set and dz > 0 the reference is propagated by the tool; when the state
  on z0 = 2648.2 mm is outside the box of the training tracks the tool returns false and the ARKN runs.
- `PrKalmanFilter.cu/.cuh`: `full_fit`, `full_kalman_filter` and `refit_outliers_full_fit` pass the
  pointer through; only the magnet crossing of the second pass (last UT hit → first SciFi hit,
  PrKalmanFilter.cu:382 in !2407) uses it. The first-pass crossing (T → UT, dz < 0) keeps the ARKN.
- Two new properties of `full_kalman_filter_t`: `magnet_nn_weights` (path of the JSON file;
  **empty = off, the default**) and `magnet_nn_polarity_of_training` (+1: trained on the up map).
  The weights are read once on the host (nlohmann::json), copied to the device
  (`Allen::malloc`/`Allen::memcpy`, 35 kB for width 64). Polarity used =
  sign(`constants.magnet_polarity`) × polarity of training (Allen: −1 down, +1 up), exact because
  the v8r1 maps are exact negatives.
- `AllenConf.secondary_vertex_reconstruction.make_full_kalman_long(..., magnet_nn_weights="")`.
- `test/unit_tests/generic/src/TestMagnetCrossingNN.cpp`: Catch2 tests with no input data: a zero
  network is the plain fixed-step Nystrom; the full 5×5 transport matrix of the tool equals central
  finite differences to 1e-6 (z-only toy field, where the fixed-field chaining is exact); the
  polarity symmetry is bit-exact. Checked first with a plain g++ build (3/3 pass).

## Build

CPU build in the LHCb stack, against the nightly `lhcb-master/3236` (2026-11-26) on CVMFS,
`x86_64_v3-el9-gcc15-opt`, LCG 108c, gcc 15.2 (scripts `allen_patch/build_allen.sh`,
`allen_patch/rebuild_allen.sh`). **The build succeeds** (373 targets, `make install`, no error, no
warning in the new code), MR !2407 itself compiles against today's master stack. The Catch2 tests
`unit_tests "[magnet_nn]"` pass: 51 assertions in 3 test cases (`allen_patch/unit_tests_magnet_nn.log`).
Allen's pre-commit hooks (clang-format, copyright) pass on the commit (`05be341ff` locally).

Note on CPU use: the first `make -j6` went through ninja, which ignored the make job server and
ran about 57 compiles at once on stbc-i3 for some minutes; I could only renice it (stopping it was
refused by the permission check). Later builds call `ninja -j6` directly.

## What is missing for the physics comparison

- **The input sample.** Allen's tests read `2026_expected_MinBias_Sim10g_mdf` (TestFileDB:
  `root://eoslhcb.cern.ch//eos/lhcb/cern-swtest/lhcb/swtest/2026_expected_MinBias_Sim10g_mdf/...`).
  From stbc-i3 the read is refused: `[3010] Unable to give access - user access restricted -
  unauthorized identity used`. There is no `/eos` mount, no Kerberos ticket (`klist`: none) and no
  grid proxy (`~/.globus/usercert.pem` exists, `voms-proxy-init` needs George's passphrase).
  Needed: `voms-proxy-init -voms lhcb` (or `kinit gscriven@CERN.CH`) on stbc, then
  `xrdcp` of the MDF (one file, 15,548 events) to `/tmp`, or run on lxplus.
- **MC truth for a resolution.** The MDF has detector banks only (no MC banks). A momentum
  resolution needs the DIGI version with MC (`2026_expected_MinBias_Sim10g_digi`, same EOS area)
  and the Allen MC checking (lbexec with `run_hlt1.py` and an MC options file, or the PrChecker of
  Moore on the Allen tracks). Without MC, the off/on comparison can still be made on the fitted q/p,
  chi2/ndof and the fraction of failed fits, per track.
- Run command once the sample is local (CPU build, sequence of the MR):

      ./build.x86_64_v3-el9-gcc15-opt/run lbexec \
         $ALLEN_INSTALL_DIR/python/tests/options/run_hlt1.py:main options.yaml -- \
         --sequence=hlt1_pp_forward_then_matching_and_downstream_with_fullkf_1300kHz_rich

  with `input_files` set to the local MDF, once with the default and once with
  `make_full_kalman_long.bind(magnet_nn_weights=".../linear_nystrom_12_2x64.json")`.

## Not done / open

- The T → UT crossing (first pass) still uses the ARKN: a reverse network, or a Newton inversion of
  the forward tool (2–3 forward calls with the matrix) is needed to replace it.
- GPU: with textures on (the CUDA default) the field is read with 9-bit interpolation weights; the
  networks are trained on the exact trilinear read. Not tested (no GPU build tonight).
- The networks are trained on the linear (trilinear) map, v8r1 up, the planes 2648.2 → 7826 mm.
