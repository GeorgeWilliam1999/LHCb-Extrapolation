// A host-side copy of Allen's adaptive Runge-Kutta-Nystrom extrapolator with its
// Jacobian, as PrKF (MR !2407) calls it:
// Extrapolators::AdaptiveRKNystromExtrapolator::propagate_base<withJacobian>,
// extrapolate and evaluateRKStep (device/kalman/ParKalman/include/RungeKuttaExtrapolator.cuh)
// and evaluateJacobianRKN (device/event_model/kalman/include/ExtrapolatorCommon.cuh).
//
// R = float is the copy; R = double is the same arithmetic in double. The state is
// a Vector5 of Allen (q/p in 1/MeV); the Jacobian columns of q/p come back in 1/MeV.
#pragma once

#include "rkn_common.h"

namespace allen_nn {

  struct ArknSettings {
    float inv_tolerance = 1.f / 0.005f; // mm
    float min_stepsize  = 10.f;
    float max_stepsize  = 1000.f;
    float sigma         = 5.5f;
    float min_stepscale = 0.125f;
    float max_stepscale = 4.f;
    int   max_steps     = 1000;
    float max_slope     = 10.f;
    float max_curvature = 1.f / 1000.f;
  };

  struct ArknCounts {
    int accepted = 0, rejected = 0, field_reads = 0;
  };

  template <class R, bool withJacobian, class Field>
  ALLEN_NN_HD inline void arkn_propagate( R x[5], R J[5][5], const R z_in, const R z_out, const Field& field,
                                          bool& fit_success, const ArknSettings& set = ArknSettings{},
                                          ArknCounts* counts = nullptr ) {
    if ( !fit_success || ( x[4] != x[4] ) ) {
      fit_success = false;
      return;
    }
    if ( withJacobian ) set_identity( J );
    R s[5] = { x[0], x[1], x[2], x[3], x[4] * R( c_light ) };
    R z    = z_in;

    R   dir     = ( z_out > z ) ? R( 1 ) : R( -1 );
    R   absstep = fmin( R( set.max_stepsize ), fabs( z_out - z ) );
    R   k[4][2] = {};
    R   B[3][3] = {};
    int rk_step_counter = 0;
    bool reuse_k0 = false;
    (void)reuse_k0; // k0 is recomputed from the same state and field, which gives the same value
    int reads = 0, accepted = 0, rejected = 0;

    while ( rk_step_counter < set.max_steps && fabs( z - z_out ) > R( 1e-3 ) ) {
      if ( rk_step_counter == 0 ) {
        field( s[0], s[1], z, B[0][0], B[0][1], B[0][2] );
        ++reads;
      }
      const R err = nystrom_stages( s, z, dir * absstep, B, k, field );
      reads += 2;

      bool    success      = ( absstep <= R( set.min_stepsize ) );
      const R prev_absstep = absstep;
      if ( !success ) {
        const R errorOverTolerance = R( set.inv_tolerance ) * err;
        success                    = ( errorOverTolerance <= R( set.sigma ) );
        R stepfactor               = R( set.max_stepscale );
        if ( errorOverTolerance > 1 ) {
          stepfactor = R( 1 ) / sqrt( sqrt( errorOverTolerance ) );
        } else if ( errorOverTolerance > 0 ) {
          stepfactor = pow( errorOverTolerance, R( -0.2 ) );
        }
        stepfactor = fmax( R( set.min_stepscale ), fmin( stepfactor, R( set.max_stepscale ) ) );
        absstep *= stepfactor;
        absstep = fmax( R( set.min_stepsize ), fmin( absstep, R( set.max_stepsize ) ) );
      }

      if ( success ) {
        const R step = dir * prev_absstep;
        if ( withJacobian ) evaluate_jacobian_rkn( J, step, s[2], s[3], k, s[4], B );
        nystrom_update( s, step, k );
        z += step;
        for ( int c = 0; c < 3; ++c ) B[0][c] = B[2][c];
        ++accepted;
      } else {
        ++rejected;
      }

      const R z_diff = z_out - z;
      if ( absstep - ( dir * z_diff ) > 0 ) {
        absstep = fabs( z_diff );
        dir     = ( z_out > z ) ? R( 1 ) : R( -1 );
      }
      if ( ( fmax( fabs( s[2] ), fabs( s[3] ) ) > R( set.max_slope ) ) ||
           ( fabs( s[4] * B[2][1] ) > R( set.max_curvature ) ) ||
           ( ( rk_step_counter == set.max_steps - 1 ) && ( !success ) ) ) {
        fit_success = false;
        break;
      }
      ++rk_step_counter;
    }
    if ( counts ) *counts = ArknCounts{ accepted, rejected, reads };
    if ( !fit_success ) return;
    x[0] = s[0];
    x[1] = s[1];
    x[2] = s[2];
    x[3] = s[3];
    if ( withJacobian )
      for ( int i = 0; i < 4; ++i ) J[i][4] *= R( c_light );
  }

} // namespace allen_nn
