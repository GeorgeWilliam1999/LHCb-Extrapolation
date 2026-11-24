// Copies of the two Runge-Kutta-Nystrom integrators of Allen (HLT1).
//
// Source: Allen v9r3, device/kalman/ParKalman/include/RungeKuttaExtrapolator.cuh
// and device/event_model/kalman/include/ExtrapolatorCommon.cuh.
//
//   AllenAdaptiveNystrom   AdaptiveRKNystromExtrapolator::extrapolate, with
//       evaluateRKStep and the state part of evaluateStateJacobianRKN:
//       tolerance 0.005 mm, steps 10 mm to 1 m, sigma 5.5, step scale 0.125 to
//       4. Three field points a step; the last is the first of the next step.
//       In Allen v9r3 only the test algorithm extrapolate_states calls it.
//   AllenFastNystrom       RungeKuttaNystromExtrapolator::propagate with
//       make_fast_step: fixed steps of 500 mm and ONE field read a step, at the
//       middle of the straight line of the step. The HLT1 T-track code calls it
//       (device/ttrack/prepare_tracks).
//
// Allen runs in single precision: R = float is the copy. R = double is the same
// arithmetic in double precision, for the check against the Python copy. The
// transport matrix is not computed here.
//
// The q/p of the state is the q/p of LHCb in 1/MeV times c_light, as Allen
// forms it on entry. The field comes in tesla, so kappa = 1e-3 is folded into
// q/p one time per call.
#pragma once

#include <algorithm>
#include <cmath>

#include "../classical/integrators.h"

namespace fx {

  namespace allen {
    template <class R>
    struct Slopes {
      R dtx, dty;
    };

    // slope_derivative of ExtrapolatorCommon.cuh
    template <class R>
    inline Slopes<R> slope_derivative( R tx, R ty, R qop, R Bx, R By, R Bz ) {
      const R tx2  = tx * tx;
      const R ty2  = ty * ty;
      const R norm = std::sqrt( R( 1 ) + tx2 + ty2 );
      const R ax   = norm * ( ty * ( tx * Bx + Bz ) - ( R( 1 ) + tx2 ) * By );
      const R ay   = norm * ( -tx * ( ty * By + Bz ) + ( R( 1 ) + ty2 ) * Bx );
      return { qop * ax, qop * ay };
    }

    // gamma of RungeKuttaNystromExtrapolator
    template <class R>
    inline Slopes<R> gamma( R tx, R ty, R Bx, R By, R Bz ) {
      const R norm = std::sqrt( R( 1 ) + tx * tx + ty * ty );
      return { norm * ( tx * ty * Bx - ( R( 1 ) + tx * tx ) * By + ty * Bz ),
               norm * ( ( R( 1 ) + ty * ty ) * Bx - tx * ty * By - tx * Bz ) };
    }
  } // namespace allen

  template <class R, class Field>
  struct AllenAdaptiveNystrom {
    const Field* field;
    R            z_in, z_out;
    R            inv_tolerance = R( 1 ) / R( 0.005 ); // in mm
    R            min_stepsize  = R( 10 );
    R            max_stepsize  = R( 1000 );
    R            sigma         = R( 5.5 );
    R            min_stepscale = R( 0.125 );
    R            max_stepscale = R( 4 );
    int          max_steps     = 1000;
    mutable int  last_steps = 0, last_failed = 0;

    inline State<R> operator()( State<R> s ) const {
      using allen::Slopes;
      const R qop = static_cast<R>( KAPPA * Field::tesla_per_unit ) * s.qop;
      R       z   = z_in;
      R       dir = ( z_out > z ) ? R( 1 ) : R( -1 );
      R       absstep = std::min( max_stepsize, std::abs( z_out - z ) );

      Slopes<R> k[4] = {};
      R         B[3][3] = {};
      int       rk_step_counter = 0;
      bool      reuse_k0        = false;
      int       steps = 0, failed = 0;

      while ( rk_step_counter < max_steps && std::abs( z - z_out ) > R( 1e-3 ) ) {
        // evaluateRKStep
        const R dz = dir * absstep;
        if ( rk_step_counter == 0 ) ( *field )( s.x, s.y, z, B[0][0], B[0][1], B[0][2] );
        if ( !reuse_k0 ) k[0] = allen::slope_derivative( s.tx, s.ty, qop, B[0][0], B[0][1], B[0][2] );
        const R dz2    = dz * dz;
        const R halfdz = R( 0.5 ) * dz;
        ( *field )( s.x + halfdz * s.tx + R( 0.125 ) * dz2 * k[0].dtx, s.y + halfdz * s.ty + R( 0.125 ) * dz2 * k[0].dty,
                    z + halfdz, B[1][0], B[1][1], B[1][2] );
        for ( int stage = 1; stage < 3; stage++ ) {
          k[stage] = allen::slope_derivative( s.tx + halfdz * k[stage - 1].dtx, s.ty + halfdz * k[stage - 1].dty, qop,
                                              B[1][0], B[1][1], B[1][2] );
        }
        ( *field )( s.x + dz * s.tx + R( 0.5 ) * dz2 * k[2].dtx, s.y + dz * s.ty + R( 0.5 ) * dz2 * k[2].dty, z + dz,
                    B[2][0], B[2][1], B[2][2] );
        k[3] = allen::slope_derivative( s.tx + dz * k[2].dtx, s.ty + dz * k[2].dty, qop, B[2][0], B[2][1], B[2][2] );
        const R err_x = ( k[0].dtx - k[1].dtx - k[2].dtx + k[3].dtx ) * dz2;
        const R err_y = ( k[0].dty - k[1].dty - k[2].dty + k[3].dty ) * dz2;

        bool    success      = ( absstep <= min_stepsize );
        const R prev_absstep = absstep;
        if ( !success ) {
          const R errorOverTolerance = inv_tolerance * std::max( std::abs( err_x ), std::abs( err_y ) );
          success                    = ( errorOverTolerance <= sigma );
          R stepfactor               = max_stepscale;
          if ( errorOverTolerance > 1 ) {
            stepfactor = R( 1 ) / std::sqrt( std::sqrt( errorOverTolerance ) );
          } else if ( errorOverTolerance > 0 ) {
            stepfactor = std::pow( errorOverTolerance, R( -0.2 ) );
          }
          stepfactor = std::max( min_stepscale, std::min( stepfactor, max_stepscale ) );
          absstep *= stepfactor;
          absstep = std::max( min_stepsize, std::min( absstep, max_stepsize ) );
        }

        if ( success ) {
          // evaluateStateJacobianRKN, the state part
          const R step  = dir * prev_absstep;
          const R fpos  = step * step * ( R( 1 ) / R( 6 ) );
          const R fslo  = step * ( R( 1 ) / R( 6 ) );
          const R dposx = ( k[0].dtx + k[1].dtx + k[2].dtx ) * fpos;
          const R dposy = ( k[0].dty + k[1].dty + k[2].dty ) * fpos;
          const R dslox = ( k[0].dtx + k[1].dtx * R( 2 ) + k[2].dtx * R( 2 ) + k[3].dtx ) * fslo;
          const R dsloy = ( k[0].dty + k[1].dty * R( 2 ) + k[2].dty * R( 2 ) + k[3].dty ) * fslo;
          s.x           = s.x + step * s.tx + dposx;
          s.y           = s.y + step * s.ty + dposy;
          z += step;
          s.tx += dslox;
          s.ty += dsloy;
          B[0][0] = B[2][0], B[0][1] = B[2][1], B[0][2] = B[2][2];
          reuse_k0 = false;
          ++steps;
        } else {
          reuse_k0 = true;
          ++failed;
        }

        const R z_diff = z_out - z;
        if ( absstep - ( dir * z_diff ) > 0 ) {
          absstep = std::abs( z_diff );
          dir     = ( z_out > z ) ? R( 1 ) : R( -1 );
        }
        ++rk_step_counter;
      }
      last_steps  = steps;
      last_failed = failed;
      return s;
    }
  };

  template <class R, class Field>
  struct AllenFastNystrom {
    const Field* field;
    R            z_in, z_out;
    R            step_size = R( 500 );
    int          max_steps = 100;
    mutable int  last_steps = 0, last_failed = 0;

    inline State<R> operator()( State<R> s ) const {
      using allen::Slopes;
      const R qop = static_cast<R>( KAPPA * Field::tesla_per_unit ) * s.qop;
      R       z   = z_in;
      const R dir = ( z_out > z ) ? R( 1 ) : R( -1 );
      int     steps = 0;
      for ( int i_iter = 0; i_iter < max_steps; i_iter++ ) {
        const R dz = dir * std::min( step_size, std::abs( z_out - z ) );
        // make_fast_step
        R Bx, By, Bz;
        ( *field )( s.x + R( 0.5 ) * s.tx * dz, s.y + R( 0.5 ) * s.ty * dz, z + R( 0.5 ) * dz, Bx, By, Bz );
        constexpr R c[4] = { R( 0 ), R( 0.5 ), R( 0.5 ), R( 1 ) };
        Slopes<R>   k[4];
        {
          const Slopes<R> g = allen::gamma( s.tx, s.ty, Bx, By, Bz );
          k[0]              = { g.dtx * qop, g.dty * qop };
        }
        for ( int stage = 1; stage < 4; stage++ ) {
          const R         f = dz * c[stage];
          const Slopes<R> g = allen::gamma( s.tx + k[stage - 1].dtx * f, s.ty + k[stage - 1].dty * f, Bx, By, Bz );
          k[stage]          = { g.dtx * qop, g.dty * qop };
        }
        const R fpos = dz * dz * ( R( 1 ) / R( 6 ) );
        const R fslo = dz * ( R( 1 ) / R( 6 ) );
        const R dRx  = s.tx * dz + ( k[0].dtx + k[1].dtx + k[2].dtx ) * fpos;
        const R dRy  = s.ty * dz + ( k[0].dty + k[1].dty + k[2].dty ) * fpos;
        const R dTx  = ( k[0].dtx + k[1].dtx * R( 2 ) + k[2].dtx * R( 2 ) + k[3].dtx ) * fslo;
        const R dTy  = ( k[0].dty + k[1].dty * R( 2 ) + k[2].dty * R( 2 ) + k[3].dty ) * fslo;
        s.x += dRx;
        s.y += dRy;
        z += dz;
        s.tx += dTx;
        s.ty += dTy;
        ++steps;
        if ( std::abs( z - z_out ) < R( 1 ) ) break;
      }
      last_steps = steps;
      return s;
    }
  };

} // namespace fx
