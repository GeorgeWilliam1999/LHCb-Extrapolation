// A copy of the adaptive Runge-Kutta-Nystrom integrator of the HLT2 software.
//
// Source: Rec v41r3, Tr/TrackExtrapolators/src/TrackSTEPExtrapolator.cpp
// (`extrapolate`, `evaluateRKStep`, `evaluateRKStepJacobian`,
// `evaluateDerivatives`, `evaluateDerivativesJacobian`): the STEP algorithm of
// ATLAS (Lund, Bugge, Gavrilenko and Strandlie), with the default properties of
// the tool:
//
//   Tolerance 0.005 mm   MinStep 10 mm   MaxStep 1 m   InitialStep 1 m
//   Sigma 5.5   MinStepScale 0.125   MaxStepScale 4   StepScaleSafetyFactor 1
//   MaxNumSteps 1000   MaxSlope 10   MaxCurvature 1/m   UseFieldLastStep true
//
// A step has four stages and three field points: the start, the middle (used by
// two stages) and the end. The field at the end of a step is used again as the
// field at the start of the next step, from the third step on (the condition
// in the source is `cache.laststep > 0`, so the second step reads its own
// start field). A step that failed keeps its first stage.
//
// Allen's AdaptiveRKNystromExtrapolator is the single-precision twin of this
// tool; LHCb compares the two in a Moore test
// (compare_hlt1_hlt2_extrapolator_states.py). In Rec v41r3 the RICH
// reconstruction uses this tool, and the extrapolator tests use it as the
// reference; the Kalman fit still reaches TrackRungeKuttaExtrapolator through
// TrackMasterExtrapolator by default.
//
// The transport matrix is computed when `with_jacobian` is set, as in the
// original (2 x 3 blocks for the positions and for the slopes).
//
// Units as in rec_cash_karp.h: with the field read of LHCb (RecField) the
// factor in front of q/p is exactly 1.
#pragma once

#include <algorithm>
#include <cmath>

#include "../classical/integrators.h"
#include "rec_cash_karp.h"

namespace fx {

  template <class Field>
  struct RecSTEP {
    const Field* field;
    double       z_in, z_out;
    // the properties of the tool
    double toleranceX       = 0.005;
    double minRKStep        = 10.;
    double maxRKStep        = 1000.;
    double initialRKStep    = 1000.;
    double sigma            = 5.5;
    double minStepScale     = 0.125;
    double maxStepScale     = 4.0;
    double safetyFactor     = 1.0;
    std::size_t maxNumRKSteps = 1000;
    double maxSlope         = 10.0;
    double maxCurvature     = 1. / 1000.;
    bool   useFieldLastStep = true;
    bool   with_jacobian    = false;
    mutable int    last_steps = 0, last_failed = 0;
    mutable bool   last_success = true;
    mutable double last_jacobian[4][3] = {}; // rows x, y, tx, ty; columns tx0, ty0, q/p

    struct RKState {
      double x, y, tx, ty, qop, z;
    };
    struct RKStage {
      RKState state;
      double  dtx, dty; // derivative (dtxdz, dtydz)
      double  B[3];
    };
    struct RKCache {
      RKStage stage[4];
      int     laststep = -1;
      int     step     = 0;
    };

    static inline void evaluateDerivatives( RKStage& stage ) {
      const double tx  = stage.state.tx;
      const double ty  = stage.state.ty;
      const double qop = stage.state.qop;
      const double Bx = stage.B[0], By = stage.B[1], Bz = stage.B[2];
      const double tx2 = tx * tx;
      const double ty2 = ty * ty;
      const double qopnorm = qop * std::sqrt( 1.0 + tx2 + ty2 );
      stage.dtx            = qopnorm * ( ty * ( tx * Bx + Bz ) - ( 1.0 + tx2 ) * By );
      stage.dty            = qopnorm * ( -tx * ( ty * By + Bz ) + ( 1.0 + ty2 ) * Bx );
    }

    inline void read( RKStage& stage ) const {
      ( *field )( stage.state.x, stage.state.y, stage.state.z, stage.B[0], stage.B[1], stage.B[2] );
    }

    inline void evaluateRKStep( const double dz, RKState& pin, double err[2], RKCache& cache ) const {
      RKStage& stage0 = cache.stage[0];
      if ( cache.laststep == cache.step ) {
        // the previous step failed: the first stage is kept
      } else {
        stage0.state = pin;
        if ( cache.laststep > 0 && useFieldLastStep ) {
          stage0.B[0] = cache.stage[3].B[0], stage0.B[1] = cache.stage[3].B[1], stage0.B[2] = cache.stage[3].B[2];
        } else {
          read( stage0 );
        }
        evaluateDerivatives( stage0 );
        cache.laststep = cache.step;
      }

      const double dzdz   = dz * dz;
      const double halfdz = 0.5 * dz;

      RKStage& stage1 = cache.stage[1];
      stage1.state    = pin;
      stage1.state.z += halfdz;
      stage1.state.x += halfdz * pin.tx + 0.125 * dzdz * stage0.dtx;
      stage1.state.y += halfdz * pin.ty + 0.125 * dzdz * stage0.dty;
      stage1.state.tx += halfdz * stage0.dtx;
      stage1.state.ty += halfdz * stage0.dty;
      read( stage1 );
      evaluateDerivatives( stage1 );

      RKStage& stage2 = cache.stage[2];
      stage2          = stage1;
      stage2.state.tx = pin.tx;
      stage2.state.ty = pin.ty;
      stage2.state.tx += halfdz * stage1.dtx;
      stage2.state.ty += halfdz * stage1.dty;
      evaluateDerivatives( stage2 );

      RKStage& stage3 = cache.stage[3];
      stage3.state    = pin;
      stage3.state.z += dz;
      stage3.state.x += dz * ( pin.tx + halfdz * stage2.dtx );
      stage3.state.y += dz * ( pin.ty + halfdz * stage2.dty );
      stage3.state.tx += dz * stage2.dtx;
      stage3.state.ty += dz * stage2.dty;
      read( stage3 );
      evaluateDerivatives( stage3 );

      pin.x += dz * pin.tx + dzdz / 6.0 * ( stage0.dtx + stage1.dtx + stage2.dtx );
      pin.y += dz * pin.ty + dzdz / 6.0 * ( stage0.dty + stage1.dty + stage2.dty );
      pin.tx += dz / 6.0 * ( stage0.dtx + 2.0 * ( stage1.dtx + stage2.dtx ) + stage3.dtx );
      pin.ty += dz / 6.0 * ( stage0.dty + 2.0 * ( stage1.dty + stage2.dty ) + stage3.dty );
      pin.z += dz;

      err[0] = dzdz * ( stage0.dtx - stage1.dtx - stage2.dtx + stage3.dtx );
      err[1] = dzdz * ( stage0.dty - stage1.dty - stage2.dty + stage3.dty );
    }

    // evaluateDerivativesJacobian: deriv is 2 x 3; `tm` is the slope block of the transport matrix
    static inline void evaluateDerivativesJacobian( const RKStage& stage, const double tm[2][3], double deriv[2][3] ) {
      const double tx  = stage.state.tx;
      const double ty  = stage.state.ty;
      const double qop = stage.state.qop;
      const double Bx = stage.B[0], By = stage.B[1], Bz = stage.B[2];

      const double tx2 = tx * tx;
      const double ty2 = ty * ty;

      const double n2 = 1.0 + tx2 + ty2;
      const double n  = std::sqrt( n2 );

      const double txBx = tx * Bx;
      const double txBy = tx * By;
      const double tyBy = ty * By;
      const double tyBx = ty * Bx;

      const double Ax = n * ( ty * ( txBx + Bz ) - ( 1 + tx2 ) * By );
      const double Ay = n * ( -tx * ( tyBy + Bz ) + ( 1 + ty2 ) * Bx );

      const double inv_n2 = 1.0 / n2;
      const double Ax_n2  = Ax * inv_n2;
      const double Ay_n2  = Ay * inv_n2;

      const double dAxdTx = Ax_n2 * tx + n * ( tyBx - 2 * txBy );
      const double dAxdTy = Ax_n2 * ty + n * ( txBx + Bz );

      const double dAydTx = Ay_n2 * tx + n * ( -tyBy - Bz );
      const double dAydTy = Ay_n2 * ty + n * ( -txBy + 2 * tyBx );

      deriv[0][0] = qop * ( tm[0][0] * dAxdTx + tm[1][0] * dAxdTy );
      deriv[1][0] = qop * ( tm[0][0] * dAydTx + tm[1][0] * dAydTy );
      deriv[0][1] = qop * ( tm[0][1] * dAxdTx + tm[1][1] * dAxdTy );
      deriv[1][1] = qop * ( tm[0][1] * dAydTx + tm[1][1] * dAydTy );
      deriv[0][2] = Ax + qop * ( tm[0][2] * dAxdTx + tm[1][2] * dAxdTy );
      deriv[1][2] = Ay + qop * ( tm[0][2] * dAydTx + tm[1][2] * dAydTy );
    }

    static inline void evaluateRKStepJacobian( const double dz, double xm[2][3], double tm[2][3], const RKCache& cache ) {
      double k0[2][3], k1[2][3], k2[2][3], k3[2][3], tmp[2][3];
      evaluateDerivativesJacobian( cache.stage[0], tm, k0 );
      for ( int r = 0; r < 2; ++r )
        for ( int c = 0; c < 3; ++c ) tmp[r][c] = tm[r][c] + 0.5 * dz * k0[r][c];
      evaluateDerivativesJacobian( cache.stage[1], tmp, k1 );
      for ( int r = 0; r < 2; ++r )
        for ( int c = 0; c < 3; ++c ) tmp[r][c] = tm[r][c] + 0.5 * dz * k1[r][c];
      evaluateDerivativesJacobian( cache.stage[2], tmp, k2 );
      for ( int r = 0; r < 2; ++r )
        for ( int c = 0; c < 3; ++c ) tmp[r][c] = tm[r][c] + dz * k2[r][c];
      evaluateDerivativesJacobian( cache.stage[3], tmp, k3 );
      for ( int r = 0; r < 2; ++r )
        for ( int c = 0; c < 3; ++c ) {
          xm[r][c] += dz * tm[r][c] + dz * dz / 6.0 * ( k0[r][c] + k1[r][c] + k2[r][c] );
          tm[r][c] += dz / 6.0 * ( k0[r][c] + 2 * ( k1[r][c] + k2[r][c] ) + k3[r][c] );
        }
    }

    template <class R>
    inline State<R> operator()( const State<R>& in ) const {
      RKState state{ static_cast<double>( in.x ),
                     static_cast<double>( in.y ),
                     static_cast<double>( in.tx ),
                     static_cast<double>( in.ty ),
                     ( KAPPA * Field::tesla_per_unit ) * static_cast<double>( in.qop ),
                     z_in };
      const double zout = z_out;

      double xm[2][3] = {}, tm[2][3] = {};
      tm[0][0]        = 1; // dTxdTx0
      tm[1][1]        = 1; // dTydTy0

      const double totalStep = zout - state.z;
      double       absstep   = std::min( std::abs( totalStep ), initialRKStep );
      const int    direction = totalStep > 0 ? +1 : -1;
      const double toleranceX_inv = 1.0 / toleranceX;

      RKCache     rkcache;
      double      err[2] = { 0, 0 };
      std::size_t numfailedstep = 0;
      bool        ok            = true;

      while ( ok && std::abs( state.z - zout ) > 1e-6 ) { // TrackParameters::propagationTolerance
        const RKState prevstate = state;
        evaluateRKStep( absstep * direction, state, err, rkcache );

        bool success = ( absstep <= minRKStep );
        if ( !success ) {
          const double normdx             = std::abs( err[0] ) * toleranceX_inv;
          const double normdy             = std::abs( err[1] ) * toleranceX_inv;
          const double errorOverTolerance = std::max( normdx, normdy );
          success                         = ( errorOverTolerance <= sigma );
          if ( !success ) ++numfailedstep;
          double stepfactor( maxStepScale );
          if ( errorOverTolerance > 1 ) {
            stepfactor = std::max( minStepScale, safetyFactor / std::sqrt( std::sqrt( errorOverTolerance ) ) );
          } else {
            if ( errorOverTolerance > 0 ) {
              stepfactor = std::min( maxStepScale, safetyFactor * rec::invfifthroot( errorOverTolerance ) );
            }
          }
          absstep *= stepfactor;
          absstep = std::max( minRKStep, std::min( absstep, maxRKStep ) );
        }

        if ( success ) {
          const double thisstep = state.z - prevstate.z;
          if ( with_jacobian ) evaluateRKStepJacobian( thisstep, xm, tm, rkcache );
          ++rkcache.step;
        } else {
          state = prevstate;
        }

        const double z_diff = zout - state.z;
        if ( absstep - ( direction * z_diff ) > 0 ) absstep = std::abs( z_diff );

        if ( std::max( std::abs( state.tx ), std::abs( state.ty ) ) > maxSlope ) {
          ok = false;
        } else if ( std::abs( state.qop * rkcache.stage[0].B[1] ) > maxCurvature ) {
          // q/p here holds the unit factor of the reader, so the product is that of the original
          ok = false;
        } else if ( ( numfailedstep + static_cast<std::size_t>( rkcache.step ) ) >= maxNumRKSteps ) {
          ok = false;
        }
      }
      last_steps   = ok ? rkcache.step : -1; // -1: the original returns a failure
      last_failed  = static_cast<int>( numfailedstep );
      last_success = ok;
      if ( with_jacobian ) {
        for ( int c = 0; c < 3; ++c ) {
          last_jacobian[0][c] = xm[0][c];
          last_jacobian[1][c] = xm[1][c];
          last_jacobian[2][c] = tm[0][c];
          last_jacobian[3][c] = tm[1][c];
        }
      }
      if ( !ok ) return in; // the original leaves the state as it was and returns a failure
      return { static_cast<R>( state.x ), static_cast<R>( state.y ), static_cast<R>( state.tx ),
               static_cast<R>( state.ty ), in.qop };
    }
  };

} // namespace fx
