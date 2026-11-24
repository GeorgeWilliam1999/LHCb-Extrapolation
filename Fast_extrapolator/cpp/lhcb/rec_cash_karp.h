// A copy of the adaptive Runge-Kutta integrator of the HLT2 software.
//
// Source: Rec v41r3, Tr/TrackExtrapolators/src/TrackRungeKuttaExtrapolator.cpp:
// `extrapolate`, `evaluateStep` and `evaluateDerivatives`, with the default
// properties of the tool:
//
//   RKScheme CashKarp (6 stages, orders 5 and 4)     Tolerance 0.001 mm
//   RelToleranceTx 5e-5      MinStep 10 mm           MaxStep 1 m
//   InitialStep 1 m          Sigma 5.5               MinStepScale 0.125
//   MaxStepScale 4           StepScaleSafetyFactor 1 CorrectNumSteps false
//
// TrackMasterExtrapolator, which the HLT2 Kalman fit calls, uses this tool for
// every propagation longer than 100 mm. The first stage of a failed step is
// used again, as in the original. The growth of the step uses
// FastRoots::invfifthroot of LHCbMath (LHCb v60r3), copied below. Material and
// energy loss belong to TrackMasterExtrapolator and are not part of this
// integrator.
//
// The transport matrix (evaluateStepJacobian and evaluateDerivativesJacobian
// of the original) is computed when `with_jacobian` is set, for every accepted
// step, as the original does when the caller asks for it. The Kalman fit asks
// for it. The original takes no derivative of the field with respect to the
// position, and neither does this copy.
//
// The state of this file is the state of the experiment: its q/p is the q/p
// of LHCb in 1/MeV times c_light, which is what the original forms on entry.
// With the field read of LHCb (RecField, Gaudi units) the factor in front of
// q/p is exactly 1, and every operation is that of the original. With a reader
// that gives tesla, the factor kappa = 1e-3 is folded into q/p.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "../classical/integrators.h"

namespace fx {

  namespace rec {

    // FastRoots::invfifthroot( double ) of LHCbMath.
    inline double invfifthroot( double x ) noexcept {
      const double xa = std::abs( x );
      std::int64_t i;
      std::memcpy( &i, &xa, sizeof( i ) );
      i = 0x4cb8bc2ffc470ddb - i / 5;
      double r;
      std::memcpy( &r, &i, sizeof( r ) );
      r += ( r - xa * ( ( r * r ) * ( r * r ) * ( r * r ) ) ) / 5;
      r += ( r - xa * ( ( r * r ) * ( r * r ) * ( r * r ) ) ) / 5;
      r += ( r - xa * ( ( r * r ) * ( r * r ) * ( r * r ) ) ) / 5;
      r += ( r - xa * ( ( r * r ) * ( r * r ) * ( r * r ) ) ) / 5;
      return std::copysign( r, x );
    }

    // The Butcher tableau CashKarp of the original.
    constexpr int    NStages = 6;
    constexpr double a[15]   = { 1. / 5,                                                              //
                                 3. / 40,       9. / 40,                                              //
                                 3. / 10,       -9. / 10,   6. / 5,                                   //
                                 -11. / 54,     5. / 2,     -70. / 27,    35. / 27,                   //
                                 1631. / 55296, 175. / 512, 575. / 13824, 44275. / 110592, 253. / 4096 };
    constexpr double b[6]      = { 37. / 378, 0., 250. / 621, 125. / 594, 0., 512. / 1771 };
    constexpr double b_star[6] = { 2825. / 27648, 0., 18575. / 48384, 13525. / 55296, 277. / 14336, 1. / 4 };

    struct Vec4 {
      double v[4];
    };

  } // namespace rec

  template <class Field>
  struct RecCashKarp {
    const Field* field;
    double       z_in, z_out;
    // the properties of the tool
    double toleranceX     = 0.001;
    double relToleranceTx = 5e-5;
    double minRKStep      = 10.;
    double maxRKStep      = 1000.;
    double initialRKStep  = 1000.;
    double sigma          = 5.5;
    double minStepScale   = 0.125;
    double maxStepScale   = 4.0;
    double safetyFactor   = 1.0;
    // With the transport matrix, as the Kalman fit asks for it. The matrix is
    // the 4 x 3 block of the original: the derivatives of (x, y, tx, ty) with
    // respect to (tx0, ty0, q/p), with q/p in the unit of the integration.
    bool           with_jacobian = false;
    mutable int    last_steps = 0, last_failed = 0;
    mutable double last_jacobian[4][3] = {};

    struct Stage {
      rec::Vec4 state;
      double    B[3];
      rec::Vec4 derivative;
    };

    // evaluateDerivatives: (tx, ty, qop Ax, qop Ay); the field of the stage is kept
    inline void evaluate( Stage& stage, double z, double qop ) const {
      ( *field )( stage.state.v[0], stage.state.v[1], z, stage.B[0], stage.B[1], stage.B[2] );
      const double Bx = stage.B[0], By = stage.B[1], Bz = stage.B[2];
      const double tx = stage.state.v[2], ty = stage.state.v[3];
      const double tx2 = tx * tx, ty2 = ty * ty;
      const double norm = std::sqrt( 1 + tx2 + ty2 );
      const double Ax   = norm * ( ty * ( tx * Bx + Bz ) - ( 1 + tx2 ) * By );
      const double Ay   = norm * ( -tx * ( ty * By + Bz ) + ( 1 + ty2 ) * Bx );
      stage.derivative  = { { tx, ty, qop * Ax, qop * Ay } };
    }

    // evaluateDerivativesJacobian: the derivative along z of the 4 x 3 matrix `jacobian`
    static inline void derivative_of_the_jacobian( const Stage& stage, double qop, const double jacobian[4][3],
                                                   double out[4][3] ) {
      const double tx = stage.state.v[2];
      const double ty = stage.state.v[3];
      const double Bx = stage.B[0], By = stage.B[1], Bz = stage.B[2];

      const double tx2 = tx * tx;
      const double ty2 = ty * ty;

      const double n2 = 1 + tx2 + ty2;
      const double n  = std::sqrt( n2 );

      const double txBx = tx * Bx;
      const double txBy = tx * By;
      const double tyBy = ty * By;
      const double tyBx = ty * Bx;

      const double Ax = n * ( ty * ( txBx + Bz ) - ( 1 + tx2 ) * By );
      const double Ay = n * ( -tx * ( tyBy + Bz ) + ( 1 + ty2 ) * Bx );

      const double Ax_n2 = Ax / n2;
      const double Ay_n2 = Ay / n2;

      const double dAxdTx = Ax_n2 * tx + n * ( tyBx - 2 * txBy );
      const double dAxdTy = Ax_n2 * ty + n * ( txBx + Bz );

      const double dAydTx = Ay_n2 * tx + n * ( -tyBy - Bz );
      const double dAydTy = Ay_n2 * ty + n * ( -txBy + 2 * tyBx );

      // columns: Tx0, Ty0, QoP0; rows: X, Y, Tx, Ty
      for ( int c = 0; c < 3; ++c ) {
        out[0][c] = jacobian[2][c];
        out[1][c] = jacobian[3][c];
        out[2][c] = qop * ( jacobian[2][c] * dAxdTx + jacobian[3][c] * dAxdTy );
        out[3][c] = qop * ( jacobian[2][c] * dAydTx + jacobian[3][c] * dAydTy );
      }
      out[2][2] = Ax + qop * ( jacobian[2][2] * dAxdTx + jacobian[3][2] * dAxdTy );
      out[3][2] = Ay + qop * ( jacobian[2][2] * dAydTx + jacobian[3][2] * dAydTy );
    }

    // evaluateStepJacobian for the six stages of the scheme
    static inline void step_of_the_jacobian( double dz, double qop, double jacobian[4][3], const Stage stages[] ) {
      using namespace rec;
      double k[NStages][4][3];
      for ( int m = 0; m != NStages; ++m ) {
        double        input[4][3];
        const double* row = a + m * ( m - 1 ) / 2;
        for ( int r = 0; r < 4; ++r )
          for ( int c = 0; c < 3; ++c ) {
            double sum = jacobian[r][c];
            for ( int j = 0; j < m; ++j ) sum += row[j] * k[j][r][c];
            input[r][c] = sum;
          }
        double d[4][3];
        derivative_of_the_jacobian( stages[m], qop, input, d );
        for ( int r = 0; r < 4; ++r )
          for ( int c = 0; c < 3; ++c ) k[m][r][c] = dz * d[r][c];
      }
      for ( int m = 0; m != NStages; ++m )
        for ( int r = 0; r < 4; ++r )
          for ( int c = 0; c < 3; ++c ) jacobian[r][c] += b[m] * k[m][r][c];
    }

    template <class R>
    inline State<R> operator()( const State<R>& in ) const {
      using namespace rec;
      Vec4         state{ { static_cast<double>( in.x ), static_cast<double>( in.y ), static_cast<double>( in.tx ),
                            static_cast<double>( in.ty ) } };
      const double qop = ( KAPPA * Field::tesla_per_unit ) * static_cast<double>( in.qop );
      double       z   = z_in;
      const double zout = z_out;

      double jacobian[4][3] = {};
      jacobian[2][0]        = 1; // dTxdTx0
      jacobian[3][1]        = 1; // dTydTy0

      const double totalStep   = zout - z;
      const double toleranceTx = toleranceX / std::abs( totalStep );
      double       absstep     = std::min( std::abs( totalStep ), initialRKStep );
      const int    direction   = totalStep > 0 ? +1 : -1;

      Stage stages[NStages];
      bool  reuse_first = false; // the first stage of a step that failed is used again
      int   steps = 0, failed = 0;

      while ( std::abs( z - zout ) > 1e-6 ) { // TrackParameters::propagationTolerance
        const double dz = absstep * direction;
        Vec4         k[NStages];
        if ( !reuse_first ) {
          stages[0].state = state;
          evaluate( stages[0], z, qop );
        }
        for ( int c = 0; c < 4; ++c ) k[0].v[c] = dz * stages[0].derivative.v[c];
        for ( int m = 1; m != NStages; ++m ) {
          Vec4          stage = state;
          double        stage_z = z;
          const double* row = a + m * ( m - 1 ) / 2;
          for ( int j = 0; j < m; ++j ) {
            for ( int c = 0; c < 4; ++c ) stage.v[c] += row[j] * k[j].v[c];
            stage_z += row[j] * dz;
          }
          stages[m].state = stage;
          evaluate( stages[m], stage_z, qop );
          for ( int c = 0; c < 4; ++c ) k[m].v[c] = dz * stages[m].derivative.v[c];
        }
        Vec4 err{ { 0, 0, 0, 0 } };
        Vec4 next = state;
        for ( int m = 0; m != NStages; ++m ) {
          for ( int c = 0; c < 4; ++c ) {
            err.v[c] += ( b[m] - b_star[m] ) * k[m].v[c];
            next.v[c] += b[m] * k[m].v[c];
          }
        }

        bool success = ( absstep <= minRKStep );
        if ( !success ) {
          const double normdx  = std::abs( err.v[0] ) / toleranceX;
          const double normdy  = std::abs( err.v[1] ) / toleranceX;
          const double deltatx = next.v[2] - state.v[2];
          const double normdtx = std::abs( err.v[2] ) / ( toleranceTx + std::abs( deltatx ) * relToleranceTx );
          const double errorOverTolerance = std::max( normdx, std::max( normdy, normdtx ) );
          success                         = ( errorOverTolerance <= sigma );
          double stepfactor( 1 );
          if ( errorOverTolerance > 1 ) {
            stepfactor = std::max( minStepScale, safetyFactor / std::sqrt( std::sqrt( errorOverTolerance ) ) );
          } else {
            if ( errorOverTolerance > 0 ) {
              stepfactor = std::min( maxStepScale, safetyFactor * rec::invfifthroot( errorOverTolerance ) );
            } else {
              stepfactor = maxStepScale;
            }
          }
          absstep = std::max( minRKStep, std::min( absstep * stepfactor, maxRKStep ) );
        }
        if ( success ) {
          if ( with_jacobian ) step_of_the_jacobian( dz, qop, jacobian, stages );
          state = next;
          z += dz;
          ++steps;
          reuse_first = false;
        } else {
          ++failed;
          reuse_first = true;
        }
        if ( absstep > direction * ( zout - z ) ) absstep = std::abs( zout - z );
      }
      last_steps  = steps;
      last_failed = failed;
      if ( with_jacobian )
        for ( int r = 0; r < 4; ++r )
          for ( int c = 0; c < 3; ++c ) last_jacobian[r][c] = jacobian[r][c];
      return { static_cast<R>( state.v[0] ), static_cast<R>( state.v[1] ), static_cast<R>( state.v[2] ),
               static_cast<R>( state.v[3] ), in.qop };
    }
  };

} // namespace fx
