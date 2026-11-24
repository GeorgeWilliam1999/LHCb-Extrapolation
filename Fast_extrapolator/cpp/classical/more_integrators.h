// More classical integrators, for the frontier that a network must beat.
//
//   Nystrom3<R, Field>   fourth-order Runge-Kutta-Nystrom with three field
//       points a step: the start, the middle and the end of the step. The end
//       of one step is the start of the next, so a crossing of N steps reads
//       the field 2 N + 1 times, against 4 N for RK4. The step is the step of
//       the adaptive integrator of Allen (evaluateRKStep and
//       evaluateStateJacobianRKN of RungeKuttaExtrapolator.cuh); here the
//       steps are fixed. The planes of the steps are given, so that steps of
//       unequal length can be used.
//   RK4Plan<R, Field>    classic RK4 on given planes.
//   StageSweeps<R, Field>  the Gauss-Legendre scheme with q stages on N equal
//       steps, solved with sweeps of its stage map and no network:
//
//           Y(0)   = the straight line from the start of the step
//           Y(k+1) = S0 + dz A f(Y(k))            k = 0 .. K - 1
//           end    = S0 + dz b f(Y(K))
//
//       It reads the field (K + 1) N q times. With many sweeps it is the
//       exact scheme, the floor of every network on that layout. With few it
//       is the stage network with a physics layer without its network.
#pragma once

#include <cmath>
#include <vector>

#include "integrators.h"

namespace fx {

  template <class R, class Field>
  struct Nystrom3 {
    const Field*   field;
    std::vector<R> planes; // the start plane, the end of every step
    mutable int    last_steps = 0, last_failed = 0;

    static inline void slopes( R tx, R ty, R qop, const R B[3], R& dtx, R& dty ) {
      const R tx2  = tx * tx;
      const R ty2  = ty * ty;
      const R norm = std::sqrt( R( 1 ) + tx2 + ty2 );
      dtx          = qop * ( norm * ( ty * ( tx * B[0] + B[2] ) - ( R( 1 ) + tx2 ) * B[1] ) );
      dty          = qop * ( norm * ( -tx * ( ty * B[1] + B[2] ) + ( R( 1 ) + ty2 ) * B[0] ) );
    }

    inline State<R> operator()( State<R> s ) const {
      const R   qop = static_cast<R>( KAPPA * Field::tesla_per_unit ) * s.qop;
      const int n   = static_cast<int>( planes.size() ) - 1;
      R         B0[3], B1[3], B2[3];
      ( *field )( s.x, s.y, planes[0], B0[0], B0[1], B0[2] );
      for ( int i = 0; i < n; ++i ) {
        const R z = planes[i], dz = planes[i + 1] - planes[i];
        const R dz2 = dz * dz, half = R( 0.5 ) * dz;
        R       k0x, k0y, k1x, k1y, k2x, k2y, k3x, k3y;
        slopes( s.tx, s.ty, qop, B0, k0x, k0y );
        ( *field )( s.x + half * s.tx + R( 0.125 ) * dz2 * k0x, s.y + half * s.ty + R( 0.125 ) * dz2 * k0y, z + half,
                    B1[0], B1[1], B1[2] );
        slopes( s.tx + half * k0x, s.ty + half * k0y, qop, B1, k1x, k1y );
        slopes( s.tx + half * k1x, s.ty + half * k1y, qop, B1, k2x, k2y );
        ( *field )( s.x + dz * s.tx + R( 0.5 ) * dz2 * k2x, s.y + dz * s.ty + R( 0.5 ) * dz2 * k2y, z + dz, B2[0],
                    B2[1], B2[2] );
        slopes( s.tx + dz * k2x, s.ty + dz * k2y, qop, B2, k3x, k3y );
        const R fpos = dz2 * ( R( 1 ) / R( 6 ) ), fslo = dz * ( R( 1 ) / R( 6 ) );
        s.x          = s.x + dz * s.tx + ( k0x + k1x + k2x ) * fpos;
        s.y          = s.y + dz * s.ty + ( k0y + k1y + k2y ) * fpos;
        s.tx += ( k0x + k1x * R( 2 ) + k2x * R( 2 ) + k3x ) * fslo;
        s.ty += ( k0y + k1y * R( 2 ) + k2y * R( 2 ) + k3y ) * fslo;
        B0[0] = B2[0], B0[1] = B2[1], B0[2] = B2[2];
      }
      last_steps = n;
      return s;
    }
  };

  template <class R, class Field>
  struct RK4Plan {
    const Field*   field;
    std::vector<R> planes;
    mutable int    last_steps = 0, last_failed = 0;

    inline State<R> operator()( State<R> s ) const {
      const R   k = static_cast<R>( KAPPA * Field::tesla_per_unit ) * s.qop;
      const int n = static_cast<int>( planes.size() ) - 1;
      for ( int i = 0; i < n; ++i ) {
        const R z = planes[i], h = planes[i + 1] - planes[i];
        R       a1x, a1y, a2x, a2y, a3x, a3y, a4x, a4y;
        slope_rates( *field, s.x, s.y, s.tx, s.ty, k, z, a1x, a1y );
        const R half = R( 0.5 ) * h;
        const R tx2 = s.tx + half * a1x, ty2 = s.ty + half * a1y;
        slope_rates( *field, s.x + half * s.tx, s.y + half * s.ty, tx2, ty2, k, z + half, a2x, a2y );
        const R tx3 = s.tx + half * a2x, ty3 = s.ty + half * a2y;
        slope_rates( *field, s.x + half * tx2, s.y + half * ty2, tx3, ty3, k, z + half, a3x, a3y );
        const R tx4 = s.tx + h * a3x, ty4 = s.ty + h * a3y;
        slope_rates( *field, s.x + h * tx3, s.y + h * ty3, tx4, ty4, k, z + h, a4x, a4y );
        const R sixth = h / R( 6.0 );
        const R nx = s.x + sixth * ( s.tx + 2 * tx2 + 2 * tx3 + tx4 );
        const R ny = s.y + sixth * ( s.ty + 2 * ty2 + 2 * ty3 + ty4 );
        const R ntx = s.tx + sixth * ( a1x + 2 * a2x + 2 * a3x + a4x );
        const R nty = s.ty + sixth * ( a1y + 2 * a2y + 2 * a3y + a4y );
        s.x = nx, s.y = ny, s.tx = ntx, s.ty = nty;
      }
      last_steps = n;
      return s;
    }
  };

  constexpr int MOST_STAGES_OF_A_SWEEP = 16;

  template <class R, class Field>
  struct StageSweeps {
    const Field*   field;
    R              z0, z1;
    int            steps = 1, stages = 2, sweeps = 1;
    std::vector<R> nodes, weights, stage_matrix; // q, q, q * q: the Gauss-Legendre tableau
    mutable int    last_steps = 0, last_failed = 0;

    inline State<R> operator()( State<R> s ) const {
      const int q  = stages;
      const R   dz = ( z1 - z0 ) / steps;
      const R   k  = static_cast<R>( KAPPA * Field::tesla_per_unit ) * s.qop;
      R         Y[MOST_STAGES_OF_A_SWEEP][4], F[MOST_STAGES_OF_A_SWEEP][4];
      for ( int n = 0; n < steps; ++n ) {
        const R start = z0 + n * dz;
        const R S0[4] = { s.x, s.y, s.tx, s.ty };
        for ( int j = 0; j < q; ++j ) {
          const R offset = nodes[j] * dz;
          Y[j][0] = s.x + s.tx * offset, Y[j][1] = s.y + s.ty * offset, Y[j][2] = s.tx, Y[j][3] = s.ty;
        }
        for ( int sweep = 0;; ++sweep ) {
          for ( int j = 0; j < q; ++j ) {
            F[j][0] = Y[j][2];
            F[j][1] = Y[j][3];
            slope_rates( *field, Y[j][0], Y[j][1], Y[j][2], Y[j][3], k, start + nodes[j] * dz, F[j][2], F[j][3] );
          }
          if ( sweep == sweeps ) break;
          for ( int j = 0; j < q; ++j )
            for ( int c = 0; c < 4; ++c ) {
              R sum = 0;
              for ( int m = 0; m < q; ++m ) sum += stage_matrix[j * q + m] * F[m][c];
              Y[j][c] = S0[c] + dz * sum;
            }
        }
        R end[4];
        for ( int c = 0; c < 4; ++c ) {
          R sum = 0;
          for ( int j = 0; j < q; ++j ) sum += weights[j] * F[j][c];
          end[c] = S0[c] + dz * sum;
        }
        s.x = end[0], s.y = end[1], s.tx = end[2], s.ty = end[3];
      }
      last_steps = steps;
      return s;
    }
  };

} // namespace fx
