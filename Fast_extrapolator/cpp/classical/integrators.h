// The state, the equation of motion and the fixed-step classical integrators.
//
// A state is (x, y, tx, ty, q/p) on a plane of constant z: x and y in mm, tx
// and ty the slopes dx/dz and dy/dz, and q/p = 0.299792458 * charge / momentum
// in GeV (the q/p of LHCb in 1/MeV times c_light). With the field in tesla,
//
//   dtx/dz = kappa (q/p) sqrt(1 + tx^2 + ty^2) (tx ty Bx - (1 + tx^2) By + ty Bz)
//   dty/dz = kappa (q/p) sqrt(1 + tx^2 + ty^2) ((1 + ty^2) Bx - tx ty By - tx Bz)
//
// with kappa = 1e-3. A field reader states its unit (`tesla_per_unit`), and
// every integrator folds kappa * tesla_per_unit into q/p. The arithmetic of
// `slope_rates` and of `RK4` follows the present Python package term by term
// (equation_of_motion/lhcb.py and evaluation/time_per_event.py,
// rk4_whole_track), so that the compiled result equals the Python result to
// rounding.
#pragma once

#include <cmath>

namespace fx {

  constexpr double KAPPA = 1.0e-3;

  template <class R>
  struct State {
    R x, y, tx, ty, qop;
  };

  // The rates of the two slopes at one point; k = kappa * q/p.
  template <class R, class Field>
  inline void slope_rates( const Field& field, R x, R y, R tx, R ty, R k, R z, R& dtx, R& dty ) {
    R bx, by, bz;
    field( x, y, z, bx, by, bz );
    const R N = std::sqrt( 1 + tx * tx + ty * ty );
    dtx       = k * N * ( tx * ty * bx - ( 1 + tx * tx ) * by + ty * bz );
    dty       = k * N * ( ( 1 + ty * ty ) * bx - tx * ty * by - tx * bz );
  }

  // Classic fourth-order Runge-Kutta with `steps` equal steps from z0 to z1: 4 field reads a step.
  template <class R, class Field>
  struct RK4 {
    const Field* field;
    R            z0, z1;
    int          steps;
    mutable int  last_steps = 0, last_failed = 0;

    inline State<R> operator()( State<R> s ) const {
      const R h = ( z1 - z0 ) / steps;
      const R k = static_cast<R>( KAPPA * Field::tesla_per_unit ) * s.qop;
      for ( int n = 0; n < steps; ++n ) {
        const R z = z0 + n * h;
        R       a1x, a1y, a2x, a2y, a3x, a3y, a4x, a4y;
        // stage 1
        slope_rates( *field, s.x, s.y, s.tx, s.ty, k, z, a1x, a1y );
        // stage 2
        const R half = R( 0.5 ) * h;
        const R tx2 = s.tx + half * a1x, ty2 = s.ty + half * a1y;
        slope_rates( *field, s.x + half * s.tx, s.y + half * s.ty, tx2, ty2, k, z + half, a2x, a2y );
        // stage 3
        const R tx3 = s.tx + half * a2x, ty3 = s.ty + half * a2y;
        slope_rates( *field, s.x + half * tx2, s.y + half * ty2, tx3, ty3, k, z + half, a3x, a3y );
        // stage 4
        const R tx4 = s.tx + h * a3x, ty4 = s.ty + h * a3y;
        slope_rates( *field, s.x + h * tx3, s.y + h * ty3, tx4, ty4, k, z + h, a4x, a4y );
        const R sixth = h / R( 6.0 );
        const R nx = s.x + sixth * ( s.tx + 2 * tx2 + 2 * tx3 + tx4 );
        const R ny = s.y + sixth * ( s.ty + 2 * ty2 + 2 * ty3 + ty4 );
        const R ntx = s.tx + sixth * ( a1x + 2 * a2x + 2 * a3x + a4x );
        const R nty = s.ty + sixth * ( a1y + 2 * a2y + 2 * a3y + a4y );
        s.x = nx, s.y = ny, s.tx = ntx, s.ty = nty;
      }
      last_steps = steps;
      return s;
    }
  };

} // namespace fx
