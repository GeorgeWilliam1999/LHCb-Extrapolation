// The pieces that the adaptive RKN of Allen and the magnet-crossing tool share.
//
// Units are those of Allen's AdaptiveRKNystromExtrapolator: positions in mm, q/p
// in 1/MeV on the outside and q/p * c_light on the inside, the field in the unit
// of the LHCb field grid (Gaudi: 1 T = 1e-3). With these units the slope
// derivative needs no other factor.
//
// A field is any object with
//     void operator()(R x, R y, R z, R& bx, R& by, R& bz) const
// in the Gaudi unit. In Allen it is a thin adapter of
// MagneticField::Magfield::fieldVectorLinearInterpolation.
//
// A Jacobian is R J[5][5], J[i][j] = d out_i / d in_j, rows and columns
// (x, y, tx, ty, q/p), as Extrapolators::State::Jacobian.
//
// Everything is a plain function; with nvcc or hipcc it is __host__ __device__.
#pragma once

#include <cmath>

#ifndef ALLEN_NN_HD
#if defined( __CUDACC__ ) || defined( __HIPCC__ )
#define ALLEN_NN_HD __host__ __device__
#else
#define ALLEN_NN_HD
#endif
#endif

namespace allen_nn {

  // The float overloads of the math functions (a bare ::pow or ::sqrt can be the
  // double one on the host, which rounds differently from Allen's powf, sqrtf).
  using std::ceil;
  using std::exp;
  using std::fabs;
  using std::fmax;
  using std::fmin;
  using std::pow;
  using std::sqrt;
  using std::tanh;

  // Extrapolators::c_light of Allen, rounded to float as Allen rounds it.
  constexpr float c_light = 299.792458f;

  template <class R>
  ALLEN_NN_HD inline void set_identity( R J[5][5] ) {
    for ( int i = 0; i < 5; ++i )
      for ( int j = 0; j < 5; ++j ) J[i][j] = ( i == j ) ? R( 1 ) : R( 0 );
  }

  // C = A B
  template <class R>
  ALLEN_NN_HD inline void multiply( const R A[5][5], const R B[5][5], R C[5][5] ) {
    for ( int i = 0; i < 5; ++i )
      for ( int j = 0; j < 5; ++j ) {
        R sum = 0;
        for ( int m = 0; m < 5; ++m ) sum += A[i][m] * B[m][j];
        C[i][j] = sum;
      }
  }

  // slope_derivative of ExtrapolatorCommon.cuh (q/p already times c_light)
  template <class R>
  ALLEN_NN_HD inline void slope_derivative( R tx, R ty, R qop, const R B[3], R& dtx, R& dty ) {
    const R tx2  = tx * tx;
    const R ty2  = ty * ty;
    const R norm = sqrt( R( 1 ) + tx2 + ty2 );
    const R ax   = norm * ( ty * ( tx * B[0] + B[2] ) - ( R( 1 ) + tx2 ) * B[1] );
    const R ay   = norm * ( -tx * ( ty * B[1] + B[2] ) + ( R( 1 ) + ty2 ) * B[0] );
    dtx          = qop * ax;
    dty          = qop * ay;
  }

  // State::Jacobian::JacobianDerivatives of ExtrapolatorCommon.cuh, term by term.
  // m[2][3] are the 2 x 3 blocks (rows tx, ty or x, y; columns tx0, ty0, q/p).
  template <class R>
  ALLEN_NN_HD inline void jacobian_derivatives( R tx_matrix[2][3], R pos_jac[2][3], R slope_jac[2][3],
                                                const int stage, const R dz, const R sx, const R sy, const R qop,
                                                const R B[3] ) {
    const R norm    = sqrt( R( 1 ) + sx * sx + sy * sy );
    const R factorY = sx * sy * B[0] - ( R( 1 ) + sx * sx ) * B[1] + sy * B[2];
    const R factorX = ( R( 1 ) + sy * sy ) * B[0] - sx * sy * B[1] - sx * B[2];

    const R factorXDerivTx = sx / norm * factorX - norm * ( sy * B[1] + B[2] );
    const R factorXDerivTy = sy / norm * factorX + norm * ( 2 * sy * B[0] - sx * B[1] );
    const R factorYDerivTx = sx / norm * factorY + norm * ( sy * B[0] - 2 * sx * B[1] );
    const R factorYDerivTy = sy / norm * factorY + norm * ( sx * B[0] + B[2] );

    const R a11 = qop * ( factorYDerivTx * tx_matrix[0][0] + factorYDerivTy * tx_matrix[1][0] );
    const R a21 = qop * ( factorXDerivTx * tx_matrix[0][0] + factorXDerivTy * tx_matrix[1][0] );
    const R a12 = qop * ( factorYDerivTx * tx_matrix[0][1] + factorYDerivTy * tx_matrix[1][1] );
    const R a22 = qop * ( factorXDerivTx * tx_matrix[0][1] + factorXDerivTy * tx_matrix[1][1] );
    const R e1  = factorY * norm + qop * ( factorYDerivTx * tx_matrix[0][2] + factorYDerivTy * tx_matrix[1][2] );
    const R e2  = factorX * norm + qop * ( factorXDerivTx * tx_matrix[0][2] + factorXDerivTy * tx_matrix[1][2] );

    if ( stage != 3 ) {
      const R f = dz * dz / R( 6 );
      pos_jac[0][0] += f * a11;
      pos_jac[0][1] += f * a12;
      pos_jac[0][2] += f * e1;
      pos_jac[1][0] += f * a21;
      pos_jac[1][1] += f * a22;
      pos_jac[1][2] += f * e2;
      const R prefactor = ( stage == 0 ) || ( stage == 1 ) ? R( 0.5 ) : R( 1 );
      tx_matrix[0][0]   = prefactor * dz * a11;
      tx_matrix[0][1]   = prefactor * dz * a12;
      tx_matrix[0][2]   = prefactor * dz * e1;
      tx_matrix[1][0]   = prefactor * dz * a21;
      tx_matrix[1][1]   = prefactor * dz * a22;
      tx_matrix[1][2]   = prefactor * dz * e2;
    }
    const R g = ( dz / R( 6 ) ) * ( ( stage == 1 ) || ( stage == 2 ) ? R( 2 ) : R( 1 ) );
    slope_jac[0][0] += g * a11;
    slope_jac[0][1] += g * a12;
    slope_jac[0][2] += g * e1;
    slope_jac[1][0] += g * a21;
    slope_jac[1][1] += g * a22;
    slope_jac[1][2] += g * e2;
  }

  // State::Jacobian::evaluateJacobianRKN: the Jacobian of one Nystrom step with the
  // field held at the three values of the step, added to the running J. It is
  // called with the slopes before the step and the k of the step.
  template <class R>
  ALLEN_NN_HD inline void evaluate_jacobian_rkn( R J[5][5], const R dz, const R tx, const R ty, const R k[4][2],
                                                 const R qop, const R B[3][3] ) {
    R pos_jac[2][3]   = {};
    R slope_jac[2][3] = {};
    R orig[2][3], t[2][3];
    for ( int r = 0; r < 2; ++r )
      for ( int c = 0; c < 3; ++c ) orig[r][c] = t[r][c] = J[2 + r][2 + c];
    jacobian_derivatives( t, pos_jac, slope_jac, 0, dz, tx, ty, qop, B[0] );
    for ( int r = 0; r < 2; ++r )
      for ( int c = 0; c < 3; ++c ) t[r][c] += orig[r][c];
    jacobian_derivatives( t, pos_jac, slope_jac, 1, dz, tx + k[0][0] * R( 0.5 ) * dz, ty + k[0][1] * R( 0.5 ) * dz, qop, B[1] );
    for ( int r = 0; r < 2; ++r )
      for ( int c = 0; c < 3; ++c ) t[r][c] += orig[r][c];
    jacobian_derivatives( t, pos_jac, slope_jac, 2, dz, tx + k[1][0] * R( 0.5 ) * dz, ty + k[1][1] * R( 0.5 ) * dz, qop, B[1] );
    for ( int r = 0; r < 2; ++r )
      for ( int c = 0; c < 3; ++c ) t[r][c] += orig[r][c];
    jacobian_derivatives( t, pos_jac, slope_jac, 3, dz, tx + k[2][0] * dz, ty + k[2][1] * dz, qop, B[2] );
    for ( int r = 0; r < 2; ++r )
      for ( int c = 0; c < 3; ++c ) {
        J[r][2 + c] += pos_jac[r][c] + dz * orig[r][c];
        J[2 + r][2 + c] += slope_jac[r][c];
      }
  }

  // One Nystrom step of length dz from the state s (x, y, tx, ty, q/p * c_light) at z,
  // three field points: B[0] at the start (given), B[1] at the middle, B[2] at the end
  // (evaluateRKStep of RungeKuttaExtrapolator.cuh). k receives the four slope derivatives.
  // Returns the error estimate max(|err_x|, |err_y|) of the ARKN.
  template <class R, class Field>
  ALLEN_NN_HD inline R nystrom_stages( const R s[5], const R z, const R dz, R B[3][3], R k[4][2],
                                       const Field& field ) {
    const R dz2    = dz * dz;
    const R halfdz = R( 0.5 ) * dz;
    slope_derivative( s[2], s[3], s[4], B[0], k[0][0], k[0][1] );
    field( s[0] + halfdz * s[2] + R( 0.125 ) * dz2 * k[0][0], s[1] + halfdz * s[3] + R( 0.125 ) * dz2 * k[0][1],
           z + halfdz, B[1][0], B[1][1], B[1][2] );
    for ( int stage = 1; stage < 3; ++stage )
      slope_derivative( s[2] + halfdz * k[stage - 1][0], s[3] + halfdz * k[stage - 1][1], s[4], B[1], k[stage][0],
                        k[stage][1] );
    field( s[0] + dz * s[2] + R( 0.5 ) * dz2 * k[2][0], s[1] + dz * s[3] + R( 0.5 ) * dz2 * k[2][1], z + dz, B[2][0],
           B[2][1], B[2][2] );
    slope_derivative( s[2] + dz * k[2][0], s[3] + dz * k[2][1], s[4], B[2], k[3][0], k[3][1] );
    const R ex = ( k[0][0] - k[1][0] - k[2][0] + k[3][0] ) * dz2;
    const R ey = ( k[0][1] - k[1][1] - k[2][1] + k[3][1] ) * dz2;
    return fabs( ex ) > fabs( ey ) ? fabs( ex ) : fabs( ey );
  }

  // evaluateStateRKN: the state update of an accepted step.
  template <class R>
  ALLEN_NN_HD inline void nystrom_update( R s[5], const R dz, const R k[4][2] ) {
    const R fpos = dz * dz * ( R( 1 ) / R( 6 ) );
    const R fslo = dz * ( R( 1 ) / R( 6 ) );
    s[0]         = s[0] + dz * s[2] + ( k[0][0] + k[1][0] + k[2][0] ) * fpos;
    s[1]         = s[1] + dz * s[3] + ( k[0][1] + k[1][1] + k[2][1] ) * fpos;
    s[2] += ( k[0][0] + k[1][0] * R( 2 ) + k[2][0] * R( 2 ) + k[3][0] ) * fslo;
    s[3] += ( k[0][1] + k[1][1] * R( 2 ) + k[2][1] * R( 2 ) + k[3][1] ) * fslo;
  }

} // namespace allen_nn
