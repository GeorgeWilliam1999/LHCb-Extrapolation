// The magnet crossing of the Kalman fit: K fixed Nystrom steps plus one network
// correction, with its transport matrix. A drop-in for
// AdaptiveRKNystromExtrapolator::propagate_jac on the crossing from the last UT
// hit to the first SciFi hit ("way A"):
//
//     z_in  --leg-->  z0 = 2648.2  --crossing-->  z1 = 7826  --leg-->  z_out
//
// The legs are fixed Nystrom steps of at most `leg_step` mm (the hits are a few
// tens of mm from the planes, so a leg is one step). The crossing is
//
//     s_K  = K equal Nystrom steps from z0 to z1 (the field read as the ARKN reads it)
//     end  = s_K + sigma_out * (q/p / sigma_qop) * NN(s0 / sigma_in)
//
// with NN a tanh network (depth hidden layers of `width`, a linear last layer),
// as trained by fastex/train.py (start "nystrom", scale "times_qop"); s0 is the
// state on z0 with q/p * c_light, the q/p of the experiment.
//
// The transport matrix is the product of the three parts. A Nystrom part is
// chained step by step with the field held fixed, as LHCb does
// (evaluate_jacobian_rkn); the network part is its exact derivative, all five
// columns. The q/p columns come back in 1/MeV.
//
// Polarity: the two LHCb maps are exact negatives of each other (v8r1, checked),
// so a network trained on one serves the other with q/p -> -q/p. The Nystrom
// parts read the real field and need nothing; `polarity` is -1 when the network
// was trained on the other polarity.
//
// Outside the domain of the training tracks (|q/p| above the largest of the
// training set, slopes or positions outside its box) the function returns false
// and changes nothing, so that the caller falls back on the ARKN.
#pragma once

#include "rkn_common.h"

namespace allen_nn {

  constexpr int MOST_WIDTH  = 64;
  constexpr int MOST_HIDDEN = 3;  // hidden layers
  constexpr int MOST_STEPS  = 16; // Nystrom steps of the crossing

  template <class R>
  struct CrossingNetwork {
    int steps = 0, depth = 0, width = 0;
    R   z0 = 0, z1 = 0;
    R   planes[MOST_STEPS + 1] = {};
    R   inv_scale_in[5]        = {}; // 1 / sigma_in
    R   scale_out[4]           = {};
    R   inv_scale_qop          = 0;
    // the box of the training tracks on z0 (largest absolute values); q/p * c_light
    R most_qop = 0, most_tx = 0, most_ty = 0, most_x = 0, most_y = 0;
    R leg_step = 250;
    // fall back on the ARKN when the error estimate of the ARKN, (k0 - k1 - k2 + k3) dz^2,
    // of any step of the crossing is above this (mm): the fixed steps then miss a feature
    // of the field (large |y| and |ty|, near the yoke). Infinite: off.
    R most_step_error = R( 1e30 );
    // the corner rule (George, 2026-11-27): when |y| of the straight line from the state on z0 at
    // z = corner_z is above corner_y, or the state is outside the box, the crossing is corner_steps
    // plain Nystrom steps (no network, no ARKN). corner_y infinite: off.
    R   corner_y = R( 1e30 ), corner_z = R( 5000 );
    int corner_steps = 48;
    // 1: the transport matrix has the derivative of the network; 0: the matrix of the K
    // Nystrom steps only (the network moves the end by micrometres; its derivative is far
    // below the error of the field-held-fixed chaining, and costs 4-5 network calls)
    int network_jacobian = 1;
    // weights, row-major [out][in]
    R W_first[MOST_WIDTH * 5]                         = {};
    R b_first[MOST_WIDTH]                             = {};
    R W_hidden[MOST_HIDDEN - 1][MOST_WIDTH * MOST_WIDTH] = {};
    R b_hidden[MOST_HIDDEN - 1][MOST_WIDTH]           = {};
    R W_last[4 * MOST_WIDTH]                          = {};
    R b_last[4]                                       = {};
  };

  // tanh from one exponential (|error| < 1e-7 in float): the tanh of the C library is
  // several times slower than exp on the host.
  template <class R>
  ALLEN_NN_HD inline R tanh_fast( const R a ) {
    return R( 1 ) - R( 2 ) / ( exp( R( 2 ) * a ) + R( 1 ) );
  }

  // b + sum_m W[m] h[m] with eight independent partial sums, so that the sum is not
  // one chain of dependent multiply-adds (the compiler keeps the order of a plain sum).
  template <class R>
  ALLEN_NN_HD inline R dot_plus( const R* W, const R* h, const int n, const R b ) {
    R   acc[8] = {};
    int m      = 0;
    for ( ; m + 8 <= n; m += 8 )
      for ( int k = 0; k < 8; ++k ) acc[k] += W[m + k] * h[m + k];
    for ( ; m < n; ++m ) acc[0] += W[m] * h[m];
    return b + ( ( ( acc[0] + acc[1] ) + ( acc[2] + acc[3] ) ) + ( ( acc[4] + acc[5] ) + ( acc[6] + acc[7] ) ) );
  }

  // The network and, when D is given, its derivative D[4][5] = d NN / d u.
  template <class R>
  ALLEN_NN_HD inline void network_forward( const CrossingNetwork<R>& net, const R u[5], R out[4], R ( *D )[5] ) {
    const int w = net.width;
    R         h[MOST_WIDTH], g[MOST_WIDTH];
    R         M[MOST_WIDTH][5], N[MOST_WIDTH][5]; // d h / d u
    for ( int i = 0; i < w; ++i ) {
      R a = net.b_first[i];
      for ( int j = 0; j < 5; ++j ) a += net.W_first[i * 5 + j] * u[j];
      h[i] = tanh_fast( a );
      if ( D ) {
        const R d = R( 1 ) - h[i] * h[i];
        for ( int j = 0; j < 5; ++j ) M[i][j] = d * net.W_first[i * 5 + j];
      }
    }
    for ( int l = 0; l < net.depth - 1; ++l ) {
      const R* W = net.W_hidden[l];
      for ( int i = 0; i < w; ++i ) {
        g[i] = tanh_fast( dot_plus( W + i * w, h, w, net.b_hidden[l][i] ) );
      }
      if ( D ) {
        for ( int i = 0; i < w; ++i ) {
          const R d = R( 1 ) - g[i] * g[i];
          for ( int j = 0; j < 5; ++j ) {
            R a = 0;
            for ( int m = 0; m < w; ++m ) a += W[i * w + m] * M[m][j];
            N[i][j] = d * a;
          }
        }
        for ( int i = 0; i < w; ++i )
          for ( int j = 0; j < 5; ++j ) M[i][j] = N[i][j];
      }
      for ( int i = 0; i < w; ++i ) h[i] = g[i];
    }
    for ( int o = 0; o < 4; ++o ) {
      out[o] = dot_plus( net.W_last + o * w, h, w, net.b_last[o] );
      if ( D )
        for ( int j = 0; j < 5; ++j ) {
          R b = 0;
          for ( int m = 0; m < w; ++m ) b += net.W_last[o * w + m] * M[m][j];
          D[o][j] = b;
        }
    }
  }

  // Fixed Nystrom steps on the planes p[0..n] (the field at the start in B0, the
  // field at the end left in B0). The state has q/p * c_light. J (or null) is the
  // running Jacobian of the part, chained as the ARKN chains it.
  // Returns the largest error estimate of the steps (mm).
  template <class R, class Field>
  ALLEN_NN_HD inline R nystrom_fixed( R s[5], const R* p, const int n, R B0[3], const Field& field, R ( *J )[5] ) {
    R B[3][3], k[4][2];
    R most = 0;
    for ( int c = 0; c < 3; ++c ) B[0][c] = B0[c];
    for ( int i = 0; i < n; ++i ) {
      const R dz  = p[i + 1] - p[i];
      const R err = nystrom_stages( s, p[i], dz, B, k, field );
      if ( err > most ) most = err;
      if ( J ) evaluate_jacobian_rkn( J, dz, s[2], s[3], k, s[4], B );
      nystrom_update( s, dz, k );
      for ( int c = 0; c < 3; ++c ) B[0][c] = B[2][c];
    }
    for ( int c = 0; c < 3; ++c ) B0[c] = B[0][c];
    return most;
  }

  // A leg from za to zb in equal steps of at most leg_step.
  template <class R, class Field>
  ALLEN_NN_HD inline void nystrom_leg( R s[5], const R za, const R zb, const R leg_step, R B0[3], const Field& field,
                                       R ( *J )[5] ) {
    const R length = zb - za;
    if ( fabs( length ) <= R( 1e-3 ) ) return;
    int n = static_cast<int>( ceil( fabs( length ) / leg_step ) );
    if ( n < 1 ) n = 1;
    if ( n > MOST_STEPS ) n = MOST_STEPS;
    R p[MOST_STEPS + 1];
    for ( int i = 0; i <= n; ++i ) p[i] = za + length * ( R( i ) / R( n ) );
    p[n] = zb;
    nystrom_fixed( s, p, n, B0, field, J );
  }

  template <class R>
  ALLEN_NN_HD inline bool in_domain( const CrossingNetwork<R>& net, const R s[5] ) {
    return fabs( s[4] ) <= net.most_qop && fabs( s[2] ) <= net.most_tx && fabs( s[3] ) <= net.most_ty &&
           fabs( s[0] ) <= net.most_x && fabs( s[1] ) <= net.most_y;
  }

  // The crossing alone, z0 -> z1. s has q/p * c_light; B0 is the field at the start
  // (read here when has_B0 is false). J (or null) receives d end / d s0 in the
  // units of s (q/p * c_light). Returns the largest step error estimate (mm).
  template <class R, class Field>
  ALLEN_NN_HD inline R crossing( const CrossingNetwork<R>& net, R s[5], R B0[3], bool has_B0, const Field& field,
                                    const R polarity, R ( *J )[5] ) {
    const R s0[5] = { s[0], s[1], s[2], s[3], s[4] };
    if ( !has_B0 ) field( s[0], s[1], net.z0, B0[0], B0[1], B0[2] );
    if ( J ) set_identity( J );
    const R step_error = nystrom_fixed( s, net.planes, net.steps, B0, field, J );

    const R q    = polarity * s0[4];
    const R u[5] = { s0[0] * net.inv_scale_in[0], s0[1] * net.inv_scale_in[1], s0[2] * net.inv_scale_in[2],
                     s0[3] * net.inv_scale_in[3], q * net.inv_scale_in[4] };
    R       out[4], D[4][5];
    network_forward( net, u, out, ( J && net.network_jacobian ) ? D : nullptr );
    const R f = q * net.inv_scale_qop;
    for ( int o = 0; o < 4; ++o ) s[o] += net.scale_out[o] * f * out[o];
    if ( J && net.network_jacobian ) {
      for ( int o = 0; o < 4; ++o ) {
        const R so = net.scale_out[o];
        for ( int j = 0; j < 4; ++j ) J[o][j] += so * f * D[o][j] * net.inv_scale_in[j];
        // d/dq of so * (q / sigma_qop) * NN(.., q / sigma_in), then dq/ds0[4] = polarity
        J[o][4] += polarity * so * ( net.inv_scale_qop * out[o] + f * D[o][4] * net.inv_scale_in[4] );
      }
    }
    return step_error;
  }

  // The crossing z0 -> z1 as corner_steps equal plain Nystrom steps (the corner rule), in chunks of
  // MOST_STEPS so that no long plane array is needed; J (or null) is d end / d s0.
  template <class R, class Field>
  ALLEN_NN_HD inline void plain_crossing( const CrossingNetwork<R>& net, R s[5], R B0[3], const Field& field,
                                          R ( *J )[5] ) {
    const int n = net.corner_steps;
    R         p[MOST_STEPS + 1];
    for ( int done = 0; done < n; done += MOST_STEPS ) {
      const int m = ( n - done < MOST_STEPS ) ? n - done : MOST_STEPS;
      for ( int i = 0; i <= m; ++i ) p[i] = net.z0 + ( net.z1 - net.z0 ) * ( R( done + i ) / R( n ) );
      nystrom_fixed( s, p, m, B0, field, J );
    }
  }

  // Way A with the interface of ARKN::propagate_jac: x is a Vector5 of Allen
  // (q/p in 1/MeV), J the 5 x 5 transport matrix (q/p columns in 1/MeV). Returns
  // false (and leaves x, J and fit_success alone) when the state on z0 is outside
  // the domain of the network or a step of the crossing has an error estimate above
  // most_step_error; the caller then calls the ARKN.
  template <class R, bool withJacobian, class Field>
  ALLEN_NN_HD inline bool propagate_magnet( const CrossingNetwork<R>& net, R x[5], R J[5][5], const R z_in,
                                            const R z_out, const Field& field, const R polarity, bool& fit_success ) {
    if ( !fit_success || ( x[4] != x[4] ) ) {
      fit_success = false;
      return true;
    }
    R s[5] = { x[0], x[1], x[2], x[3], x[4] * R( c_light ) };
    R B0[3];
    field( s[0], s[1], z_in, B0[0], B0[1], B0[2] );
    R J1[5][5], J2[5][5], J3[5][5], T[5][5];
    if ( withJacobian ) set_identity( J1 ), set_identity( J3 );
    nystrom_leg( s, z_in, net.z0, net.leg_step, B0, field, withJacobian ? J1 : nullptr );
    const bool corner_rule = net.corner_y < R( 1e29 );
    if ( corner_rule && ( !in_domain( net, s ) || fabs( s[1] + s[3] * ( net.corner_z - net.z0 ) ) > net.corner_y ) ) {
      if ( withJacobian ) set_identity( J2 );
      plain_crossing( net, s, B0, field, withJacobian ? J2 : nullptr );
    } else {
      if ( !in_domain( net, s ) ) return false;
      if ( crossing( net, s, B0, true, field, polarity, withJacobian ? J2 : nullptr ) > net.most_step_error ) return false;
    }
    nystrom_leg( s, net.z1, z_out, net.leg_step, B0, field, withJacobian ? J3 : nullptr );

    if ( !( fabs( s[2] ) <= R( 10 ) && fabs( s[3] ) <= R( 10 ) && s[0] == s[0] && s[1] == s[1] ) ) {
      fit_success = false;
      return true;
    }
    x[0] = s[0];
    x[1] = s[1];
    x[2] = s[2];
    x[3] = s[3];
    if ( withJacobian ) {
      multiply( J2, J1, T );
      multiply( J3, T, J );
      for ( int i = 0; i < 4; ++i ) J[i][4] *= R( c_light );
    }
    return true;
  }

} // namespace allen_nn
