// A start plus a correction: the form of the methods of stage 2.
//
//     end state = start(start state) + correction(start state)
//
// The start is the straight line (no field read) or the Nystrom integrator
// with N equal steps. The correction is a function of the start state alone:
//
//   polynomial   u_j = input_j / scale_j            j = x, y, tx, ty, q/p
//                correction_c = u_qop * sum over terms of
//                               a_c,term  L_a(u_x) L_b(u_y) L_c(u_tx) L_d(u_ty) L_e(u_qop)
//                with L the Legendre polynomials and a + b + c + d + e <= degree.
//                The terms are in the order of the nested loops a, b, c, d, e
//                (python/fastex/correction.py, `terms_of`). The factor u_qop
//                makes the correction zero for a track with q/p = 0, which is
//                a straight line for every start.
//   network      one call of a tanh network on the scaled start state; its
//                four outputs times a fixed scale, and times q/p over its
//                scale when the folder says so.
//
// A folder of <exports>/corrections/ holds one correction: meta, scale_of_inputs.f64,
// and coefficients.f64 (polynomial: 4 x terms, component by component) or the
// layers of the network and scale_of_outputs.f64.
#pragma once

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "../classical/more_integrators.h"
#include "../common/io.h"
#include "../networks/networks.h"

namespace fx {

  constexpr int HIGHEST_DEGREE_OF_A_CORRECTION = 15;

  // The polynomial is evaluated without a branch that depends on the data, and
  // with one multiplication a term.
  //
  // The products of the first four Legendre polynomials of a term are built
  // level by level, and each level is sorted by its total degree s:
  //     p2: L_a(u_x) L_b(u_y)             group s holds a = 0 .. s, b = s - a
  //     p3: p2 entry of degree s' <= s, times L_{s - s'}(u_tx)
  //     p4: p3 entry of degree s' <= s, times L_{s - s'}(u_ty)
  // so the group s of a level is the first groups of the level before, each
  // times one number: a few products of a vector with a number. A product is
  // written as whole vectors; what it writes past its own length is
  // overwritten by the next one.
  //
  // An entry of p4 of degree s belongs to the terms with e = 0 .. degree - s.
  // The 4 (degree + 1) sums g[e][component] = sum over the entries of
  // a_component,term * entry are laid out with e first, so the vector k of g
  // needs only the entries of p4 with s <= degree - e, a leading part of p4.
  // The correction is u_qop * sum over e of g[e][component] L_e(u_qop).
  //
  // The order of the additions differs from the order of the terms; the result
  // differs from python/fastex/correction.py in the last digits.
  template <class R>
  struct PolynomialCorrection {
    using V = typename simd<R>::V;
    static constexpr int N    = simd<R>::N;
    static constexpr int MOST = HIGHEST_DEGREE_OF_A_CORRECTION + 1;

    int                         degree = 0; // of the sum; the correction has one more power of q/p
    R                           inverse_scale[5];
    // the recurrence of the Legendre polynomials: L_{k+1} = first[k] u L_k - second[k] L_{k-1}
    R                           first[MOST], second[MOST];
    std::size_t                 terms = 0;
    std::vector<int>            start2, start3, start4; // where the group s of a level starts; one more for the end
    std::vector<std::vector<R>> weights;                // for the vector k of g: N numbers an entry of p4
    std::vector<int>            entries;                // the leading part of p4 that the vector k needs
    mutable std::vector<R>      p2, p3, p4;

    PolynomialCorrection() {
      for ( int k = 0; k < MOST; ++k ) {
        first[k]  = static_cast<R>( ( 2.0 * k + 1.0 ) / ( k + 1.0 ) );
        second[k] = static_cast<R>( k / ( k + 1.0 ) );
      }
    }

    static std::size_t number_of_terms( int degree ) {
      std::size_t n = 0;
      for ( int a = 0; a <= degree; ++a )
        for ( int b = 0; b <= degree - a; ++b )
          for ( int c = 0; c <= degree - a - b; ++c )
            for ( int d = 0; d <= degree - a - b - c; ++d ) n += static_cast<std::size_t>( degree - a - b - c - d + 1 );
      return n;
    }

    // `coefficients`: component c at c * terms, the terms in the order of the
    // nested loops a, b, c, d, e.
    void prepare( int the_degree, const std::vector<double>& coefficients ) {
      degree = the_degree;
      terms  = number_of_terms( degree );
      if ( coefficients.size() != 4 * terms ) throw std::runtime_error( "the coefficients do not fit the degree" );
      // the number of a term from its exponents, in the order of the nested loops
      const int                side = degree + 1;
      std::vector<std::size_t> number( static_cast<std::size_t>( side ) * side * side * side * side, 0 );
      const auto at = [side]( int a, int b, int c, int d, int e ) {
        return ( ( ( static_cast<std::size_t>( a ) * side + b ) * side + c ) * side + d ) * side + e;
      };
      std::size_t t = 0;
      for ( int a = 0; a <= degree; ++a )
        for ( int b = 0; b <= degree - a; ++b )
          for ( int c = 0; c <= degree - a - b; ++c )
            for ( int d = 0; d <= degree - a - b - c; ++d )
              for ( int e = 0; e <= degree - a - b - c - d; ++e ) number[at( a, b, c, d, e )] = t++;
      // the entries of each level, group by group
      struct Entry {
        int a, b, c, d;
      };
      std::vector<Entry> of2, of3, of4;
      start2.assign( 1, 0 );
      start3.assign( 1, 0 );
      start4.assign( 1, 0 );
      for ( int s = 0; s <= degree; ++s ) {
        for ( int a = 0; a <= s; ++a ) of2.push_back( { a, s - a, 0, 0 } );
        start2.push_back( static_cast<int>( of2.size() ) );
      }
      for ( int s = 0; s <= degree; ++s ) {
        for ( int lower = 0; lower <= s; ++lower )
          for ( int i = start2[lower]; i < start2[lower + 1]; ++i ) of3.push_back( { of2[i].a, of2[i].b, s - lower, 0 } );
        start3.push_back( static_cast<int>( of3.size() ) );
      }
      for ( int s = 0; s <= degree; ++s ) {
        for ( int lower = 0; lower <= s; ++lower )
          for ( int i = start3[lower]; i < start3[lower + 1]; ++i )
            of4.push_back( { of3[i].a, of3[i].b, of3[i].c, s - lower } );
        start4.push_back( static_cast<int>( of4.size() ) );
      }
      const std::size_t room = 2 * static_cast<std::size_t>( N );
      p2.assign( of2.size() + room, R( 0 ) );
      p3.assign( of3.size() + room, R( 0 ) );
      p4.assign( of4.size() + room, R( 0 ) );
      // the weights: slot q of g is e = q / 4, component = q % 4
      const int vectors = ( 4 * ( degree + 1 ) + N - 1 ) / N;
      weights.assign( static_cast<std::size_t>( vectors ), {} );
      entries.assign( static_cast<std::size_t>( vectors ), 0 );
      for ( int k = 0; k < vectors; ++k ) {
        const int lowest_e = ( k * N ) / 4;
        entries[k]         = ( lowest_e > degree ) ? 0 : start4[degree - lowest_e + 1];
        weights[k].assign( static_cast<std::size_t>( entries[k] ) * N, R( 0 ) );
        for ( int j = 0; j < entries[k]; ++j ) {
          const Entry& p = of4[static_cast<std::size_t>( j )];
          const int    s = p.a + p.b + p.c + p.d;
          for ( int lane = 0; lane < N; ++lane ) {
            const int e = ( k * N + lane ) / 4, component = ( k * N + lane ) % 4;
            if ( e > degree - s ) continue;
            weights[k][static_cast<std::size_t>( j ) * N + lane] = static_cast<R>(
                coefficients[static_cast<std::size_t>( component ) * terms + number[at( p.a, p.b, p.c, p.d, e )]] );
          }
        }
      }
    }

    // to[i] = from[i] * value for i < length, written as whole vectors
    static inline void times( R* to, const R* from, int length, R value ) {
      for ( int i = 0; i < length; i += N ) simd<R>::store( to + i, simd<R>::load( from + i ) * value );
    }

    // sum over j < m of w[j] (a vector) * x[j], as eight partial sums
    static inline V weighted_sum( const R* w, const R* x, int m ) {
      V   a0 = V{} + R( 0 ), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
      int j = 0;
      for ( ; j + 8 <= m; j += 8, w += 8 * N ) {
        a0 += simd<R>::load( w ) * x[j];
        a1 += simd<R>::load( w + N ) * x[j + 1];
        a2 += simd<R>::load( w + 2 * N ) * x[j + 2];
        a3 += simd<R>::load( w + 3 * N ) * x[j + 3];
        a4 += simd<R>::load( w + 4 * N ) * x[j + 4];
        a5 += simd<R>::load( w + 5 * N ) * x[j + 5];
        a6 += simd<R>::load( w + 6 * N ) * x[j + 6];
        a7 += simd<R>::load( w + 7 * N ) * x[j + 7];
      }
      for ( ; j < m; ++j, w += N ) a0 += simd<R>::load( w ) * x[j];
      return ( ( a0 + a1 ) + ( a2 + a3 ) ) + ( ( a4 + a5 ) + ( a6 + a7 ) );
    }

    inline void operator()( const State<R>& s, R out[4] ) const {
      const R u[5] = { s.x * inverse_scale[0], s.y * inverse_scale[1], s.tx * inverse_scale[2],
                       s.ty * inverse_scale[3], s.qop * inverse_scale[4] };
      R       L[5][MOST + 2 * N];
      for ( int j = 0; j < 5; ++j ) {
        L[j][0] = R( 1 );
        L[j][1] = u[j];
        for ( int k = 1; k < degree; ++k ) L[j][k + 1] = first[k] * u[j] * L[j][k] - second[k] * L[j][k - 1];
        for ( int k = degree + 1; k < MOST + 2 * N; ++k ) L[j][k] = R( 0 );
      }
      // level 2: the group s is L_a(u_x) for a = 0 .. s, times L_{s - a}(u_y); the second
      // table is read backwards, so it is turned round first
      R turned[MOST + 2 * N];
      for ( int k = 0; k <= degree; ++k ) turned[k] = L[1][degree - k];
      for ( int k = degree + 1; k < MOST + 2 * N; ++k ) turned[k] = R( 0 );
      for ( int g = 0; g <= degree; ++g )
        for ( int i = 0; i <= g; i += N )
          simd<R>::store( p2.data() + start2[g] + i,
                          simd<R>::load( L[0] + i ) * simd<R>::load( turned + ( degree - g ) + i ) );
      // levels 3 and 4: the group s is the groups 0 .. s of the level before, each times one number
      for ( int g = 0; g <= degree; ++g )
        for ( int lower = 0; lower <= g; ++lower )
          times( p3.data() + start3[g] + start2[lower], p2.data() + start2[lower], lower + 1, L[2][g - lower] );
      for ( int g = 0; g <= degree; ++g )
        for ( int lower = 0; lower <= g; ++lower )
          times( p4.data() + start4[g] + start3[lower], p3.data() + start3[lower],
                 start3[lower + 1] - start3[lower], L[3][g - lower] );
      R                 sums[4 * MOST + N];
      const std::size_t vectors = weights.size();
      for ( std::size_t k = 0; k < vectors; ++k )
        simd<R>::store( sums + k * N, weighted_sum( weights[k].data(), p4.data(), entries[k] ) );
      for ( int component = 0; component < 4; ++component ) {
        R sum = 0;
        for ( int e = 0; e <= degree; ++e ) sum += sums[4 * e + component] * L[4][e];
        out[component] = u[4] * sum;
      }
    }
  };

  template <class R>
  struct NetworkCorrection {
    Body<R> body;
    R       inverse_scale_of_inputs[5];
    R       scale_of_outputs[4];
    bool    times_qop            = false;
    R       inverse_scale_of_qop = 1;

    inline void operator()( const State<R>& s, R out[4] ) const {
      const R input[5] = { s.x * inverse_scale_of_inputs[0], s.y * inverse_scale_of_inputs[1],
                           s.tx * inverse_scale_of_inputs[2], s.ty * inverse_scale_of_inputs[3],
                           s.qop * inverse_scale_of_inputs[4] };
      R       raw[4];
      body.forward( input, raw );
      const R factor = times_qop ? s.qop * inverse_scale_of_qop : R( 1 );
      for ( int c = 0; c < 4; ++c ) out[c] = raw[c] * scale_of_outputs[c] * factor;
    }
  };

  template <class R, class Field>
  struct CorrectedStart {
    Nystrom3<R, Field>      integrator; // used when `steps` > 0
    R                       z0 = 0, z1 = 0;
    int                     steps      = 0; // 0: the straight line
    bool                    polynomial = true;
    PolynomialCorrection<R> of_polynomial;
    NetworkCorrection<R>    of_network;
    mutable int             last_steps = 0, last_failed = 0;

    inline State<R> operator()( const State<R>& s ) const {
      State<R> end;
      if ( steps < 0 ) {
        end = { R( 0 ), R( 0 ), R( 0 ), R( 0 ), s.qop }; // no start: the correction is the end state
      } else if ( steps == 0 ) {
        const R length = z1 - z0;
        end            = { s.x + length * s.tx, s.y + length * s.ty, s.tx, s.ty, s.qop };
      } else {
        end = integrator( s );
      }
      R correction[4];
      if ( polynomial )
        of_polynomial( s, correction );
      else
        of_network( s, correction );
      end.x += correction[0];
      end.y += correction[1];
      end.tx += correction[2];
      end.ty += correction[3];
      last_steps = steps > 0 ? steps : 0;
      return end;
    }
  };

  template <class R, class Field>
  inline CorrectedStart<R, Field> read_corrected_start( const std::string& folder, const Field& field, double z0,
                                                        double z1 ) {
    const Meta               meta = read_meta( folder + "/meta" );
    CorrectedStart<R, Field> m;
    m.z0 = static_cast<R>( z0 );
    m.z1 = static_cast<R>( z1 );
    if ( std::abs( meta.number( "first_plane_mm" ) - z0 ) > 1e-9 || std::abs( meta.number( "last_plane_mm" ) - z1 ) > 1e-9 )
      throw std::runtime_error( "the correction in " + folder + " is for other planes" );
    const std::string start = meta.text( "start" );
    m.integrator.field      = &field;
    if ( start == "none" ) {
      m.steps = -1;
    } else if ( start == "straight_line" ) {
      m.steps = 0;
    } else if ( start == "nystrom" ) {
      m.steps = static_cast<int>( meta.integer( "steps" ) );
      for ( int i = 0; i <= m.steps; ++i )
        m.integrator.planes.push_back( static_cast<R>( z0 + i * ( ( z1 - z0 ) / m.steps ) ) );
    } else {
      throw std::runtime_error( "a start is none, straight_line or nystrom, not " + start );
    }
    const auto        inputs     = read_f64( folder + "/scale_of_inputs.f64" );
    const std::string correction = meta.text( "correction" );
    if ( correction == "polynomial" ) {
      m.polynomial = true;
      auto& p      = m.of_polynomial;
      const int degree = static_cast<int>( meta.integer( "degree" ) );
      if ( degree < 1 || degree > HIGHEST_DEGREE_OF_A_CORRECTION ) throw std::runtime_error( "degree out of range" );
      for ( int j = 0; j < 5; ++j ) p.inverse_scale[j] = static_cast<R>( 1.0 / inputs.at( j ) );
      p.prepare( degree, read_f64( folder + "/coefficients.f64" ) );
    } else if ( correction == "network" ) {
      m.polynomial = false;
      auto& n      = m.of_network;
      n.body       = read_body<R>( folder, meta );
      for ( int j = 0; j < 5; ++j ) n.inverse_scale_of_inputs[j] = static_cast<R>( 1.0 / inputs.at( j ) );
      const auto outputs = read_f64( folder + "/scale_of_outputs.f64" );
      for ( int c = 0; c < 4; ++c ) n.scale_of_outputs[c] = static_cast<R>( outputs.at( c ) );
      n.times_qop = meta.integer( "times_qop" ) != 0;
      if ( n.times_qop ) n.inverse_scale_of_qop = static_cast<R>( 1.0 / meta.number( "scale_of_qop" ) );
    } else {
      throw std::runtime_error( "a correction is polynomial or network, not " + correction );
    }
    return m;
  }

} // namespace fx
