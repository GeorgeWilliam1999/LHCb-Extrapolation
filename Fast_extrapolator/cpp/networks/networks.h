// The forward pass of the trained networks.
//
//   Body<R>            `depth` layers of `width` units, each followed by tanh,
//                      then a linear layer: body_of_layers of the present
//                      package (networks/stage_network.py).
//   WholeCrossing<R>   the whole-crossing network: one call from the first
//                      plane to the last, no field read
//                      (networks/whole_crossing_network.py, direct states).
//   StageChain<R, F>   the stage network applied to itself over the layout
//                      (networks/stage_network.py and self_chain.py), with the
//                      output forms direct_states and
//                      straight_line_plus_scaled_correction, the end state
//                      summed from the stages or predicted, and the physics
//                      layer (stage_network_with_a_physics_layer.py).
//
// The arithmetic follows the Python term by term. Two places differ at the
// level of rounding: a layer sums its products in its own order (the Python
// uses a library routine with another order), and the mean over the field
// samples of the per-track scale is a plain sum.
//
// A layer is written for speed, because the cost of one network call decides
// which networks can compete: the weights of one input lie side by side, the
// outputs are handled in 256-bit vectors, and eight vectors of outputs are
// summed side by side in registers.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#if defined( __AVX2__ )
#  include <immintrin.h>
#endif

#include "../classical/integrators.h"
#include "../common/io.h"

namespace fx {

  // The activation of the hidden layers.
  //   Tanh         tanh of the C library, one value at a time (about 17 ns each)
  //   TanhFromExp  tanh(x) = 1 - 2 / (exp(2 x) + 1), four doubles or eight floats at
  //                a time, with the vector exponential of the C library (libmvec).
  //                The absolute error is that of the rounding (1e-16 in double,
  //                1e-7 in float); the relative error is larger near zero.
  //   None         no activation; for the time of the layers alone
  enum class Activation { Tanh, TanhFromExp, None };

  // 256-bit vectors: four doubles or eight floats, handled by the compiler
  // (AVX2 on the nodes; plain code where the processor has none).
  template <class R>
  struct simd {
    static constexpr int N = 32 / static_cast<int>( sizeof( R ) );
    typedef R            V __attribute__( ( vector_size( 32 ) ) );
    static inline V      load( const R* p ) {
      V v;
      std::memcpy( &v, p, sizeof( v ) );
      return v;
    }
    static inline void store( R* p, V v ) { std::memcpy( p, &v, sizeof( v ) ); }
  };

#if defined( __AVX2__ )
  extern "C" {
  __m256d _ZGVdN4v_exp( __m256d );
  __m256  _ZGVdN8v_expf( __m256 );
  }
  inline simd<double>::V vector_exp( simd<double>::V x ) { return (simd<double>::V)_ZGVdN4v_exp( (__m256d)x ); }
  inline simd<float>::V  vector_exp( simd<float>::V x ) { return (simd<float>::V)_ZGVdN8v_expf( (__m256)x ); }
#else
  template <class V>
  inline V vector_exp( V x ) {
    for ( unsigned i = 0; i < sizeof( V ) / sizeof( x[0] ); ++i ) x[i] = std::exp( x[i] );
    return x;
  }
#endif

  // One layer. The outputs are padded to a whole number of vectors, and the
  // weights of one input lie side by side: w[j * padded + i] = W[i][j].
  template <class R>
  struct Dense {
    int            in = 0, out = 0, padded = 0;
    std::vector<R> w;
    std::vector<R> b; // padded with zeros

    void shape( int n_in, int n_out ) {
      in     = n_in;
      out    = n_out;
      padded = ( ( n_out + simd<R>::N - 1 ) / simd<R>::N ) * simd<R>::N;
      w.assign( static_cast<std::size_t>( in ) * padded, R( 0 ) );
      b.assign( padded, R( 0 ) );
    }
    R& weight( int i, int j ) { return w[static_cast<std::size_t>( j ) * padded + i]; }
  };

  template <class R>
  struct Body {
    using V = typename simd<R>::V;
    static constexpr int N = simd<R>::N;

    std::vector<Dense<R>>  layers; // the activation follows every layer but the last
    Activation             activation = Activation::Tanh;
    mutable std::vector<R> scratch_a, scratch_b;

    void prepare() {
      std::size_t widest = 0;
      for ( const auto& layer : layers ) widest = std::max( widest, static_cast<std::size_t>( layer.padded ) );
      scratch_a.assign( widest, R( 0 ) );
      scratch_b.assign( widest, R( 0 ) );
    }

    int number_of_inputs() const { return layers.front().in; }
    int number_of_outputs() const { return layers.back().out; }

    long multiply_adds() const {
      long total = 0;
      for ( const auto& layer : layers ) total += static_cast<long>( layer.in ) * layer.out;
      return total;
    }

    long hidden_units() const {
      long total = 0;
      for ( std::size_t l = 0; l + 1 < layers.size(); ++l ) total += layers[l].out;
      return total;
    }

    // Eight vectors of outputs at a time. Each output is summed over the inputs
    // in their order, as in a plain loop; the eight sums run side by side.
    static inline void eight_vectors( const R* w, std::ptrdiff_t stride, const R* bias, const R* x, int n_in, R* y ) {
      V a0 = simd<R>::load( bias ), a1 = simd<R>::load( bias + N ), a2 = simd<R>::load( bias + 2 * N ),
        a3 = simd<R>::load( bias + 3 * N ), a4 = simd<R>::load( bias + 4 * N ), a5 = simd<R>::load( bias + 5 * N ),
        a6 = simd<R>::load( bias + 6 * N ), a7 = simd<R>::load( bias + 7 * N );
      for ( int j = 0; j < n_in; ++j, w += stride ) {
        const R xj = x[j];
        a0 += simd<R>::load( w ) * xj;
        a1 += simd<R>::load( w + N ) * xj;
        a2 += simd<R>::load( w + 2 * N ) * xj;
        a3 += simd<R>::load( w + 3 * N ) * xj;
        a4 += simd<R>::load( w + 4 * N ) * xj;
        a5 += simd<R>::load( w + 5 * N ) * xj;
        a6 += simd<R>::load( w + 6 * N ) * xj;
        a7 += simd<R>::load( w + 7 * N ) * xj;
      }
      simd<R>::store( y, a0 );
      simd<R>::store( y + N, a1 );
      simd<R>::store( y + 2 * N, a2 );
      simd<R>::store( y + 3 * N, a3 );
      simd<R>::store( y + 4 * N, a4 );
      simd<R>::store( y + 5 * N, a5 );
      simd<R>::store( y + 6 * N, a6 );
      simd<R>::store( y + 7 * N, a7 );
    }

    // One vector of outputs. The sum over the inputs is split into four partial
    // sums, so that a layer with few outputs and many inputs does not wait on
    // one chain of additions. The order of the additions differs from a plain
    // loop; the result differs in the last digit.
    static inline void one_vector( const R* w, std::ptrdiff_t stride, const R* bias, const R* x, int n_in, R* y ) {
      V   a0 = simd<R>::load( bias ), a1 = V{} + R( 0 ), a2 = a1, a3 = a1;
      int j = 0;
      for ( ; j + 4 <= n_in; j += 4, w += 4 * stride ) {
        a0 += simd<R>::load( w ) * x[j];
        a1 += simd<R>::load( w + stride ) * x[j + 1];
        a2 += simd<R>::load( w + 2 * stride ) * x[j + 2];
        a3 += simd<R>::load( w + 3 * stride ) * x[j + 3];
      }
      for ( ; j < n_in; ++j, w += stride ) a0 += simd<R>::load( w ) * x[j];
      simd<R>::store( y, ( a0 + a1 ) + ( a2 + a3 ) );
    }

    // input: number_of_inputs() values; output: number_of_outputs() values
    inline void forward( const R* input, R* output ) const {
      const R*          x       = input;
      R*                bufs[2] = { scratch_a.data(), scratch_b.data() };
      const std::size_t last    = layers.size() - 1;
      for ( std::size_t l = 0; l < layers.size(); ++l ) {
        const Dense<R>&      layer  = layers[l];
        R*                   y      = bufs[l & 1];
        const int            n_in   = layer.in, padded = layer.padded;
        const R*             w      = layer.w.data();
        const R*             bias   = layer.b.data();
        const std::ptrdiff_t stride = padded;
        int                  i      = 0;
        for ( ; i + 8 * N <= padded; i += 8 * N ) eight_vectors( w + i, stride, bias + i, x, n_in, y + i );
        for ( ; i < padded; i += N ) one_vector( w + i, stride, bias + i, x, n_in, y + i );
        if ( l != last ) {
          if ( activation == Activation::Tanh ) {
            for ( int k = 0; k < layer.out; ++k ) y[k] = std::tanh( y[k] );
          } else if ( activation == Activation::TanhFromExp ) {
            for ( int k = 0; k < padded; k += N ) {
              const V v = simd<R>::load( y + k );
              const V e = vector_exp( v + v );
              simd<R>::store( y + k, R( 1 ) - R( 2 ) / ( e + R( 1 ) ) );
            }
          }
        }
        x = y;
      }
      const int n_out = layers[last].out;
      for ( int k = 0; k < n_out; ++k ) output[k] = x[k];
    }
  };

  template <class R>
  inline std::vector<R> as_type( const std::vector<double>& values ) {
    return std::vector<R>( values.begin(), values.end() );
  }

  // The layers of an exported network: W<k>.f64 (out x in, row by row) and b<k>.f64.
  template <class R>
  inline Body<R> read_body( const std::string& folder, const Meta& meta ) {
    Body<R>   body;
    const int layers = static_cast<int>( meta.integer( "layers" ) );
    for ( int k = 0; k < layers; ++k ) {
      Dense<R>           layer;
      std::istringstream shape( meta.text( "layer_" + std::to_string( k ) ) );
      int                n_out = 0, n_in = 0;
      shape >> n_out >> n_in;
      layer.shape( n_in, n_out );
      const auto W = read_f64( folder + "/W" + std::to_string( k ) + ".f64" );
      const auto b = read_f64( folder + "/b" + std::to_string( k ) + ".f64" );
      if ( W.size() != static_cast<std::size_t>( n_in ) * n_out || b.size() != static_cast<std::size_t>( n_out ) )
        throw std::runtime_error( "a layer of " + folder + " has the wrong size" );
      for ( int i = 0; i < n_out; ++i ) {
        for ( int j = 0; j < n_in; ++j ) layer.weight( i, j ) = static_cast<R>( W[static_cast<std::size_t>( i ) * n_in + j] );
        layer.b[i] = static_cast<R>( b[i] );
      }
      body.layers.push_back( std::move( layer ) );
    }
    body.prepare();
    return body;
  }

  // A body with random weights, for the cost of one call: `inputs` -> depth x width -> `outputs`.
  template <class R>
  inline Body<R> random_body( int inputs, int width, int depth, int outputs, unsigned seed = 1 ) {
    Body<R>  body;
    unsigned state = seed * 2654435761u + 12345u;
    auto     next  = [&state]() {
      state = state * 1664525u + 1013904223u;
      return static_cast<double>( state >> 8 ) / 16777216.0 - 0.5;
    };
    int n_in = inputs;
    for ( int l = 0; l <= depth; ++l ) {
      Dense<R> layer;
      layer.shape( n_in, ( l == depth ) ? outputs : width );
      const double bound = 1.0 / std::sqrt( static_cast<double>( layer.in ) );
      for ( int i = 0; i < layer.out; ++i ) {
        for ( int j = 0; j < layer.in; ++j ) layer.weight( i, j ) = static_cast<R>( 2 * bound * next() );
        layer.b[i] = static_cast<R>( 2 * bound * next() );
      }
      n_in = layer.out;
      body.layers.push_back( std::move( layer ) );
    }
    body.prepare();
    return body;
  }

  // ---------------------------------------------------------------- the whole-crossing network
  template <class R>
  struct WholeCrossing {
    Body<R>     body;
    R           scale_of_inputs[5];
    R           scale_of_outputs[4];
    mutable int last_steps = 1, last_failed = 0;

    inline State<R> operator()( const State<R>& s ) const {
      const R input[5] = { s.x / scale_of_inputs[0], s.y / scale_of_inputs[1], s.tx / scale_of_inputs[2],
                           s.ty / scale_of_inputs[3], s.qop / scale_of_inputs[4] };
      R       raw[4];
      body.forward( input, raw );
      return { raw[0] * scale_of_outputs[0], raw[1] * scale_of_outputs[1], raw[2] * scale_of_outputs[2],
               raw[3] * scale_of_outputs[3], s.qop };
    }
  };

  template <class R>
  inline WholeCrossing<R> read_whole_crossing( const std::string& folder ) {
    const Meta meta = read_meta( folder + "/meta" );
    if ( meta.text( "kind" ) != "whole_crossing_network" || meta.text( "output_form" ) != "direct_states" )
      throw std::runtime_error( folder + " is not a whole-crossing network with direct states" );
    WholeCrossing<R> network;
    network.body        = read_body<R>( folder, meta );
    const auto inputs   = read_f64( folder + "/scale_of_inputs.f64" );
    const auto outputs  = read_f64( folder + "/scale_of_outputs.f64" );
    for ( int i = 0; i < 5; ++i ) network.scale_of_inputs[i] = static_cast<R>( inputs.at( i ) );
    for ( int i = 0; i < 4; ++i ) network.scale_of_outputs[i] = static_cast<R>( outputs.at( i ) );
    return network;
  }

  // ---------------------------------------------------------------- the stage network, chained
  constexpr int MOST_STAGES = 16;

  template <class R, class Field>
  struct StageChain {
    const Field*   field = nullptr;
    Body<R>        body;
    R              scale_of_inputs[5];
    R              scale_of_outputs[4]; // the fixed scale; not used with the per-track scale
    R              middle_of_start_planes, half_range_of_start_planes;
    R              first_plane, step_length;
    int            steps = 0, stages = 0, emitted = 0;
    bool           straight_line = false, per_track = false, summed = true;
    int            samples_of_the_scale = 16;
    int            sweeps_inside        = 0;
    std::vector<R> nodes, weights, stage_matrix; // q, q, q * q
    mutable int    last_steps = 0, last_failed = 0;

    // output_forms.per_track_scale, with `samples_of_the_scale` field samples
    inline void per_track_scale( const State<R>& s, R start_plane, R scale[4] ) const {
      const R dz = step_length;
      const R tx = s.tx, ty = s.ty;
      R       sum_B = 0, sum_y = 0;
      const R norm = std::sqrt( R( 1 ) + tx * tx + ty * ty );
      for ( int m = 0; m < samples_of_the_scale; ++m ) {
        const R u  = ( static_cast<R>( m ) + R( 0.5 ) ) / static_cast<R>( samples_of_the_scale );
        const R sv = u * dz;
        R       Bx, By, Bz;
        ( *field )( s.x + tx * sv, s.y + ty * sv, start_plane + sv, Bx, By, Bz );
        Bx *= static_cast<R>( Field::tesla_per_unit );
        By *= static_cast<R>( Field::tesla_per_unit );
        Bz *= static_cast<R>( Field::tesla_per_unit );
        sum_B += std::sqrt( Bx * Bx + By * By + Bz * Bz );
        sum_y += norm * ( ( R( 1 ) + ty * ty ) * std::abs( Bx ) + std::abs( tx * ty * By ) + std::abs( tx * Bz ) );
      }
      const R I_B    = sum_B / static_cast<R>( samples_of_the_scale ) * std::abs( dz );
      const R I_y    = sum_y / static_cast<R>( samples_of_the_scale ) * std::abs( dz );
      const R qop    = std::abs( s.qop );
      const R base_x = static_cast<R>( KAPPA ) * qop * I_B;
      const R base_y = std::max( static_cast<R>( KAPPA ) * qop * I_y, R( 1e-3 ) * base_x );
      const R half   = std::abs( dz ) / R( 2 );
      scale[0]       = std::max( base_x * half, R( 1e-9 ) );
      scale[1]       = std::max( base_y * half, R( 1e-9 ) );
      scale[2]       = std::max( base_x, R( 1e-12 ) );
      scale[3]       = std::max( base_y, R( 1e-12 ) );
    }

    inline State<R> operator()( State<R> s ) const {
      const int q = stages;
      const R   k = static_cast<R>( KAPPA * Field::tesla_per_unit ) * s.qop;
      R         raw[4 * ( MOST_STAGES + 1 )];
      R         Y[MOST_STAGES][4], F[MOST_STAGES][4];
      for ( int n = 0; n < steps; ++n ) {
        const R start_plane = first_plane + n * step_length;
        const R input[6]    = { s.x / scale_of_inputs[0],
                                s.y / scale_of_inputs[1],
                                s.tx / scale_of_inputs[2],
                                s.ty / scale_of_inputs[3],
                                s.qop / scale_of_inputs[4],
                                ( start_plane - middle_of_start_planes ) / half_range_of_start_planes };
        body.forward( input, raw );
        // the output form: raw outputs -> states
        if ( straight_line ) {
          R scale[4];
          if ( per_track ) {
            per_track_scale( s, start_plane, scale );
          } else {
            for ( int c = 0; c < 4; ++c ) scale[c] = scale_of_outputs[c];
          }
          for ( int j = 0; j < emitted; ++j ) {
            const R offset = ( j < q ? nodes[j] : R( 1 ) ) * step_length;
            const R line[4] = { s.x + s.tx * offset, s.y + s.ty * offset, s.tx, s.ty };
            for ( int c = 0; c < 4; ++c ) raw[4 * j + c] = line[c] + scale[c] * raw[4 * j + c];
          }
        } else {
          for ( int j = 0; j < emitted; ++j )
            for ( int c = 0; c < 4; ++c ) raw[4 * j + c] = raw[4 * j + c] * scale_of_outputs[c];
        }
        if ( !summed ) { // the end state is the last state that the network emits
          s.x = raw[4 * q], s.y = raw[4 * q + 1], s.tx = raw[4 * q + 2], s.ty = raw[4 * q + 3];
          continue;
        }
        for ( int j = 0; j < q; ++j )
          for ( int c = 0; c < 4; ++c ) Y[j][c] = raw[4 * j + c];
        const R S0[4] = { s.x, s.y, s.tx, s.ty };
        // the rates at the stages; the physics layer passes the stages through the stage map
        for ( int sweep = 0;; ++sweep ) {
          for ( int j = 0; j < q; ++j ) {
            F[j][0] = Y[j][2];
            F[j][1] = Y[j][3];
            slope_rates( *field, Y[j][0], Y[j][1], Y[j][2], Y[j][3], k, start_plane + nodes[j] * step_length, F[j][2],
                         F[j][3] );
          }
          if ( sweep == sweeps_inside ) break;
          for ( int j = 0; j < q; ++j ) {
            for ( int c = 0; c < 4; ++c ) {
              R sum = 0;
              for ( int m = 0; m < q; ++m ) sum += stage_matrix[j * q + m] * F[m][c];
              Y[j][c] = S0[c] + step_length * sum;
            }
          }
        }
        // collect and sum
        R end[4];
        for ( int c = 0; c < 4; ++c ) {
          R sum = 0;
          for ( int j = 0; j < q; ++j ) sum += weights[j] * F[j][c];
          end[c] = S0[c] + step_length * sum;
        }
        s.x = end[0], s.y = end[1], s.tx = end[2], s.ty = end[3];
      }
      last_steps = steps;
      return s;
    }
  };

  template <class R, class Field>
  inline StageChain<R, Field> read_stage_chain( const std::string& folder, const Field& field ) {
    const Meta        meta = read_meta( folder + "/meta" );
    const std::string kind = meta.text( "kind" );
    if ( kind != "stage_network" && kind != "stage_network_with_a_physics_layer" )
      throw std::runtime_error( folder + " is not a stage network" );
    StageChain<R, Field> network;
    network.field       = &field;
    network.body        = read_body<R>( folder, meta );
    const auto inputs   = read_f64( folder + "/scale_of_inputs.f64" );
    for ( int i = 0; i < 5; ++i ) network.scale_of_inputs[i] = static_cast<R>( inputs.at( i ) );
    network.straight_line = meta.text( "output_form" ) == "straight_line_plus_scaled_correction";
    network.per_track     = meta.text( "scale" ) == "per_track";
    if ( !network.per_track ) {
      const auto outputs = read_f64( folder + "/scale_of_outputs.f64" );
      for ( int i = 0; i < 4; ++i ) network.scale_of_outputs[i] = static_cast<R>( outputs.at( i ) );
    }
    network.summed                     = meta.text( "end_state" ) == "summed_from_the_stages";
    network.sweeps_inside              = static_cast<int>( meta.integer( "sweeps_inside" ) );
    network.steps                      = static_cast<int>( meta.integer( "number_of_steps" ) );
    network.stages                     = static_cast<int>( meta.integer( "number_of_stages" ) );
    network.emitted                    = static_cast<int>( meta.integer( "states_emitted" ) );
    network.first_plane                = static_cast<R>( meta.number( "first_plane_mm" ) );
    network.step_length                = static_cast<R>( meta.number( "step_length_mm" ) );
    network.middle_of_start_planes     = static_cast<R>( meta.number( "middle_of_start_planes_mm" ) );
    network.half_range_of_start_planes = static_cast<R>( meta.number( "half_range_of_start_planes_mm" ) );
    network.nodes                      = as_type<R>( read_f64( folder + "/nodes.f64" ) );
    network.weights                    = as_type<R>( read_f64( folder + "/weights.f64" ) );
    network.stage_matrix               = as_type<R>( read_f64( folder + "/stage_matrix.f64" ) );
    if ( network.stages > MOST_STAGES || network.body.number_of_outputs() != 4 * network.emitted )
      throw std::runtime_error( folder + ": the layout does not fit" );
    return network;
  }

} // namespace fx
