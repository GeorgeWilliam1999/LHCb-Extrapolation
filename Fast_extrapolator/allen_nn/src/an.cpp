// an: the standalone program of the magnet-crossing tool (allen_nn).
//
//   an run method=M precision=float|double states=FILE z_in=Z z_out=Z out=FILE
//          [jac=FILE] [counts=FILE] [map=v8r1_up] [z_in_file=FILE z_out_file=FILE]
//       Carries every state (rows of x, y, tx, ty, q/p * c_light, float64) and writes
//       the final states (rows of 5) and, with jac=, the transport matrix as 4 x 5
//       float64 a row (rows x, y, tx, ty; columns x0, y0, tx0, ty0, q/p * c_light).
//       A method with no matrix for x0, y0 gives the identity there. With
//       z_in_file / z_out_file each track has its own planes (float64 a track).
//       counts= writes field reads, accepted and rejected steps (int64 x 3), and a
//       fourth int64: 1 when the tool fell back on the ARKN.
//
//   an gate states=FILE weights=JSON out=FILE        double, the crossing z0 -> z1:
//       rows of 5 (the K Nystrom steps), 4 (the end), 20 (d end / d s0, 4 x 5)
//
//   an time methods=M1,M2,... states=FILE z_in=Z z_out=Z out=CSV [repeats=R]
//       float, one call at a time; the lowest and the median time per call of R passes.
//
// Methods:
//   arkn               Allen's adaptive RKN with Jacobian (allen_nn/arkn.h)
//   arkn_state         the same without Jacobian
//   tool:JSON[:polarity=P][:fallback=E]   way A: legs + K Nystrom steps + network, with Jacobian;
//                      netjac=0: transport matrix of the Nystrom steps only; fallback=E overrides the step-error threshold of the ARKN fallback (mm)
//   tool_state:JSON    the same without Jacobian
//   nystrom:K          K equal Nystrom steps z_in -> z_out with the ARKN Jacobian
//   planes:Z0:Z1:...   fixed Nystrom steps on the given planes (z_in, z_out ignored), ARKN Jacobian
//   rec_step           the Rec STEP integrator (cpp/lhcb/rec_step.h, double), Jacobian
//   reference:N        RK4, N equal steps, double, the trilinear map (no Jacobian)
//   parkf              the HLT1 parametrised step (cpp/lhcb/allen_utt.h), planes 2665 -> 7826
// The field is read as LHCb reads it (RecField of cpp/field: float, Gaudi unit,
// zero outside the grid), except reference: (LinearPackage<double>, tesla).
#include <algorithm>
#include <chrono>
#include <functional>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "../../cpp/classical/integrators.h"
#include "../../cpp/classical/more_integrators.h"
#include "../../cpp/common/io.h"
#include "../../cpp/field/field_map.h"
#include "../../cpp/lhcb/allen_nystrom.h"
#include "../../cpp/lhcb/allen_utt.h"
#include "../../cpp/lhcb/rec_step.h"
#include "../include/allen_nn/arkn.h"
#include "../include/allen_nn/magnet_crossing.h"
#include "../include/allen_nn/weights_json.h"

namespace {

  const std::string g_exports = "/data/bfys/gscriven/fast_extrapolator_store/exports";

  std::map<std::string, std::string> arguments( int argc, char** argv ) {
    std::map<std::string, std::string> a;
    for ( int i = 2; i < argc; ++i ) {
      const std::string s  = argv[i];
      const auto        eq = s.find( '=' );
      if ( eq == std::string::npos ) throw std::runtime_error( "argument without '=': " + s );
      a[s.substr( 0, eq )] = s.substr( eq + 1 );
    }
    return a;
  }
  std::string get( const std::map<std::string, std::string>& a, const std::string& k, const std::string& d = "" ) {
    auto f = a.find( k );
    if ( f != a.end() ) return f->second;
    if ( d.empty() ) throw std::runtime_error( "missing argument " + k );
    return d;
  }
  std::vector<std::string> split( const std::string& s, char c ) {
    std::vector<std::string> out;
    std::stringstream        ss( s );
    std::string              part;
    while ( std::getline( ss, part, c ) ) out.push_back( part );
    return out;
  }
  double option( const std::vector<std::string>& parts, const std::string& key, double otherwise ) {
    for ( const auto& p : parts )
      if ( p.rfind( key + "=", 0 ) == 0 ) return std::stod( p.substr( key.size() + 1 ) );
    return otherwise;
  }

  // The result of one call: the end state, the 4 x 5 matrix in the units of the
  // states (q/p * c_light), the counts.
  struct Result {
    double end[5];
    double J[4][5];
    long   counts[4];
  };

  // A method as a function of (state, z_in, z_out).
  using Method = std::function<void( const double*, double, double, Result& )>;

  // The field as the tool wants it, from the RecField reader of cpp/field.
  template <class R>
  struct RecAdapter {
    const fx::RecField* f;
    mutable long        reads = 0;
    inline void         operator()( R x, R y, R z, R& bx, R& by, R& bz ) const {
      ++reads;
      ( *f )( x, y, z, bx, by, bz );
    }
  };

  struct World {
    fx::MapFile                                    map;
    std::unique_ptr<fx::RecField>                  rec;
    std::unique_ptr<fx::LinearPackage<double>>     linear;
    std::map<std::string, std::shared_ptr<void>>   keep;
    explicit World( const std::string& name ) : map( fx::read_map( g_exports + "/field_" + name ) ) {
      rec    = std::make_unique<fx::RecField>( map );
      linear = std::make_unique<fx::LinearPackage<double>>( map );
    }
  };

  void identity_xy( Result& r ) {
    for ( int i = 0; i < 4; ++i )
      for ( int j = 0; j < 5; ++j ) r.J[i][j] = 0;
    r.J[0][0] = r.J[1][1] = 1;
  }

  template <class R>
  void from_allen( const R x[5], const R J[5][5], bool with_j, Result& r ) {
    for ( int i = 0; i < 4; ++i ) r.end[i] = static_cast<double>( x[i] );
    if ( with_j )
      for ( int i = 0; i < 4; ++i ) {
        for ( int j = 0; j < 4; ++j ) r.J[i][j] = static_cast<double>( J[i][j] );
        r.J[i][4] = static_cast<double>( J[i][4] ) / static_cast<double>( allen_nn::c_light );
      }
  }

  template <class R>
  Method make_method( World& w, const std::string& spec ) {
    const auto         parts = split( spec, ':' );
    const std::string& name  = parts.at( 0 );
    if ( name == "arkn" || name == "arkn_state" ) {
      const bool with_j = name == "arkn";
      return [&w, with_j]( const double* s, double zi, double zo, Result& r ) {
        RecAdapter<R> field{ w.rec.get() };
        R             x[5] = { R( s[0] ), R( s[1] ), R( s[2] ), R( s[3] ), R( s[4] / double( allen_nn::c_light ) ) };
        R             J[5][5];
        bool          ok = true;
        allen_nn::ArknCounts c;
        if ( with_j )
          allen_nn::arkn_propagate<R, true>( x, J, R( zi ), R( zo ), field, ok, allen_nn::ArknSettings{}, &c );
        else
          allen_nn::arkn_propagate<R, false>( x, J, R( zi ), R( zo ), field, ok, allen_nn::ArknSettings{}, &c );
        identity_xy( r );
        from_allen( x, J, with_j, r );
        r.end[4] = s[4];
        r.counts[0] = c.field_reads, r.counts[1] = c.accepted, r.counts[2] = c.rejected, r.counts[3] = ok ? 0 : -1;
      };
    }
    if ( name == "tool" || name == "tool_state" ) {
      auto net      = std::make_shared<allen_nn::CrossingNetwork<R>>( allen_nn::read_crossing_network<R>( parts.at( 1 ) ) );
      w.keep[spec]  = net;
      const R pol   = R( option( parts, "polarity", 1 ) );
      if ( option( parts, "netjac", -1 ) >= 0 ) net->network_jacobian = int( option( parts, "netjac", 1 ) );
      if ( option( parts, "fallback", -1 ) > 0 ) net->most_step_error = R( option( parts, "fallback", -1 ) );
      const bool wj = name == "tool";
      return [&w, net, pol, wj]( const double* s, double zi, double zo, Result& r ) {
        RecAdapter<R> field{ w.rec.get() };
        R             x[5] = { R( s[0] ), R( s[1] ), R( s[2] ), R( s[3] ), R( s[4] / double( allen_nn::c_light ) ) };
        R             J[5][5];
        bool          ok = true;
        bool          fell_back = false;
        bool          done;
        if ( wj )
          done = allen_nn::propagate_magnet<R, true>( *net, x, J, R( zi ), R( zo ), field, pol, ok );
        else
          done = allen_nn::propagate_magnet<R, false>( *net, x, J, R( zi ), R( zo ), field, pol, ok );
        allen_nn::ArknCounts c;
        if ( !done ) {
          fell_back = true;
          field.reads = 0;
          if ( wj )
            allen_nn::arkn_propagate<R, true>( x, J, R( zi ), R( zo ), field, ok, allen_nn::ArknSettings{}, &c );
          else
            allen_nn::arkn_propagate<R, false>( x, J, R( zi ), R( zo ), field, ok, allen_nn::ArknSettings{}, &c );
        }
        identity_xy( r );
        from_allen( x, J, wj, r );
        r.end[4]    = s[4];
        r.counts[0] = field.reads, r.counts[1] = net->steps, r.counts[2] = 0, r.counts[3] = fell_back ? 1 : ( ok ? 0 : -1 );
      };
    }
    if ( name == "nystrom" ) {
      const int K = std::stoi( parts.at( 1 ) );
      return [&w, K]( const double* s, double zi, double zo, Result& r ) {
        RecAdapter<R> field{ w.rec.get() };
        R             x[5] = { R( s[0] ), R( s[1] ), R( s[2] ), R( s[3] ), R( s[4] ) };
        R             J[5][5], B0[3];
        std::vector<R> p( K + 1 );
        allen_nn::set_identity( J );
        for ( int i = 0; i <= K; ++i ) p[i] = R( zi + i * ( ( zo - zi ) / K ) );
        field( x[0], x[1], p[0], B0[0], B0[1], B0[2] );
        allen_nn::nystrom_fixed( x, p.data(), K, B0, field, J );
        identity_xy( r );
        for ( int i = 0; i < 4; ++i ) {
          r.end[i] = double( x[i] );
          for ( int j = 0; j < 5; ++j ) r.J[i][j] = double( J[i][j] );
        }
        r.end[4]    = s[4];
        r.counts[0] = field.reads, r.counts[1] = K, r.counts[2] = 0, r.counts[3] = 0;
      };
    }
    if ( name == "planes" ) { // fixed Nystrom steps on the given planes (z_in, z_out are ignored)
      std::vector<double> planes;
      for ( std::size_t i = 1; i < parts.size(); ++i ) planes.push_back( std::stod( parts[i] ) );
      const int K = int( planes.size() ) - 1;
      if ( K < 1 ) throw std::runtime_error( "planes: needs at least two planes" );
      return [&w, planes, K]( const double* s, double, double, Result& r ) {
        RecAdapter<R> field{ w.rec.get() };
        R             x[5] = { R( s[0] ), R( s[1] ), R( s[2] ), R( s[3] ), R( s[4] ) };
        R             J[5][5], B0[3];
        std::vector<R> p( K + 1 );
        allen_nn::set_identity( J );
        for ( int i = 0; i <= K; ++i ) p[i] = R( planes[i] );
        field( x[0], x[1], p[0], B0[0], B0[1], B0[2] );
        allen_nn::nystrom_fixed( x, p.data(), K, B0, field, J );
        identity_xy( r );
        for ( int i = 0; i < 4; ++i ) {
          r.end[i] = double( x[i] );
          for ( int j = 0; j < 5; ++j ) r.J[i][j] = double( J[i][j] );
        }
        r.end[4]    = s[4];
        r.counts[0] = field.reads, r.counts[1] = K, r.counts[2] = 0, r.counts[3] = 0;
      };
    }
    if ( name == "rec_step" ) {
      return [&w]( const double* s, double zi, double zo, Result& r ) {
        fx::RecSTEP<fx::RecField> m{ w.rec.get(), zi, zo };
        m.with_jacobian = true;
        const fx::State<double> e = m( fx::State<double>{ s[0], s[1], s[2], s[3], s[4] } );
        identity_xy( r );
        r.end[0] = e.x, r.end[1] = e.y, r.end[2] = e.tx, r.end[3] = e.ty, r.end[4] = s[4];
        for ( int i = 0; i < 4; ++i )
          for ( int c = 0; c < 3; ++c ) r.J[i][2 + c] = m.last_jacobian[i][c];
        r.counts[0] = 0, r.counts[1] = m.last_steps, r.counts[2] = m.last_failed, r.counts[3] = m.last_success ? 0 : -1;
      };
    }
    if ( name == "allen_adaptive_nystrom" ) { // the earlier copy, state only, for the check of arkn
      return [&w]( const double* s, double zi, double zo, Result& r ) {
        fx::AllenAdaptiveNystrom<R, fx::RecField> m{ w.rec.get(), R( zi ), R( zo ) };
        // the q/p that Allen forms on entry (1/MeV times c_light, in R), as the arkn method forms it
        const R q = R( s[4] / double( allen_nn::c_light ) ) * R( allen_nn::c_light );
        const auto e = m( fx::State<R>{ R( s[0] ), R( s[1] ), R( s[2] ), R( s[3] ), q } );
        identity_xy( r );
        r.end[0] = e.x, r.end[1] = e.y, r.end[2] = e.tx, r.end[3] = e.ty, r.end[4] = s[4];
        r.counts[0] = 0, r.counts[1] = m.last_steps, r.counts[2] = m.last_failed, r.counts[3] = 0;
      };
    }
    if ( name == "reference" ) {
      const int N = std::stoi( parts.at( 1 ) );
      return [&w, N]( const double* s, double zi, double zo, Result& r ) {
        fx::RK4<double, fx::LinearPackage<double>> m{ w.linear.get(), zi, zo, N };
        const auto e = m( fx::State<double>{ s[0], s[1], s[2], s[3], s[4] } );
        identity_xy( r );
        r.end[0] = e.x, r.end[1] = e.y, r.end[2] = e.tx, r.end[3] = e.ty, r.end[4] = s[4];
        r.counts[0] = 4L * N, r.counts[1] = N, r.counts[2] = 0, r.counts[3] = 0;
      };
    }
    if ( name == "parkf" ) {
      auto utt = std::make_shared<fx::AllenUTT>( fx::read_allen_utt( g_exports + "/allen_utt" ) );
      w.keep[spec] = utt;
      return [utt]( const double* s, double, double, Result& r ) {
        const auto e = ( *utt )( fx::State<float>{ float( s[0] ), float( s[1] ), float( s[2] ), float( s[3] ), float( s[4] ) } );
        identity_xy( r );
        r.end[0] = e.x, r.end[1] = e.y, r.end[2] = e.tx, r.end[3] = e.ty, r.end[4] = s[4];
        for ( int i = 0; i < 4; ++i ) {
          r.J[i][2] = utt->last_der_tx[i];
          r.J[i][3] = utt->last_der_ty[i];
          r.J[i][4] = utt->last_der_qop[i] / double( allen_nn::c_light );
        }
        r.counts[0] = 0, r.counts[1] = 1, r.counts[2] = 0, r.counts[3] = 0;
      };
    }
    throw std::runtime_error( "unknown method " + spec );
  }

  Method method_of( World& w, const std::string& spec, const std::string& precision ) {
    if ( precision == "double" ) return make_method<double>( w, spec );
    if ( precision == "float" ) return make_method<float>( w, spec );
    throw std::runtime_error( "precision is float or double" );
  }

  int run( const std::map<std::string, std::string>& a ) {
    World       w( get( a, "map", "v8r1_up" ) );
    const auto  states = fx::read_f64( get( a, "states" ) );
    const auto  n      = states.size() / 5;
    std::vector<double> zin( n ), zout( n );
    if ( a.count( "z_in_file" ) ) {
      zin  = fx::read_f64( get( a, "z_in_file" ) );
      zout = fx::read_f64( get( a, "z_out_file" ) );
    } else {
      std::fill( zin.begin(), zin.end(), std::stod( get( a, "z_in" ) ) );
      std::fill( zout.begin(), zout.end(), std::stod( get( a, "z_out" ) ) );
    }
    Method              m = method_of( w, get( a, "method" ), get( a, "precision", "float" ) );
    std::vector<double> out( n * 5 ), jac( n * 20 );
    std::vector<std::int64_t> counts( n * 4 );
    Result r;
    for ( std::size_t t = 0; t < n; ++t ) {
      m( &states[5 * t], zin[t], zout[t], r );
      for ( int i = 0; i < 5; ++i ) out[5 * t + i] = r.end[i];
      for ( int i = 0; i < 4; ++i )
        for ( int j = 0; j < 5; ++j ) jac[20 * t + 5 * i + j] = r.J[i][j];
      for ( int i = 0; i < 4; ++i ) counts[4 * t + i] = r.counts[i];
    }
    fx::write_raw( get( a, "out" ), out );
    if ( a.count( "jac" ) ) fx::write_raw( get( a, "jac" ), jac );
    if ( a.count( "counts" ) ) fx::write_raw( get( a, "counts" ), counts );
    return 0;
  }

  // The crossing alone in double, for the check against torch.
  int gate( const std::map<std::string, std::string>& a ) {
    World      w( get( a, "map", "v8r1_up" ) );
    const auto states = fx::read_f64( get( a, "states" ) );
    const auto n      = states.size() / 5;
    auto       net    = allen_nn::read_crossing_network<double>( get( a, "weights" ) );
    net.network_jacobian = 1; // the gate checks the derivative of the network too
    std::vector<double> out;
    RecAdapter<double>  field{ w.rec.get() };
    for ( std::size_t t = 0; t < n; ++t ) {
      double s[5], B0[3], J[5][5];
      for ( int i = 0; i < 5; ++i ) s[i] = states[5 * t + i];
      double sk[5] = { s[0], s[1], s[2], s[3], s[4] };
      double Bk[3];
      field( sk[0], sk[1], net.z0, Bk[0], Bk[1], Bk[2] );
      allen_nn::nystrom_fixed( sk, net.planes, net.steps, Bk, field, ( double( * )[5] ) nullptr );
      allen_nn::crossing( net, s, B0, false, field, 1.0, J );
      for ( int i = 0; i < 5; ++i ) out.push_back( sk[i] );
      for ( int i = 0; i < 4; ++i ) out.push_back( s[i] );
      for ( int i = 0; i < 4; ++i )
        for ( int j = 0; j < 5; ++j ) out.push_back( J[i][j] );
    }
    fx::write_raw( get( a, "out" ), out );
    return 0;
  }

  int time( const std::map<std::string, std::string>& a ) {
    World      w( get( a, "map", "v8r1_up" ) );
    const auto states  = fx::read_f64( get( a, "states" ) );
    const auto n       = states.size() / 5;
    const double zi    = std::stod( get( a, "z_in" ) ), zo = std::stod( get( a, "z_out" ) );
    const int repeats  = std::stoi( get( a, "repeats", "7" ) );
    const auto specs   = split( get( a, "methods" ), ',' );
    std::vector<Method> methods;
    for ( const auto& s : specs ) methods.push_back( method_of( w, s, "float" ) );
    std::vector<std::vector<double>> per( specs.size() );
    Result  r;
    double  sink = 0;
    for ( int rep = 0; rep < repeats; ++rep )
      for ( std::size_t k = 0; k < methods.size(); ++k ) {
        const auto t0 = std::chrono::steady_clock::now();
        for ( std::size_t t = 0; t < n; ++t ) {
          methods[k]( &states[5 * t], zi, zo, r );
          sink += r.end[0];
        }
        const auto t1 = std::chrono::steady_clock::now();
        per[k].push_back( std::chrono::duration<double, std::nano>( t1 - t0 ).count() / n );
      }
    FILE* f = std::fopen( get( a, "out" ).c_str(), "w" );
    std::fprintf( f, "method,lowest_ns,median_ns,tracks,repeats\n" );
    for ( std::size_t k = 0; k < specs.size(); ++k ) {
      auto v = per[k];
      std::sort( v.begin(), v.end() );
      std::fprintf( f, "%s,%.1f,%.1f,%zu,%d\n", specs[k].c_str(), v.front(), v[v.size() / 2], n, repeats );
    }
    std::fclose( f );
    std::fprintf( stderr, "(sink %g)\n", sink );
    return 0;
  }

} // namespace

int main( int argc, char** argv ) {
  try {
    if ( argc < 2 ) throw std::runtime_error( "an run|gate|time key=value ..." );
    const std::string mode = argv[1];
    const auto        a    = arguments( argc, argv );
    if ( mode == "run" ) return run( a );
    if ( mode == "gate" ) return gate( a );
    if ( mode == "time" ) return time( a );
    throw std::runtime_error( "unknown mode " + mode );
  } catch ( const std::exception& e ) {
    std::cerr << "an: " << e.what() << "\n";
    return 1;
  }
}
