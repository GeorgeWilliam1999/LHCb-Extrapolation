// fx: the compiled program of the fast-extrapolator experiment.
//
// Every method is one function object: a state on the first plane in, the
// state on the last plane out. The same program computes the final states (to
// score a method and to check it against Python) and measures the time.
//
//   fx run   method=M precision=P map=MAP reader=READER states=FILE z0=Z z1=Z
//            out=FILE [counts=FILE] [jacobian=FILE] [n=N]
//       carries every state of FILE (rows of x, y, tx, ty, q/p as float64) and
//       writes the final states (float64 rows of 5) and, with counts=, three
//       int64 per row: field reads, accepted steps, failed steps. With
//       jacobian=, a method that computes a transport matrix writes its 4 x 3
//       block (12 float64 a row).
//
//   fx time  list=FILE out=CSV [repeats=R] [budget=SECONDS] [cold=N]
//       FILE has one method a line as `key=value` pairs separated by blanks:
//       label= method= precision= map= reader= states= z0= z1= . Every method
//       carries its states one at a time. A pass is one loop over all states;
//       the passes of the methods take turns in a shuffled order, R times (a
//       slow method gets fewer passes, so that it stays below SECONDS; 3 at
//       least). The CSV has the lowest and the median time per track. With cold=N, N tracks
//       are also timed one call at a time after a sweep of a 64 MB buffer, so
//       that the caches hold nothing of the method.
//
//   fx micro_field map=MAP out=CSV      the time of one field read, per reader
//   fx micro_mlp   out=CSV              the time of one network call, per size
//
// Methods:  rk4:N   nystrom:N   rk4_plan:FILE   nystrom_plan:FILE   sweeps:N:q:K
//           rec_cash_karp[:jacobian=1][:tolerance=T][:relative_tolerance_tx=T]
//           rec_step[:jacobian=1][:tolerance=T]
//           allen_adaptive_nystrom[:tolerance=T]   allen_fast_nystrom[:STEP]
//           allen_utt[:full=1][:polarity=-1]   whole_crossing:RUN[:fast_tanh=1]
//           chain:RUN[:samples=K][:fast_tanh=1]   corrected:NAME[:fast_tanh=1]
// Readers:  linear  cubic  rec  allen          Precision: double  float
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <unistd.h>
#include <vector>

#include "../classical/integrators.h"
#include "../classical/more_integrators.h"
#include "../common/io.h"
#include "../corrections/corrected_start.h"
#include "../field/field_map.h"
#include "../lhcb/allen_nystrom.h"
#include "../lhcb/allen_utt.h"
#include "../lhcb/rec_cash_karp.h"
#include "../lhcb/rec_step.h"
#include "../networks/networks.h"

using namespace fx;

namespace {

  using Clock = std::chrono::steady_clock;

  std::string g_exports = "/data/bfys/gscriven/fast_extrapolator_store/exports";

  using Args = std::map<std::string, std::string>;

  Args parse( const std::vector<std::string>& words ) {
    Args args;
    for ( const auto& word : words ) {
      const auto equal = word.find( '=' );
      if ( equal == std::string::npos ) throw std::runtime_error( "not key=value: " + word );
      args[word.substr( 0, equal )] = word.substr( equal + 1 );
    }
    return args;
  }

  const std::string& need( const Args& args, const std::string& key ) {
    const auto found = args.find( key );
    if ( found == args.end() ) throw std::runtime_error( "missing " + key + "=" );
    return found->second;
  }

  std::string get( const Args& args, const std::string& key, const std::string& otherwise ) {
    const auto found = args.find( key );
    return found == args.end() ? otherwise : found->second;
  }

  std::vector<std::string> split( const std::string& text, char separator ) {
    std::vector<std::string> parts;
    std::stringstream        stream( text );
    std::string              part;
    while ( std::getline( stream, part, separator ) ) parts.push_back( part );
    return parts;
  }

  // ------------------------------------------------------------ the maps and their readers
  template <class R>
  struct Readers {
    std::unique_ptr<LinearPackage<R>> linear;
    std::unique_ptr<CubicPackage<R>>  cubic;
    std::unique_ptr<RecField>         rec;
    std::unique_ptr<AllenField>       allen;
  };

  struct World {
    std::map<std::string, MapFile>         maps;
    std::map<std::string, Readers<double>> readers_double;
    std::map<std::string, Readers<float>>  readers_float;

    const MapFile& map( const std::string& name ) {
      auto found = maps.find( name );
      if ( found == maps.end() ) found = maps.emplace( name, read_map( g_exports + "/field_" + name ) ).first;
      return found->second;
    }
    template <class R>
    Readers<R>& readers( const std::string& name ) {
      if constexpr ( std::is_same_v<R, double> )
        return readers_double[name];
      else
        return readers_float[name];
    }
  };

  template <class R, class Use>
  void with_field( World& world, const std::string& map, const std::string& reader, Use&& use ) {
    Readers<R>& r = world.readers<R>( map );
    if ( reader == "linear" ) {
      if ( !r.linear ) r.linear = std::make_unique<LinearPackage<R>>( world.map( map ) );
      use( *r.linear );
    } else if ( reader == "cubic" ) {
      if ( !r.cubic ) r.cubic = std::make_unique<CubicPackage<R>>( world.map( map ) );
      use( *r.cubic );
    } else if ( reader == "rec" ) {
      if ( !r.rec ) r.rec = std::make_unique<RecField>( world.map( map ) );
      use( *r.rec );
    } else if ( reader == "allen" ) {
      if ( !r.allen ) r.allen = std::make_unique<AllenField>( world.map( map ) );
      use( *r.allen );
    } else {
      throw std::runtime_error( "the readers are linear, cubic, rec and allen, not " + reader );
    }
  }

  // The options of a method after its name: key=value parts.
  double option( const std::vector<std::string>& parts, const std::string& key, double otherwise ) {
    for ( const auto& part : parts ) {
      if ( part.rfind( key + "=", 0 ) == 0 ) return std::stod( part.substr( key.size() + 1 ) );
    }
    return otherwise;
  }

  template <class R, class Field, class Use>
  void with_method( const std::string& method, const Field& field, double z0, double z1, Use&& use ) {
    const auto         parts = split( method, ':' );
    const std::string& name  = parts.at( 0 );
    if ( name == "rk4" ) {
      RK4<R, Field> m{ &field, static_cast<R>( z0 ), static_cast<R>( z1 ), std::stoi( parts.at( 1 ) ) };
      use( m );
    } else if ( name == "nystrom" || name == "rk4_plan" || name == "nystrom_plan" ) {
      // equal steps (nystrom:N), or the planes of a file of float64 (…_plan:FILE)
      std::vector<R> planes;
      if ( name == "nystrom" ) {
        const int n = std::stoi( parts.at( 1 ) );
        for ( int i = 0; i <= n; ++i ) planes.push_back( static_cast<R>( z0 + i * ( ( z1 - z0 ) / n ) ) );
      } else {
        planes = as_type<R>( read_f64( method.substr( name.size() + 1 ) ) );
      }
      if ( name == "rk4_plan" ) {
        RK4Plan<R, Field> m{ &field, planes };
        use( m );
      } else {
        Nystrom3<R, Field> m{ &field, planes };
        use( m );
      }
    } else if ( name == "sweeps" ) {
      // sweeps:N:q:K  the Gauss-Legendre scheme of q stages on N steps, K sweeps of its stage map
      StageSweeps<R, Field> m;
      m.field  = &field;
      m.z0     = static_cast<R>( z0 );
      m.z1     = static_cast<R>( z1 );
      m.steps  = std::stoi( parts.at( 1 ) );
      m.stages = std::stoi( parts.at( 2 ) );
      m.sweeps = std::stoi( parts.at( 3 ) );
      const std::string folder = g_exports + "/tableau_" + parts.at( 2 );
      m.nodes                  = as_type<R>( read_f64( folder + "/nodes.f64" ) );
      m.weights                = as_type<R>( read_f64( folder + "/weights.f64" ) );
      m.stage_matrix           = as_type<R>( read_f64( folder + "/stage_matrix.f64" ) );
      if ( m.stages > MOST_STAGES_OF_A_SWEEP ) throw std::runtime_error( "too many stages" );
      use( m );
    } else if ( name == "rec_cash_karp" ) {
      RecCashKarp<Field> m{ &field, z0, z1 };
      m.toleranceX    = option( parts, "tolerance", m.toleranceX );
      m.maxRKStep     = option( parts, "max_step", m.maxRKStep );
      m.minRKStep     = option( parts, "min_step", m.minRKStep );
      m.initialRKStep  = option( parts, "initial_step", m.initialRKStep );
      m.relToleranceTx = option( parts, "relative_tolerance_tx", m.relToleranceTx );
      m.with_jacobian  = option( parts, "jacobian", 0 ) != 0;
      use( m );
    } else if ( name == "rec_step" ) {
      RecSTEP<Field> m{ &field, z0, z1 };
      m.toleranceX    = option( parts, "tolerance", m.toleranceX );
      m.maxRKStep     = option( parts, "max_step", m.maxRKStep );
      m.minRKStep     = option( parts, "min_step", m.minRKStep );
      m.initialRKStep = option( parts, "initial_step", m.initialRKStep );
      m.with_jacobian = option( parts, "jacobian", 0 ) != 0;
      use( m );
    } else if ( name == "allen_adaptive_nystrom" ) {
      AllenAdaptiveNystrom<R, Field> m{ &field, static_cast<R>( z0 ), static_cast<R>( z1 ) };
      m.inv_tolerance = static_cast<R>( 1.0 / option( parts, "tolerance", 0.005 ) );
      m.max_stepsize  = static_cast<R>( option( parts, "max_step", 1000. ) );
      m.min_stepsize  = static_cast<R>( option( parts, "min_step", 10. ) );
      use( m );
    } else if ( name == "allen_fast_nystrom" ) {
      AllenFastNystrom<R, Field> m{ &field, static_cast<R>( z0 ), static_cast<R>( z1 ) };
      if ( parts.size() > 1 ) m.step_size = static_cast<R>( std::stod( parts[1] ) );
      m.max_steps = 100000;
      if ( parts.size() <= 1 ) m.max_steps = 100;
      use( m );
    } else if ( name == "allen_utt" ) {
      // the planes are those of the parameter file (2665 mm to 7826 mm), not z0 and z1
      AllenUTT m  = read_allen_utt( g_exports + "/allen_utt" );
      m.full      = option( parts, "full", 0 ) != 0;
      m.polarity  = static_cast<float>( option( parts, "polarity", 1 ) );
      use( m );
    } else if ( name == "whole_crossing" ) {
      auto m = read_whole_crossing<R>( g_exports + "/networks/" + parts.at( 1 ) );
      if ( option( parts, "fast_tanh", 0 ) != 0 ) m.body.activation = Activation::TanhFromExp;
      use( m );
    } else if ( name == "chain" ) {
      auto m = read_stage_chain<R, Field>( g_exports + "/networks/" + parts.at( 1 ), field );
      m.samples_of_the_scale = static_cast<int>( option( parts, "samples", 16 ) );
      if ( option( parts, "fast_tanh", 0 ) != 0 ) m.body.activation = Activation::TanhFromExp;
      use( m );
    } else if ( name == "corrected" ) {
      // corrected:NAME[:fast_tanh=1]  a start plus a correction, <exports>/corrections/NAME
      auto m = read_corrected_start<R, Field>( g_exports + "/corrections/" + parts.at( 1 ), field, z0, z1 );
      if ( option( parts, "fast_tanh", 0 ) != 0 ) m.of_network.body.activation = Activation::TanhFromExp;
      use( m );
    } else {
      throw std::runtime_error( "no method is named " + name );
    }
  }

  template <class R>
  std::vector<State<R>> states_of( const std::vector<double>& rows, long n ) {
    std::vector<State<R>> states( static_cast<std::size_t>( n ) );
    for ( long i = 0; i < n; ++i )
      states[i] = { static_cast<R>( rows[5 * i] ), static_cast<R>( rows[5 * i + 1] ), static_cast<R>( rows[5 * i + 2] ),
                    static_cast<R>( rows[5 * i + 3] ), static_cast<R>( rows[5 * i + 4] ) };
    return states;
  }

  // ------------------------------------------------------------ run
  template <class R>
  void run( World& world, const Args& args ) {
    const auto rows = read_f64( need( args, "states" ) );
    long       n    = static_cast<long>( rows.size() / 5 );
    if ( args.count( "n" ) ) n = std::min( n, std::stol( args.at( "n" ) ) );
    const auto                states = states_of<R>( rows, n );
    const double              z0 = std::stod( need( args, "z0" ) ), z1 = std::stod( need( args, "z1" ) );
    std::vector<double>       finals( 5 * static_cast<std::size_t>( n ) );
    std::vector<std::int64_t> counts( 3 * static_cast<std::size_t>( n ) );
    std::vector<double>       jacobians; // 12 numbers a track, for a method that has a transport matrix
    with_field<R>( world, need( args, "map" ), need( args, "reader" ), [&]( const auto& field ) {
      const Counting<std::decay_t<decltype( field )>> counted( field );
      with_method<R>( need( args, "method" ), counted, z0, z1, [&]( const auto& carry ) {
        for ( long i = 0; i < n; ++i ) {
          counted.reset();
          const State<R> f  = carry( states[i] );
          finals[5 * i]     = static_cast<double>( f.x );
          finals[5 * i + 1] = static_cast<double>( f.y );
          finals[5 * i + 2] = static_cast<double>( f.tx );
          finals[5 * i + 3] = static_cast<double>( f.ty );
          finals[5 * i + 4] = static_cast<double>( f.qop );
          counts[3 * i]     = counted.reads();
          counts[3 * i + 1] = carry.last_steps;
          counts[3 * i + 2] = carry.last_failed;
          if constexpr ( requires { carry.last_jacobian[0][0]; } ) {
            for ( int r = 0; r < 4; ++r )
              for ( int c = 0; c < 3; ++c ) jacobians.push_back( carry.last_jacobian[r][c] );
          }
        }
      } );
    } );
    write_raw( need( args, "out" ), finals );
    if ( args.count( "counts" ) ) write_raw( args.at( "counts" ), counts );
    if ( args.count( "jacobian" ) ) write_raw( args.at( "jacobian" ), jacobians );
  }

  // ------------------------------------------------------------ time
  struct Timed {
    Args                                       request;
    std::function<double()>                    pass;          // ns per track, tracks in the order of the file
    std::function<double()>                    pass_shuffled; // the same with the tracks in a random order
    std::function<double( std::size_t )>       one;           // ns of one call, track i
    std::size_t                                n = 0;
    std::vector<double>                        times, times_shuffled, cold;
  };

  template <class R>
  void prepare_timed( World& world, Timed& timed ) {
    const Args& args = timed.request;
    const auto  rows = read_f64( need( args, "states" ) );
    long        n    = static_cast<long>( rows.size() / 5 );
    if ( args.count( "n" ) ) n = std::min( n, std::stol( args.at( "n" ) ) );
    auto states   = std::make_shared<std::vector<State<R>>>( states_of<R>( rows, n ) );
    auto shuffled = std::make_shared<std::vector<State<R>>>( *states );
    std::mt19937 generator( 20261124 );
    std::shuffle( shuffled->begin(), shuffled->end(), generator );
    auto         out = std::make_shared<std::vector<State<R>>>( static_cast<std::size_t>( n ) );
    const double z0 = std::stod( need( args, "z0" ) ), z1 = std::stod( need( args, "z1" ) );
    timed.n = static_cast<std::size_t>( n );
    with_field<R>( world, need( args, "map" ), need( args, "reader" ), [&]( const auto& field ) {
      with_method<R>( need( args, "method" ), field, z0, z1, [&]( const auto& carry ) {
        const auto loop = [carry, out]( const std::vector<State<R>>& in ) -> double {
          auto&             o     = *out;
          const std::size_t count = in.size();
          const auto        t0    = Clock::now();
          for ( std::size_t i = 0; i < count; ++i ) o[i] = carry( in[i] );
          const auto t1 = Clock::now();
          return std::chrono::duration<double, std::nano>( t1 - t0 ).count() / static_cast<double>( count );
        };
        timed.pass          = [loop, states]() { return loop( *states ); };
        timed.pass_shuffled = [loop, shuffled]() { return loop( *shuffled ); };
        timed.one           = [carry, states, out]( std::size_t i ) -> double {
          const auto t0 = Clock::now();
          ( *out )[i]   = carry( ( *states )[i] );
          const auto t1 = Clock::now();
          return std::chrono::duration<double, std::nano>( t1 - t0 ).count();
        };
      } );
    } );
  }

  double median_of( std::vector<double> values ) {
    if ( values.empty() ) return 0;
    std::sort( values.begin(), values.end() );
    const std::size_t n = values.size();
    return n % 2 ? values[n / 2] : 0.5 * ( values[n / 2 - 1] + values[n / 2] );
  }

  // The time of the clock itself, for the calls that are timed one by one.
  double clock_overhead_ns() {
    std::vector<double> samples;
    for ( int i = 0; i < 2001; ++i ) {
      const auto t0 = Clock::now();
      const auto t1 = Clock::now();
      samples.push_back( std::chrono::duration<double, std::nano>( t1 - t0 ).count() );
    }
    return median_of( samples );
  }

  void time_methods( World& world, const Args& args ) {
    std::ifstream list( need( args, "list" ) );
    if ( !list ) throw std::runtime_error( "cannot open the list" );
    std::vector<Timed> methods;
    std::string        line;
    while ( std::getline( list, line ) ) {
      if ( line.empty() || line[0] == '#' ) continue;
      std::istringstream       words( line );
      std::vector<std::string> parts;
      std::string              word;
      while ( words >> word ) parts.push_back( word );
      Timed timed;
      timed.request = parse( parts );
      methods.push_back( std::move( timed ) );
    }
    for ( auto& timed : methods ) {
      if ( need( timed.request, "precision" ) == "double" )
        prepare_timed<double>( world, timed );
      else
        prepare_timed<float>( world, timed );
    }
    const int         repeats = std::stoi( get( args, "repeats", "15" ) );
    const double      budget  = std::stod( get( args, "budget", "1e9" ) ); // seconds a method, at most
    const std::size_t cold    = static_cast<std::size_t>( std::stol( get( args, "cold", "0" ) ) );
    // The warm-up pass is not timed as a result; it sets how many passes a slow
    // method gets, so that no method takes more than `budget` seconds (but 3 at least).
    std::vector<int> passes( methods.size(), repeats );
    for ( std::size_t m = 0; m < methods.size(); ++m ) {
      const double seconds = methods[m].pass() * 1e-9 * static_cast<double>( methods[m].n );
      methods[m].pass_shuffled();
      passes[m] = std::max( 3, std::min( repeats, static_cast<int>( budget / ( 2 * seconds ) ) ) );
    }
    std::mt19937             generator( 7 );
    std::vector<std::size_t> order( methods.size() );
    std::iota( order.begin(), order.end(), 0 );
    for ( int r = 0; r < repeats; ++r ) {
      std::shuffle( order.begin(), order.end(), generator );
      for ( const std::size_t m : order )
        if ( r < passes[m] ) methods[m].times.push_back( methods[m].pass() );
      std::shuffle( order.begin(), order.end(), generator );
      for ( const std::size_t m : order )
        if ( r < passes[m] ) methods[m].times_shuffled.push_back( methods[m].pass_shuffled() );
    }
    const double overhead = clock_overhead_ns();
    if ( cold > 0 ) {
      std::vector<char> buffer( 64u << 20, 1 );
      for ( std::size_t i = 0; i < cold; ++i ) {
        std::shuffle( order.begin(), order.end(), generator );
        for ( const std::size_t m : order ) {
          const std::size_t track = ( i * 37 ) % methods[m].n;
          long              sum   = 0;
          for ( std::size_t j = 0; j < buffer.size(); j += 64 ) sum += ++buffer[j];
          if ( sum == -1 ) std::puts( "" );
          methods[m].cold.push_back( methods[m].one( track ) - overhead );
        }
      }
    }
    char host[256] = "unknown";
    gethostname( host, sizeof( host ) );
    std::ofstream out( need( args, "out" ) );
    out << "label,method,precision,map,reader,tracks,repeats,ns_lowest,ns_median,ns_highest,"
           "ns_shuffled_lowest,ns_shuffled_median,cold_calls,cold_ns_median,cold_ns_lowest,"
           "clock_overhead_ns,host\n";
    for ( auto& timed : methods ) {
      const auto& t = timed.times;
      const auto& s = timed.times_shuffled;
      char        text[2048];
      std::snprintf( text, sizeof( text ), "%s,%s,%s,%s,%s,%zu,%zu,%.3f,%.3f,%.3f,%.3f,%.3f,%zu,%.1f,%.1f,%.1f,%s\n",
                     get( timed.request, "label", need( timed.request, "method" ) ).c_str(),
                     need( timed.request, "method" ).c_str(), need( timed.request, "precision" ).c_str(),
                     need( timed.request, "map" ).c_str(), need( timed.request, "reader" ).c_str(), timed.n,
                     t.size(),
                     *std::min_element( t.begin(), t.end() ), median_of( t ), *std::max_element( t.begin(), t.end() ),
                     *std::min_element( s.begin(), s.end() ), median_of( s ), timed.cold.size(),
                     median_of( timed.cold ),
                     timed.cold.empty() ? 0.0 : *std::min_element( timed.cold.begin(), timed.cold.end() ), overhead,
                     host );
      out << text;
      std::fputs( text, stdout );
    }
  }

  // ------------------------------------------------------------ the time of one field read
  template <class R, class Field>
  void time_reads( const Field& field, const std::vector<R>& points, int repeats, double& lowest, double& median,
                   double& checksum ) {
    const std::size_t   n = points.size() / 3;
    std::vector<double> times;
    R                   sum = 0;
    for ( int r = 0; r < repeats + 1; ++r ) {
      sum           = 0;
      const auto t0 = Clock::now();
      for ( std::size_t i = 0; i < n; ++i ) {
        R bx, by, bz;
        field( points[3 * i], points[3 * i + 1], points[3 * i + 2], bx, by, bz );
        sum += bx + by + bz;
      }
      const auto t1 = Clock::now();
      if ( r > 0 ) times.push_back( std::chrono::duration<double, std::nano>( t1 - t0 ).count() / static_cast<double>( n ) );
    }
    lowest   = *std::min_element( times.begin(), times.end() );
    median   = median_of( times );
    checksum = static_cast<double>( sum );
  }

  template <class R>
  void micro_field_of( World& world, const std::string& map, const std::vector<double>& track_rows, double z0,
                       double z1, std::ofstream& out ) {
    const std::size_t points = 1u << 20;
    std::mt19937_64   generator( 11 );
    std::uniform_real_distribution<double> ux( -2500., 2500. ), uy( -2000., 2000. ), uz( z0, z1 );
    std::vector<R>                         random( 3 * points );
    for ( std::size_t i = 0; i < points; ++i ) {
      random[3 * i]     = static_cast<R>( ux( generator ) );
      random[3 * i + 1] = static_cast<R>( uy( generator ) );
      random[3 * i + 2] = static_cast<R>( uz( generator ) );
    }
    // along the straight lines of the test tracks, 64 points a track, track after track
    const std::size_t tracks = track_rows.size() / 5;
    std::vector<R>    along( 3 * 64 * tracks );
    for ( std::size_t t = 0; t < tracks; ++t ) {
      for ( int m = 0; m < 64; ++m ) {
        const double s             = ( m + 0.5 ) * ( z1 - z0 ) / 64;
        along[3 * ( 64 * t + m )]     = static_cast<R>( track_rows[5 * t] + track_rows[5 * t + 2] * s );
        along[3 * ( 64 * t + m ) + 1] = static_cast<R>( track_rows[5 * t + 1] + track_rows[5 * t + 3] * s );
        along[3 * ( 64 * t + m ) + 2] = static_cast<R>( z0 + s );
      }
    }
    const char* precision = std::is_same_v<R, double> ? "double" : "float";
    for ( const std::string reader : { "linear", "cubic", "rec", "allen" } ) {
      if ( reader == "cubic" && world.map( map ).cubic.empty() ) continue;
      with_field<R>( world, map, reader, [&]( const auto& field ) {
        double lowest, median, checksum;
        time_reads<R>( field, random, 9, lowest, median, checksum );
        out << map << ',' << reader << ',' << precision << ",random points," << points << ',' << lowest << ','
            << median << ',' << checksum << '\n';
        std::printf( "%-8s %-7s %-6s random points       %8.2f ns (median %8.2f)\n", map.c_str(), reader.c_str(),
                     precision, lowest, median );
        time_reads<R>( field, along, 9, lowest, median, checksum );
        out << map << ',' << reader << ',' << precision << ",along the tracks," << along.size() / 3 << ',' << lowest
            << ',' << median << ',' << checksum << '\n';
        std::printf( "%-8s %-7s %-6s along the tracks    %8.2f ns (median %8.2f)\n", map.c_str(), reader.c_str(),
                     precision, lowest, median );
      } );
    }
  }

  void micro_field( World& world, const Args& args ) {
    const std::string map  = need( args, "map" );
    const auto        rows = read_f64( need( args, "states" ) );
    const double      z0 = std::stod( need( args, "z0" ) ), z1 = std::stod( need( args, "z1" ) );
    std::ofstream     out( need( args, "out" ) );
    out << "map,reader,precision,pattern,reads,ns_lowest,ns_median,checksum\n";
    micro_field_of<double>( world, map, rows, z0, z1, out );
    micro_field_of<float>( world, map, rows, z0, z1, out );
  }

  // ------------------------------------------------------------ the time of one network call
  template <class R>
  void micro_mlp_of( std::ofstream& out ) {
    const char*        precision = std::is_same_v<R, double> ? "double" : "float";
    const int          inputs = 6, outputs = 8;
    const std::size_t  samples = 4096;
    std::vector<R>     in( inputs * samples );
    std::mt19937       generator( 3 );
    std::uniform_real_distribution<double> uniform( -1., 1. );
    for ( auto& v : in ) v = static_cast<R>( uniform( generator ) );
    for ( const int depth : { 1, 2, 3 } ) {
      for ( const int width : { 8, 16, 32, 64, 128, 256, 512 } ) {
        for ( const int kind : { 0, 1, 2 } ) {
          const char* names[3] = { "tanh of the C library", "tanh from the vector exponential", "none" };
          Body<R>     body     = random_body<R>( inputs, width, depth, outputs );
          body.activation      = kind == 0 ? Activation::Tanh : ( kind == 1 ? Activation::TanhFromExp : Activation::None );
          const long calls = std::max<long>( 2000, 40000000L / std::max<long>( 1, body.multiply_adds() ) );
          std::vector<double> times;
          R                   sum = 0;
          for ( int r = 0; r < 8; ++r ) {
            sum           = 0;
            const auto t0 = Clock::now();
            for ( long c = 0; c < calls; ++c ) {
              R result[8] = {};
              body.forward( &in[inputs * ( static_cast<std::size_t>( c ) % samples )], result );
              sum += result[0] + result[7];
            }
            const auto t1 = Clock::now();
            if ( r > 0 ) times.push_back( std::chrono::duration<double, std::nano>( t1 - t0 ).count() / static_cast<double>( calls ) );
          }
          const double lowest = *std::min_element( times.begin(), times.end() );
          out << precision << ',' << depth << ',' << width << ',' << names[kind] << ',' << body.multiply_adds()
              << ',' << body.hidden_units() << ',' << calls << ',' << lowest << ',' << median_of( times ) << ','
              << static_cast<double>( sum ) << '\n';
          std::printf( "%-6s depth %d width %4d %-33s %8ld multiply-adds  %9.1f ns a call (median %9.1f)\n",
                       precision, depth, width, names[kind], body.multiply_adds(), lowest, median_of( times ) );
        }
      }
    }
  }

  void micro_mlp( const Args& args ) {
    std::ofstream out( need( args, "out" ) );
    out << "precision,depth,width,activation,multiply_adds,hidden_units,calls,ns_lowest,ns_median,checksum\n";
    micro_mlp_of<double>( out );
    micro_mlp_of<float>( out );
  }

} // namespace

int main( int argc, char** argv ) {
  try {
    if ( argc < 2 ) throw std::runtime_error( "fx run | time | micro_field | micro_mlp  key=value ..." );
    const std::string        command = argv[1];
    std::vector<std::string> words( argv + 2, argv + argc );
    const Args               args = parse( words );
    if ( args.count( "exports" ) ) g_exports = args.at( "exports" );
    World world;
    if ( command == "run" ) {
      if ( need( args, "precision" ) == "double" )
        run<double>( world, args );
      else
        run<float>( world, args );
    } else if ( command == "time" ) {
      time_methods( world, args );
    } else if ( command == "micro_field" ) {
      micro_field( world, args );
    } else if ( command == "micro_mlp" ) {
      micro_mlp( args );
    } else {
      throw std::runtime_error( "no command is named " + command );
    }
  } catch ( const std::exception& error ) {
    std::fprintf( stderr, "fx: %s\n", error.what() );
    return 1;
  }
  return 0;
}
