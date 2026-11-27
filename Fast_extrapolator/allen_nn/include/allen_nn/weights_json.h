// Host side: read the weights of a magnet-crossing network from its JSON file
// (written by allen_nn/python/export_json.py) into a CrossingNetwork<R>.
//
// The parser is a small one for this file (objects, arrays, numbers, strings,
// true/false/null), so that the tool needs no library. In Allen the same file can
// be read with nlohmann::json; the fields are the same.
#pragma once

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "magnet_crossing.h"

namespace allen_nn {

  struct Json {
    enum Kind { Null, Bool, Number, String, Array, Object } kind = Null;
    double                      number = 0;
    std::string                 text;
    std::vector<Json>           items;
    std::map<std::string, Json> fields;

    const Json& operator[]( const std::string& key ) const {
      auto found = fields.find( key );
      if ( found == fields.end() ) throw std::runtime_error( "weights file: no field " + key );
      return found->second;
    }
    bool has( const std::string& key ) const { return fields.count( key ) != 0; }
  };

  namespace detail {
    struct Parser {
      const std::string& t;
      std::size_t        i = 0;
      void               blank() {
        while ( i < t.size() && std::isspace( static_cast<unsigned char>( t[i] ) ) ) ++i;
      }
      [[noreturn]] void fail( const char* what ) {
        throw std::runtime_error( std::string( "weights file: " ) + what + " at " + std::to_string( i ) );
      }
      std::string string() {
        if ( t[i] != '"' ) fail( "a string expected" );
        std::string out;
        for ( ++i; i < t.size() && t[i] != '"'; ++i ) {
          if ( t[i] == '\\' ) ++i;
          out += t[i];
        }
        ++i;
        return out;
      }
      Json value() {
        blank();
        Json v;
        if ( i >= t.size() ) fail( "end of file" );
        const char c = t[i];
        if ( c == '{' ) {
          v.kind = Json::Object;
          ++i;
          blank();
          if ( t[i] == '}' ) return ++i, v;
          for ( ;; ) {
            blank();
            const std::string key = string();
            blank();
            if ( t[i++] != ':' ) fail( "':' expected" );
            v.fields[key] = value();
            blank();
            if ( t[i] == ',' ) {
              ++i;
              continue;
            }
            if ( t[i++] != '}' ) fail( "'}' expected" );
            return v;
          }
        }
        if ( c == '[' ) {
          v.kind = Json::Array;
          ++i;
          blank();
          if ( t[i] == ']' ) return ++i, v;
          for ( ;; ) {
            v.items.push_back( value() );
            blank();
            if ( t[i] == ',' ) {
              ++i;
              continue;
            }
            if ( t[i++] != ']' ) fail( "']' expected" );
            return v;
          }
        }
        if ( c == '"' ) {
          v.kind = Json::String;
          v.text = string();
          return v;
        }
        if ( t.compare( i, 4, "true" ) == 0 ) return i += 4, v.kind = Json::Bool, v.number = 1, v;
        if ( t.compare( i, 5, "false" ) == 0 ) return i += 5, v.kind = Json::Bool, v;
        if ( t.compare( i, 4, "null" ) == 0 ) return i += 4, v;
        char*       end   = nullptr;
        const char* start = t.c_str() + i;
        v.number          = std::strtod( start, &end );
        if ( end == start ) fail( "a value expected" );
        v.kind = Json::Number;
        i += static_cast<std::size_t>( end - start );
        return v;
      }
    };
  } // namespace detail

  inline Json read_json( const std::string& path ) {
    std::ifstream in( path );
    if ( !in ) throw std::runtime_error( "cannot open " + path );
    std::stringstream buffer;
    buffer << in.rdbuf();
    const std::string text = buffer.str();
    detail::Parser    p{ text };
    return p.value();
  }

  // The network of a weights file, in precision R. The planes of the crossing are
  // computed in double and rounded, as the training computed its starts.
  template <class R>
  inline CrossingNetwork<R> read_crossing_network( const std::string& path ) {
    const Json         j = read_json( path );
    CrossingNetwork<R> net;
    net.steps = static_cast<int>( j["steps"].number );
    net.depth = static_cast<int>( j["depth"].number );
    net.width = static_cast<int>( j["width"].number );
    if ( net.steps < 1 || net.steps > MOST_STEPS || net.depth < 1 || net.depth > MOST_HIDDEN || net.width < 1 ||
         net.width > MOST_WIDTH )
      throw std::runtime_error( "weights file: steps, depth or width out of range" );
    const double z0 = j["z0"].number, z1 = j["z1"].number;
    net.z0          = static_cast<R>( z0 );
    net.z1          = static_cast<R>( z1 );
    for ( int i = 0; i <= net.steps; ++i ) net.planes[i] = static_cast<R>( z0 + i * ( ( z1 - z0 ) / net.steps ) );
    for ( int a = 0; a < 5; ++a ) net.inv_scale_in[a] = static_cast<R>( 1.0 / j["scale_of_inputs"].items.at( a ).number );
    for ( int a = 0; a < 4; ++a ) net.scale_out[a] = static_cast<R>( j["scale_of_outputs"].items.at( a ).number );
    net.inv_scale_qop = static_cast<R>( 1.0 / j["scale_of_qop"].number );
    const Json& d     = j["domain"];
    net.most_qop      = static_cast<R>( d["most_abs_qop"].number );
    net.most_tx       = static_cast<R>( d["most_abs_tx"].number );
    net.most_ty       = static_cast<R>( d["most_abs_ty"].number );
    net.most_x        = static_cast<R>( d["most_abs_x"].number );
    net.most_y        = static_cast<R>( d["most_abs_y"].number );
    if ( j.has( "leg_step_mm" ) ) net.leg_step = static_cast<R>( j["leg_step_mm"].number );
    if ( j.has( "network_jacobian" ) ) net.network_jacobian = static_cast<int>( j["network_jacobian"].number );
    if ( j.has( "fallback_step_error_mm" ) ) net.most_step_error = static_cast<R>( j["fallback_step_error_mm"].number );

    const auto& layers = j["layers"].items;
    if ( static_cast<int>( layers.size() ) != net.depth + 1 ) throw std::runtime_error( "weights file: layers" );
    auto matrix = [&]( const Json& W, int rows, int cols, R* out ) {
      if ( static_cast<int>( W.items.size() ) != rows ) throw std::runtime_error( "weights file: rows" );
      for ( int r = 0; r < rows; ++r ) {
        if ( static_cast<int>( W.items[r].items.size() ) != cols ) throw std::runtime_error( "weights file: columns" );
        for ( int c = 0; c < cols; ++c ) out[r * cols + c] = static_cast<R>( W.items[r].items[c].number );
      }
    };
    auto vector = [&]( const Json& b, int n, R* out ) {
      if ( static_cast<int>( b.items.size() ) != n ) throw std::runtime_error( "weights file: bias" );
      for ( int r = 0; r < n; ++r ) out[r] = static_cast<R>( b.items[r].number );
    };
    const int w = net.width;
    matrix( layers[0]["W"], w, 5, net.W_first );
    vector( layers[0]["b"], w, net.b_first );
    for ( int l = 1; l < net.depth; ++l ) {
      matrix( layers[l]["W"], w, w, net.W_hidden[l - 1] );
      vector( layers[l]["b"], w, net.b_hidden[l - 1] );
    }
    matrix( layers[net.depth]["W"], 4, w, net.W_last );
    vector( layers[net.depth]["b"], 4, net.b_last );
    return net;
  }

} // namespace allen_nn
