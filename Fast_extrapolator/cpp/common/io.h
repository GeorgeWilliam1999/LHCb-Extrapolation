// Reading and writing the flat files of the experiment.
//
// Every array is one file of raw little-endian numbers (.f64 float64, .i64
// int64) and every folder has a `meta` file of lines `key value`. The files
// are written by python/fastex/export.py.
#pragma once

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fx {

  template <class T>
  inline std::vector<T> read_raw( const std::string& path ) {
    std::ifstream in( path, std::ios::binary | std::ios::ate );
    if ( !in ) throw std::runtime_error( "cannot open " + path );
    const std::streamsize bytes = in.tellg();
    if ( bytes % static_cast<std::streamsize>( sizeof( T ) ) != 0 )
      throw std::runtime_error( "the size of " + path + " is not a whole number of values" );
    std::vector<T> values( static_cast<std::size_t>( bytes ) / sizeof( T ) );
    in.seekg( 0 );
    in.read( reinterpret_cast<char*>( values.data() ), bytes );
    if ( !in ) throw std::runtime_error( "cannot read " + path );
    return values;
  }

  inline std::vector<double>       read_f64( const std::string& path ) { return read_raw<double>( path ); }
  inline std::vector<std::int64_t> read_i64( const std::string& path ) { return read_raw<std::int64_t>( path ); }

  template <class T>
  inline void write_raw( const std::string& path, const std::vector<T>& values ) {
    std::ofstream out( path, std::ios::binary );
    if ( !out ) throw std::runtime_error( "cannot write " + path );
    out.write( reinterpret_cast<const char*>( values.data() ),
               static_cast<std::streamsize>( values.size() * sizeof( T ) ) );
  }

  // The `meta` file of a folder: lines `key value`; the value is the rest of the line.
  struct Meta {
    std::map<std::string, std::string> values;
    std::string                        path;

    bool has( const std::string& key ) const { return values.count( key ) != 0; }

    const std::string& text( const std::string& key ) const {
      const auto found = values.find( key );
      if ( found == values.end() ) throw std::runtime_error( "no key " + key + " in " + path );
      return found->second;
    }
    double number( const std::string& key ) const { return std::stod( text( key ) ); }
    long   integer( const std::string& key ) const { return std::stol( text( key ) ); }
  };

  inline Meta read_meta( const std::string& path ) {
    std::ifstream in( path );
    if ( !in ) throw std::runtime_error( "cannot open " + path );
    Meta        meta;
    std::string line;
    meta.path = path;
    while ( std::getline( in, line ) ) {
      const auto space = line.find( ' ' );
      if ( space == std::string::npos ) continue;
      meta.values[line.substr( 0, space )] = line.substr( space + 1 );
    }
    return meta;
  }

} // namespace fx
