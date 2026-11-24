// The magnetic field map and four ways to read it.
//
// A map is a regular grid of 81 x 81 x 146 points, 100 mm apart, with
// (Bx, By, Bz) on every point, in the unit of the map file (the Gaudi unit:
// 1 tesla = 1e-3). Every reader returns the field in tesla at a point in mm.
//
//   LinearPackage<R>  trilinear, with the arithmetic of the present Python
//                     package (FieldMap.__call__) in the same order: the cell
//                     index is clamped to the grid, so the field is held
//                     constant outside it. R is double or float.
//   CubicPackage<R>   tricubic B-spline through the grid points, with the
//                     arithmetic of the present package (CubicFieldMap); the
//                     coefficients come from the export (scipy spline_filter).
//   RecField          the read of LHCb's Detector project,
//                     MagneticFieldGrid::fieldVectorLinearInterpolation
//                     (Detector v3r19, Core/src/MagneticFieldGrid.cpp): the
//                     position is cast to float, four floats are handled side
//                     by side, and the field is zero outside the grid. The
//                     HLT2 integrator reads the field this way.
//   AllenField        the read of Allen, Magfield::fieldVectorLinearInterpolation
//                     without textures (Allen v9r3, MagneticField.cuh): float,
//                     three separate arrays, zero outside the grid.
//
// Units. LinearPackage, CubicPackage and AllenField return tesla (AllenField
// holds its arrays in tesla, so it does no more work than its original).
// RecField returns the unit of the map file, the Gaudi unit, with the scale
// factor 1 of the conditions, exactly as the original: any scaling in single
// precision would change its last digits. Every reader states its unit in
// `tesla_per_unit`, and an integrator folds kappa * tesla_per_unit into q/p one
// time per call. For RecField that product is exactly 1, so the integrator
// works with q/p times c_light and the field in Gaudi units, as Rec does.
#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>

#include "../common/io.h"

namespace fx {

  struct MapFile {
    int                 nx = 0, ny = 0, nz = 0;
    double              inv[3]{}, low[3]{};
    double              scale_to_tesla = 1;
    std::vector<double> grid;  // Bx, then By, then Bz; each nz * ny * nx values, x fastest; file unit
    std::vector<double> cubic; // B-spline coefficients, same layout, tesla; empty when not exported
    std::string         name;
    std::size_t         points() const { return static_cast<std::size_t>( nx ) * ny * nz; }
  };

  inline MapFile read_map( const std::string& folder ) {
    const Meta meta = read_meta( folder + "/meta" );
    MapFile    map;
    map.name = meta.text( "name" );
    map.nx   = static_cast<int>( meta.integer( "nx" ) );
    map.ny   = static_cast<int>( meta.integer( "ny" ) );
    map.nz   = static_cast<int>( meta.integer( "nz" ) );
    const char* letters[3] = { "x", "y", "z" };
    for ( int a = 0; a < 3; ++a ) {
      map.inv[a] = meta.number( std::string( "inverse_cell_size_" ) + letters[a] );
      map.low[a] = meta.number( std::string( "lowest_corner_" ) + letters[a] );
    }
    map.scale_to_tesla = meta.number( "scale_to_tesla" );
    map.grid           = read_f64( folder + "/grid.f64" );
    if ( map.grid.size() != 3 * map.points() ) throw std::runtime_error( "grid.f64 has the wrong size" );
    if ( meta.integer( "has_cubic" ) ) {
      map.cubic = read_f64( folder + "/cubic.f64" );
      if ( map.cubic.size() != 3 * map.points() ) throw std::runtime_error( "cubic.f64 has the wrong size" );
    }
    return map;
  }

  // (Bx, By, Bz, 0) on every grid point, side by side.
  template <class R>
  inline std::vector<std::array<R, 4>> side_by_side( const std::vector<double>& three, std::size_t points ) {
    std::vector<std::array<R, 4>> b( points );
    for ( std::size_t i = 0; i < points; ++i )
      b[i] = { static_cast<R>( three[i] ), static_cast<R>( three[points + i] ),
               static_cast<R>( three[2 * points + i] ), R( 0 ) };
    return b;
  }

  // ---------------------------------------------------------------- trilinear, package arithmetic
  template <class R>
  class LinearPackage {
  public:
    static constexpr double tesla_per_unit = 1.0;
    explicit LinearPackage( const MapFile& map )
        : m_nx( map.nx ), m_ny( map.ny ), m_nz( map.nz ), m_scale( static_cast<R>( map.scale_to_tesla ) ),
          m_b( side_by_side<R>( map.grid, map.points() ) ) {
      for ( int a = 0; a < 3; ++a ) {
        m_inv[a] = static_cast<R>( map.inv[a] );
        m_low[a] = static_cast<R>( map.low[a] );
      }
    }

    // The call converts its arguments to R and the field back, so that an
    // integrator in one precision can read a map held in the other.
    template <class T>
    inline void operator()( T x, T y, T z, T& bx, T& by, T& bz ) const {
      R rx, ry, rz;
      read( static_cast<R>( x ), static_cast<R>( y ), static_cast<R>( z ), rx, ry, rz );
      bx = static_cast<T>( rx );
      by = static_cast<T>( ry );
      bz = static_cast<T>( rz );
    }

    inline void read( R x, R y, R z, R& bx, R& by, R& bz ) const {
      const R fx = ( x - m_low[0] ) * m_inv[0];
      const R fy = ( y - m_low[1] ) * m_inv[1];
      const R fz = ( z - m_low[2] ) * m_inv[2];
      const long ix = clamp_index( static_cast<long>( fx ), m_nx - 2 );
      const long iy = clamp_index( static_cast<long>( fy ), m_ny - 2 );
      const long iz = clamp_index( static_cast<long>( fz ), m_nz - 2 );
      const R    tx = clamp_unit( fx - static_cast<R>( ix ) );
      const R    ty = clamp_unit( fy - static_cast<R>( iy ) );
      const R    tz = clamp_unit( fz - static_cast<R>( iz ) );
      const std::size_t i000 = static_cast<std::size_t>( ix ) +
                               static_cast<std::size_t>( m_nx ) *
                                   ( static_cast<std::size_t>( iy ) + static_cast<std::size_t>( m_ny ) * iz );
      const std::size_t dy = m_nx, dz = static_cast<std::size_t>( m_nx ) * m_ny;
      const R* c000 = m_b[i000].data();
      const R* c100 = m_b[i000 + 1].data();
      const R* c010 = m_b[i000 + dy].data();
      const R* c110 = m_b[i000 + dy + 1].data();
      const R* c001 = m_b[i000 + dz].data();
      const R* c101 = m_b[i000 + dz + 1].data();
      const R* c011 = m_b[i000 + dz + dy].data();
      const R* c111 = m_b[i000 + dz + dy + 1].data();
      R        out[4];
      for ( int c = 0; c < 4; ++c ) {
        const R c00 = c000[c] * ( 1 - tx ) + c100[c] * tx;
        const R c10 = c010[c] * ( 1 - tx ) + c110[c] * tx;
        const R c01 = c001[c] * ( 1 - tx ) + c101[c] * tx;
        const R c11 = c011[c] * ( 1 - tx ) + c111[c] * tx;
        const R c0  = c00 * ( 1 - ty ) + c10 * ty;
        const R c1  = c01 * ( 1 - ty ) + c11 * ty;
        out[c]      = ( c0 * ( 1 - tz ) + c1 * tz ) * m_scale;
      }
      bx = out[0];
      by = out[1];
      bz = out[2];
    }

  private:
    static inline long clamp_index( long i, long highest ) { return i < 0 ? 0 : ( i > highest ? highest : i ); }
    static inline R    clamp_unit( R t ) { return t < 0 ? R( 0 ) : ( t > 1 ? R( 1 ) : t ); }

    int                           m_nx, m_ny, m_nz;
    R                             m_inv[3], m_low[3], m_scale;
    std::vector<std::array<R, 4>> m_b;
  };

  // ---------------------------------------------------------------- tricubic B-spline, package arithmetic
  template <class R>
  class CubicPackage {
  public:
    static constexpr double tesla_per_unit = 1.0;
    explicit CubicPackage( const MapFile& map )
        : m_nx( map.nx ), m_ny( map.ny ), m_nz( map.nz ), m_c( side_by_side<R>( map.cubic, map.points() ) ) {
      if ( map.cubic.empty() ) throw std::runtime_error( "the map " + map.name + " has no cubic coefficients" );
      for ( int a = 0; a < 3; ++a ) {
        m_inv[a] = static_cast<R>( map.inv[a] );
        m_low[a] = static_cast<R>( map.low[a] );
      }
    }

    template <class T>
    inline void operator()( T x, T y, T z, T& bx, T& by, T& bz ) const {
      R rx, ry, rz;
      read( static_cast<R>( x ), static_cast<R>( y ), static_cast<R>( z ), rx, ry, rz );
      bx = static_cast<T>( rx );
      by = static_cast<T>( ry );
      bz = static_cast<T>( rz );
    }

    inline void read( R x, R y, R z, R& bx, R& by, R& bz ) const {
      const R position[3] = { x, y, z };
      const int n[3]      = { m_nx, m_ny, m_nz };
      R         w[3][4];
      long      index[3][4];
      for ( int a = 0; a < 3; ++a ) {
        const R    f = ( position[a] - m_low[a] ) * m_inv[a];
        const R    fl = std::floor( f );
        const long i  = static_cast<long>( fl );
        const R    t = f - fl, t2 = t * t, t3 = t2 * t, u = 1 - t;
        w[a][0] = u * u * u / 6;
        w[a][1] = ( 3 * t3 - 6 * t2 + 4 ) / 6;
        w[a][2] = ( -3 * t3 + 3 * t2 + 3 * t + 1 ) / 6;
        w[a][3] = t3 / 6;
        for ( int o = 0; o < 4; ++o ) {
          const long j = i + o - 1;
          index[a][o]  = j < 0 ? 0 : ( j > n[a] - 1 ? n[a] - 1 : j );
        }
      }
      R total[4] = { 0, 0, 0, 0 };
      for ( int c = 0; c < 4; ++c ) {
        for ( int b = 0; b < 4; ++b ) {
          const std::size_t base =
              static_cast<std::size_t>( m_nx ) *
              ( static_cast<std::size_t>( index[1][b] ) + static_cast<std::size_t>( m_ny ) * index[2][c] );
          R row[4] = { 0, 0, 0, 0 };
          for ( int a = 0; a < 4; ++a ) {
            const R* p = m_c[base + static_cast<std::size_t>( index[0][a] )].data();
            for ( int k = 0; k < 4; ++k ) row[k] += w[0][a] * p[k];
          }
          const R wzy = w[2][c] * w[1][b];
          for ( int k = 0; k < 4; ++k ) total[k] += wzy * row[k];
        }
      }
      bx = total[0];
      by = total[1];
      bz = total[2];
    }

  private:
    int                           m_nx, m_ny, m_nz;
    R                             m_inv[3], m_low[3];
    std::vector<std::array<R, 4>> m_c;
  };

  // ---------------------------------------------------------------- the read of LHCb's Detector project
  class RecField {
  public:
    static constexpr double tesla_per_unit = 1000.0; // the Gaudi unit of the field: 1 tesla = 1e-3

    explicit RecField( const MapFile& map ) : m_B( side_by_side<float>( map.grid, map.points() ) ) {
      if ( map.scale_to_tesla != tesla_per_unit ) throw std::runtime_error( "RecField needs a map in Gaudi units" );
      for ( int a = 0; a < 3; ++a ) {
        m_min[a]     = static_cast<float>( map.low[a] );
        m_invDxyz[a] = static_cast<float>( map.inv[a] );
      }
      m_Nxyz = { map.nx, map.ny, map.nz, 0 };
    }

    template <class R>
    inline void operator()( R x, R y, R z, R& bx, R& by, R& bz ) const {
      const float abc0 = ( static_cast<float>( x ) - m_min[0] ) * m_invDxyz[0];
      const float abc1 = ( static_cast<float>( y ) - m_min[1] ) * m_invDxyz[1];
      const float abc2 = ( static_cast<float>( z ) - m_min[2] ) * m_invDxyz[2];
      const auto  i    = static_cast<std::int32_t>( abc0 );
      const auto  j    = static_cast<std::int32_t>( abc1 );
      const auto  k    = static_cast<std::int32_t>( abc2 );
      if ( i >= 0 && i < m_Nxyz[0] - 1 && //
           j >= 0 && j < m_Nxyz[1] - 1 && //
           k >= 0 && k < m_Nxyz[2] - 1 ) {
        const auto  ijk000 = m_Nxyz[0] * ( m_Nxyz[1] * k + j ) + i;
        const auto  ijk001 = m_Nxyz[0] * ( m_Nxyz[1] * ( k + 1 ) + j ) + i;
        const auto  ijk010 = m_Nxyz[0] * ( m_Nxyz[1] * k + j + 1 ) + i;
        const auto  ijk011 = m_Nxyz[0] * ( m_Nxyz[1] * ( k + 1 ) + j + 1 ) + i;
        const float h1x    = abc0 - static_cast<float>( i );
        const float h1y    = abc1 - static_cast<float>( j );
        const float h1z    = abc2 - static_cast<float>( k );
        const float h0x    = 1.0f - h1x;
        const float h0y    = 1.0f - h1y;
        const float h0z    = 1.0f - h1z;
        const auto  h00    = splat( h0x * h0y );
        const auto  h01    = splat( h0x * h1y );
        const auto  h10    = splat( h1x * h0y );
        const auto  h11    = splat( h1x * h1y );
        const auto  edge_j0 = [&]( auto ijk ) { return madd( h10, load( m_B[ijk + 1] ), h00 * load( m_B[ijk] ) ); };
        const auto  edge_j1 = [&]( auto ijk ) { return madd( h11, load( m_B[ijk + 1] ), h01 * load( m_B[ijk] ) ); };
        const float_4v bf   = load( m_scaleFactor ) * madd( splat( h1z ), edge_j0( ijk001 ) + edge_j1( ijk011 ),
                                                            splat( h0z ) * ( edge_j0( ijk000 ) + edge_j1( ijk010 ) ) );
        bx = static_cast<R>( bf[0] );
        by = static_cast<R>( bf[1] );
        bz = static_cast<R>( bf[2] );
        return;
      }
      bx = by = bz = R( 0 );
    }

  private:
    using float_4v = float __attribute__( ( vector_size( 16 ) ) );
    static inline float_4v load( const std::array<float, 4>& a ) {
      float_4v v;
      std::memcpy( &v, a.data(), sizeof( v ) );
      return v;
    }
    static inline float_4v splat( float a ) { return float_4v{ a, a, a, a }; }
    static inline float_4v madd( float_4v a, float_4v b, float_4v c ) {
      float_4v r;
      for ( int i = 0; i < 4; ++i ) r[i] = std::fma( a[i], b[i], c[i] );
      return r;
    }

    std::array<float, 4>              m_scaleFactor = { 1, 1, 1, 0 };
    std::vector<std::array<float, 4>> m_B;
    std::array<float, 4>              m_min     = { 0., 0., 0., 0. };
    std::array<float, 4>              m_invDxyz = { 0., 0., 0., 0. };
    std::array<std::int32_t, 4>       m_Nxyz    = { 0, 0, 0, 0 };
  };

  // ---------------------------------------------------------------- the read of Allen
  class AllenField {
  public:
    static constexpr double tesla_per_unit = 1.0;
    explicit AllenField( const MapFile& map ) : Nx( map.nx ), Ny( map.ny ), Nz( map.nz ) {
      const std::size_t n = map.points();
      Bx.resize( n );
      By.resize( n );
      Bz.resize( n );
      for ( std::size_t i = 0; i < n; ++i ) {
        Bx[i] = static_cast<float>( map.grid[i] * map.scale_to_tesla );
        By[i] = static_cast<float>( map.grid[n + i] * map.scale_to_tesla );
        Bz[i] = static_cast<float>( map.grid[2 * n + i] * map.scale_to_tesla );
      }
      minX  = static_cast<float>( map.low[0] );
      minY  = static_cast<float>( map.low[1] );
      minZ  = static_cast<float>( map.low[2] );
      invDx = static_cast<float>( map.inv[0] );
      invDy = static_cast<float>( map.inv[1] );
      invDz = static_cast<float>( map.inv[2] );
    }

    template <class R>
    inline void operator()( R px, R py, R pz, R& bx, R& by, R& bz ) const {
      const float x = ( static_cast<float>( px ) - minX ) * invDx;
      const float y = ( static_cast<float>( py ) - minY ) * invDy;
      const float z = ( static_cast<float>( pz ) - minZ ) * invDz;

      const int ix = (int)x;
      const int iy = (int)y;
      const int iz = (int)z;

      if ( ix < 0 || iy < 0 || iz < 0 || ix >= Nx - 1 || iy >= Ny - 1 || iz >= Nz - 1 ) {
        bx = by = bz = R( 0 );
        return;
      }

      const int i000 = Nx * ( Ny * iz + iy ) + ix;
      const int i001 = Nx * ( Ny * ( iz + 1 ) + iy ) + ix;
      const int i010 = Nx * ( Ny * iz + iy + 1 ) + ix;
      const int i011 = Nx * ( Ny * ( iz + 1 ) + iy + 1 ) + ix;

      const float h1x = x - ix;
      const float h1y = y - iy;
      const float h1z = z - iz;

      const float h0x = 1.0f - h1x;
      const float h0y = 1.0f - h1y;
      const float h0z = 1.0f - h1z;

      const float h00 = h0x * h0y;
      const float h01 = h0x * h1y;
      const float h10 = h1x * h0y;
      const float h11 = h1x * h1y;

      bx = static_cast<R>( h0z * ( h00 * Bx[i000] + h10 * Bx[i000 + 1] + h01 * Bx[i010] + h11 * Bx[i010 + 1] ) +
                           h1z * ( h00 * Bx[i001] + h10 * Bx[i001 + 1] + h01 * Bx[i011] + h11 * Bx[i011 + 1] ) );
      by = static_cast<R>( h0z * ( h00 * By[i000] + h10 * By[i000 + 1] + h01 * By[i010] + h11 * By[i010 + 1] ) +
                           h1z * ( h00 * By[i001] + h10 * By[i001 + 1] + h01 * By[i011] + h11 * By[i011 + 1] ) );
      bz = static_cast<R>( h0z * ( h00 * Bz[i000] + h10 * Bz[i000 + 1] + h01 * Bz[i010] + h11 * Bz[i010 + 1] ) +
                           h1z * ( h00 * Bz[i001] + h10 * Bz[i001 + 1] + h01 * Bz[i011] + h11 * Bz[i011 + 1] ) );
    }

  private:
    std::vector<float> Bx, By, Bz;
    int                Nx, Ny, Nz;
    float              invDx, invDy, invDz;
    float              minX, minY, minZ;
  };

  // ---------------------------------------------------------------- a reader that counts its reads
  template <class Field>
  class Counting {
  public:
    static constexpr double tesla_per_unit = Field::tesla_per_unit;
    explicit Counting( const Field& field ) : m_field( &field ) {}
    template <class R>
    inline void operator()( R x, R y, R z, R& bx, R& by, R& bz ) const {
      ++m_reads;
      ( *m_field )( x, y, z, bx, by, bz );
    }
    long reads() const { return m_reads; }
    void reset() const { m_reads = 0; }

  private:
    const Field* m_field;
    mutable long m_reads = 0;
  };

} // namespace fx
