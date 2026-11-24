// A copy of the parametrised step of HLT1 from the Upstream Tracker to the
// SciFi tracker: the only method of LHCb that crosses the magnet with no field
// read.
//
// Source: Allen v9r3,
//   device/event_model/kalman/include/KalmanParametrizations.cuh
//       (KalmanParametrizations: read_params_UTT, Read, UttOffset, compute_state)
//   device/kalman/ParKalman/include/ParKalmanMethods.cuh
//       (extrapUTT and ExtrapolateUTT)
//   device/event_model/kalman/include/ParKalmanDefinitions.cuh (the sizes)
// with the parameter files that Allen v9r3 loads from PARAM/ParamFiles:
//   ParametrizedKalmanFit/25v0/params_UTT_v0.tab   the tabulated polynomials
//   ParametrizedKalmanFit/25v1/params.json         UTT_META and UTTFParams
//
// The step goes from z = ZINI to z = ZFIN of the parameter file (2665 mm to
// 7826 mm). The plane (x, y) on ZINI is cut into 60 x 50 bins. Each bin holds
// polynomials in fq = q/p * PMIN, of degree 9 for x and tx and 7 for y and ty,
// with terms that are linear in two reduced slopes ux and uy. The coefficients
// are interpolated between six neighbouring bins. The final state is the
// straight line plus these polynomials.
//
//   AllenUTT, variant "polynomial"  extrapUTT alone: the tabulated polynomials
//                                   with the q/p of the state.
//   AllenUTT, variant "full"        ExtrapolateUTT: first q/p is changed with
//                                   two parameters (dev_pars[18] and [19]; the
//                                   source says that the state holds the q/p at
//                                   the primary vertex), then extrapUTT, then
//                                   terms in q/p from UTTFParams are added to
//                                   x, y, tx and ty.
//
// Allen computes in single precision (KalmanFloat is float); so does this
// copy. The sums for the transport matrix inside compute_state are kept, so
// the time of this copy includes them, as in Allen. The noise matrix Q of
// ExtrapolateUTT is not computed.
//
// The q/p of our states is the q/p of LHCb in 1/MeV times c_light; the copy
// divides by c_light on entry.
#pragma once

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../classical/integrators.h"
#include "../common/io.h"

namespace fx {

  namespace allen_utt {

    using KalmanFloat = float;

    constexpr int nBinXMax = 60;
    constexpr int nBinYMax = 50;
    constexpr int DEGx1    = 7;
    constexpr int DEGx2    = 9;
    constexpr int DEGy1    = 5;
    constexpr int DEGy2    = 7;

    struct KalmanParametrizations {
      std::vector<float> x00, x10, x01, tx00, tx10, tx01;
      std::vector<float> y00, y10, y01, ty00, ty10, ty01;

      float ZINI = 0, ZFIN = 0;
      float PMIN = 0;
      float BENDX = 0, BENDX_X2 = 0, BENDX_Y2 = 0, BENDY_XY = 0;
      float Txmax = 0, Tymax = 0, XFmax = 0, Xmax = 0, Ymax = 0;
      float Dtxy = 0;
      int   Nbinx = 0, Nbiny = 0;
      int   XGridOption = 0, YGridOption = 0;
      int   DEGX1 = 0, DEGX2 = 0, DEGY1 = 0, DEGY2 = 0;

      KalmanParametrizations()
          : x00( nBinXMax * nBinYMax * DEGx2 )
          , x10( nBinXMax * nBinYMax * DEGx1 )
          , x01( nBinXMax * nBinYMax * DEGx1 )
          , tx00( nBinXMax * nBinYMax * DEGx2 )
          , tx10( nBinXMax * nBinYMax * DEGx1 )
          , tx01( nBinXMax * nBinYMax * DEGx1 )
          , y00( nBinXMax * nBinYMax * DEGy2 )
          , y10( nBinXMax * nBinYMax * DEGy1 )
          , y01( nBinXMax * nBinYMax * DEGy1 )
          , ty00( nBinXMax * nBinYMax * DEGy2 )
          , ty10( nBinXMax * nBinYMax * DEGy1 )
          , ty01( nBinXMax * nBinYMax * DEGy1 ) {}

      inline int UttOffset( const int& x_bin, const int& y_bin, const int& deg ) const {
        return deg * ( nBinYMax * x_bin + y_bin );
      }

      int Read( int ix, int iy, std::istream& inFile ) {
        auto GetPar = []( std::istream& ss ) -> float {
          std::string s;
          ss >> s;
          return std::stof( s );
        };
        int bi = UttOffset( ix, iy, DEGx2 );
        for ( int i = 0; i < DEGx2; i++ ) { this->x00[bi + i] = GetPar( inFile ); }
        for ( int i = 0; i < DEGx2; i++ ) { this->tx00[bi + i] = 1e-3f * GetPar( inFile ); }
        bi = UttOffset( ix, iy, DEGx1 );
        for ( int i = 0; i < DEGx1; i++ ) { this->x10[bi + i] = GetPar( inFile ); }
        for ( int i = 0; i < DEGx1; i++ ) { this->x01[bi + i] = GetPar( inFile ); }
        for ( int i = 0; i < DEGx1; i++ ) { this->tx10[bi + i] = 1e-3f * GetPar( inFile ); }
        for ( int i = 0; i < DEGx1; i++ ) { this->tx01[bi + i] = 1e-3f * GetPar( inFile ); }
        bi = UttOffset( ix, iy, DEGy2 );
        for ( int i = 0; i < DEGy2; i++ ) { this->y00[bi + i] = GetPar( inFile ); }
        for ( int i = 0; i < DEGy2; i++ ) { this->ty00[bi + i] = 1e-3f * GetPar( inFile ); }
        bi = UttOffset( ix, iy, DEGy1 );
        for ( int i = 0; i < DEGy1; i++ ) { this->y10[bi + i] = GetPar( inFile ); }
        for ( int i = 0; i < DEGy1; i++ ) { this->y01[bi + i] = GetPar( inFile ); }
        for ( int i = 0; i < DEGy1; i++ ) { this->ty10[bi + i] = 1e-3f * GetPar( inFile ); }
        for ( int i = 0; i < DEGy1; i++ ) { this->ty01[bi + i] = 1e-3f * GetPar( inFile ); }
        if ( inFile.eof() ) return 0;
        return 1;
      }

      void read_params_UTT( const std::string& file ) {
        std::ifstream myfile( file );
        if ( !myfile.is_open() ) throw std::runtime_error( "Failed to open parameter file " + file );
        myfile >> ZINI >> ZFIN >> PMIN >> BENDX >> BENDX_X2 >> BENDX_Y2 >> BENDY_XY >> Txmax >> Tymax >> XFmax >> Dtxy;
        myfile >> Nbinx >> Nbiny >> XGridOption >> YGridOption >> DEGX1 >> DEGX2 >> DEGY1 >> DEGY2;
        if ( Nbinx > nBinXMax || Nbiny > nBinYMax || DEGX1 != DEGx1 || DEGX2 != DEGx2 || DEGY1 != DEGy1 ||
             DEGY2 != DEGy2 )
          throw std::runtime_error( "the parameter file does not fit the sizes of Allen v9r3" );
        for ( int ix = 0; ix < Nbinx; ix++ ) {
          for ( int iy = 0; iy < Nbiny; iy++ ) {
            int ec = Read( ix, iy, myfile );
            if ( ec == 0 ) throw std::runtime_error( "Something went wrong in reading the parameter file." );
          }
        }
        Xmax = ZINI * Txmax;
        Ymax = ZINI * Tymax;
      }

      // Template helper to compute polynomial UTT extrapolation
      // DEG0: Polynomial degree for arr00 coefficients
      // DEG1: Polynomial degree for arr10/arr01 coefficients
      template <int DEG0, int DEG1>
      inline void compute_state( const float* __restrict__ arr00, const float* __restrict__ arr10,
                                 const float* __restrict__ arr01, const int ix, const int iy, const KalmanFloat gx,
                                 const KalmanFloat gy, const int sx, const int sy, const int rx, const int ry,
                                 const KalmanFloat ux, const KalmanFloat uy, const KalmanFloat fq,
                                 KalmanFloat& state_out, KalmanFloat& der_qop_out, KalmanFloat& der_tx_out,
                                 KalmanFloat& der_ty_out ) const {
        // Precompute interpolation scalers (same for all 3 coefficient sets)
        const KalmanFloat scaler0 = ( ( gy * gy - gy ) * ( (KalmanFloat)0.5 ) + ( !ry ) * sx * gx * gy );
        const KalmanFloat scaler1 = ( ( (KalmanFloat)1 ) - ( gx * gx + gy * gy ) + sx * sy * gx * gy );
        const KalmanFloat scaler2 = ( ( gy * gy + gy ) * ( (KalmanFloat)0.5 ) - ry * sx * gx * gy );
        const KalmanFloat scaler3 = ( ( gx * gx - gx ) * ( (KalmanFloat)0.5 ) + ( !rx ) * sy * gx * gy );
        const KalmanFloat scaler4 = ( sx * sy * gx * gy );
        const KalmanFloat scaler5 = ( ( gx * gx + gx ) * ( (KalmanFloat)0.5 ) - rx * sy * gx * gy );

        // Process 00 coefficients (degree DEG0)
        KalmanFloat ff = 1;
        for ( int deg = 0; deg < DEG0; deg++ ) {
          // Interpolate coefficient from 6 grid points
          const int         bi_center = UttOffset( ix, iy, DEG0 );
          const KalmanFloat coef =
              arr00[UttOffset( ix, iy - 1, DEG0 ) + deg] * scaler0 + arr00[bi_center + deg] * scaler1 +
              arr00[UttOffset( ix, iy + 1, DEG0 ) + deg] * scaler2 + arr00[UttOffset( ix - 1, iy, DEG0 ) + deg] * scaler3 +
              arr00[UttOffset( ix + sx, iy + sy, DEG0 ) + deg] * scaler4 +
              arr00[UttOffset( ix + 1, iy, DEG0 ) + deg] * scaler5;

          der_qop_out += ( deg + 1 ) * coef * ff;
          ff *= fq;
          state_out += coef * ff;
        }

        // Process 10 coefficients (degree DEG1, ux term)
        ff = 1;
        for ( int deg = 0; deg < DEG1; deg++ ) {
          const int         bi_center = UttOffset( ix, iy, DEG1 );
          const KalmanFloat coef =
              arr10[UttOffset( ix, iy - 1, DEG1 ) + deg] * scaler0 + arr10[bi_center + deg] * scaler1 +
              arr10[UttOffset( ix, iy + 1, DEG1 ) + deg] * scaler2 + arr10[UttOffset( ix - 1, iy, DEG1 ) + deg] * scaler3 +
              arr10[UttOffset( ix + sx, iy + sy, DEG1 ) + deg] * scaler4 +
              arr10[UttOffset( ix + 1, iy, DEG1 ) + deg] * scaler5;

          const KalmanFloat term = coef * ux;
          der_qop_out += ( deg + 1 ) * term * ff;
          ff *= fq;
          state_out += term * ff;
          der_tx_out += coef * ff;
        }

        // Process 01 coefficients (degree DEG1, uy term)
        ff = 1;
        for ( int deg = 0; deg < DEG1; deg++ ) {
          const int         bi_center = UttOffset( ix, iy, DEG1 );
          const KalmanFloat coef =
              arr01[UttOffset( ix, iy - 1, DEG1 ) + deg] * scaler0 + arr01[bi_center + deg] * scaler1 +
              arr01[UttOffset( ix, iy + 1, DEG1 ) + deg] * scaler2 + arr01[UttOffset( ix - 1, iy, DEG1 ) + deg] * scaler3 +
              arr01[UttOffset( ix + sx, iy + sy, DEG1 ) + deg] * scaler4 +
              arr01[UttOffset( ix + 1, iy, DEG1 ) + deg] * scaler5;

          const KalmanFloat term = coef * uy;
          der_qop_out += ( deg + 1 ) * term * ff;
          ff *= fq;
          state_out += term * ff;
          der_ty_out += coef * ff;
        }
      }
    };

    // extrapUTT of ParKalmanMethods.cuh; `polarity` is tI.m_polarity.
    inline void extrapUTT( const KalmanParametrizations* kalman_params, const float* dev_UTT_META, KalmanFloat& x,
                           KalmanFloat& y, KalmanFloat& tx, KalmanFloat& ty, KalmanFloat qOp, KalmanFloat* der_tx,
                           KalmanFloat* der_ty, KalmanFloat* der_qop, KalmanFloat polarity ) {
      KalmanFloat qop = qOp * polarity;

      KalmanFloat xx( 0 ), yy( 0 ), dx, dy, ux, uy;
      int         ix, iy;
      KalmanFloat zi = dev_UTT_META[0];
      KalmanFloat zf = dev_UTT_META[1];

      // XGridOption and YGridOption are just always 1.
      xx = x / ( zi * dev_UTT_META[7] );
      yy = y / ( zi * dev_UTT_META[8] );

      dx = dev_UTT_META[11] * ( xx + ( (KalmanFloat)1.0 ) ) / ( (KalmanFloat)2.0 );
      ix = dx;
      dx -= ix;
      dy = dev_UTT_META[12] * ( yy + ( (KalmanFloat)1.0 ) ) / ( (KalmanFloat)2.0 );
      iy = dy;
      dy -= iy;

      KalmanFloat DtxyInv = ( (KalmanFloat)1.0 ) / dev_UTT_META[10];
      KalmanFloat ziInv   = ( (KalmanFloat)1.0 ) / zi;
      KalmanFloat bendx   = dev_UTT_META[3] + dev_UTT_META[4] * ( x * ziInv ) * ( x * ziInv ) +
                          dev_UTT_META[5] * ( y * ziInv ) * ( y * ziInv );
      KalmanFloat bendy = dev_UTT_META[6] * ( x * ziInv ) * ( y * ziInv );
      ux                = ( tx - x * ziInv - bendx * qop ) * DtxyInv;
      uy                = ( ty - y * ziInv - bendy * qop ) * DtxyInv;

      KalmanFloat gx, gy;
      gx = dx - ( (KalmanFloat)0.5 );
      gy = dy - ( (KalmanFloat)0.5 );
      if ( ix <= 0 ) {
        ix = 1;
        gx -= (KalmanFloat)1.;
      }
      if ( ix >= dev_UTT_META[11] - 1 ) {
        ix = dev_UTT_META[11] - 2;
        gx += (KalmanFloat)1.;
      }
      if ( iy <= 0 ) {
        iy = 1;
        gy -= (KalmanFloat)1.;
      }
      if ( iy >= dev_UTT_META[12] - 1 ) {
        iy = dev_UTT_META[12] - 2;
        gy += (KalmanFloat)1.;
      }

      const int rx = ( gx >= 0 );
      const int sx = 2 * rx - 1;
      const int ry = ( gy >= 0 );
      const int sy = 2 * ry - 1;

      x = x + tx * ( zf - zi );
      y = y + ty * ( zf - zi );

      // Initialize derivative accumulators
      for ( int k = 0; k < 4; k++ ) der_tx[k] = der_ty[k] = der_qop[k] = 0;

      // corrections to straight line -------------------------
      const KalmanFloat fq = qop * dev_UTT_META[2];

      // State 0: x position
      kalman_params->compute_state<DEGx2, DEGx1>( kalman_params->x00.data(), kalman_params->x10.data(),
                                                  kalman_params->x01.data(), ix, iy, gx, gy, sx, sy, rx, ry, ux, uy,
                                                  fq, x, der_qop[0], der_tx[0], der_ty[0] );
      // State 2: tx slope
      kalman_params->compute_state<DEGx2, DEGx1>( kalman_params->tx00.data(), kalman_params->tx10.data(),
                                                  kalman_params->tx01.data(), ix, iy, gx, gy, sx, sy, rx, ry, ux, uy,
                                                  fq, tx, der_qop[2], der_tx[2], der_ty[2] );
      // State 1: y position
      kalman_params->compute_state<DEGy2, DEGy1>( kalman_params->y00.data(), kalman_params->y10.data(),
                                                  kalman_params->y01.data(), ix, iy, gx, gy, sx, sy, rx, ry, ux, uy,
                                                  fq, y, der_qop[1], der_tx[1], der_ty[1] );
      // State 3: ty slope
      kalman_params->compute_state<DEGy2, DEGy1>( kalman_params->ty00.data(), kalman_params->ty10.data(),
                                                  kalman_params->ty01.data(), ix, iy, gx, gy, sx, sy, rx, ry, ux, uy,
                                                  fq, ty, der_qop[3], der_tx[3], der_ty[3] );

      for ( int k = 0; k < 4; k++ ) {
        der_qop[k] *= dev_UTT_META[2];
        der_tx[k] *= DtxyInv;
        der_ty[k] *= DtxyInv;
      }
      // from straight line
      der_tx[0] += zf - zi;
      der_ty[1] += zf - zi;
      der_tx[2] += 1;
      der_ty[3] += 1;

      der_qop[0] = polarity * der_qop[0];
      der_qop[1] = polarity * der_qop[1];
      der_qop[2] = polarity * der_qop[2];
      der_qop[3] = polarity * der_qop[3];
    }

  } // namespace allen_utt

  struct AllenUTT {
    allen_utt::KalmanParametrizations params;
    float                             UTT_META[19]{};
    float                             UTTF[20]{};
    float                             polarity = 1.f;
    bool                              full     = false; // false: extrapUTT alone; true: ExtrapolateUTT
    mutable int                       last_steps = 1, last_failed = 0;
    mutable float                     last_der_tx[4]{}, last_der_ty[4]{}, last_der_qop[4]{};

    float z_in() const { return UTT_META[0]; }
    float z_out() const { return UTT_META[1]; }

    template <class R>
    inline State<R> operator()( const State<R>& in ) const {
      using allen_utt::KalmanFloat;
      // the state of Allen: x, y, tx, ty and q/p in 1/MeV
      KalmanFloat x[5] = { static_cast<float>( in.x ), static_cast<float>( in.y ), static_cast<float>( in.tx ),
                           static_cast<float>( in.ty ), static_cast<float>( in.qop ) / 299.792458f };
      if ( !full ) {
        allen_utt::extrapUTT( &params, UTT_META, x[0], x[1], x[2], x[3], x[4], last_der_tx, last_der_ty, last_der_qop,
                              polarity );
      } else {
        // ExtrapolateUTT, without the transport and noise matrices
        const float* dev_pars = UTTF;
        float        par      = dev_pars[18];
        KalmanFloat  qopHere  = x[4] + x[4] * ( (KalmanFloat)1e-4 ) * par + x[4] * fabsf( x[4] ) * dev_pars[19];
        allen_utt::extrapUTT( &params, UTT_META, x[0], x[1], x[2], x[3], qopHere, last_der_tx, last_der_ty,
                              last_der_qop, polarity );
        par = dev_pars[0] * polarity;
        x[3] += par * x[4];
        par = dev_pars[3] * polarity * ( (KalmanFloat)1e2 );
        x[1] += par * x[4];
        par = dev_pars[6] * polarity;
        x[2] += par * x[4];
        par = dev_pars[7] * x[4] * ( (KalmanFloat)1e5 );
        x[2] += par * x[4];
        par = dev_pars[8] * polarity * x[4] * x[4] * ( (KalmanFloat)1e8 );
        x[2] += par * x[4];
        par = dev_pars[9] * polarity * ( (KalmanFloat)1e2 );
        x[0] += par * x[4];
        par = dev_pars[10] * x[4] * ( (KalmanFloat)1e5 );
        x[0] += par * x[4];
        par = dev_pars[11] * polarity * x[4] * x[4] * ( (KalmanFloat)1e10 );
        x[0] += par * x[4];
      }
      return { static_cast<R>( x[0] ), static_cast<R>( x[1] ), static_cast<R>( x[2] ), static_cast<R>( x[3] ), in.qop };
    }
  };

  // The export: <folder>/meta has the path of the .tab file; UTT_META.f64 has 19 numbers, UTTF.f64 has 20.
  inline AllenUTT read_allen_utt( const std::string& folder ) {
    const Meta meta = read_meta( folder + "/meta" );
    AllenUTT   utt;
    utt.params.read_params_UTT( meta.text( "tab" ) );
    const auto META = read_f64( folder + "/UTT_META.f64" );
    const auto UTTF = read_f64( folder + "/UTTF.f64" );
    if ( META.size() != 19 || UTTF.size() != 20 ) throw std::runtime_error( "the export of the parametrised step has the wrong size" );
    for ( int i = 0; i < 19; ++i ) utt.UTT_META[i] = static_cast<float>( META[i] );
    for ( int i = 0; i < 20; ++i ) utt.UTTF[i] = static_cast<float>( UTTF[i] );
    return utt;
  }

} // namespace fx
