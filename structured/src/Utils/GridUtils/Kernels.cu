#pragma once
#include "uammd.cuh"

namespace uammd{
namespace structured{
namespace kernels{

  class Gaussian{
    real h;
    real prefactor;
    real tau;
  public:
    int support;
    real rmax;
    Gaussian(real sigma, real h, int support):
      tau(-0.5/(sigma*sigma)),
      h(h),
      rmax(0.5*h*support),
      support(support),
      prefactor(rsqrt(2.0*M_PI*sigma*sigma)){}
    
    Gaussian(real sigma, real tolerance, Box box){
      real Lx         = box.boxSize.x;
      real best_h     = adviseGridSize(sigma, tolerance);
      int nCells      = round(Lx/best_h);
      int3 nCells3    = nextFFTWiseSize3D(make_int3(nCells));
      real sigma2     = sigma*sigma;
      real targetRmax = sqrt(-2*sigma2*log(sqrt(2*M_PI*sigma2)*tolerance));
      
      h         = Lx/nCells3.x;
      support   = static_cast<int>(ceil(2*targetRmax/h));
      rmax      = 0.5*h*support;
      tau       = -0.5/(sigma2);
      prefactor = rsqrt(2.0*M_PI*sigma2);
    }
    
    __host__ __device__ real phi(real r, real3 pos = real3()){
      if (r>rmax) return 0.0;
      else return prefactor*exp(tau*r*r);
    }

    real getCellSize(){
      return h;
    }

    int getSupport(){
      return support;
    }

    int getSupport(real3 pos, int3 cell){
      return support;
    }

    static real computeUpsampling(real tolerance){
      real amin = 0.55;
      real amax = 1.65;
      real x = -log10(3*tolerance)/10.0;
      real factor = std::min(amin + x*(amax-amin), amax);
      return factor;
    }
    
    static real adviseGridSize(real sigma, real tolerance){
      real factor = computeUpsampling(tolerance);
      return sigma/factor;
    }
  };

  enum class Direction { x, y, z };
  
  template <Direction gradientDirection>
  class GaussianGradient {
    real h;
    real prefactor;
    real tau;
  public:
    int support;
    real rmax;
    GaussianGradient(real sigma, real h, int support):
      tau(-0.5/(sigma*sigma)),
      h(h),
      rmax(0.5*h*support),
      support(support),
      prefactor(rsqrt(2.0*M_PI*sigma*sigma)){}
    
    //The sign - in phiX,Y,Z arises because in UAMMD r = x_blob - x_cell,
    //and here it should be x_cell - x_blob
    __host__ __device__ real phiX(real r, real3 pos = real3()){
      if (r>rmax) return 0.0;
      real val = prefactor*exp(tau*r*r);
      if constexpr (gradientDirection == Direction::x) val *= real(-2) * r * tau;
      return val;
    }

    
    __host__ __device__ real phiY(real r, real3 pos = real3()){
      if (r>rmax) return 0.0;
      real val = prefactor*exp(tau*r*r);
      if constexpr (gradientDirection == Direction::y) val *= real(-2) * r * tau;
      return val;
    }

    
    __host__ __device__ real phiZ(real r, real3 pos = real3()){
      if (r>rmax) return 0.0;
      real val = prefactor*exp(tau*r*r);
      if constexpr (gradientDirection == Direction::z) val *= real(-2) * r * tau;
      return val;
    }

    real getCellSize(){
      return h;
    }

    int getSupport(){
      return support;
    }

    int getSupport(real3 pos, int3 cell){
      return support;
    }
  };
  
  class Peskin_4p{
	real h;
    real invh;
  public:
    static constexpr int support = 4;
    
	Peskin_4p(real h_in, Box box){
      real Lx         = box.boxSize.x;
      real best_h     = h_in;
      int nCells      = round(Lx/best_h);
      int3 nCells3    = nextFFTWiseSize3D(make_int3(nCells));
      h               = Lx/nCells3.x;
      invh            = 1./h;
    }
                               
    __host__  __device__ real phi(real rr, real3 pos = real3()) const{
      const real r = fabs(rr)*invh;
      constexpr real onediv8 = real(0.125);
      if(r<real(1.0)){
        return invh*onediv8*(real(3.0) - real(2.0)*r + sqrt(real(1.0)+real(4.0)*r*(real(1.0)-r)));
      }
      else if(r<real(2.0)){
        return invh*onediv8*(real(5.0) - real(2.0)*r - sqrt(real(-7.0) + real(12.0)*r-real(4.0)*r*r));
      }
      else return 0;
    }

    real getCellSize(){
      return h;
    }
    
    int getSupport(){
      return support;
    }
    
    int getSupport(real3 pos, int3 cell){
      return support;
    }    
  };

  template <Direction gradientDirection>
  class Peskin_4pGradient{
	real h;

    __host__ __device__ double sgn(double x) {
      if (x > 0.0) return 1.0;
      if (x < 0.0) return -1.0;
      return 0.0;
    }
    
    __host__  __device__ real phiBase(real rr, real3 pos = real3()) const{
      const real invh = 1.0/h;
      const real r = fabs(rr)*invh;
      constexpr real onediv8 = real(0.125);
      if(r<real(1.0)){
        return invh*onediv8*(real(3.0) - real(2.0)*r + sqrt(real(1.0)+real(4.0)*r*(real(1.0)-r)));
      }
      else if(r<real(2.0)){
        return invh*onediv8*(real(5.0) - real(2.0)*r - sqrt(real(-7.0) + real(12.0)*r-real(4.0)*r*r));
      }
      else return 0;
    }

    __host__ __device__ real derivPhiBase(real rr) {
      double val = 0.0;
      const real invh = 1.0/h;
      double ar = std::fabs(rr)*invh;
      if (ar == 0.0) {
        return 0.0;
      }
      
      if (ar <= 1.0){
        double tmp = std::sqrt(1.0 + 4.0*ar - 4.0*ar*ar);
        double dphi_dar = (-2.0 + (4.0 - 8.0*ar) / (2.0 * tmp)) * invh / (8.0*h);
        val = -sgn(rr) * dphi_dar;
      }
      else if (ar <= 2.0){
        double tmp = std::sqrt(-7.0 + 12.0*ar - 4.0*ar*ar);
        double dphi_dar = (-2.0 - (12.0 - 8.0*ar) / (2.0 * tmp)) * invh / (8.0*h);
        val = -sgn(rr) * dphi_dar;
      }
      return val;
    }
      
  public:
    static constexpr int support = 4;
    
    Peskin_4pGradient(real h_in, Box box){
      real Lx         = box.boxSize.x;
      real best_h     = h_in;
      int nCells      = round(Lx/best_h);
      int3 nCells3    = nextFFTWiseSize3D(make_int3(nCells));
      h               = Lx/nCells3.x;     
    }

    __host__ __device__ real phiX(real r, real3 pos = real3()){
      real val;
      if constexpr (gradientDirection == Direction::x) val = derivPhiBase(r);
      else val = phiBase(r);
      return val;
    }

    
    __host__ __device__ real phiY(real r, real3 pos = real3()){
      real val;
      if constexpr (gradientDirection == Direction::y) val = derivPhiBase(r);
      else val = phiBase(r);
      return val;
    }

    
    __host__ __device__ real phiZ(real r, real3 pos = real3()){
      real val;
      if constexpr (gradientDirection == Direction::z) val = derivPhiBase(r); 
      else val = phiBase(r);
      return val;
    }
	

    real getCellSize(){
      return h;
    }
    
    int getSupport(){
      return support;
    }
    
    int getSupport(real3 pos, int3 cell){
      return support;
    }    
  };

  class BetaSpline3 {
    real h;
    real invh;
    
  public:
    static constexpr int support = 4;
    
    BetaSpline3(real h_in, Box box) {
        real Lx         = box.boxSize.x;
        real best_h     = h_in;
        int nCells      = round(Lx / best_h);
        int3 nCells3    = nextFFTWiseSize3D(make_int3(nCells));
        h               = Lx / nCells3.x;
        invh            = 1.0 / h;
    }

    __host__ __device__ real phi(real rr, real3 pos = real3()) const {
        const real r = fabs(rr) * invh;
        if (r <= real(1.0)) {
          return invh * (0.5 * r * r * r - r * r + 2.0 / 3.0);
            
        } else if (r <= real(2.0)) {
          return invh * (-1.0 / 6.0 * r * r * r + r * r - 2.0 * r + 4.0 / 3.0);            
        } else {
          return 0.0;
        }
    }


    real getCellSize() {
        return h;
    }

    int getSupport() {
        return support;
    }

    int getSupport(real3 pos, int3 cell){
      return support;
    }    
};

template <Direction gradientDirection>
class BetaSpline3Gradient {
  real h;
  real invh;
  real invh2;

    __host__ __device__ double sgn(double x) {
        if (x > 0.0) return 1.0;
        if (x < 0.0) return -1.0;
        return 0.0;
    }

    __host__ __device__ real phiBase(real rr, real3 pos = real3()) const {
        const real r = fabs(rr) * invh;
        if (r <= real(1.0)) {
            return invh * (0.5 * r * r * r - r * r + 2.0 / 3.0);
        } else if (r <= real(2.0)) {
            return invh * (-1.0 / 6.0 * r * r * r + r * r - 2.0 * r + 4.0 / 3.0);
        } else {
            return 0.0;
        }
    }

    __host__ __device__ real derivPhiBase(real rr) {
        const real r = fabs(rr) * invh;
        if (r <= real(1.0)) {
            return -invh2 * sgn(rr) * (1.5 * r * r - 2.0 * r);
        } else if (r <= real(2.0)) {
            return -invh2 * sgn(rr) * (-0.5 * r * r + 2.0 * r - 2.0);
        } else {
            return 0.0;
        }
    }

public:
    static constexpr int support = 4;

    BetaSpline3Gradient(real h_in, Box box) {
        real Lx         = box.boxSize.x;
        real best_h     = h_in;
        int nCells      = round(Lx / best_h);
        int3 nCells3    = nextFFTWiseSize3D(make_int3(nCells));
        h               = Lx / nCells3.x;
        invh             = 1.0 / (h);
        invh2            = 1.0 / (h * h);
    }

    __host__ __device__ real phiX(real r, real3 pos = real3()) {
        if constexpr (gradientDirection == Direction::x) {
            return derivPhiBase(r);
        } else {
            return phiBase(r);
        }
    }

    __host__ __device__ real phiY(real r, real3 pos = real3()) {
        if constexpr (gradientDirection == Direction::y) {
            return derivPhiBase(r);
        } else {
            return phiBase(r);
        }
    }

    __host__ __device__ real phiZ(real r, real3 pos = real3()) {
        if constexpr (gradientDirection == Direction::z) {
            return derivPhiBase(r);
        } else {
            return phiBase(r);
        }
    }

    real getCellSize() {
        return h;
    }

    int getSupport() {
        return support;
    }

  int getSupport(real3 pos, int3 cell){
    return support;
  }    
};

  class Peskin_6p{
  private:
    real h;
    real invh;
    
    static constexpr real K     = 0.7140750929766081;
    static constexpr real alpha = 28.0;
    
    // Métodos auxiliares marcados como const y __host__ __device__
    __host__ __device__ real computeBeta(real r) const {
      return 9.0/4.0 - 3.0/2.0 * (K + r*r) + (22.0/3.0 - 7*K)*r - 7.0/3.0 * r*r*r;
    }
    
    __host__ __device__ real computeGamma(real r) const {
      real term1 = -11.0/32.0 * r*r;
      real term2 = 3.0/32.0 * (2*K + r*r) * r*r;
      real term3 = 1.0/72.0 * std::pow((3*K - 1)*r + r*r*r, 2);
      real term4 = 1.0/18.0 * std::pow((4 - 3*K)*r - r*r*r, 2);
      return term1 + term2 + term3 + term4;
    }
    
    __host__ __device__ real sign(real x) const {
      return (x >= 0.0) ? 1.0 : -1.0;
    }
    
    __host__ __device__ real phiRminus3(real r) const {
      real beta_val  = computeBeta(r);
      real gamma_val = computeGamma(r);
      real term      = std::sqrt(beta_val*beta_val - 112.0 * gamma_val);
      return (-beta_val + sign(1.5 - K) * term) / (2.0 * alpha);
    }
    
    __host__ __device__ real phiRminus2(real r) const {
      return -3.0 * phiRminus3(r) - 1.0/16.0 + (K + r*r)/8.0 + (3*K - 1)*r/12.0 + r*r*r/12.0;
    }
    
    __host__ __device__ real phiRminus1(real r) const {
      return 2.0 * phiRminus3(r) + 1.0/4.0 + (4 - 3*K)*r/6.0 - r*r*r/6.0;
    }
    
    __host__ __device__ real phiR(real r) const {
      return 2.0 * phiRminus3(r) + 5.0/8.0 - (K + r*r) / 4.0;
    }
    
    __host__ __device__ real phiRplus1(real r) const {
      return -3.0 * phiRminus3(r) + 1.0/4.0 - (4 - 3*K)*r/6.0 + r*r*r/6.0;
    }
    
    __host__ __device__ real phiRplus2(real r) const {
      return phiRminus3(r) - 1.0/16.0 + (K + r*r)/8.0 - (3*K - 1)*r/12.0 - r*r*r/12.0;
    }
    
  public:
    static constexpr int support = 6;
    
    // Constructor que inicializa h e invh usando una caja dada
    Peskin_6p(real h_in, Box box) {
      real Lx      = box.boxSize.x;  // Se asume que box tiene boxSize.x
      real best_h  = h_in;
      int nCells   = round(Lx / best_h);
      int3 nCells3 = nextFFTWiseSize3D(make_int3(nCells));
      h            = Lx / nCells3.x;
      invh         = 1.0 / h;
    }
    
    __host__ __device__ real phi(real rr, real3 pos = real3()) const {
      real r = rr * invh;
      if      (r >= -3.0 && r < -2.0) return phiRminus3(r + 3.0)*invh;
      else if (r >= -2.0 && r < -1.0) return phiRminus2(r + 2.0)*invh;
      else if (r >= -1.0 && r <  0.0) return phiRminus1(r + 1.0)*invh;
      else if (r >=  0.0 && r <  1.0) return phiR(r)*invh;
      else if (r >=  1.0 && r <  2.0) return phiRplus1(r - 1.0)*invh;
      else if (r >=  2.0 && r <  3.0) return phiRplus2(r - 2.0)*invh;
      else                            return 0.0;
    }
    
    __host__ __device__ real getCellSize() const {
      return h;
    }
    
    __host__ __device__ int getSupport() const {
      return support;
    }
    
    __host__ __device__ int getSupport(real3 pos, int3 cell) const {
      return support;
    }
  };

  
  template <Direction gradientDirection>
  class Peskin_6pGradient{
  private:
    real h;
    real invh;
  
    static constexpr real K     = 0.7140750929766081;
    static constexpr real alpha = 28.0;
    
    // Métodos auxiliares para cálculos del kernel
    __host__ __device__ real beta(real r) const {
      return 9.0/4.0 - 3.0/2.0 * (K + r*r) + (22.0/3.0 - 7*K)*r - 7.0/3.0 * r*r*r;
    }
    
    __host__ __device__ real gamma(real r) const {
      real term1 = -11.0/32.0 * r*r;
      real term2 = 3.0/32.0 * (2*K + r*r) * r*r;
      real term3 = 1.0/72.0 * std::pow((3*K - 1)*r + r*r*r, 2);
      real term4 = 1.0/18.0 * std::pow((4 - 3*K)*r - r*r*r, 2);
      return term1 + term2 + term3 + term4;
    }
 
    __host__ __device__ real dbeta(real rr) const {
      return (22.0/3.0 - 7*K) - 3.0*rr - 7.0*rr*rr;
    }
    
    __host__ __device__ real dgamma(real r) const {
      return (1.0/4.0) * (((161.0/36.0) - (59.0/6.0)*K + 5.0*K*K) * r +
                          (-(109.0/24.0) + 5.0*K) * (4.0/3.0) * std::pow(r, 3) +
                          (5.0/3.0) * std::pow(r, 5));
    }
    
    __host__ __device__ real sign(real x) const {
      return (x >= 0.0) ? 1.0 : -1.0;
    }
    
    __host__ __device__ real phiRminus3(real r) const {
      real beta_val  = beta(r);
      real gamma_val = gamma(r);
      real term      = std::sqrt(beta_val*beta_val - 112.0 * gamma_val);
      return (-beta_val + sign(1.5 - K) * term) / (2.0 * alpha);
    }
    
    __host__ __device__ real phiRminus2(real r) const {
      return -3.0 * phiRminus3(r) - 1.0/16.0 + (K + r*r)/8.0 + (3*K - 1)*r/12.0 + r*r*r/12.0;
    }
    
    __host__ __device__ real phiRminus1(real r) const {
      return 2.0 * phiRminus3(r) + 1.0/4.0 + (4 - 3*K)*r/6.0 - r*r*r/6.0;
    }
    
    __host__ __device__ real phiR(real r) const {
      return 2.0 * phiRminus3(r) + 5.0/8.0 - (K + r*r) / 4.0;
    }
    
    __host__ __device__ real phiRplus1(real r) const {
      return -3.0 * phiRminus3(r) + 1.0/4.0 - (4 - 3*K)*r/6.0 + r*r*r/6.0;
    }
    
    __host__ __device__ real phiRplus2(real r) const {
      return phiRminus3(r) - 1.0/16.0 + (K + r*r)/8.0 - (3*K - 1)*r/12.0 - r*r*r/12.0;
    }
          
    __host__ __device__ real dphi_r_minus_3(real rr) const {
      real discr = std::pow(beta(rr), 2) - 4.0 * alpha * gamma(rr);
      real pm3   = (-beta(rr) + sign(1.5 - K)*std::sqrt(discr)) / (2.0 * alpha);
      return -(dbeta(rr)*pm3 + dgamma(rr)) / (2.0*alpha*pm3 + beta(rr));
    }
  
    __host__ __device__ real dphi_r_minus_2(real rr) const {
      return -3.0*dphi_r_minus_3(rr) + (1.0/12.0)*(3*K - 1) + (1.0/4.0)*rr + (1.0/4.0)*rr*rr;
    }
  
    __host__ __device__ real dphi_r_minus_1(real rr) const {
      return 2.0*dphi_r_minus_3(rr) + (1.0/6.0)*(4 - 3*K) - (1.0/2.0)*rr*rr;
    }
    
    __host__ __device__ real dphi_r(real rr) const {
      return 2.0*dphi_r_minus_3(rr) - (1.0/2.0)*rr;
    }
    
    __host__ __device__ real dphi_r_plus_1(real rr) const {
      return -3.0*dphi_r_minus_3(rr) - (1.0/6.0)*(4 - 3*K) + (1.0/2.0)*rr*rr;
    }
    
    __host__ __device__ real dphi_r_plus_2(real rr) const {
      return dphi_r_minus_3(rr) - (1.0/12.0)*(3*K - 1) + (1.0/4.0)*rr - (1.0/4.0)*rr*rr;
    }
    
    __host__ __device__ real phiBase(real rr, real3 pos = real3()) const {
      real r = rr * invh;
      if      (r >= -3.0 && r < -2.0) return phiRminus3(r + 3.0)*invh;
      else if (r >= -2.0 && r < -1.0) return phiRminus2(r + 2.0)*invh;
      else if (r >= -1.0 && r <  0.0) return phiRminus1(r + 1.0)*invh;
      else if (r >=  0.0 && r <  1.0) return phiR(r)*invh;
      else if (r >=  1.0 && r <  2.0) return phiRplus1(r - 1.0)*invh;
      else if (r >=  2.0 && r <  3.0) return phiRplus2(r - 2.0)*invh;
      else                            return 0.0;
    }
    
    __host__ __device__ real derivPhiBase(real rr, real3 pos = real3()) const {
      real r = rr * invh;
      if      (r >= -3.0 && r < -2.0) return dphi_r_minus_3(r + 3.0)*invh*invh;
      else if (r >= -2.0 && r < -1.0) return dphi_r_minus_2(r + 2.0)*invh*invh;
      else if (r >= -1.0 && r <  0.0) return dphi_r_minus_1(r + 1.0)*invh*invh;
      else if (r >=  0.0 && r <  1.0) return dphi_r(r)*invh*invh;
      else if (r >=  1.0 && r <  2.0) return dphi_r_plus_1(r - 1.0)*invh*invh;
      else if (r >=  2.0 && r <  3.0) return dphi_r_plus_2(r - 2.0)*invh*invh;
      else                            return 0.0;
    }
    
    
  public:
    static constexpr int support = 6;
    
    // Constructor que inicializa h e invh usando una caja dada
    __host__ Peskin_6pGradient(real h_in, Box box) {
      real Lx      = box.boxSize.x;  // Se asume que box tiene boxSize.x
      real best_h  = h_in;
      int nCells   = round(Lx / best_h);
      int3 nCells3 = nextFFTWiseSize3D(make_int3(nCells));
      h            = Lx / nCells3.x;
      invh         = 1.0 / h;
    }

    __host__ __device__ real phiX(real r, real3 pos = real3()) {
      if constexpr (gradientDirection == Direction::x) {
        //printf("X: r->%f dphi(r)->%f\n", r*invh, derivPhiBase(r)/(invh*invh));
        return derivPhiBase(r);
      } else {
        return phiBase(r);
      }
    }
    
    __host__ __device__ real phiY(real r, real3 pos = real3()) {
      if constexpr (gradientDirection == Direction::y) {
        //printf("Y: r->%f dphi(r)->%f\n", r*invh, derivPhiBase(r)/(invh*invh));
        return derivPhiBase(r);
      } else {
        return phiBase(r);
      }
    }
    
    __host__ __device__ real phiZ(real r, real3 pos = real3()) {
      if constexpr (gradientDirection == Direction::z) {
        //printf("Z: r->%f dphi(r)->%f\n", r*invh, derivPhiBase(r)/(invh*invh));
        return derivPhiBase(r);
      } else {
        return phiBase(r);
      }
    }
    
    __host__ __device__ real getCellSize() const {
      return h;
    }
    
    __host__ __device__ int getSupport() const {
      return support;
    }
    
    __host__ __device__ int getSupport(real3 pos, int3 cell) const {
      return support;
    }
  };
  
}}}
