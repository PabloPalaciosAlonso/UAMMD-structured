#pragma once
#include "uammd.cuh"
#include "GlobalData/GlobalData.cuh"
#include <variant>

namespace uammd{
namespace structured{
namespace Interactor{
namespace LongRange{
namespace Dipolar{
namespace detail{

  enum class Direction { x, y, z };
  
  class GaussianBase{
    real h;
    real prefactor;
    real tau;
  public:
    int support;
    real rmax;
    
    GaussianBase(real sigma, real h, real rmax, int support):
      h(h), rmax(rmax), support(support),
      tau(-0.5/(sigma*sigma)), prefactor(rsqrt(2.0*M_PI*sigma*sigma)){
      System::log<System::DEBUG>("[GaussianBase] Gaussian Base kernel initialized.");
    }
    
    __host__ __device__ real phi(real r, real3 pos = real3()){
      if (r>rmax) return 0.0;
      else return prefactor*exp(tau*r*r);
    }

    real getCellSize() const{
      return h;
    }

    int getSupport() const{
      return support;
    }

    int getSupport (real3 pos, int3 cell){
      return support;
    }
  };
  
  template <Direction gradientDirection>
  class GaussianGradient {
    real h;
    real prefactor;
    real tau;
  public:
    int support;
    real rmax;
    GaussianGradient(real sigma, real h, real rmax, int support):
      h(h), rmax(rmax), support(support),
      tau(-0.5/(sigma*sigma)), prefactor(rsqrt(2.0*M_PI*sigma*sigma)){
      switch (gradientDirection) {
      case Direction::x:
        System::log<System::DEBUG>("[GaussianGradient] Gaussian gradient kernel initialized in x-direction.");
        break;
      case Direction::y:
        System::log<System::DEBUG>("[GaussianGradient] Gaussian gradient kernel initialized in y-direction.");
        break;
      case Direction::z:
        System::log<System::DEBUG>("[GaussianGradient] Gaussian gradient kernel initialized in z-direction.");
        break;
      default:
        System::log<System::WARNING>("[GaussianGradient] Unknown direction.");
        break;
      }
    }
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

  struct Gaussian {
    GaussianBase                             kernel;
    GaussianGradient<Direction::x>           grad_x;
    GaussianGradient<Direction::y>           grad_y;
    GaussianGradient<Direction::z>           grad_z;
    
    Gaussian(std::shared_ptr<GlobalData> gd, DataEntry& data)
      :Gaussian(InitializeParameters(gd, data)) {}

  private:
    struct Parameters { real sigma, h, rmax; int support; };
    
    static real computeUpsampling(real tolerance){
      real amin   = 0.55;
      real amax   = 1.65;
      real x      = -log10(3*tolerance)/10.0;
      real factor = std::min(amin + x*(amax-amin), amax);
      return factor;
    }
    
    static real adviseGridSize(real sigma, real tolerance){
      real factor = computeUpsampling(tolerance);
      return sigma/factor;
    }
    
    static Parameters InitializeParameters(std::shared_ptr<GlobalData> gd, DataEntry& data) {
      Box box        = gd->getEnsemble()->getBox();
      real radius    = data.getParameter<real>("radius", -1.0);
      real sigma     = data.getParameter<real>("sigma",  -1.0);
      real tolerance = data.getParameter<real>("tolerance");
      
      if (radius < 0 && sigma < 0)
        System::log<System::CRITICAL>("[Gaussian Kernel] I need either the radius or the kernel width sigma.");      
      if (radius > 0 && sigma > 0)
        System::log<System::CRITICAL>("[Gaussian Kernel] Radius and kernel width provided. I just need one.");
      
      if (sigma < 0)
        sigma = radius / std::cbrt(6 * std::sqrt(M_PI));
      
      real Lx      = box.boxSize.x;
      real best_h  = adviseGridSize(sigma, tolerance);
      int  nCells  = std::round(Lx / best_h);
      int3 nCells3 = nextFFTWiseSize3D(make_int3(nCells));
      
      real h         = Lx / nCells3.x;
      real sigma2    = sigma * sigma;
      real targetR   = std::sqrt(-2 * sigma2 *
                                 std::log(std::sqrt(2 * M_PI * sigma2) * tolerance));
      int  support   = static_cast<int>(std::ceil(2 * targetR / h));
      real rmax      = 0.5 * h * support;
      
      return {sigma, h, rmax, support};
    }

    Gaussian(const Parameters& p):kernel(p.sigma, p.h, p.rmax, p.support),
                                  grad_x  (p.sigma, p.h, p.rmax, p.support),
                                  grad_y  (p.sigma, p.h, p.rmax, p.support),
                                  grad_z  (p.sigma, p.h, p.rmax, p.support){}
  };

  
  class Peskin_3pBase{
    real h;
    real invh;
  public:
    static constexpr int support = 3;

    Peskin_3pBase(real h):
      h(h), invh(1.0/h){
      System::log<System::DEBUG>("[Peskin_3pBase] Peskin_3p Base kernel initialized.");
    }
    
    __host__ __device__ real phi(real rr, real3 pos = real3()) const {
      const real r = fabs(rr) * invh;
      if (r < real(0.5)) {
        constexpr real onediv3 = real(1 / 3.0);
        return invh * onediv3 *
          (real(1.0) + sqrt(real(1.0) + real(-3.0) * r * r));
      } else if (r < real(1.5)) {
        constexpr real onediv6 = real(1 / 6.0);
        const real omr = real(1.0) - r;
        return invh * onediv6 *
          (real(5.0) - real(3.0) * r -
           sqrt(real(1.0) + real(-3.0) * omr * omr));
      } else
        return 0;
    }
    
    real getCellSize() const {
      return h;
    }
    
    int getSupport() const {
      return support;
    }
    
    int getSupport(real3 pos, int3 cell) const {
      return support;
    }    
  };

  template <Direction gradientDirection>
  class Peskin_3pGradient {
    real h;
    real invh;
    Peskin_3pBase base;
    
    __host__ __device__ real derivPhiBase(real rr) const {
      real r    = fabs(rr) * invh;
      real sgn  = (rr >= real(0)) ? real(1) : real(-1);
      real dphidx = real(0);
      
      if (r < real(0.5)) {
        /* f(r) = 1/3 [1 + sqrt(1 - 3 r²)] */
        real root = sqrt(real(1.0) - real(3.0) * r * r);
        real dfdr = -r / root;                   // –r / √(1-3r²)
        dphidx = invh * invh * dfdr * sgn;       // dφ/dx = invh²·df/dr·sign
      }
      else if (r < real(1.5)) {
        /* f(r) = 1/6 [5 - 3 r - √(1 - 3 (1-r)²)] */
        real omr  = real(1.0) - r;
        real root = sqrt(real(1.0) - real(3.0) * omr * omr);
        real dfdr = (-real(3.0) - real(3.0) * omr / root) / real(6.0);
        dphidx = invh * invh * dfdr * sgn;
      }
      return -dphidx;
    }
    
  public:
    static constexpr int support = 3;
    Peskin_3pGradient(real h):
      base(h), h(h), invh(1.0/h){
      switch (gradientDirection) {
      case Direction::x:
        System::log<System::DEBUG>("[Peskin_3pGradient] Peskin_3p gradient kernel initialized in x-direction.");
        break;
      case Direction::y:
        System::log<System::DEBUG>("[Peskin_3pGradient] Peskin_3p gradient kernel initialized in y-direction.");
        break;
      case Direction::z:
        System::log<System::DEBUG>("[Peskin_3pGradient] Peskin_3p gradient kernel initialized in z-direction.");
        break;
      default:
        System::log<System::WARNING>("[Peskin_3pGradient] Unknown direction.");
        break;
      }
    }
    
    //The sign - in phiX,Y,Z arises because in UAMMD r = x_blob - x_cell,
    //and here it should be x_cell - x_blob
    __host__ __device__ real phiX(real r, real3 pos = real3()){
      real val;
      if constexpr (gradientDirection == Direction::x) val = derivPhiBase(r);
      else val = base.phi(r);
      return val;
    }
    
    __host__ __device__ real phiY(real r, real3 pos = real3()){
      real val;
      if constexpr (gradientDirection == Direction::y) val = derivPhiBase(r);
      else val = base.phi(r);
      return val;
    }
    
    __host__ __device__ real phiZ(real r, real3 pos = real3()){
      real val;
      if constexpr (gradientDirection == Direction::z) val = derivPhiBase(r);
      else val = base.phi(r);
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
  
  struct Peskin_3p {
    Peskin_3pBase                             kernel;
    Peskin_3pGradient<Direction::x>           grad_x;
    Peskin_3pGradient<Direction::y>           grad_y;
    Peskin_3pGradient<Direction::z>           grad_z;
    
    Peskin_3p(std::shared_ptr<GlobalData> gd, DataEntry& data)
      :Peskin_3p(InitializeParameters(gd, data)) {}

  private:
    struct Parameters {real h;};
        
    static Parameters InitializeParameters(std::shared_ptr<GlobalData> gd, DataEntry& data) {      
      Box box       = gd->getEnsemble()->getBox();
      real radius   = data.getParameter<real>("radius", -1.0);
      real target_h = data.getParameter<real>("cellSize", -1.0);
      
      if (radius == -1.0 and target_h == -1.0){
        System::log<System::CRITICAL>("[Peskin_3p Kernel] I need either the radius or the cell size.");
      }
      
      if (radius>0 and target_h > 0){
        System::log<System::CRITICAL>("[Peskin_3p Kernel] Radius and cell size provided. I just need one.");
      }
      
      if (target_h<0){
        target_h   = radius * (pow(M_PI/6, 1./3.));
      }
      
      real Lx         = box.boxSize.x;
      int nCells      = round(Lx/target_h);
      int3 nCells3    = nextFFTWiseSize3D(make_int3(nCells));
      real h          = Lx/nCells3.x;
      return {h};
    }

    Peskin_3p(const Parameters& p):
      kernel(p.h), grad_x(p.h), grad_y(p.h), grad_z(p.h){
    }
  };

  
  class Peskin_4pBase{
    real h;
    real invh;
  public:
    static constexpr int support = 4;
    Peskin_4pBase(real h): h(h), invh(1.0/h){
      System::log<System::DEBUG>("[Peskin_4pBase] Peskin_4p Base kernel initialized.");
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
    
    real getCellSize() const {
      return h;
    }
    
    int getSupport() const {
      return support;
    }
    
    int getSupport(real3 pos, int3 cell) const {
      return support;
    }   
  };
  
  template <Direction gradientDirection>
  class Peskin_4pGradient{
    real h;
    Peskin_4pBase base;
    __host__ __device__ real sgn(real x) {
      if (x > 0.0) return 1.0;
      if (x < 0.0) return -1.0;
      return 0.0;
    }
    
    __host__ __device__ real derivPhiBase(real rr) {
      real val = 0.0;
      const real invh = 1.0/h;
      real ar = std::fabs(rr)*invh;
      if (ar == 0.0) {
        return 0.0;
      }
      
      if (ar <= 1.0){
        real tmp = std::sqrt(1.0 + 4.0*ar - 4.0*ar*ar);
        real dphi_dar = (-2.0 + (4.0 - 8.0*ar) / (2.0 * tmp)) * invh / (8.0*h);
        val = -sgn(rr) * dphi_dar;
      }
      else if (ar <= 2.0){
        real tmp = std::sqrt(-7.0 + 12.0*ar - 4.0*ar*ar);
        real dphi_dar = (-2.0 - (12.0 - 8.0*ar) / (2.0 * tmp)) * invh / (8.0*h);
        val = -sgn(rr) * dphi_dar;
      }
      return val;
    }
      
  public:
    static constexpr int support = 4;
    
    Peskin_4pGradient(real h): h(h), base(h){
      switch (gradientDirection) {
      case Direction::x:
        System::log<System::DEBUG>("[Peskin_4pGradient] Peskin_4p gradient kernel initialized in x-direction.");
        break;
      case Direction::y:
        System::log<System::DEBUG>("[Peskin_4pGradient] Peskin_4p gradient kernel initialized in y-direction.");
        break;
      case Direction::z:
        System::log<System::DEBUG>("[Peskin_4pGradient] Peskin_4p gradient kernel initialized in z-direction.");
        break;
      default:
        System::log<System::WARNING>("[Peskin_4pGradient] Unknown direction.");
        break;
      }
    }
    
    __host__ __device__ real phiX(real r, real3 pos = real3()){
      real val;
      if constexpr (gradientDirection == Direction::x) val = derivPhiBase(r);
      else val = base.phi(r);
      return val;
    }

    __host__ __device__ real phiY(real r, real3 pos = real3()){
      real val;
      if constexpr (gradientDirection == Direction::y) val = derivPhiBase(r);
      else val = base.phi(r);
      return val;
    }

    __host__ __device__ real phiZ(real r, real3 pos = real3()){
      real val;
      if constexpr (gradientDirection == Direction::z) val = derivPhiBase(r);
      else val = base.phi(r);      
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

  struct Peskin_4p {
    Peskin_4pBase                             kernel;
    Peskin_4pGradient<Direction::x>           grad_x;
    Peskin_4pGradient<Direction::y>           grad_y;
    Peskin_4pGradient<Direction::z>           grad_z;
    
    Peskin_4p(std::shared_ptr<GlobalData> gd, DataEntry& data)
      :Peskin_4p(InitializeParameters(gd, data)) {}

  private:
    struct Parameters {real h;};
        
    static Parameters InitializeParameters(std::shared_ptr<GlobalData> gd, DataEntry& data) {      
      Box box       = gd->getEnsemble()->getBox();
      real radius   = data.getParameter<real>("radius", -1.0);
      real target_h = data.getParameter<real>("cellSize", -1.0);
      
      if (radius == -1.0 and target_h == -1.0){
        System::log<System::CRITICAL>("[Peskin_4p Kernel] I need either the radius or the cell size.");
      }
      
      if (radius>0 and target_h > 0){
        System::log<System::CRITICAL>("[Peskin_4p Kernel] Radius and cell size provided. I just need one.");
      }
      
      if (target_h<0){
        target_h   = radius * (pow(9*M_PI/128, 1./3.));
      }
      
      real Lx         = box.boxSize.x;
      int nCells      = round(Lx/target_h);
      int3 nCells3    = nextFFTWiseSize3D(make_int3(nCells));
      real h          = Lx/nCells3.x;
      return {h};
    }

    Peskin_4p(const Parameters& p):
      kernel(p.h), grad_x(p.h), grad_y(p.h), grad_z(p.h){
    }
  };

  class Peskin_6pBase{
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
      return (-beta_val + term) / (2.0 * alpha);
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
    Peskin_6pBase(real h):h(h), invh(1./h) {}
    
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
    Peskin_6pBase base;
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
      return (-beta_val + term) / (2.0 * alpha);
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
      real pm3   = (-beta(rr) + std::sqrt(discr)) / (2.0 * alpha);
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
    
    
    __host__ __device__ real derivPhiBase(real rr, real3 pos = real3()) const {
      real r = rr * invh;
      if      (r >= -3.0 && r < -2.0) return -dphi_r_minus_3(r + 3.0)*invh*invh;
      else if (r >= -2.0 && r < -1.0) return -dphi_r_minus_2(r + 2.0)*invh*invh;
      else if (r >= -1.0 && r <  0.0) return -dphi_r_minus_1(r + 1.0)*invh*invh;
      else if (r >=  0.0 && r <  1.0) return -dphi_r(r)*invh*invh;
      else if (r >=  1.0 && r <  2.0) return -dphi_r_plus_1(r - 1.0)*invh*invh;
      else if (r >=  2.0 && r <  3.0) return -dphi_r_plus_2(r - 2.0)*invh*invh;
      else                            return 0.0;
    }
    
    
  public:
    static constexpr int support = 6;
    
    // Constructor que inicializa h e invh usando una caja dada
    Peskin_6pGradient(real h):
      h(h), invh(1./h), base(h){}

    __host__ __device__ real phiX(real r, real3 pos = real3()) {
      if constexpr (gradientDirection == Direction::x) {
        return derivPhiBase(r);
      } else {
        return base.phi(r);
      }
    }
    
    __host__ __device__ real phiY(real r, real3 pos = real3()) {
      if constexpr (gradientDirection == Direction::y) {
        return derivPhiBase(r);
      } else {
        return base.phi(r);
      }
    }
    
    __host__ __device__ real phiZ(real r, real3 pos = real3()) {
      if constexpr (gradientDirection == Direction::z) {
        return derivPhiBase(r);
      } else {
        return base.phi(r);
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

  struct Peskin_6p {
    Peskin_6pBase                             kernel;
    Peskin_6pGradient<Direction::x>           grad_x;
    Peskin_6pGradient<Direction::y>           grad_y;
    Peskin_6pGradient<Direction::z>           grad_z;
    
    Peskin_6p(std::shared_ptr<GlobalData> gd, DataEntry& data)
      :Peskin_6p(InitializeParameters(gd, data)) {}

  private:
    struct Parameters {real h;};
        
    static Parameters InitializeParameters(std::shared_ptr<GlobalData> gd, DataEntry& data) {      
      Box box       = gd->getEnsemble()->getBox();
      real radius   = data.getParameter<real>("radius", -1.0);
      real target_h = data.getParameter<real>("cellSize", -1.0);
      
      if (radius == -1.0 and target_h == -1.0){
        System::log<System::CRITICAL>("[Peskin_6p Kernel] I need either the radius or the cell size.");
      }
      
      if (radius>0 and target_h > 0){
        System::log<System::CRITICAL>("[Peskin_6p Kernel] Radius and cell size provided. I just need one.");
      }
      
      if (target_h<0){
        target_h   = radius * (pow(9*M_PI/128, 1./3.));
      }
      
      real Lx         = box.boxSize.x;
      int nCells      = round(Lx/target_h);
      int3 nCells3    = nextFFTWiseSize3D(make_int3(nCells));
      real h          = Lx/nCells3.x;
      return {h};
    }

    Peskin_6p(const Parameters& p):
      kernel(p.h), grad_x(p.h), grad_y(p.h), grad_z(p.h){
    }
  };

  
  enum class KernelName {Gaussian, Peskin_3p, Peskin_4p, Peskin_6p};
  using KernelVariant = std::variant<Gaussian, Peskin_3p, Peskin_4p, Peskin_6p>;

  inline KernelName kernelFromString(std::string s){
    for(auto& c: s) c = std::tolower(c);
    if(s=="gaussian") return KernelName::Gaussian;
    if(s=="peskin_3p") return KernelName::Peskin_3p;
    if(s=="peskin_4p") return KernelName::Peskin_4p;
    if(s=="peskin_6p") return KernelName::Peskin_6p;
    System::log<System::CRITICAL>("[Kernel] Unknown kernel %s.", s.c_str());
    std::terminate();
  }
  
  
  inline KernelVariant initializeKernel(std::shared_ptr<GlobalData> gd,
                                        DataEntry& data){
    
    std::string kernelStr = data.getParameter<std::string>("kernel");
    KernelName kind       = kernelFromString(kernelStr);
    
    switch(kind){
    case KernelName::Gaussian:           return Gaussian(gd, data);
    case KernelName::Peskin_3p:          return Peskin_3p(gd, data);
    case KernelName::Peskin_4p:          return Peskin_4p(gd, data);
    case KernelName::Peskin_6p:          return Peskin_6p(gd, data);
    default: System::log<System::CRITICAL>("[Kernel] Unknown kernel.");
    }
    std::terminate();
  }

  inline int getSupport(const KernelVariant& kernel) {
    return std::visit([](const auto& ker) {
      return ker.kernel.getSupport();
    }, kernel);
  }
  
  inline real getCellSize(const KernelVariant& kernel) {
    return std::visit([](const auto& ker) {
      return ker.kernel.getCellSize();
    }, kernel);
  }
  
}}}}}}
