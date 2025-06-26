#include "uammd.cuh"

#include "misc/IBM.cuh"

#include "System/ExtendedSystem.cuh"
#include "GlobalData/GlobalData.cuh"

#include "Interactor/Interactor.cuh"
#include "Interactor/InteractorFactory.cuh"

#include "Definitions/SFINAE.cuh"
#include "Utils/Containers/SetUtils.cuh"

#include "Definitions/Computations.cuh"
#include "Definitions/Types.cuh"

#include "Utils/GridUtils/FFTManager.cu"
//#include "Utils/GridUtils/SpreadInterp.cu"
#include "Utils/GridUtils/Kernels.cu"


namespace uammd{
namespace structured{
namespace Interactor{

  namespace DipolarMagnetic_ns{

    __device__ int3 indexToWaveNumber(int i, int3 nk){
      int ikx = i%(nk.x/2+1);
      int iky = (i/(nk.x/2+1))%nk.y;
      int ikz = i/((nk.x/2+1)*nk.y);
      ikx -= nk.x*(ikx >= (nk.x/2+1));
      iky -= nk.y*(iky >= (nk.y/2+1));
      ikz -= nk.z*(ikz >= (nk.z/2+1));
      return make_int3(ikx, iky, ikz);
    }
    
    __device__ real3 waveNumberToWaveVector(int3 ik, real3 L){
      return (real(2.0)*real(M_PI)/L)*make_real3(ik.x, ik.y, ik.z);
    }
    
    
    /*Apply the projection operator to a wave number with a certain complex factor.
      res = (I-\hat{k}^\hat{k})·factor*/
    __device__ complex3 projectFourier(real3 k, complex3 factor){
      
      complex3 res;
      const real invk2 = real(1.0)/dot(k,k);

      const real3 fr = make_real3(factor.x.real(), factor.y.real(), factor.z.real());
      const real kfr = dot(k,fr)*invk2;
      const real3 vr = (fr-k*kfr);
      
      const real3 fi = make_real3(factor.x.imag(), factor.y.imag(), factor.z.imag());
      const real kfi = dot(k,fi)*invk2;
      const real3 vi = (fi-k*kfi);
      
      res.x = complex{vr.x, vi.x};
      res.y = complex{vr.y, vi.y};
      res.z = complex{vr.z, vi.z};
      return res;
    }
    
    __global__ void magnetizationFourier2Field(const complex3* gridMagnetization,
                                               complex3* gridField,
                                               real permeability,
                                               Grid grid){
      
      const int id      = blockIdx.x*blockDim.x + threadIdx.x;      
      const int3 nCells = grid.cellDim;
      if(id>=(nCells.z*nCells.y*(nCells.x/2+1))) return;
      const real B = permeability/real(nCells.x*nCells.y*nCells.z);
      if(id == 0){
        gridField[0] = real(2.0)/real(3.0)*B*gridMagnetization[0];	
        return;
      }
      
      const int3 waveNumber = indexToWaveNumber(id, nCells);
      const real3 k         = waveNumberToWaveVector(waveNumber, grid.box.boxSize);
      complex3 factor       = gridMagnetization[id];
      gridField[id]         = projectFourier(k, factor)*B;
    }
  }
  
  
  
  
  class DipolarMagnetic: public Interactor{

    template<class T>
    using cached_vector     = uammd::uninitialized_cached_vector<T>;
    // using Gaussian          = kernels::BetaSpline3;
    // using GaussianGradientX = kernels::BetaSpline3Gradient<kernels::Direction::x>;
    // using GaussianGradientY = kernels::BetaSpline3Gradient<kernels::Direction::y>;
    // using GaussianGradientZ = kernels::BetaSpline3Gradient<kernels::Direction::z>;

    using Gaussian          = kernels::Gaussian;
    using GaussianGradientX = kernels::GaussianGradient<kernels::Direction::x>;
    using GaussianGradientY = kernels::GaussianGradient<kernels::Direction::y>;
    using GaussianGradientZ = kernels::GaussianGradient<kernels::Direction::z>;

    // using Gaussian          = kernels::Peskin_6p;
    // using GaussianGradientX = kernels::Peskin_6pGradient<kernels::Direction::x>;
    // using GaussianGradientY = kernels::Peskin_6pGradient<kernels::Direction::y>;
    // using GaussianGradientZ = kernels::Peskin_6pGradient<kernels::Direction::z>;
    
  private:

    std::shared_ptr<Gaussian> kernel;
    Grid grid;
    std::unique_ptr<FFTManager> FFT;
    
    std::unique_ptr<IBM<Gaussian>> ibm;
    std::unique_ptr<IBM<GaussianGradientX>> ibmGradX;
    std::unique_ptr<IBM<GaussianGradientY>> ibmGradY;
    std::unique_ptr<IBM<GaussianGradientZ>> ibmGradZ;
        

    cached_vector<complex3> convolveFourier(cached_vector<complex3>& gridMagnetizationFourier, cudaStream_t st){
      const int3 nCells       = grid.cellDim;
      int Nthreads            = 128;
      int Nblocks             = (nCells.z*nCells.y*(nCells.x/2+1))/Nthreads +1;
      cached_vector<complex3> gridFieldFourier(gridMagnetizationFourier.size());
      
      auto d_gridFieldFourier         = (complex3*) thrust::raw_pointer_cast(gridFieldFourier.data());
      auto d_gridMagnetizationFourier = (complex3*) thrust::raw_pointer_cast(gridMagnetizationFourier.data());
      
      DipolarMagnetic_ns::magnetizationFourier2Field<<<Nblocks, Nthreads, 0, st>>> (d_gridMagnetizationFourier,
                                                                                    d_gridFieldFourier,
                                                                                    permeability,
                                                                                    grid);
      return gridFieldFourier;
    }
        
  protected:
    
    std::shared_ptr<GlobalData>     gd;
    
    ////////////////////////////////////
    
    Box box;

    real permeability;
    real volume;
    ////////////////////////////////////
    //Warnings
    
    bool warningEnergy           = false;
    bool warningForce            = false;
    bool warningLambdaDerivative = false;
    bool warningStress           = false;
    bool warningHessian          = false;
    bool warningMagneticField    = false;
    bool warningPairwiseForces   = false;

  public:
      
    DipolarMagnetic(std::shared_ptr<GlobalData>           gd,
                    std::shared_ptr<ParticleGroup>        pg,
                    DataEntry& data,
                    std::string name):Interactor(pg,"DipolarMagnetic: \"" +name+"\""),
                                      gd(gd){
      box = gd->getEnsemble()->getBox();
        
      // Read input parameters
      permeability     = data.getParameter<real>("permeability");
      real coreRadius  = data.getParameter<real>("coreRadius");
      real tolerance   = data.getParameter<real>("tolerance",1e-4);

      volume       = 4./3.*M_PI*pow(coreRadius,3);
      real sigma   = coreRadius / (pow(6*sqrt(M_PI), 1./3.));
      auto kernel  = std::make_shared<Gaussian>(sigma, tolerance, box);
      //auto kernel  = std::make_shared<Gaussian>(coreRadius*0.75, box);
      real h       = kernel->getCellSize();
      real3 L      = box.boxSize;
      int3 nCells  = {int(L.x/h), int(L.y/h), int(L.z/h)};
      grid         = Grid(box, nCells);

      int support  = kernel->getSupport();
      auto kernelGradX = std::make_shared<GaussianGradientX>(sigma, h, support);
      auto kernelGradY = std::make_shared<GaussianGradientY>(sigma, h, support);
      auto kernelGradZ = std::make_shared<GaussianGradientZ>(sigma, h, support);

      // auto kernelGradX = std::make_shared<GaussianGradientX>(coreRadius*0.75, box);
      // auto kernelGradY = std::make_shared<GaussianGradientY>(coreRadius*0.75, box);
      // auto kernelGradZ = std::make_shared<GaussianGradientZ>(coreRadius*0.75, box);
      
      FFT          = std::make_unique<FFTManager>(nCells);
      ibm          = std::make_unique<IBM<Gaussian>>(kernel, grid);
      ibmGradX     = std::make_unique<IBM<GaussianGradientX>>(kernelGradX, grid);
      ibmGradY     = std::make_unique<IBM<GaussianGradientY>>(kernelGradY, grid);
      ibmGradZ     = std::make_unique<IBM<GaussianGradientZ>>(kernelGradZ, grid);
      
      System::log<System::MESSAGE>("[SpectralDipolarMagnetic] Magnetic permeability: %f", permeability);
      System::log<System::MESSAGE>("[SpectralDipolarMagnetic] Core radius: %f", coreRadius);
      System::log<System::MESSAGE>("[SpectralDipolarMagnetic] Tolerance: %f", tolerance);
      System::log<System::MESSAGE>("[SpectralDipolarMagnetic] Box size: %f %f %f", L.x, L.y, L.z);
      System::log<System::MESSAGE>("[SpectralDipolarMagnetic] Number of cells: %d %d %d", nCells.x, nCells.y, nCells.z);
      System::log<System::MESSAGE>("[SpectralDipolarMagnetic] Cell size, h: %f ", h);
      System::log<System::MESSAGE>("[SpectralDipolarMagnetic] Support: %d ", support);
    }

    cached_vector<real3> spreadMagnetization(cudaStream_t st){
      int numberParticles = pg->getNumberParticles();
      int3 nCells         = grid.cellDim;
      int totalNCells     = nCells.x * nCells.y * nCells.z;
      cached_vector<real4> pos_cv(numberParticles);
      cached_vector<real3> magnet_cv(numberParticles);
      cached_vector<real3>  gridMagnetization(totalNCells);
      thrust::fill(gridMagnetization.begin(), gridMagnetization.end(), real3());
      
      auto dir = pd->getDir(access::location::gpu, access::mode::read);
      auto m_M = pd->getMagnetization(access::location::gpu, access::mode::read);
      
      thrust::transform(thrust::device,
                        thrust::make_zip_iterator(thrust::make_tuple(m_M.begin(), dir.begin())),
                        thrust::make_zip_iterator(thrust::make_tuple(m_M.end(), dir.end())),
                        magnet_cv.begin(),
                        [] __device__ (const thrust::tuple<real4, real4>& tpl) {
                          const real4& val = thrust::get<0>(tpl);
                          const real4& direction = thrust::get<1>(tpl);
                          //real3 rotated = rotateVector(direction, make_real3(val));
                          real3 rotated = make_real3(val);
                          return rotated * val.w;
                        });
      auto d_pos        = pd->getPos(access::location::gpu, access::mode::read).raw();
      auto d_magnet     = thrust::raw_pointer_cast(magnet_cv.data());
      auto d_gridMagnet = thrust::raw_pointer_cast(gridMagnetization.data());
      
      ibm->spread(d_pos, d_magnet, d_gridMagnet, numberParticles, st);
      return gridMagnetization;
    }
    
    cached_vector<real3> computeGridField(cached_vector<real3> &gridMagnetization, cudaStream_t st){
      auto gridMagnetizationFourier = FFT->forwardTransform(gridMagnetization);
      auto gridFieldFourier         = convolveFourier(gridMagnetizationFourier, st);
      auto gridField                = FFT->inverseTransform(gridFieldFourier);
      return gridField;
    }

    cached_vector<real3> interpolateGridField(cached_vector<real3> &gridField, cudaStream_t st){      
      int numberParticles = pg->getNumberParticles();
      int nCells          = gridField.size();
      
      cached_vector<real3> field_cv(numberParticles);
      thrust::fill(field_cv.begin(), field_cv.end(), real3());
         
      auto d_pos          = pd->getPos(access::gpu, access::read).raw();
      auto d_gridField    = thrust::raw_pointer_cast(gridField.data());
      auto d_field        = thrust::raw_pointer_cast(field_cv.data());
      
      ibm->gather(d_pos, d_field, d_gridField, numberParticles, st);
      return field_cv;
    }
    
    
    void sum(Computables comp, cudaStream_t st) override {
        
      Box box = gd->getEnsemble()->getBox();
      if (box != this->box){
        System::log<System::CRITICAL>("[DipolarMagnetic] The box has changed. This is not allowed.");
      }

      if(comp.energy == true){
        cached_vector<real3> gridMagn  = spreadMagnetization(st);
        cached_vector<real3> gridField = computeGridField(gridMagn, st);
        cached_vector<real3> field     = interpolateGridField(gridField, st);
        auto m_M                       = pd->getMagnetization(access::gpu, access::read);
        auto energy                    = pd->getEnergy(access::gpu, access::readwrite);

        thrust::transform(thrust::device,
        thrust::make_zip_iterator(thrust::make_tuple(m_M.begin(), field.begin(), energy.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(m_M.end(), field.end(), energy.end())),
        energy.begin(),
        [] __device__ (thrust::tuple<real4, real3, real> t) -> real {
          real4 m_Mi = thrust::get<0>(t);
          real3 fi   = thrust::get<1>(t);
          real  ei   = thrust::get<2>(t);
          return ei + dot(make_real3(m_Mi), fi) * m_Mi.w;
        });
      }
        

       if(comp.force == true){
         std::cout<<"No esta implementado tete\n";
         // cached_vector<real3> gridMagn  = spreadMagnetization(st);
         // cached_vector<real3> gridField = computeGridField(gridMagn, st);
         
         // //Compute the force
         // auto d_pos                     = pd->getPos(access::gpu, access::read).raw();
         // auto force                     = pd->getForce(access::gpu, access::readwrite);
         // auto magnetization             = pd->getMagnetization(access::gpu, access::read);
         
         
         // int numberParticles = pg->getNumberParticles();
         
         // cached_vector<real3> dBx(numberParticles);
         // cached_vector<real3> dBy(numberParticles);
         // cached_vector<real3> dBz(numberParticles);

         // thrust::fill(dBx.begin(), dBx.end(), real3());
         // thrust::fill(dBy.begin(), dBy.end(), real3());
         // thrust::fill(dBz.begin(), dBz.end(), real3());
         
         // auto d_gridField   = thrust::raw_pointer_cast(gridField.data());
         // auto d_dBx         = thrust::raw_pointer_cast(dBx.data());
         // auto d_dBy         = thrust::raw_pointer_cast(dBy.data());
         // auto d_dBz         = thrust::raw_pointer_cast(dBz.data());

         
         // ibmGradX->gather(d_pos, d_dBx, d_gridField, numberParticles, st);
         // ibmGradY->gather(d_pos, d_dBy, d_gridField, numberParticles, st);
         // ibmGradZ->gather(d_pos, d_dBz, d_gridField, numberParticles, st);
        
         // thrust::transform(thrust::device,
         //                  thrust::make_zip_iterator(thrust::make_tuple(dBx.begin(), dBy.begin(), dBz.begin(), force.begin(), magnetization.begin())),
         //                  thrust::make_zip_iterator(thrust::make_tuple(dBx.end(), dBy.end(), dBz.end(), force.end(), magnetization.begin())),
         //                  force.begin(),
         //                  [vol] __device__ (thrust::tuple<real3, real3, real3, real4, real4> in) ->real4 {
         //                    real3 dBxi = thrust::get<0>(in);
         //                    real3 dByi = thrust::get<1>(in);
         //                    real3 dBzi = thrust::get<2>(in);
         //                    real4 fi   = thrust::get<3>(in);
         //                    real4 m_M  = thrust::get<4>(in);

         //                    real3 ftot = m_M.w*(m_M.x*dBxi + m_M.y*dByi + m_M.z*dBzi);
         //                    printf("dBx: %f %f %f\n", dBxi.x, dBxi.y, dBxi.z);
         //                    printf("dBy: %f %f %f\n", dByi.x, dByi.y, dByi.z);
         //                    printf("dBz: %f %f %f\n", dBzi.x, dBzi.y, dBzi.z);
         //                    printf("%f %f %f\n", ftot.x, ftot.y, ftot.z);
         //                    return fi - make_real4(ftot, 0.0);});
         
       
      //   cached_vector<real3> gridMagn  = spreadMagnetization(st);
      //   cached_vector<real3> gridField = computeGridField(gridMagn, st);

      //   //Compute the force
      //   auto d_pos                     = pd->getPos(access::gpu, access::read).raw();
      //   auto force                     = pd->getForce(access::gpu, access::readwrite);

      //   cached_vector<real> gridEnergy(gridMagn.size());
      //   thrust::fill(gridEnergy.begin(), gridEnergy.end(), 0);
      //   thrust::transform(thrust::device,
      //                     gridMagn.begin(),
      //                     gridMagn.end(),
      //                     gridField.begin(),
      //                     gridEnergy.begin(),
      //                     [] __device__ (const real3& magn, const real3& field) {
      //                       return dot(magn, field);});

      //   int numberParticles = pg->getNumberParticles();
        
      //   cached_vector<real> fx(numberParticles);
      //   cached_vector<real> fy(numberParticles);
      //   cached_vector<real> fz(numberParticles);
        
      //   auto d_gridEnergy = thrust::raw_pointer_cast(gridEnergy.data());
      //   auto d_fx         = thrust::raw_pointer_cast(fx.data());
      //   auto d_fy         = thrust::raw_pointer_cast(fy.data());
      //   auto d_fz         = thrust::raw_pointer_cast(fz.data());

      //   ibmGradX->gather(d_pos, d_fx, d_gridEnergy, numberParticles, st);
      //   ibmGradY->gather(d_pos, d_fy, d_gridEnergy, numberParticles, st);
      //   ibmGradZ->gather(d_pos, d_fz, d_gridEnergy, numberParticles, st);

      //   real vol = 0.5;//this->volume;
        
      //   thrust::transform(thrust::device,
      //                     thrust::make_zip_iterator(thrust::make_tuple(fx.begin(), fy.begin(), fz.begin(), force.begin())),
      //                     thrust::make_zip_iterator(thrust::make_tuple(fx.end(), fy.end(), fz.end(), force.end())),
      //                     force.begin(),
      //                     [vol] __device__ (thrust::tuple<real, real, real, real4> in) ->real4 {
      //                       real fxi = thrust::get<0>(in)*vol*2;
      //                       real fyi = thrust::get<1>(in)*vol*2;
      //                       real fzi = thrust::get<2>(in)*vol*2;
      //                       real4 fi = thrust::get<3>(in);
      //                       printf("%f %f %f\n", fxi, fyi, fzi);
      //                       return fi - real4{fxi, fyi, fzi, 0.0};});
        //The sign "-" arises because \int \phi*grad(m*B) = -\int mB*grad(\phi)
        
        //Compute the torque
      //   cached_vector<real3> field_cv  = interpolateGridField(gridField, st);
      //   auto m_M                       = pd->getMagnetization(access::gpu, access::read);
      //   auto torque                    = pd->getTorque(access::gpu, access::readwrite);
                
      //   thrust::transform(thrust::device,
      //                     thrust::make_zip_iterator(thrust::make_tuple(m_M.begin(), field_cv.begin(), torque.begin())),
      //                     thrust::make_zip_iterator(thrust::make_tuple(m_M.end(), field_cv.end(), torque.end())),
      //                     torque.begin(),
      //                     [] __device__ (thrust::tuple<real4, real3, real4> in) ->real4 {
      //                       real4 m_Mi = thrust::get<0>(in);
      //                       real3 bi   = thrust::get<1>(in);
      //                       real4 ti   = thrust::get<2>(in);
      //                       return ti + make_real4(m_Mi.w * cross(make_real3(m_Mi), bi), 0);}); 
      // }
       }
       
      if(comp.magneticField == true){
        cached_vector<real3> gridMagn  = spreadMagnetization(st);
        cached_vector<real3> gridField = computeGridField(gridMagn, st);
        cached_vector<real3> field_cv  = interpolateGridField(gridField, st);
        auto field                     = pd->getMagneticField(access::gpu, access::readwrite);
        
        thrust::transform(thrust::device,
                          field_cv.begin(), field_cv.end(),
                          field.begin(), field.begin(),
                          [] __device__ (real3 cv, real4 f) -> real4 {
                            return f + make_real4(cv, 0.0);});
      }
      
      if(comp.lambdaDerivative == true){
        if(!warningLambdaDerivative){
          System::log<System::WARNING>("[DipolarMagnetic] (%s) Requested non-implemented computable (lambdaDerivative)",
                                       name.c_str());
          warningLambdaDerivative = true;
        }
      }

      if(comp.stress == true){
        if(!warningStress){
          System::log<System::WARNING>("[DipolarMagnetic] (%s) Requested non-implemented computable (stress)",
                                       name.c_str());
          warningStress = true;
        }
      }

      if(comp.hessian == true){
        if(!warningHessian){
          System::log<System::WARNING>("[DipolarMagnetic] (%s) Requested non-implemented computable (hessian)",
                                       name.c_str());
          warningHessian = true;
        }
      }

      if(comp.pairwiseForce == true){
        if(!warningPairwiseForces){
          System::log<System::WARNING>("[DipolarMagnetic] (%s) Requested non-implemented computable (pairwiseForces)",
                                       name.c_str());
          warningPairwiseForces = true;
        }
      }


    }
  };

}}}

REGISTER_INTERACTOR(
                    LongRange,DipolarMagnetic,
                    uammd::structured::Interactor::DipolarMagnetic
                    )
