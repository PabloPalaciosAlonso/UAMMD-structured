#include "uammd.cuh"
#include "misc/TabulatedFunction.cuh"
#include "misc/IBM.cuh"

#include "System/ExtendedSystem.cuh"
#include "GlobalData/GlobalData.cuh"

#include "Interactor/Interactor.cuh"
#include "Interactor/InteractorFactory.cuh"

#include "Definitions/SFINAE.cuh"
#include "Utils/Containers/SetUtils.cuh"

#include "Definitions/Computations.cuh"
#include "Definitions/Types.cuh"
#include "DipolarInteractions/DipolarMagnetic_impl.cuh"

namespace uammd{
namespace structured{
namespace Interactor{
namespace LongRange{
namespace Dipolar{
    
  class DipolarMagnetic: public Interactor{

    template<class T>
    using cached_vector     = uammd::uninitialized_cached_vector<T>;
    
    detail::KernelVariant kernelGroup;
    std::shared_ptr<detail::FFTManager> FFT;
    Grid grid;    
    TabulatedFunction<real> spuriousSelfForce;
    std::shared_ptr<GlobalData>     gd;
    
    Box box;
    real permeability;
    real radius;

    bool firstCall               = true;    
    bool warningEnergy           = false;
    bool warningForce            = false;
    bool warningLambdaDerivative = false;
    bool warningStress           = false;
    bool warningHessian          = false;
    bool warningMagneticField    = false;
    bool warningPairwiseForces   = false;
    
    cached_vector<real3> particleMagnetizationToGridMagnetization(cudaStream_t st){
      int numberParticles = pg->getNumberParticles();
      int3 nCells         = grid.cellDim;
      int totalNCells     = nCells.x * nCells.y * nCells.z;
      cached_vector<real3>  gridMagnetization(totalNCells);
      detail::fillWithZero(gridMagnetization);
      auto d_m_M        = pd->getMagnetization(access::location::gpu, access::mode::read).raw();
      auto d_pos        = pd->getPos(access::location::gpu, access::mode::read).raw();
      auto d_gridMagnet = thrust::raw_pointer_cast(gridMagnetization.data());
      
      detail::particleMagnetizationToGridMagnetization(d_pos, d_m_M, d_gridMagnet,
                                                                   kernelGroup, grid,
                                                                   numberParticles, st);
      return gridMagnetization;
    }

    cached_vector<complex3> gridMagnetizationFourierToGridFieldFourier(cached_vector<complex3>& gridMagnetizationFourier, cudaStream_t st){
      cached_vector<complex3> gridFieldFourier(gridMagnetizationFourier.size());
      auto d_gridFieldFourier         = (complex3*) thrust::raw_pointer_cast(gridFieldFourier.data());
      auto d_gridMagnetizationFourier = (complex3*) thrust::raw_pointer_cast(gridMagnetizationFourier.data());
      detail::gridMagnetizationFourierToGridFieldFourier(d_gridMagnetizationFourier,
                                                                     d_gridFieldFourier,
                                                                     grid, permeability,
                                                                     st);
      return gridFieldFourier;
    }
    
    //TODO::Make a veresion of this function usable with raw pointers to put in the impl file.
    //Then write the functions particleMagnetizationToGridMagnetization in the impl file 
    cached_vector<real3> gridMagnetizationToGridField(cached_vector<real3> &gridMagnetization, cudaStream_t st){
      auto gridMagnetizationFourier = FFT->forwardTransform(gridMagnetization);
      auto gridFieldFourier         = gridMagnetizationFourierToGridFieldFourier(gridMagnetizationFourier, st);
      auto gridField                = FFT->inverseTransform(gridFieldFourier);
      return gridField;
    }
    
    cached_vector<real3> particleMagnetizationToGridField(cudaStream_t &st){
      cached_vector<real3> gridMagn  = particleMagnetizationToGridMagnetization(st);
      cached_vector<real3> gridField = gridMagnetizationToGridField(gridMagn, st);
      return gridField;
    }

    cached_vector<real3> interpolateGridField(cached_vector<real3> &gridField, cudaStream_t st){ 
      int numberParticles = pg->getNumberParticles();            
      cached_vector<real3> field_cv(numberParticles);
      detail::fillWithZero(field_cv);
      auto d_pos          = pd->getPos(access::gpu, access::read).raw();
      auto d_gridField    = thrust::raw_pointer_cast(gridField.data());
      auto d_field        = thrust::raw_pointer_cast(field_cv.data());
      detail::gridFieldToParticleField(d_pos, d_field, d_gridField,
                                                   kernelGroup, grid, numberParticles, st);      
      return field_cv;
    }

    void gridFieldToParticleEnergy(cached_vector<real3> &gridField, cudaStream_t &st) {
      cached_vector<real3> magneticField = interpolateGridField(gridField, st);
      auto magnetization = pd->getMagnetization(access::gpu, access::read);
      auto energy        = pd->getEnergy(access::gpu, access::readwrite);
      thrust::for_each(
                       thrust::cuda::par.on(st),
                       thrust::make_zip_iterator(
                                                 thrust::make_tuple(magneticField.begin(),
                                                                    magnetization.begin(),
                                                                    energy.begin())),
                       thrust::make_zip_iterator(
                                                 thrust::make_tuple(magneticField.end(),
                                                                    magnetization.end(),
                                                                    energy.end())),
                       detail::ComputeEnergy{});
    }

    void gridFieldToParticleField(cached_vector<real3> &gridField, cudaStream_t &st){
      cached_vector<real3> magneticField_cv = interpolateGridField(gridField, st);
      auto field                            = pd->getMagneticField(access::gpu, access::readwrite);

      thrust::transform(thrust::cuda::par.on(st),
                        magneticField_cv.begin(),
                        magneticField_cv.end(),
                        field.begin(),
                        field.begin(),
                        detail::SumReal3ToReal4{});
    }

    void gridFieldToParticleForce(cached_vector<real3> &gridField, cudaStream_t &st){
      int numberParticles = pg->getNumberParticles();
      cached_vector<real3> force_cv(numberParticles);
      detail::fillWithZero(force_cv);
      auto d_pos           = pd->getPos(access::gpu, access::read).raw();
      auto d_magnetization = pd->getMagnetization(access::gpu, access::read).raw();
      auto d_gridField     = thrust::raw_pointer_cast(gridField.data());
      auto d_force_cv      = thrust::raw_pointer_cast(force_cv.data());
      
      detail::gridFieldToCorrectedParticleForce(d_pos, d_magnetization,
                                                            d_gridField, d_force_cv,
                                                            kernelGroup,
                                                            spuriousSelfForce, grid,
                                                            numberParticles, st);

      auto force = pd->getForce(access::gpu, access::readwrite);
      
      thrust::transform(thrust::cuda::par.on(st),
                        force_cv.begin(),
                        force_cv.end(),
                        force.begin(),
                        force.begin(),
                        detail::SumReal3ToReal4{});
    }
     
    
  public:
    
    DipolarMagnetic(std::shared_ptr<GlobalData>           gd,
                    std::shared_ptr<ParticleGroup>        pg,
                    DataEntry& data,
                    std::string name):Interactor(pg,"DipolarMagnetic: \"" +name+"\""),
                                      gd(gd), kernelGroup(detail::initializeKernel(gd, data)){
      box = gd->getEnsemble()->getBox();
        
      // Read input parameters
      permeability          = data.getParameter<real>("permeability");
      radius                = data.getParameter<real>("radius",1);
      
      real h       = detail::getCellSize(kernelGroup);
      int support  = detail::getSupport(kernelGroup);
      real3 L      = box.boxSize;
      int3 nCells  = {static_cast<int>(round(L.x/h)),
        static_cast<int>(round(L.y/h)),
        static_cast<int>(round(L.z/h))};
      grid         = Grid(box, nCells);
      FFT          = std::make_shared<detail::FFTManager>(nCells);

      System::log<System::MESSAGE>("[SpectralDipolarMagnetic] Magnetic permeability: %f", permeability);
      System::log<System::MESSAGE>("[SpectralDipolarMagnetic] Box size: %f %f %f", L.x, L.y, L.z);
      System::log<System::MESSAGE>("[SpectralDipolarMagnetic] Number of cells: %d %d %d", nCells.x, nCells.y, nCells.z);
      System::log<System::MESSAGE>("[SpectralDipolarMagnetic] Cell size, h: %f ", h);
      System::log<System::MESSAGE>("[SpectralDipolarMagnetic] Support: %d ", support);
    }
    
    cached_vector<real3> interpolateGridForceFinniteDifferences(cached_vector<real3> &gridForce,
                                                                cudaStream_t st){      
      int numberParticles = pg->getNumberParticles();
      int nCells          = gridForce.size();
      
      cached_vector<real3> force_cv(numberParticles);
      detail::fillWithZero(force_cv);
      auto d_pos          = pd->getPos(access::gpu, access::read).raw();
      auto d_gridForce    = thrust::raw_pointer_cast(gridForce.data());
      auto d_force        = thrust::raw_pointer_cast(force_cv.data());
      
      std::visit([&](auto& ker){
        using K = std::decay_t<decltype(ker.kernel)>;
        auto kernPtr = std::make_shared<K>(ker.kernel);
        IBM<K> ibm(kernPtr, grid);
        ibm.gather(d_pos, d_force, d_gridForce, numberParticles, st);
      }, kernelGroup);
      return force_cv;
    }
        
    void sum(Computables comp, cudaStream_t st) override {
      
      Box box = gd->getEnsemble()->getBox();
      if (box != this->box){
        System::log<System::CRITICAL>("[DipolarMagnetic] The box has changed. This is not allowed.");
      }
      
      if (comp.energy or comp.force or comp.magneticField){
        cached_vector<real3> gridField = particleMagnetizationToGridField(st);
        if (comp.energy){
          gridFieldToParticleEnergy(gridField, st);
        }

        if (comp.force){
          if (firstCall){
            auto functor = [this, st](real z_div_h){
              return detail::computeParticleSpuriousSelfForce(kernelGroup, FFT,
                                                              grid, permeability,
                                                              z_div_h, st);
            };
            spuriousSelfForce = TabulatedFunction<real>(2000, 0.0, 1.0, functor);
            firstCall = false;
          }
            gridFieldToParticleForce(gridField, st);
        }      
      
        if (comp.magneticField){
          gridFieldToParticleField(gridField, st);
        }
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
  
}}}}}

REGISTER_INTERACTOR(LongRange,DipolarMagnetic,
                    uammd::structured::Interactor::LongRange::Dipolar::DipolarMagnetic)
