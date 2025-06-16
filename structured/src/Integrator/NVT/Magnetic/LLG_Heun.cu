#include "System/ExtendedSystem.cuh"
#include "GlobalData/GlobalData.cuh"
#include "ParticleData/ExtendedParticleData.cuh"
#include "ParticleData/ParticleGroup.cuh"
#include "ParticleGroup/ParticleGroupUtils.cuh"

#include "Integrator/IntegratorBase.cuh"
#include "Integrator/IntegratorFactory.cuh"
#include "Integrator/IntegratorUtils.cuh"

#include "utils/container.h"
#include "utils/quaternion.cuh"
#include "Landau_Lifshitz_Gilbert.cuh"

namespace uammd{
namespace structured{
namespace Integrator{
namespace NVT{
namespace Magnetic{
  namespace LLG{
    namespace Heun_ns{

      enum class SubStepType {Predictor, Corrector};    
    
      __forceinline__ __global__ void integrateLLG(real4* dir, real4* field, real4* magnetization,
                                                   real3* initialMagnetization, real* anisotropy,
                                                   ParticleGroup::IndexIterator indexIterator,
                                                   real dt, real kbT, real damping, real msat,
                                                   real gyroRatio, int currentStep, int seed, int N,
                                                   SubStepType subStep){
      
        int id = blockIdx.x*blockDim.x+threadIdx.x;
        if(id>=N) return;
        int i = indexIterator[id];
        real4 m_and_M = magnetization[i];
        real3 mi = make_real3(m_and_M);
        real Mi = m_and_M.w;
        if (Mi == real(0.0)) return;
        Quat diri = dir[i];
        real anisotropy_i = anisotropy[i];
        if (subStep==SubStepType::Predictor){
          initialMagnetization[i] = mi;
        }
        real3 bi = make_real3(field[i]);
        //Moves from the frame of the laboratory to the frame of the particle.
        bi = rotateVector(Quat(dir[i]).getConjugate(), bi);
        real fluctuationsAmplitude = sqrt(2*kbT*damping/(gyroRatio*Mi*dt));
        bi += computeThermalField(fluctuationsAmplitude, currentStep, seed, id);
        real3 anisotropyField = computeAnisotropyField(mi, anisotropy_i/msat);
        bi+=anisotropyField;
        real3 dmi = computeMagnetizationDerivative(bi,mi,damping, gyroRatio)*dt;
        if (subStep==SubStepType::Predictor){
          mi+=dmi;
          field[i] = real4();
        } else if (subStep==SubStepType::Corrector) {
          real3 mi_t0 = initialMagnetization[i];
          mi = real(0.5)*(mi_t0 + mi + dmi);
          mi *= rsqrt(dot(mi,mi));
        }
        magnetization[i] = make_real4(mi, Mi);
      }
    }
  }
  
  class LLG_Heun: public IntegratorBaseMagnetic{
  private:
    real damping;
    real msat;
    real gyroRatio;
    thrust::device_vector<real3> magnetizationCopy;
    uint seed;
    
    void updateHalfStep(LLG::Heun_ns::SubStepType subStep){
      auto field          = pd->getMagneticField(access::location::gpu, access::mode::read).raw();
      auto dir            = pd->getDir(access::location::gpu, access::mode::read).raw();
      auto anisotropy     = pd->getAnisotropy(access::location::gpu, access::mode::read).raw();
      auto magnetization  = pd->getMagnetization(access::location::gpu, access::mode::readwrite).raw();
      auto groupIterator  = pg->getIndexIterator(access::location::gpu);
      auto initMagnet_ptr = thrust::raw_pointer_cast(magnetizationCopy.data());
      uint currentStep    = gd->getFundamental()->getCurrentStep();
      
      int BLOCKSIZE       = 128;
      int numberParticles = pg->getNumberParticles();
      uint Nthreads       = BLOCKSIZE<numberParticles?BLOCKSIZE:numberParticles;
      uint Nblocks        = numberParticles/Nthreads +  ((numberParticles%Nthreads!=0)?1:0);
      LLG::Heun_ns::integrateLLG<<<Nblocks, Nthreads, 0, stream>>>(dir, field, magnetization,
                                                                   initMagnet_ptr, anisotropy,
                                                                   groupIterator, dt, kBT,
                                                                   damping, msat, gyroRatio,
                                                                   currentStep, seed,
                                                                   numberParticles, subStep);
      
    }
    
  public:
    LLG_Heun(std::shared_ptr<GlobalData>    gd,
             std::shared_ptr<ParticleGroup> pg,
             DataEntry& data,
             std::string name):IntegratorBaseMagnetic(gd, pg, data, name){
      //Read input parameters
      damping   = data.getParameter<real>("damping");
      msat      = data.getParameter<real>("msat");
      gyroRatio = data.getParameter<real>("gyroRatio");
      seed      = gd->getSystem()->getSeed();
      System::log<System::MESSAGE>("[LLG_Heun] Damping parameter, α: %f", damping);
      System::log<System::MESSAGE>("[LLG_Heun] Saturation magnetization: %f", msat);
      System::log<System::MESSAGE>("[LLG_Heun] Gyromagnetic ratio, γ: %f", gyroRatio);
      magnetizationCopy.resize(pg->getNumberParticles());
      
    }
    
    
    void updateMagnetization() override {
      int currentStep  = this->gd->getFundamental()->getCurrentStep();
      real currentTime = this->gd->getFundamental()->getSimulationTime();
      
      updateHalfStep(LLG::Heun_ns::SubStepType::Predictor);
      resetMagneticField();

      this->gd->getFundamental()->setCurrentStep(currentStep + 1);
      this->gd->getFundamental()->setSimulationTime(currentTime + this->dt);
      
      updateMagneticField();
      //The currentStep and currenteTime must be updated by the forwardTime function
      this->gd->getFundamental()->setCurrentStep(currentStep);
      this->gd->getFundamental()->setSimulationTime(currentTime);
      
      updateHalfStep(LLG::Heun_ns::SubStepType::Corrector);
      
    }
  };
  
}}}}}

  REGISTER_INTEGRATOR(
    Magnetic,LLG_Heun,
    uammd::structured::Integrator::NVT::Magnetic::LLG_Heun
)
