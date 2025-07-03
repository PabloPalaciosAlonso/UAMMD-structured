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
    namespace Euler_ns{
      
      __forceinline__ __global__ void integrateLLG(real4* dir, real4*  field,
                                                   real4* magnetization,
                                                   ParticleGroup::IndexIterator indexIterator,
                                                   real dt, real kbT, real damping, real msat, real gyroRatio,
                                                   int currentStep, int seed,  int N){
        
        int id = blockIdx.x*blockDim.x+threadIdx.x;
        if(id>=N) return;
        int i             = indexIterator[id];
        real4 m_and_M     = magnetization[i];
        real3 mi          = make_real3(m_and_M);
        real Mi           = m_and_M.w;
        if (Mi == real(0.0)) return;

        //Moves from the frame of the laboratory to the frame of the particle.
        real3 bi = make_real3(field[i]);
        
        real fluctuationsAmplitude = sqrt(2*kbT*damping/(gyroRatio*Mi*dt));
        real3 thermalField         = computeThermalField(fluctuationsAmplitude, currentStep, seed, id);
        bi+= thermalField;

        real3 dmi = computeMagnetizationDerivative(bi, mi, damping, gyroRatio)*dt;
        mi += dmi;
        magnetization[i] = make_real4(mi*rsqrt(dot(mi,mi)), Mi);
      }
    }
  }

  class LLG_Euler: public IntegratorBaseMagnetic{
  private:
    real damping;
    real msat;
    real gyroRatio;

  public:
    LLG_Euler(std::shared_ptr<GlobalData>    gd,
              std::shared_ptr<ParticleGroup> pg,
              DataEntry& data,
              std::string name):IntegratorBaseMagnetic(gd, pg, data, name){
      //Read input parameters
      damping   = data.getParameter<real>("damping");
      msat      = data.getParameter<real>("msat");
      gyroRatio = data.getParameter<real>("gyroRatio");
      System::log<System::MESSAGE>("[LLG_Euler] Damping parameter, α: %f", damping);
      System::log<System::MESSAGE>("[LLG_Euler] Saturation magnetization: %f", msat);
      System::log<System::MESSAGE>("[LLG_Euler] Gyromagnetic ratio, γ: %f", gyroRatio);
    }
      
          
    void updateMagnetization() override {
        
      auto field         = pd->getMagneticField(access::location::gpu, access::mode::read).raw();
      auto dir           = pd->getDir(access::location::gpu, access::mode::read).raw();
      auto magnetization = pd->getMagnetization(access::location::gpu, access::mode::readwrite).raw();
      auto groupIterator  = pg->getIndexIterator(access::location::gpu);
      uint currentStep    = gd->getFundamental()->getCurrentStep();
      uint seed           = gd->getSystem()->getSeed();
      int BLOCKSIZE       = 128;
      int numberParticles = pg->getNumberParticles();
      uint Nthreads       = BLOCKSIZE<numberParticles?BLOCKSIZE:numberParticles;
      uint Nblocks        = numberParticles/Nthreads +  ((numberParticles%Nthreads!=0)?1:0);
      LLG::Euler_ns::integrateLLG<<<Nblocks, Nthreads, 0, stream>>>(dir, field,
                                                                   magnetization, groupIterator, dt,
                                                                   kBT, damping, msat, gyroRatio,
                                                                   currentStep, seed, numberParticles);
    }
  };
    
}}}}}

  REGISTER_INTEGRATOR(
    Magnetic,LLG_Euler,
    uammd::structured::Integrator::NVT::Magnetic::LLG_Euler
)
