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
  
  class RigidDipole : public IntegratorBaseMagnetic{
  private:
    real3 magneticMomentDir;
    
  public:
    RigidDipole(std::shared_ptr<GlobalData>    gd,
                std::shared_ptr<ParticleGroup> pg,
                DataEntry&                     data,
                std::string                    name):IntegratorBaseMagnetic(gd, pg, data, name){
        
        magneticMomentDir = data.getParameter<real3>("magneticMomentDir");
        System::log<System::MESSAGE>("[RigidDipole] The magnetic moments are aligned with the axis [%f %f %f] in each particle frame",
                                     magneticMomentDir.x,
                                     magneticMomentDir.y,
                                     magneticMomentDir.z);
    }

    void updateMagnetization() override {
      
      auto dir = pd->getDir(access::location::gpu, access::mode::read);
      auto m_M = pd->getMagnetization(access::location::gpu, access::mode::readwrite);
      
      thrust::transform(thrust::device, dir.begin(), dir.end(),
                        m_M.begin(), m_M.begin(),
                        [mdir = magneticMomentDir] __device__ (const real4 &q, const real4 &m_in)
                        {return make_real4(rotateVector(Quat(q), mdir), m_in.w);});
    }
    
    void processInteractions() override {
      auto magneticField = pd->getMagneticField(access::location::gpu, access::mode::read);
      auto torque        = pd->getTorque(access::location::gpu, access::mode::readwrite);
      auto m_M           = pd->getMagnetization(access::location::gpu, access::mode::read);
      auto nParticles    = pg->getNumberParticles();
      
      thrust::transform(thrust::device, thrust::make_counting_iterator<int>(0),
                        thrust::make_counting_iterator<int>(nParticles),
                        torque.begin(),
                        [=] __device__ (int i) {
                          real3 m_i     = make_real3(m_M[i]);
                          real  M_i     = m_M[i].w;
                          real3 field_i = make_real3(magneticField[i]);
                          real3 t_i     = make_real3(torque[i]);
                          return make_real4(t_i + M_i * cross(m_i, field_i), 0);});
    }
  };
  
}}}}}

  REGISTER_INTEGRATOR(
    Magnetic,RigidDipole,
    uammd::structured::Integrator::NVT::Magnetic::RigidDipole
)
