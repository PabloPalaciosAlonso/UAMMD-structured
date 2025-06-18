#include "System/ExtendedSystem.cuh"
#include "GlobalData/GlobalData.cuh"
#include "ParticleData/ExtendedParticleData.cuh"
#include "ParticleData/ParticleGroup.cuh"
#include "ParticleGroup/ParticleGroupUtils.cuh"

#include "Integrator/IntegratorBase.cuh"
#include "Integrator/IntegratorFactory.cuh"
#include "Integrator/IntegratorUtils.cuh"
#include "Integrator/NVT/Brownian/EulerMaruyamaRigidBody.cu"

namespace uammd{
namespace structured{
namespace Integrator{
namespace NVT{
namespace MagneticMotion{
    
  class Brownian : public IntegratorBaseMagneticMotion{
    
  private:
    
    std::shared_ptr<NVT::Brownian::EulerMaruyamaRigidBody>  brownian;
    bool firstStep = true;
  public:

    Brownian(std::shared_ptr<GlobalData> gd,
             std::shared_ptr<ParticleGroup>  pg,
             DataEntry& data,
             std::string name):IntegratorBaseMagneticMotion(gd,pg,data,name){
      std::string magneticIntegratorSubType = data.getParameter<std::string>("magneticIntegrator");
      System::log<System::MESSAGE>("[Magnetic-Brownian] Created \"%s\" Brownian integrator coupled to \"%s\" magnetic integrator",
                                   name.c_str(), magneticIntegratorSubType.c_str());
      
      brownian = std::make_shared<NVT::Brownian::EulerMaruyamaRigidBody>(gd,pg,data,name);
    }
    
    void forwardTime() override {
      if (firstStep){
        loadInteractorsToIntegrator(magneticIntegrator);
        loadUpdatablesToIntegrator(magneticIntegrator);
        firstStep = false;
      }
      updateForceTorqueMagneticField();
      magneticIntegrator->processInteractions();
      brownian->integrationStep(); //Already sets forces and torques to zero
      magneticIntegrator->updateMagnetization();
      magneticIntegrator->resetMagneticField();
      this->gd->getFundamental()->setCurrentStep(this->gd->getFundamental()->getCurrentStep()+1);
      this->gd->getFundamental()->setSimulationTime(this->gd->getFundamental()->getSimulationTime()+this->dt);
    }
  };
  
}}}}}

REGISTER_INTEGRATOR(
    MagneticMotion,Brownian,
    uammd::structured::Integrator::NVT::MagneticMotion::Brownian
)
