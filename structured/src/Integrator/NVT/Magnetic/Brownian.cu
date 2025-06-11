#include "Integrator/NVT/Brownian/EulerMaruyamaRigidBody.cu"
#include "MagneticBase.cuh"

namespace uammd{
namespace structured{
namespace Integrator{
namespace NVT{
namespace Magnetic{

  class Brownian : public NVT::Brownian::EulerMaruyamaRigidBody{

  private:

    std::unique_ptr<MagneticBase> magnetic;
    bool firstStep = true;

  public:

    Brownian(std::shared_ptr<GlobalData> gd,
	     std::shared_ptr<ParticleGroup>  pg,
	     DataEntry& data,
	     std::string name):Brownian::EulerMaruyamaRigidBody(gd,pg,data,name){

      typename MagneticBase::Parameters parameters;

      parameters.stream = this->stream;
      parameters.dt     = this->dt;
      parameters.kBT		= this->kBT;
      parameters.damping = data.getParameter<real>("damping", -1);
      parameters.msat = data.getParameter<real>("msat");
      parameters.gyroRatio = data.getParameter<real>("gyroRatio", -1);
      parameters.magneticIntegrationAlgorithm = data.getParameter<std::string>("magneticIntegrationAlgorithm","NotSelected");

      magnetic = std::make_unique<MagneticBase>(gd,pg,parameters,name);

    }

    void forwardTime() override {

      if(firstStep){
	//Load all interactors into magnetic
	for(auto& interactor : this->getInteractors()){
	  magnetic->addInteractor(interactor);
	}

	//Load all updatables into magnetic
	for(auto& updatable : this->getUpdatables()){
	  magnetic->addUpdatable(updatable);
	}
    
	firstStep = false;
      }
      this->updateForce();
      magnetic->updateMagnetization();

      EulerMaruyamaRigidBody::integrationStep(); //Integration step set forces and torques to zero !!!
      this->gd->getFundamental()->setCurrentStep(this->gd->getFundamental()->getCurrentStep()+1);
      this->gd->getFundamental()->setSimulationTime(this->gd->getFundamental()->getSimulationTime()+this->dt);
      magnetic->resetMagneticField();
    }

  };

}}}}}

REGISTER_INTEGRATOR(
    Magnetic,Brownian,
    uammd::structured::Integrator::NVT::Magnetic::Brownian
)
