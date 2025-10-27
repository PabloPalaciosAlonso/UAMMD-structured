#include "System/ExtendedSystem.cuh"
#include "GlobalData/GlobalData.cuh"
#include "ParticleData/ExtendedParticleData.cuh"
#include "ParticleData/ParticleGroup.cuh"
#include "ParticleGroup/ParticleGroupUtils.cuh"

#include "Integrator/IntegratorBase.cuh"
#include "Integrator/IntegratorFactory.cuh"
#include "Integrator/IntegratorUtils.cuh"

#include "Integrator/BDHI/BDHI_EulerMaruyama.cuh"
#include "Integrator/BDHI/BDHI_FCM.cuh"

namespace uammd{
namespace structured{
namespace Integrator{
namespace NVT{
namespace MagneticMotion{

  class ForceCouplingMethod : public IntegratorBaseMagneticMotion{
    using Kernel       = BDHI::FCM_ns::Kernels::Gaussian;
    using KernelTorque = BDHI::FCM_ns::Kernels::GaussianTorque;
    using FCM_super    = BDHI::FCM_impl<Kernel, KernelTorque>;

  private:
    std::shared_ptr<FCM_super> fcm;
    bool firstStep = true;
  public:

    ForceCouplingMethod(std::shared_ptr<GlobalData>           gd,
			std::shared_ptr<ParticleGroup>        pg,
			DataEntry& data,
			std::string name):IntegratorBaseMagneticMotion(gd,pg,data,name){

      bool resizeBox           = data.getParameter<bool>("resizeBox", false);

      //Set up the bdhi integrator
      FCM_super::Parameters bdhiParams;
      bdhiParams.temperature = this ->kBT;
      bdhiParams.viscosity = data.getParameter<real>("viscosity");
      bdhiParams.hydrodynamicRadius = data.getParameter<real>("hydrodynamicRadius", -1.0);
      bdhiParams.tolerance          = data.getParameter<real>("tolerance", 1e-3);
      bdhiParams.cells              = make_int3(data.getParameter<real3>("cells",
									 make_real3(-1, -1, -1)));
      bdhiParams.adaptBoxSize       = data.getParameter<bool>("adaptBoxSize", false);
      bdhiParams.box = gd->getEnsemble()->getBox();
      auto grid      = uammd::BDHI::detail::initializeGrid<Kernel>(bdhiParams);
      bdhiParams.box = grid.box;
      bdhiParams.cells = grid.cellDim;
      bdhiParams.kernel = uammd::BDHI::detail::initializeKernel<Kernel>(bdhiParams, grid);


      bdhiParams.hydrodynamicRadius = bdhiParams.kernel->fixHydrodynamicRadius(bdhiParams.hydrodynamicRadius,
                                                                               grid.cellSize.x);
      bdhiParams.kernelTorque = uammd::BDHI::detail::initializeKernelTorque<KernelTorque>(bdhiParams,
											  grid);
      if (bdhiParams.adaptBoxSize){
        gd->getEnsemble()->setBox(grid.box);
        
      }
      fcm = std::make_unique<FCM_super>(bdhiParams);
    }

    void updatePosition(){
      int numberParticles = pg->getNumberParticles();
      real dt = gd->getFundamental()->getTimeStep();
      auto indexIter = pg->getIndexIterator(access::location::gpu);
      auto pos = pd->getPos(access::location::gpu, access::mode::readwrite).raw();
      auto dir = pd->getDir(access::location::gpu, access::mode::readwrite).raw();
      auto force = pd->getForce(access::location::gpu, access::mode::readwrite).raw();
      auto torque = pd->getTorque(access::location::gpu, access::mode::readwrite).raw();
      auto disp = fcm->computeHydrodynamicDisplacements(pos, force, torque,
							numberParticles, kBT,
							rsqrt(dt), stream);
      auto linearVelocities = disp.first;
      auto angularVelocities = disp.second;
      real3* d_linearV = thrust::raw_pointer_cast(linearVelocities.data());
      real3* d_angularV = thrust::raw_pointer_cast(angularVelocities.data());
      int BLOCKSIZE = 128;
      int nthreads = BLOCKSIZE<numberParticles?BLOCKSIZE:numberParticles;
      int nblocks = numberParticles/nthreads + ((numberParticles%nthreads!=0)?1:0);
      uammd::BDHI::FCM_ns::integrateEulerMaruyamaD<<<nblocks, nthreads, 0, stream>>>(pos,
										     dir,
										     indexIter,
										     d_linearV,
										     d_angularV,
										     numberParticles,
										     dt);
      CudaCheckError();
    }

    void forwardTime() override {

      if (firstStep){
        loadInteractorsToIntegrator(magneticIntegrator);
        loadUpdatablesToIntegrator(magneticIntegrator);
        firstStep = false;
      }
      
      updateForceTorqueMagneticField();
      magneticIntegrator->processInteractions();
      updatePosition();
      magneticIntegrator->updateMagnetization();
      resetForceTorqueMagneticField();
      this->gd->getFundamental()->setCurrentStep(this->gd->getFundamental()->getCurrentStep()+1);
      this->gd->getFundamental()->setSimulationTime(this->gd->getFundamental()->getSimulationTime()+this->dt);
    }

  };

}}}}}

REGISTER_INTEGRATOR(
    MagneticMotion,ForceCouplingMethod,
    uammd::structured::Integrator::NVT::MagneticMotion::ForceCouplingMethod
)

