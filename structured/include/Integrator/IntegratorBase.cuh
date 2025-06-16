#pragma once

#include "uammd.cuh"

#include "System/ExtendedSystem.cuh"
#include "GlobalData/GlobalData.cuh"
#include "ParticleData/ParticleGroup.cuh"

#include "Integrator/IntegratorUtils.cuh"
#include "Integrator/IntegratorFactory.cuh"

#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>

namespace uammd{
namespace structured{
namespace Integrator{

    class IntegratorBase: public Integrator{

        protected:

            cudaStream_t stream;

            std::shared_ptr<GlobalData> gd;

            real dt;

        public:

            IntegratorBase(std::shared_ptr<GlobalData>    gd,
                           std::shared_ptr<ParticleGroup> pg,
                           DataEntry& data,
                           std::string name);

            void resetEnergy();
            void resetForce();
            void resetTorque();

            void updateEnergy();
            void updateForce();

            virtual void forwardTime() override = 0;

    };

  class IntegratorBaseNVT: public IntegratorBase {

        protected:

            real temperature;
            real kBT;

        public:

            IntegratorBaseNVT(std::shared_ptr<GlobalData>           gd,
                              std::shared_ptr<ParticleGroup>        pg,
                              DataEntry& data,
                              std::string name);

            virtual void forwardTime() override = 0;

  };

  class IntegratorBaseMagnetic: public IntegratorBaseNVT {

  public:
    
    IntegratorBaseMagnetic(std::shared_ptr<GlobalData>           gd,
                           std::shared_ptr<ParticleGroup>        pg,
                           DataEntry& data,
                           std::string name);
    
    void resetMagneticField();
    void updateMagneticField();
    
    virtual void forwardTime() override;
    virtual void updateMagnetization() = 0;
    
  };


  class IntegratorBaseMagneticMotion: public IntegratorBaseNVT {
    
  protected:
    std::shared_ptr<IntegratorBaseMagnetic> magneticIntegrator;
    
  public:
    
    IntegratorBaseMagneticMotion(std::shared_ptr<GlobalData>           gd,
                                 std::shared_ptr<ParticleGroup>        pg,
                                 DataEntry& data,
                                 std::string name);
    
    bool isMagneticIntegratorAvailable(std::string magneticIntegratorSubType);
    void loadInteractorsToIntegrator(std::shared_ptr<Integrator> integrator);
    void loadUpdatablesToIntegrator(std::shared_ptr<Integrator> integrator);
    void updateForceTorqueMagneticField();
    void resetForceTorqueMagneticField();
    
    virtual void forwardTime() = 0;
  };

}}}

