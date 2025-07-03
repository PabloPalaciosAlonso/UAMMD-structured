#include "System/ExtendedSystem.cuh"
#include "GlobalData/GlobalData.cuh"
#include "ParticleData/ExtendedParticleData.cuh"
#include "ParticleData/ParticleGroup.cuh"

#include "SimulationStep/SimulationStep.cuh"
#include "SimulationStep/SimulationStepFactory.cuh"

#include "Utils/Measures/MeasuresBasic.cuh"

namespace uammd{
namespace structured{
namespace SimulationStep{
namespace SimulationMeasures{

  //We should write a SimulationStepBase_Force in UAMMD-structured
  //and derive this class from that
  class ForceOverParticles : public SimulationStepBase_EnergyForceTorque{
    
  private:
    
    std::string   outputFilePath;
    std::ofstream outputFile;
    int startStep;
    
    
    void computeForce(cudaStream_t st){
      for(auto& interactor : this->topology->getInteractors()){
	    //Create computable
	    uammd::Interactor::Computables compTmp;
	    compTmp.force = true;
	    interactor.second->sum(compTmp,st);
	  }
    }

    void writeForce(){        
      outputFile<<"#\n";
      auto force          = this->pd->getForce(access::location::cpu, access::mode::read);
      auto id             = this->pd->getId(access::location::cpu, access::mode::read);
      int numberParticles = this->pg->getNumberParticles();
      fori(0, numberParticles){
        outputFile<<id[i]<<" "<<make_real3(force[i])<<"\n";
      }
    }  
    
  public:
    
    ForceOverParticles(std::shared_ptr<ParticleGroup>  pg,
                       std::shared_ptr<IntegratorManager> integrator,
                       std::shared_ptr<ForceField>    ff,
                       DataEntry& data,
                       std::string name):SimulationStepBase_EnergyForceTorque(pg,integrator,ff,data,name){
      
      outputFilePath = data.getParameter<std::string>("outputFilePath");
      startStep      = data.getParameter<int>("startStep", 0);
    }
    
    void init(cudaStream_t st) override{

      bool isFileEmpty = Backup::openFile(this->sys, outputFilePath, outputFile);

      //If the file did not exist, we can write the header here.
      if(isFileEmpty){
        outputFile << "#id Fx Fy Fz" << std::endl;
      }
    }

    void applyStep(ullint step, cudaStream_t st) override{
      if (step>=startStep){
        copyToTmp(st);
        setZero(st);
        computeForce(st);
        writeForce();
        copyFromTmp(st);
      }
    }
  };
  
}}}}

REGISTER_SIMULATION_STEP(
                         MechanicalMeasure,ForceOverParticles,
                         uammd::structured::SimulationStep::SimulationMeasures::ForceOverParticles
                         )
