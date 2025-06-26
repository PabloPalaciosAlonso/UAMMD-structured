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

  //We should write a SimulationStepBase_MagneticField in UAMMD-structured
  //and derive this class from that
  class MeasureMagneticField : public SimulationStepBase{
    
  private:
    
    std::string   outputFilePath;
    std::ofstream outputFile;
    int startStep;
    uninitialized_cached_vector<real4> fieldTmp;
    
    void copyToTmp(cudaStream_t st){
      int N = this->pd->getNumParticles();

        fieldTmp.resize(N);
        auto field = this->pd->getMagneticField(access::location::gpu, access::mode::read);
        thrust::copy(thrust::cuda::par.on(st),
                     field.begin(),
                     field.end(),
                     fieldTmp.begin());
    }
    
    void copyFromTmp(cudaStream_t st){
      
      int N = this->pd->getNumParticles();
      auto field = this->pd->getMagneticField(access::location::gpu, access::mode::write);
      thrust::copy(thrust::cuda::par.on(st),
                   fieldTmp.begin(),
                   fieldTmp.end(),
                   field.begin());
    }
    
    void setZero(cudaStream_t st){
      auto field = this->pd->getMagneticField(access::location::gpu, access::mode::write);
      thrust::fill(thrust::cuda::par.on(st),
                   field.begin(),
                   field.end(),
                   real4());
    }

    void computeField(cudaStream_t st){
      for(auto& interactor : this->topology->getInteractors()){
	    //Create computable
	    uammd::Interactor::Computables compTmp;
	    compTmp.magneticField = true;
	    interactor.second->sum(compTmp,st);
	  }
    }

    void writeField(){        
      outputFile<<"#\n";
      auto field          = this->pd->getMagneticField(access::location::cpu, access::mode::read);
      auto id             = this->pd->getId(access::location::cpu, access::mode::read);
      int numberParticles = this->pg->getNumberParticles();
      fori(0, numberParticles){
        outputFile<<id[i]<<" "<<make_real3(field[i])<<"\n";
      }
    }  
    
  public:
    
    MeasureMagneticField(std::shared_ptr<ParticleGroup>  pg,
                         std::shared_ptr<IntegratorManager> integrator,
                         std::shared_ptr<ForceField>    ff,
                         DataEntry& data,
                         std::string name):SimulationStepBase(pg,integrator,ff,data,name){

      outputFilePath = data.getParameter<std::string>("outputFilePath");
      startStep      = data.getParameter<int>("startStep", 0);
    }

    void init(cudaStream_t st) override{

      bool isFileEmpty = Backup::openFile(this->sys, outputFilePath, outputFile);

      //If the file did not exist, we can write the header here.
      if(isFileEmpty){
        outputFile << "#id Bx By Bz" << std::endl;
      }
    }

    void applyStep(ullint step, cudaStream_t st) override{
      if (step>=startStep){
        copyToTmp(st);
        setZero(st);
        computeField(st);
        writeField();
        copyFromTmp(st);
      }
    }
  };
  
}}}}

REGISTER_SIMULATION_STEP(
                         MagneticMeasure,MeasureMagneticField,
                         uammd::structured::SimulationStep::SimulationMeasures::MeasureMagneticField
                         )
