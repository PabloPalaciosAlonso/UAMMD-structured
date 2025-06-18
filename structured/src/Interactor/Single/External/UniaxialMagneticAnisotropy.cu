#include "uammd.cuh"
#include "utils/quaternion.cuh"

#include "System/ExtendedSystem.cuh"
#include "GlobalData/GlobalData.cuh"
#include "ParticleData/ExtendedParticleData.cuh"
#include "ParticleData/ParticleGroup.cuh"

#include "Interactor/Single/SingleInteractor.cuh"
#include "Interactor/Single/External/External.cuh"
#include "Interactor/InteractorFactory.cuh"

namespace uammd{
namespace structured{
namespace Potentials{
namespace External{

  struct UniaxialMagneticAnisotropy_{

    //Computational data
    struct ComputationalData{
      real4* dir;
      real* anisotropy;
      real4* magnetization;
      real* volume;      
    };
    
    struct StorageData{
      thrust::device_vector<real> anisotropy;
      thrust::device_vector<real> volume;
      
    };
    
    static __host__ ComputationalData getComputationalData(std::shared_ptr<GlobalData>    gd,
                                                           std::shared_ptr<ParticleGroup> pg,
                                                           const StorageData&  storage,
                                                           const Computables& comp,
                                                           const cudaStream_t& st){
      
      ComputationalData computational;
      std::shared_ptr<ParticleData> pd = pg->getParticleData();

      computational.magnetization = pd->getMagnetization(access::location::gpu,access::mode::read).raw();
      computational.dir           = pd->getDir(access::location::gpu,access::mode::read).raw();
      computational.anisotropy    = (real*)thrust::raw_pointer_cast(storage.anisotropy.data());
      computational.volume        = (real*)thrust::raw_pointer_cast(storage.volume.data());
      return computational;
    }

    static __host__ StorageData getStorageData(std::shared_ptr<GlobalData>    gd,
						 std::shared_ptr<ParticleGroup> pg,
						 DataEntry& data){
      
      StorageData storage;
      storage.anisotropy               = data.getParameter<std::vector<real>>("anisotropy");
      std::shared_ptr<ParticleData> pd = pg->getParticleData();
      const int N                      = pg->getNumberParticles();
      auto radius                      = pd->getRadius(access::location::gpu,access::mode::read);

      storage.volume.resize(N);
      thrust::transform(thrust::device, radius.begin(),
                        radius.end(), storage.volume.begin(),                        
                        [] __device__ (real r) {return (4.0 / 3.0) * M_PI * r * r * r;});

      return storage;
    }
    
    static inline __device__ real energy(int index_i, const ComputationalData& computational){
      const real  volume     = computational.volume[index_i];
      const real  anisotropy = computational.anisotropy[index_i];
      const Quat diri        = computational.dir[index_i];
      const real4 m_and_M    = computational.magnetization[index_i];

      real3 um     = make_real3(m_and_M);
      real3 uk     = diri.getVz();            
      real ctheta  = dot(um,uk);
      real ctheta2 = min(ctheta*ctheta, real(1.0));
      real e       = anisotropy * volume*(real(1.0)-ctheta2);
      return e;
    }

    static inline __device__ real4 magneticField(const int index_i,const ComputationalData& computational){
      const real  anisotropy = computational.anisotropy[index_i];
      const Quat diri        = computational.dir[index_i];
      const real4 m_and_M    = computational.magnetization[index_i];
      const real  volume     = computational.volume[index_i];
      
      real  m0    = m_and_M.w;
      real3 um    = make_real3(m_and_M);
      real3 uk    = diri.getVz();

      real3 field = real(2.0)*anisotropy*volume/m0*dot(um,uk)*uk;
      return make_real4(field, 0);
    }

    static inline __device__ ForceTorque forceTorque(const int index_i,const ComputationalData& computational){
      ForceTorque forceTorque;
      forceTorque.force  = make_real4(0.0);

      const real  volume     = computational.volume[index_i];
      const real  anisotropy = computational.anisotropy[index_i];
      const Quat diri        = computational.dir[index_i];
      const real4 m_and_M    = computational.magnetization[index_i];
      
      real3 um     = make_real3(m_and_M);
      real3 uk     = diri.getVz();
      real3 torque = -real(2.0) * anisotropy * volume * dot(um, uk)*cross(um, uk);
      forceTorque.torque = make_real4(torque, real(0.0));
      
      return forceTorque;
	  }

  };

  using UniaxialMagneticAnisotropy = ExternalForceTorqueMagneticField_<UniaxialMagneticAnisotropy_>;

}}}}

REGISTER_SINGLE_INTERACTOR(
    External,UniaxialMagneticAnisotropy,
    uammd::structured::Interactor::SingleInteractor<uammd::structured::Potentials::External::UniaxialMagneticAnisotropy>
)
