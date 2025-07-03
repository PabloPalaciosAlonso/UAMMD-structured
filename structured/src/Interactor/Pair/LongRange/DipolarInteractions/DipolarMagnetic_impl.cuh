#pragma once

#include "uammd.cuh"
#include "misc/TabulatedFunction.cuh"
#include "misc/IBM.cuh"

#include "System/ExtendedSystem.cuh"
#include "GlobalData/GlobalData.cuh"

#include "Interactor/Interactor.cuh"
#include "Interactor/InteractorFactory.cuh"

#include "Definitions/SFINAE.cuh"
#include "Utils/Containers/SetUtils.cuh"

#include "Definitions/Computations.cuh"
#include "Definitions/Types.cuh"

#include "utils.cuh"
#include "FFTManager.cuh"
#include "Kernels.cuh"

namespace uammd{
namespace structured{
namespace Interactor{
namespace LongRange{
namespace Dipolar{
namespace detail{

  /*Apply the projection operator to a wave number with a certain complex factor.
    res = (I-\hat{k}^\hat{k})·factor*/
  __device__ complex3 projectFourier(real3 k, complex3 factor){
      
    complex3 res;
    const real invk2 = real(1.0)/dot(k,k);
    
    const real3 fr = make_real3(factor.x.real(), factor.y.real(), factor.z.real());
    const real kfr = dot(k,fr)*invk2;
    const real3 vr = (fr-k*kfr);
    
    const real3 fi = make_real3(factor.x.imag(), factor.y.imag(), factor.z.imag());
    const real kfi = dot(k,fi)*invk2;
    const real3 vi = (fi-k*kfi);
    
    res.x = complex{vr.x, vi.x};
    res.y = complex{vr.y, vi.y};
    res.z = complex{vr.z, vi.z};
    return res;
  }
  
  __global__ void magnetizationFourierToField(const complex3* gridMagnetization,
                                             complex3* gridField,
                                             real permeability,
                                             Grid grid){
    
      const int id      = blockIdx.x*blockDim.x + threadIdx.x;      
      const int3 nCells = grid.cellDim;
      if(id>=(nCells.z*nCells.y*(nCells.x/2+1))) return;
      const real B = permeability/real(nCells.x*nCells.y*nCells.z);
      if(id == 0){
        gridField[0] = real(2.0)/real(3.0)*B*gridMagnetization[0];	
        return;
      }
      
      const int3 waveNumber = indexToWaveNumber(id, nCells);
      const real3 k         = waveNumberToWaveVector(waveNumber, grid.box.boxSize);
      complex3 factor       = gridMagnetization[id];
      gridField[id]         = projectFourier(k, factor)*B;
    }

  
  template<class Kernel>
  void particleMagnetizationToGridMagnetization(real4* d_pos,
                                                real4* d_particleMagnetization,   // (x,y,z)=direction, w=magnitude
                                                real3* d_gridMagnetization,
                                                std::shared_ptr<Kernel> &kernel,
                                                Grid grid,
                                                int numberParticles,
                                                cudaStream_t st){
    
    cached_vector<real3> particleMagneticMoment(numberParticles);
    auto d_particleMagneticMoment = particleMagneticMoment.data();
    
    thrust::transform(thrust::cuda::par.on(st),        
                      d_particleMagnetization,
                      d_particleMagnetization + numberParticles,
                      d_particleMagneticMoment,
                      [] __device__ (const real4& v) {
                        return make_real3(v) * v.w;});

    IBM<Kernel> ibm(kernel, grid);
    ibm.spread(d_pos, d_particleMagneticMoment,   // real3*
               d_gridMagnetization, numberParticles,
               st);
}

  
  void particleMagnetizationToGridMagnetization(real4* d_pos,
                                                real4* d_particleMagnetization,   // (x,y,z)=direction, w=magnitude
                                                real3* d_gridMagnetization,
                                                KernelVariant kernelGroup,
                                                Grid grid,
                                                int numberParticles,
                                                cudaStream_t &st){
    
    std::visit([&](auto& ker){
      using K = std::decay_t<decltype(ker.kernel)>;
      auto kernPtr = std::make_shared<K>(ker.kernel);
      particleMagnetizationToGridMagnetization(d_pos, d_particleMagnetization,
                                               d_gridMagnetization,
                                               kernPtr, grid,
                                               numberParticles, st);
      
    }, kernelGroup);    
  }


  template<class Kernel>
  void gridFieldToParticleField(real4* d_pos, real3* d_particleField, real3* d_gridField,
                                std::shared_ptr<Kernel> &kernel, Grid grid,
                                int numberParticles, cudaStream_t &st){
    
    IBM<Kernel> ibm(kernel, grid);
    ibm.gather(d_pos, d_particleField, d_gridField, numberParticles, st);
  }
  
  void gridFieldToParticleField(real4* d_pos, real3* d_particleField, real3* d_gridField, 
                                KernelVariant kernelGroup, Grid grid,
                                int numberParticles, cudaStream_t &st){
    
    std::visit([&](auto& ker){
      using K = std::decay_t<decltype(ker.kernel)>;
      auto kernPtr = std::make_shared<K>(ker.kernel);
      gridFieldToParticleField(d_pos, d_particleField,
                               d_gridField,
                               kernPtr, grid,
                               numberParticles, st);
      
    }, kernelGroup);    
  }
  
  void gridMagnetizationFourierToGridFieldFourier(complex3* d_gridMagnetizationFourier,
                                                  complex3* d_gridFieldFourier,
                                                  Grid grid, real permeability,
                                                  cudaStream_t &st){
    const int3 nCells       = grid.cellDim;
    int Nthreads            = 128;
    int Nblocks             = (nCells.z*nCells.y*(nCells.x/2+1))/Nthreads +1;
    magnetizationFourierToField<<<Nblocks, Nthreads, 0, st>>> (d_gridMagnetizationFourier,
                                                               d_gridFieldFourier,
                                                               permeability,
                                                               grid);
  }

  template<class Kernel>
  void gridFieldToParticleGradientField(real4* d_pos,
                                        real3* d_gridField,
                                        real3* d_particleDerivField,
                                        std::shared_ptr<Kernel> &kernelGradient,
                                        Grid grid, int numberParticles,
                                        cudaStream_t &st){
    
    
    IBM<Kernel> ibm(kernelGradient, grid);
    ibm.gather(d_pos, d_particleDerivField, d_gridField, numberParticles, st);
  }
  
  void gridFieldToParticleForce(real4* d_pos,
                                real4* d_particleMagnetization,
                                real3* d_gridField,
                                real3* d_particleForces,
                                KernelVariant kernelGroup,
                                Grid grid, int numberParticles,
                                cudaStream_t &st){
    
    cached_vector<real3> gradBx(numberParticles);
    cached_vector<real3> gradBy(numberParticles);
    cached_vector<real3> gradBz(numberParticles);
    detail::fillWithZero(gradBx);
    detail::fillWithZero(gradBy);
    detail::fillWithZero(gradBz);
      
    auto d_gradBx = thrust::raw_pointer_cast(gradBx.data());
    auto d_gradBy = thrust::raw_pointer_cast(gradBy.data());
    auto d_gradBz = thrust::raw_pointer_cast(gradBz.data());
    
    std::visit([&](auto& ker){
      using Kx = std::decay_t<decltype(ker.grad_x)>;
      using Ky = std::decay_t<decltype(ker.grad_y)>;
      using Kz = std::decay_t<decltype(ker.grad_z)>;
      
      auto kernPtr_x = std::make_shared<Kx>(ker.grad_x);
      auto kernPtr_y = std::make_shared<Ky>(ker.grad_y);
      auto kernPtr_z = std::make_shared<Kz>(ker.grad_z);
      
      gridFieldToParticleGradientField(d_pos, d_gridField, d_gradBx,
                                       kernPtr_x, grid,
                                       numberParticles, st);
      
      gridFieldToParticleGradientField(d_pos, d_gridField, d_gradBy,
                                       kernPtr_z, grid,
                                       numberParticles, st);
      
      gridFieldToParticleGradientField(d_pos, d_gridField, d_gradBz,
                                       kernPtr_z, grid,
                                       numberParticles, st);      
      
    }, kernelGroup);
    
    thrust::transform(thrust::cuda::par.on(st),
                      thrust::make_counting_iterator<int>(0),
                      thrust::make_counting_iterator<int>(numberParticles),
                      d_particleForces,
                      ForceFromFieldGradient{d_gradBx, d_gradBy, d_gradBz, d_particleMagnetization});   
  }

  
  __global__ void removeSpuriousForceD(real4* pos, real4* magnetization, real3* force,
                                       TabulatedFunction<real> spurious_self_force,
                                       real cellSize, int numberParticles){
    
    const int id    = blockIdx.x*blockDim.x + threadIdx.x;      
    if (id>=numberParticles) return;
    real3 posi      = make_real3(pos[id]);
    real4 m_M       = magnetization[id];
    real3 mdir      = make_real3(m_M);
    
    real xh         = posi.x / cellSize - floor(posi.x / cellSize);     
    real yh         = posi.y / cellSize - floor(posi.y / cellSize);     
    real zh         = posi.z / cellSize - floor(posi.z / cellSize);     
    
    real3 spuriousF;
    spuriousF.x = spurious_self_force(xh)*m_M.w*mdir.x;
    spuriousF.y = spurious_self_force(yh)*m_M.w*mdir.y;
    spuriousF.z = spurious_self_force(zh)*m_M.w*mdir.z;
    force[id]-= spuriousF;
  }    
  
  //Computes the force acting on each particle and removes the spurious self force that appears
  //when particles are not in the center of each a cell.
  void gridFieldToCorrectedParticleForce(real4* d_pos,
                                         real4* d_particleMagnetization,
                                         real3* d_gridField,
                                         real3* d_particleForce,
                                         KernelVariant kernelGroup,
                                         TabulatedFunction<real> spuriousSelfForce,
                                         Grid grid, int numberParticles,
                                         cudaStream_t st){
    
    gridFieldToParticleForce(d_pos, d_particleMagnetization,
                             d_gridField,
                             d_particleForce,
                             kernelGroup, grid,
                             numberParticles, st);
    
    real cellSize       = grid.cellSize.z;
    int Nthreads        = 128;
    int Nblocks          = (numberParticles + Nthreads - 1) / Nthreads;
    removeSpuriousForceD<<<Nblocks, Nthreads, 0, st>>>(d_pos, d_particleMagnetization,
                                                       d_particleForce, spuriousSelfForce,
                                                       cellSize, numberParticles);
  }
  
  real computeParticleSpuriousSelfForce(KernelVariant kernelGroup,
                                        std::shared_ptr<FFTManager>& FFT,
                                        Grid grid, real permeability,
                                        real posInCellDiv_h, cudaStream_t st){
    int numberParticles = 1;
    real lbox  = grid.box.boxSize.x;
    real h     = grid.cellSize.x;
    int nCells = grid.cellDim.x * grid.cellDim.y * grid.cellDim.z;
    cached_vector<real4> pos(numberParticles);
    cached_vector<real4> m_M(numberParticles);
    pos[0] = make_real4(0.0, 0.0, -lbox/2 + posInCellDiv_h*h, 0.0);
    m_M[0] = make_real4(0.0, 0.0, 1.0, 1.0);
    
    cached_vector<real3> gridMagnetization(nCells);
    detail::fillWithZero(gridMagnetization);
    auto d_pos = thrust::raw_pointer_cast(pos.data());
       
    {
      auto d_gridMagnetization = thrust::raw_pointer_cast(gridMagnetization.data());
      auto d_m_M = thrust::raw_pointer_cast(m_M.data());
      particleMagnetizationToGridMagnetization(d_pos, d_m_M,
                                               d_gridMagnetization,
                                               kernelGroup, grid,
                                               numberParticles, st);
    }
    
    cached_vector<complex3> gridMagnetizationFourier = FFT->forwardTransform(gridMagnetization);
    cached_vector<complex3> gridFieldFourier(gridMagnetizationFourier.size());

    auto d_gridMagnetFourier = thrust::raw_pointer_cast(gridMagnetizationFourier.data());

    {
    auto d_gridFieldFourier  = thrust::raw_pointer_cast(gridFieldFourier.data());

    gridMagnetizationFourierToGridFieldFourier(d_gridMagnetFourier,
                                               d_gridFieldFourier,
                                               grid, permeability, st);
    }
    cached_vector<real3> gridField = FFT->inverseTransform(gridFieldFourier);
    auto d_gridField = thrust::raw_pointer_cast(gridField.data());

    cached_vector<real3> particleForces(numberParticles);
    particleForces[0] = real3();
    auto d_particleForces = thrust::raw_pointer_cast(particleForces.data());
    auto d_m_M = thrust::raw_pointer_cast(m_M.data());
    gridFieldToParticleForce(d_pos, d_m_M,
                             d_gridField,
                             d_particleForces,
                             kernelGroup, grid,
                             numberParticles, st);
    return real3(particleForces[0]).z;    
  }

}}}}}}
    
