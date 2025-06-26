#pragma once
#include "uammd.cuh"
#include "utils/Grid.cuh"
#include "utils/container.h"
#include "misc/IBM.cuh"

namespace uammd{
namespace structured{

  template<class T>
  using cached_vector = uammd::uninitialized_cached_vector<T>;
  
  template<typename T, typename Kernel>
  cached_vector<T> spread(cached_vector<real4> &pos,
                          cached_vector<T>  &magnitude,
                          std::shared_ptr<Kernel>& kernel,
                          Grid grid, 
                          cudaStream_t st = 0){
    
    int3 nCells         = grid.cellDim;
    int numberParticles = magnitude.size();
    cached_vector<T> gridMagnitude(nCells.x*nCells.y*nCells.z);
    thrust::fill(thrust::cuda::par.on(st), gridMagnitude.begin(), gridMagnitude.end(), T());

    auto d_pos           = thrust::raw_pointer_cast(pos.data());
    auto d_magnitude     = thrust::raw_pointer_cast(magnitude.data());
    auto d_gridMagnitude = thrust::raw_pointer_cast(gridMagnitude.data());

    IBM<Kernel> ibm(kernel, grid);
    ibm.spread(d_pos, d_magnitude, d_gridMagnitude, numberParticles, st);
    CudaCheckError();
    return gridMagnitude;    
  }
  
  template<typename T, typename Kernel>
  cached_vector<T> interpolate(cached_vector<real4> &pos,
                               cached_vector<T>  &gridMagnitude,
                               std::shared_ptr<Kernel>& kernel,
                               Grid grid, 
                               cudaStream_t st = 0){
    
    int numberParticles = pos.size();
    
    cached_vector<T> magnitude(numberParticles);
    thrust::fill(thrust::cuda::par.on(st), magnitude.begin(), magnitude.end(), T());
    
    auto d_pos           = thrust::raw_pointer_cast(pos.data());
    auto d_magnitude     = thrust::raw_pointer_cast(magnitude.data());
    auto d_gridMagnitude = thrust::raw_pointer_cast(gridMagnitude.data());
    
    IBM<Kernel> ibm(kernel, grid);
    ibm.gather(d_pos, d_magnitude, d_gridMagnitude, numberParticles, st);
    CudaCheckError();
    return magnitude;    
  }
}}
