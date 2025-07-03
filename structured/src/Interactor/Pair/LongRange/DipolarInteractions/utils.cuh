#pragma once
#include "uammd.cuh"
#include "utils/container.h"

namespace uammd{
namespace structured{
namespace Interactor{
namespace LongRange{
namespace Dipolar{
namespace detail{

  template<class T>
  using cached_vector     = uammd::uninitialized_cached_vector<T>;

  struct ComputeEnergy {
    __device__
    void operator()(thrust::tuple<real3, real4, real&> t) const {
      const real3& B = thrust::get<0>(t);
      const real4& m = thrust::get<1>(t);
      real& e        = thrust::get<2>(t);
      e += dot(B, make_real3(m)) * m.w;
    }
  };
  
  struct SumReal3ToReal4{
    __device__
    real4 operator()(const real3& r3_in, real4 r4_out) const {
      return make_real4(r3_in, real(0.0)) + r4_out;
    }
  };
  
  struct ForceFromFieldGradient{
    real3* gradBx;
    real3* gradBy;
    real3* gradBz;
    real4* m;
    
    __device__
    real3 operator()(int i) const {
      real3 gradBxi = gradBx[i];
      real3 gradByi = gradBy[i];
      real3 gradBzi = gradBz[i];
      real4 mi   = m[i];
      return mi.w * (mi.x * gradBxi + mi.y * gradByi + mi.z * gradBzi);
    }
  };

  template<class T>
  void fillWithZero(cached_vector<T> &vec){
    thrust::fill(vec.begin(), vec.end(), T());
  }

    

  __device__ int3 indexToWaveNumber(int i, int3 nk){
    int ikx = i%(nk.x/2+1);
    int iky = (i/(nk.x/2+1))%nk.y;
    int ikz = i/((nk.x/2+1)*nk.y);
    ikx -= nk.x*(ikx >= (nk.x/2+1));
    iky -= nk.y*(iky >= (nk.y/2+1));
    ikz -= nk.z*(ikz >= (nk.z/2+1));
    return make_int3(ikx, iky, ikz);
  }
  
  __device__ real3 waveNumberToWaveVector(int3 ik, real3 L){
    return (real(2.0)*real(M_PI)/L)*make_real3(ik.x, ik.y, ik.z);
    }

    /* Given the id of a cell in 3D computes the id in 1D*/
    __host__ __device__ int getCell(int cellx, int celly, int cellz,
                                    int3 n){
      return ((cellx+n.x)%n.x)+n.x*((celly+n.y)%n.y)+n.x*n.y*((cellz+n.z)%n.z);
    }
    
    __host__ __device__ int getCell(int3 cell, int3 n){
      return getCell(cell.x, cell.y, cell.z, n);
    }
  
}}}}}}
