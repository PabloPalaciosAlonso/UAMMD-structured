#pragma once

#include "utils/container.h"
#include "utils/quaternion.cuh"

namespace uammd{
namespace structured{
namespace Integrator{
namespace NVT{
namespace Magnetic{

  namespace LLG{
    /* Computes dm/dt = -gyroRation/(1+damping^2)*(m x b+ damping * m x m x b)*/
    inline __device__ real3 computeMagnetizationDerivative(real3 field, real3 magnetization,
                                                           real damping, real gyroRatio){
      real3 mxbi    = cross(magnetization,field);
      real3 amxmxbi = damping*cross(magnetization,mxbi);
      real3 dmi     = -gyroRatio/(1+damping*damping)*(mxbi+amxmxbi);
      return dmi;
    }

    /* Computes the thermal field i.e. an hypothetial field to consider the thermal fluctuations
     * in the system. The noise follow a Gaussian distribution with mean 0 and standard deviation
     * dW = sqrt(2*k_BT*damping/gyroRation*msat*V*dt)
     */
    inline __device__ real3 computeThermalField(real fluctuationsAmplitude,
                                                int step, uint seed, int id){
      Saru rng(seed,id,step);
      real3 randomField = make_real3(rng.gf(0, fluctuationsAmplitude),
                                     rng.gf(0, fluctuationsAmplitude).x);
      return randomField;
    }

    // /* Computes the anisotropy field. i.e. an internal field that pushes the magnetization
    //  * towards the direction of the anisotropy axis. B_anis = 2*anisCons/msat*(m·u)u
    //  */
    // inline __device__ real3 computeAnisotropyField(real3 mi, real anisConst_div_msat){
    //   real dotm_u = mi.z; //In the particles frame the easy axis is in the z axis
    //   real fieldAnisParticle_z = real(2.0)*anisConst_div_msat*dotm_u;
    //   return {0, 0, fieldAnisParticle_z};
    // }

    // inline __device__ real3 computeAnisotropyField(Quat diri, real3 mi, real anisConst_div_msat){
    //   real3 vz = diri.getVz();
    //   real dotm_u = dot(mi, vz); //In the particles frame the easy axis is in the z axis
    //   real3 fieldAnisParticle = real(2.0)*anisConst_div_msat*dotm_u*vz;
    //   return fieldAnisParticle;
    // }
  }
}}}}}
