#pragma once

#include "utils/quaternion.cuh"
#include "Definitions/Computations.cuh"

namespace uammd{
namespace structured{
namespace Potentials{
namespace BasicPotentials{


    struct Morse{

        static inline __device__ real3 force(const real3& rij, const real& r2,
                                             const real& e,const real& r0,const real& D){

            const real  r = sqrt(r2);
            const real dr = r-r0;
            const real factor = exp(-dr/D);

            const real fmod = real(2.0)*(e/D)*(real(1.0)-factor)*factor;

            return fmod*rij/r;
        }

        static inline __device__ real energy(const real3& rij, const real& r2,
                                             const real& e,const real& r0,const real& D){

            const real dr = sqrt(r2)-r0;
            const real factor = real(1.0)-exp(-dr/D);

            return e*(factor*factor-real(1.0));

        }
    };

}}}}
