/* Raul P. Pelaez. 2022
   Definition of the complex type and related functions.

 */
#ifndef UAMMD_COMPLEX_CUH
#define UAMMD_COMPLEX_CUH
#include"global/defines.h"
#include<thrust/complex.h>
namespace uammd{
namespace structured{
  
  //Currently the complex type is just an alias to the one provided by thrust, which is compatible with
  //the std one but works in device functions
  using complex  = thrust::complex<real>;

  struct complex3{
    complex x,y,z;

    friend inline  __device__ __host__ complex3 operator+(const complex3 &a, const complex3 &b){
      return {a.x + b.x, a.y + b.y, a.z + b.z};
    }
    friend inline  __device__ __host__ void operator+=(complex3 &a, const complex3 &b){
      a.x += b.x; a.y += b.y; a.z += b.z;
    }
    
    friend inline  __device__ __host__ complex3 operator-(const complex3 &a, const complex3 &b){
      return {a.x - b.x, a.y - b.y, a.z - b.z};
    }
    friend inline  __device__ __host__ void operator-=(complex3 &a, const complex3 &b){
      a.x -= b.x; a.y -= b.y; a.z -= b.z;
    }

    
    friend inline  __device__ __host__ complex3 operator*(const complex3 &a, real b){
      complex3 res;
      res.x = a.x * b;
      res.y = a.y * b;
      res.z = a.z * b;
      return res;
    }
    friend inline  __device__ __host__ complex3 operator*(real b, const complex3 &a){
      return a*b;
    }
    friend inline  __device__ __host__ void operator*=(complex3 &a, real b){
      a = a*b;
    }


    friend inline  __device__ __host__ complex3 operator/(const complex3 &a, real b){
      complex3 res;
      res.x = a.x / b;
      res.y = a.y / b;
      res.z = a.z / b;
      return res;
    }
    friend inline  __device__ __host__ complex3 operator/(real b, const complex3 &a){
      complex3 res;
      res.x = b/a.x;
      res.y = b/a.y;
      res.z = b/a.z;
      return res;
    }
    friend inline  __device__ __host__ void operator/=(complex3 &a, real b){
      a = a/b;
    }
  };
}}

#endif
