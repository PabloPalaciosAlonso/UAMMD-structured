#pragma once

#include "uammd.cuh"
#include "utils/cufftPrecisionAgnostic.h"
#include "utils/cufftComplex3.cuh"
#include "utils/cufftDebug.h"
#include "utils/Grid.cuh"
#include "utils/container.h"
#include "complex.cuh"

#include "System/ExtendedSystem.cuh"


namespace uammd {
namespace structured{
namespace Interactor{
namespace LongRange{
namespace Dipolar{
namespace detail{
    
  class FFTManager {
    template<class T>
    using cached_vector = uammd::uninitialized_cached_vector<T>;
    using cufftReal     = cufftReal_t<real>;
    using cufftComplex  = cufftComplex_t<real>;
    using cufftComplex3 = cufftComplex3_t<real>;
    
  public:
    FFTManager(int3 nCells):
      nCells(nCells){}

    FFTManager(int3 nCells, cudaStream_t stream):
      nCells(nCells),
      stream(stream){}
    
    ~FFTManager() {
      if (planForwardScalar) cufftDestroy(planForwardScalar);
      if (planForwardVector) cufftDestroy(planForwardVector);
      if (planInverseScalar) cufftDestroy(planInverseScalar);
      if (planInverseVector) cufftDestroy(planInverseVector);
    }

    void setStream(cudaStream_t st){
      stream = st;
    }

    bool isStreamSet() {
      return stream != 0;
    }
    
    cached_vector<complex> forwardTransform(cached_vector<real>& gridReal) {
      if (!planForwardScalar) {
        createForwardTransformPlanScalar();
        cufftSetStream(planForwardScalar, stream);
      }

      cached_vector<complex> gridFourier((nCells.x / 2 + 1) * nCells.y * nCells.z);
      thrust::fill(thrust::cuda::par.on(stream), gridFourier.begin(), gridFourier.end(), complex());

      auto d_gridFourier = (cufftComplex*)thrust::raw_pointer_cast(gridFourier.data());
      auto d_gridReal    = (cufftReal*)thrust::raw_pointer_cast(gridReal.data());

      CufftSafeCall(cufftExecReal2Complex<real>(planForwardScalar, d_gridReal, d_gridFourier));
      return gridFourier;
    }

    cached_vector<complex3> forwardTransform(cached_vector<real3>& gridReal) {
      if (!planForwardVector) {
        createForwardTransformPlanVector();
        cufftSetStream(planForwardVector, stream);
      }

      cached_vector<complex3> gridFourier((nCells.x / 2 + 1) * nCells.y * nCells.z);
      thrust::fill(thrust::cuda::par.on(stream), gridFourier.begin(), gridFourier.end(), complex3());

      auto d_gridFourier = (cufftComplex*)thrust::raw_pointer_cast(gridFourier.data());
      auto d_gridReal    = (cufftReal*)thrust::raw_pointer_cast(gridReal.data());

      CufftSafeCall(cufftExecReal2Complex<real>(planForwardVector, d_gridReal, d_gridFourier));
      return gridFourier;
    }

    cached_vector<real> inverseTransform(cached_vector<complex>& gridFourier) {
      if (!planInverseScalar) {
        createInverseTransformPlanScalar();
        cufftSetStream(planInverseScalar, stream);
      }
      
      cached_vector<real> gridReal(nCells.x * nCells.y * nCells.z);
      thrust::fill(thrust::cuda::par.on(stream), gridReal.begin(), gridReal.end(), real());
      
      auto d_gridFourier = (cufftComplex*)thrust::raw_pointer_cast(gridFourier.data());
      auto d_gridReal    = (cufftReal*)thrust::raw_pointer_cast(gridReal.data());

      CufftSafeCall(cufftExecComplex2Real<real>(planInverseScalar, d_gridFourier, d_gridReal));
      return gridReal;
    }


    cached_vector<real3> inverseTransform(cached_vector<complex3>& gridFourier) {
      if (!planInverseVector) {
        createInverseTransformPlanVector();
        cufftSetStream(planInverseVector, stream);
      }

      cached_vector<real3> gridReal(nCells.x * nCells.y * nCells.z);
      thrust::fill(thrust::cuda::par.on(stream), gridReal.begin(), gridReal.end(), real3());

      auto d_gridFourier = (cufftComplex*)thrust::raw_pointer_cast(gridFourier.data());
      auto d_gridReal    = (cufftReal*)thrust::raw_pointer_cast(gridReal.data());
      
      CufftSafeCall(cufftExecComplex2Real<real>(planInverseVector, d_gridFourier, d_gridReal));
      return gridReal;
    }

  private:
    int3 nCells;
    cudaStream_t stream = 0;
    cufftHandle planForwardScalar = 0;
    cufftHandle planForwardVector = 0;
    cufftHandle planInverseScalar = 0;
    cufftHandle planInverseVector = 0;

    void createForwardTransformPlanScalar(){
      CufftSafeCall(cufftCreate(&planForwardScalar));
      //CufftSafeCall(cufftSetAutoAllocation(planForwardScalar, 0));

      size_t cufftWorkSize = 0;
      int3 cdtmp           = {nCells.z, nCells.y, nCells.x};
      
      CufftSafeCall(cufftMakePlan3d(planForwardScalar, cdtmp.x, cdtmp.y, cdtmp.z,
                                    CUFFT_Real2Complex<real>::value, &cufftWorkSize));
      // cached_vector<char> cufftWorkArea;
      // cufftWorkArea.resize(cufftWorkSize);
      // auto d_cufftWorkArea = thrust::raw_pointer_cast(cufftWorkArea.data());
      // CufftSafeCall(cufftSetWorkArea(planForwardScalar, (void*)d_cufftWorkArea));    
    }
    
    void createForwardTransformPlanVector(){
      int3 cdtmp   = {nCells.z, nCells.y, nCells.x};
      int3 inembed = {nCells.z, nCells.y, nCells.x};
      int3 oembed  = {nCells.z, nCells.y, nCells.x/2+1};
      
      // Create the forward FFT plan
      size_t cufftWorkSize = 0;
      
      CufftSafeCall(cufftCreate(&planForwardVector));
      //CufftSafeCall(cufftSetAutoAllocation(planForwardVector, 0));
      CufftSafeCall(cufftMakePlanMany(planForwardVector,
                                      3, &cdtmp.x, /*Three dimensional FFT*/
                                      &inembed.x,
                                      /*Each FFT starts in 1+previous FFT index. FFTx in 0*/
                                      3, 1, //Each element separated by three others x0 y0 z0 x1 y1 z1...
                                      /*Same format in the output*/
                                      &oembed.x,
                                      3, 1,
                                      /*Perform 3 direct Batched FFTs*/
                                      CUFFT_Real2Complex<real>::value, 3,
                                      &cufftWorkSize));
      
      // cached_vector<char> cufftWorkArea;
      // cufftWorkArea.resize(cufftWorkSize);
      // auto d_cufftWorkArea = thrust::raw_pointer_cast(cufftWorkArea.data());
      // CufftSafeCall(cufftSetWorkArea(planForwardVector, (void*)d_cufftWorkArea));    
    }

    void createInverseTransformPlanScalar(){
      CufftSafeCall(cufftCreate(&planInverseScalar));
      //CufftSafeCall(cufftSetAutoAllocation(planInverseScalar, 0));
      
      size_t cufftWorkSize = 0;
      int3 cdtmp           = {nCells.z, nCells.y, nCells.x};
      
      // Crear el plan para la transformada inversa
      CufftSafeCall(cufftMakePlan3d(planInverseScalar, cdtmp.x, cdtmp.y, cdtmp.z,
                                    CUFFT_Complex2Real<real>::value, &cufftWorkSize));
      
      // Configurar el área de trabajo
      // cached_vector<char> cufftWorkArea;
      // cufftWorkArea.resize(cufftWorkSize);
      // auto d_cufftWorkArea = thrust::raw_pointer_cast(cufftWorkArea.data());
      // CufftSafeCall(cufftSetWorkArea(planInverseScalar, (void*)d_cufftWorkArea));    
    }
    
    void createInverseTransformPlanVector(){
      // Dimensions for the inverse FFT
      int3 cdtmp   = {nCells.z, nCells.y, nCells.x};
      int3 oembed  = {nCells.z, nCells.y, nCells.x};  // Output embedding for real data
      int3 inembed = {nCells.z, nCells.y, nCells.x / 2 + 1};  // Input embedding for complex data
      
      // Create the inverse FFT plan
      size_t cufftWorkSize = 0;
      
      CufftSafeCall(cufftCreate(&planInverseVector));
      //CufftSafeCall(cufftSetAutoAllocation(planInverseVector, 0));  // Manual memory allocation for work area
      
      CufftSafeCall(cufftMakePlanMany(planInverseVector,
                                      3, &cdtmp.x, /*Three-dimensional FFT*/
                                      &inembed.x,
                                      /*Each complex FFT starts at the appropriate offset*/
                                      3, 1, // Stride for the complex input data (real and imaginary parts)
                                      &oembed.x,
                                      3, 1, // Stride for the real output data
                                      /*Perform 3 batched inverse FFTs*/
                                      CUFFT_Complex2Real<real>::value, 3, 
                                      &cufftWorkSize));
      
      // Allocate work area and set it for the inverse plan
      // cached_vector<char> cufftWorkArea;
      // cufftWorkArea.resize(cufftWorkSize);
      // auto d_cufftWorkArea = thrust::raw_pointer_cast(cufftWorkArea.data());
      // CufftSafeCall(cufftSetWorkArea(planInverseVector, (void*)d_cufftWorkArea));
    }
  };
  
}}}}}}
