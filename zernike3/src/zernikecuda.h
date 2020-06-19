#include<iostream>
#include<cufft.h>

#define NX 64
#define NY 128
#define NZ 128
#define BATCH 10
#define NRANK 3

using namespace std;

class zcuda{
  public:
    zcuda()
    {
      // cout<<"creating zcuda"<<endl;
      // cufftHandle plan;


      cout<<"running cuda fft"<<endl;
      // cufftComplex* data;

      // // cufftPlanMany

      cufftHandle plan;
      cufftComplex *data;
      int n[NRANK] = {NX, NY, NZ};

      cudaMalloc((void**)&data, sizeof(cufftComplex)*NX*NY*NZ*BATCH);
      if (cudaGetLastError() != cudaSuccess){
        fprintf(stderr, "Cuda error: Failed to allocate\n");
        return;	
      }

      /* Create a 3D FFT plan. */
      if (cufftPlanMany(&plan, NRANK, n, 
            NULL, 1, NX*NY*NZ, // *inembed, istride, idist 
            NULL, 1, NX*NY*NZ, // *onembed, ostride, odist
            CUFFT_C2C, BATCH) != CUFFT_SUCCESS){
        fprintf(stderr, "CUFFT error: Plan creation failed");
        return;	
      }	

      /* Use the CUFFT plan to transform the signal in place. */
      if (cufftExecC2C(plan, data, data, CUFFT_FORWARD) != CUFFT_SUCCESS){
        fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
        return;	
      }

      if (cudaDeviceSynchronize() != cudaSuccess){
        fprintf(stderr, "Cuda error: Failed to synchronize\n");
        return;	
      }	

      cufftDestroy(plan);
      cudaFree(data);
      cout<<"cuda free called"<<endl;

    }
};
