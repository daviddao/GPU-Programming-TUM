/**
 * Emanuele Rodola
 * TU Munich
 * Jun 2015
 */
#include "mex.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

__global__
void testKernel(float* res) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    
    int index = x + y + z;
    
    if(threadIdx.x != 0)
        res[0] = threadIdx.x;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray*prhs[])
{

    float res[1];
    res[0] = 0;
    
    float* d_res;
    cudaMalloc(&d_res,1 * sizeof(float));
    
    cudaMemcpy(d_res, res, sizeof(float), cudaMemcpyHostToDevice);

    dim3 block = dim3(32, 8, 1);
    dim3 grid = dim3(1,2,1);

    testKernel <<<grid, block>>> (d_res);
    
    cudaMemcpy(res, d_res, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_res);
    
    mexPrintf("it worked: %f \n",res[0]);
    


}