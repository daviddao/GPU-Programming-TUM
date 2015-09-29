/**
 * David Dao, Johannes Rausch, Michal Szymczak
 * TU Munich
 * Sep 2015
 */

#include <stdio.h>
#include "mex.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cula_device.h>
#include <cula_lapack_device.h>

using namespace std;

void checkStatus(culaStatus status)
{
char buf[80];

if(!status)
return;

culaGetErrorInfoString(status, culaGetErrorInfo(), buf, sizeof(buf));
printf("%s\n %d", buf, status);

culaShutdown();
//exit(EXIT_FAILURE);
}

// cuda error checking
string prev_file = "";
int prev_line = 0;
#define CUDA_CHECK cuda_check(__FILE__,__LINE__)
void cuda_check(string file, int line)
{
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
    {
        cout << endl << file << ", line " << line << ": " << cudaGetErrorString(e) << " (" << e << ")" << endl;
        if (prev_line>0) cout << "Previous CUDA call:" << endl << prev_file << ", line " << prev_line << endl;
        cudaDeviceReset();
        //exit(0);
    }
    prev_file = file;
    prev_line = line;
}

/* Input arguments */
#define IN_A		prhs[0]
#define IN_B		prhs[1]

/* Output arguments */
//#define OUT_X		plhs[0]

/* Gateway routine */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
  float *matrix_A, *matrix_B;
  float *d_matrix_A, *d_matrix_B;
  int     N, D;
  
  /* Get the sizes of each input argument */
  N = mxGetM(IN_A);
  D = mxGetN(IN_B);
  
    /* Assign pointers to the input arguments */
  matrix_A      = (float*) mxGetPr(IN_A);
  matrix_B      = (float*) mxGetPr(IN_B);
  
  /* Create the new arrays and set the output pointers to them */
//  OUT_X     = mxCreateDoubleMatrix(N, D, mxREAL);
  
  /* Assign pointers to the output arguments */
//  matrix_X      = mxGetPr(OUT_X);

//  int matrix_pivot[N];
  int *d_matrix_pivot;
  culaStatus status;
  status = culaInitialize();
  checkStatus(status);
  
  cudaMalloc(&d_matrix_A, N * N * sizeof(float)); CUDA_CHECK;
  cudaMalloc(&d_matrix_B, N * D * sizeof(float)); CUDA_CHECK;
  cudaMalloc(&d_matrix_pivot, N * sizeof(int)); CUDA_CHECK;
  cudaMemcpy(d_matrix_A, matrix_A, N * N * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
  cudaMemcpy(d_matrix_B, matrix_B, N * D * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
 
  status = culaDeviceSgesv(N, D, d_matrix_A, N, d_matrix_pivot, d_matrix_B, N);
  checkStatus(status);
  culaShutdown();
  
  cudaMemcpy(matrix_B, d_matrix_B, N * D * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
  
  cudaFree(d_matrix_A);
  cudaFree(d_matrix_B);
  cudaFree(d_matrix_pivot);
  
  return;
}



