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
using namespace std;

#define	max(A, B)	((A) > (B) ? (A) : (B))
#define	min(A, B)	((A) < (B) ? (A) : (B))

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
        exit(1);
    }
    prev_file = file;
    prev_line = line;
}


/* Input arguments */
#define IN_A		prhs[0]
#define IN_B		prhs[1]

/* Output arguments */
#define OUT_X		plhs[0]

/* Gateway routine */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
  double *matrix_A, *matrix_B, *matrix_X;
  int     N, D;
  
  /* Get the sizes of each input argument */
  N = mxGetM(IN_A);
  D = mxGetN(IN_B);
  
  /* Create the new arrays and set the output pointers to them */
  //OUT_X     = mxCreateDoubleMatrix(N, D, mxREAL);
    OUT_X     = mxCreateDoubleMatrix(N, N, mxREAL);


    /* Assign pointers to the input arguments */
  matrix_A      = mxGetPr(IN_A);
  matrix_B      = mxGetPr(IN_B);
  
  /* Assign pointers to the output arguments */
  matrix_X      = mxGetPr(OUT_X);
  
  /* Do the actual computations in a subroutine */
    //allocate memory on GPU
    double* d_matrix_A;
    double* d_matrix_X;
    double *d_Aarray = new double[1];
    double **dd_Aarray;
    dd_Aarray = (double**)malloc(sizeof(double*));
    int* d_infoArray;
    
    
    cublasStatus_t status;
    cublasHandle_t handle;
    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) 
    {
        printf("error1 %i\n",status);
        return;
    }

    cudaMalloc((void**)&dd_Aarray[0], N*N*sizeof(double)); CUDA_CHECK;
    cudaMemcpy(dd_Aarray[0], matrix_A, N * N * sizeof(double), cudaMemcpyHostToDevice); CUDA_CHECK;
    
    //cudaMalloc(&d_matrix_A, N * N * sizeof(double)); CUDA_CHECK;
   // d_Aarray = d_matrix_A;

    cudaMalloc(&d_matrix_X, N * N * sizeof(double)); CUDA_CHECK;
    cudaMalloc(&d_infoArray, N * N * sizeof(int)); CUDA_CHECK;   
   

    //cudaMemcpy(d_matrix_A, matrix_A, N * N * sizeof(double), cudaMemcpyHostToDevice); CUDA_CHECK;
    //cudaMemcpy(dd_Aarray, d_Aarray, sizeof(double*), cudaMemcpyDeviceToDevice); CUDA_CHECK;

   /* for(int i=0;i<N;i++)
    {
        for( int j=0;j<N;j++)
        {
            printf("%.2f ", matrix_A[i*N + j]);
        }
    printf("\n");
    }*/

    status = cublasDgetrfBatched(handle,
                               N, 
                               dd_Aarray,
                               N, 
                               NULL,
                               d_infoArray,
                               1);
    if (status != CUBLAS_STATUS_SUCCESS) 
    {
        printf("error2 %i\n",status);
        return;
    }

    cudaMemcpy(matrix_X, dd_Aarray[0], N * sizeof(double), cudaMemcpyDeviceToHost);  CUDA_CHECK;

    cublasDestroy(handle);
    cudaFree(d_matrix_A);
    cudaFree(d_matrix_X);
    delete[] d_Aarray;
    free(dd_Aarray);
    
  return;
}



