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
#include "cublas_v2.h"
#include "cusolverDn.h"
#include "time.h"
using namespace std;

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

/*
 *  solve A*x = b by LU with partial pivoting
 *
 */
/* Input arguments */
#define IN_A		prhs[0]
#define IN_B		prhs[1]

/* Output arguments */
//#define OUT_X		plhs[0]

/* Gateway routine */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
	double *Acopy, *Bcopy;
	double *A, *B;
    cusolverDnHandle_t handle;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    int bufferSize = 0;
    int *info = NULL;
    double *buffer = NULL;
    int *ipiv = NULL; // pivoting sequence
    int h_info = 0;
    int n = 0;
    int d = 0;
    int lda = 0;
    
	/* Get the sizes of each input argument */
	n = mxGetM(IN_A);
	d = mxGetN(IN_B);
	lda = n;
	
	  /* Assign pointers to the input arguments */
	Acopy      = mxGetPr(IN_A);
	Bcopy      = mxGetPr(IN_B);

    cusolver_status = cusolverDnCreate(&handle);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    
    cusolverDnDgetrf_bufferSize(handle, n, n, (double*)Acopy, lda, &bufferSize); CUDA_CHECK;

    cudaMalloc(&info, sizeof(int)); CUDA_CHECK;
    cudaMalloc(&buffer, sizeof(double)*bufferSize); CUDA_CHECK;
    cudaMalloc(&A, sizeof(double)*lda*n); CUDA_CHECK;
    cudaMalloc(&ipiv, sizeof(int)*n); CUDA_CHECK;
    cudaMalloc(&B, sizeof(double)*n*d); CUDA_CHECK;
    

    // prepare a copy of A because getrf will overwrite A with L
    cudaMemcpy(A, Acopy, sizeof(double)*lda*n, cudaMemcpyHostToDevice); CUDA_CHECK;
    cudaMemset(info, 0, sizeof(int)); CUDA_CHECK;

    cusolverDnDgetrf(handle, n, n, A, lda, buffer, ipiv, info); CUDA_CHECK;
    cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost); CUDA_CHECK;

    if ( 0 != h_info ){
        fprintf(stderr, "Error: LU factorization failed\n");
    }

    cudaMemcpy(B, Bcopy, sizeof(double)*n*d, cudaMemcpyHostToDevice); CUDA_CHECK;
    cusolverDnDgetrs(handle, CUBLAS_OP_N, n, d, A, lda, ipiv, B, n, info); CUDA_CHECK;
    cudaDeviceSynchronize(); CUDA_CHECK;
    cudaMemcpy(Bcopy, B, sizeof(double)*n*d, cudaMemcpyDeviceToHost); CUDA_CHECK;
    
    if (info  ) { cudaFree(info  );  CUDA_CHECK;}
    if (buffer) { cudaFree(buffer);  CUDA_CHECK;}
    if (A     ) { cudaFree(A);  	 CUDA_CHECK;}
    if (B     ) { cudaFree(B);  	 CUDA_CHECK;}
    if (ipiv  ) { cudaFree(ipiv); 	 CUDA_CHECK;}

    return;
}