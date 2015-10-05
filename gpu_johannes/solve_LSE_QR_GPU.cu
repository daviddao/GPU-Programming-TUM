 
/*
 * How to compile (assume cuda is installed at /usr/local/cuda/)
 *   nvcc -c -I/usr/local/cuda/include ormqr_example.cpp 
 *   nvcc -o a.out ormqr_example.o -L/usr/local/cuda/lib64 -lcublas -lcusolver
 *
 */



#include <stdio.h>
#include "mex.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdlib.h>
#include <assert.h>
#include <cusolverDn.h>
using namespace std;

/* Input arguments */
#define IN_A		prhs[0]
#define IN_B		prhs[1]

/* Output arguments */
#define OUT_X		plhs[0]



/* Gateway routine */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
	float *A, *B;
	double *XC;
	
	/* Get the sizes of each input argument */
	const int N = mxGetM(IN_A);
	const int D = mxGetN(IN_B);
	
	/* Create the new arrays and set the output pointers to them */
	OUT_X     = mxCreateDoubleMatrix(N, D, mxREAL);
	//OUT_X     = mxCreateDoubleMatrix(N, N, mxREAL);
	float *X_h = (float*)malloc(N*D*sizeof(float));
	
	
	/* Assign pointers to the input arguments */
	A      = (float*)mxGetPr(IN_A);
	B      = (float*)mxGetPr(IN_B);
	
	/* Assign pointers to the output arguments */
	XC      = mxGetPr(OUT_X);
	
	
	//cudsHandle_t cudenseH = NULL;
	cusolverDnHandle_t cudenseH = NULL;
    cublasHandle_t cublasH = NULL;
    cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;    
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    cudaError_t cudaStat4 = cudaSuccess;
    
    const int lda = N;
    const int ldb = N;
    const int nrhs1 = D; // number of right hand side vectors
/*       | 1 2 3 |
 *   A = | 4 5 6 |
 *       | 2 1 1 |
 *
 *   x = (1 1 1)'
 *   b = (6 15 4)'
 */
 

    //double A[lda*m] = { 1.0, 4.0, 2.0, 2.0, 5.0, 1.0, 3.0, 6.0, 1.0}; 
//    double X[ldb*nrhs1] = { 1.0, 1.0, 1.0}; // exact solution
    //double B[ldb*nrhs1] = { 6.0, 15.0, 4.0}; 
    //double XC[ldb*nrhs1]; // solution matrix from GPU

    float *d_A = NULL; // linear memory of GPU  
    float *d_tau = NULL; // linear memory of GPU 
    float *d_B  = NULL; 
    int *devInfo = NULL; // info in gpu (device copy)
    float *d_work = NULL;
    int  lwork = 0; 

    int info_gpu = 0;

    const float one = 1;

    /*printf("A = (matlab base-1)\n");
    printMatrix(m, m, A, lda, "A");
    printf("=====\n");
    printf("B = (matlab base-1)\n");
    printMatrix(m, nrhs1, B, ldb, "B");
    printf("=====\n");*/

// step 1: create cudense/cublas handle
    cusolver_status = cusolverDnCreate(&cudenseH);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    cublas_status = cublasCreate(&cublasH);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);
    
// step 2: copy A and B to device
    cudaStat1 = cudaMalloc ((void**)&d_A  , sizeof(float) * lda * N);
    cudaStat2 = cudaMalloc ((void**)&d_tau, sizeof(float) * N);
    cudaStat3 = cudaMalloc ((void**)&d_B  , sizeof(float) * ldb * nrhs1);
    cudaStat4 = cudaMalloc ((void**)&devInfo, sizeof(int));
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    assert(cudaSuccess == cudaStat4);

    cudaStat1 = cudaMemcpy(d_A, A, sizeof(float) * lda * N   , cudaMemcpyHostToDevice);
    cudaStat2 = cudaMemcpy(d_B, B, sizeof(float) * ldb * nrhs1, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
 
// step 3: query working space of geqrf and ormqr
    cusolver_status = cusolverDnSgeqrf_bufferSize(
        cudenseH, 
        N, 
        N, 
        d_A, 
        lda, 
        &lwork);
    assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);
 
    cudaStat1 = cudaMalloc((void**)&d_work, sizeof(float)*lwork);
    assert(cudaSuccess == cudaStat1);

// step 4: compute QR factorization
    cusolver_status = cusolverDnSgeqrf(
        cudenseH, 
        N, 
        N, 
        d_A, 
        lda, 
        d_tau, 
        d_work, 
        lwork, 
        devInfo);
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    assert(cudaSuccess == cudaStat1);

    // check if QR is good or not
    cudaStat1 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);

    printf("after geqrf: info_gpu = %d\n", info_gpu);
    assert(0 == info_gpu);

// step 5: compute Q^T*B
    cusolver_status= cusolverDnSormqr(
        cudenseH, 
        CUBLAS_SIDE_LEFT, 
        CUBLAS_OP_T,
        N, 
        nrhs1, 
        N, 
        d_A, 
        lda,
        d_tau,
        d_B,
        ldb,
        d_work,
        lwork,
        devInfo);
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    assert(cudaSuccess == cudaStat1);
 
    // check if QR is good or not
    cudaStat1 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);

    printf("after ormqr: info_gpu = %d\n", info_gpu);
    assert(0 == info_gpu);

// step 6: compute x = R \ Q^T*B

    cublas_status = cublasStrsm(
         cublasH,
         CUBLAS_SIDE_LEFT,
         CUBLAS_FILL_MODE_UPPER,
         CUBLAS_OP_N, 
         CUBLAS_DIAG_NON_UNIT,
         N,
         nrhs1,
         &one,
         d_A,
         lda,
         d_B,
         ldb);
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);
    assert(cudaSuccess == cudaStat1);

    cudaStat1 = cudaMemcpy(X_h, d_B, sizeof(float)*ldb*nrhs1, cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);
    
    for(int i =0; i<4;i++)
    	printf("%f ", X_h[i]);
    printf("\n");
    
    for(int i=0;i<N*D;i++)
    	XC[i] = (double)X_h[i];

    //printf("X = (matlab base-1)\n");
    //printMatrix(m, nrhs1, XC, ldb, "X");

// free resources
    if (d_A    ) cudaFree(d_A);
    if (d_tau  ) cudaFree(d_tau);
    if (d_B    ) cudaFree(d_B);
    if (devInfo) cudaFree(devInfo);
    if (d_work ) cudaFree(d_work);
    free(X_h);

    if (cublasH ) cublasDestroy(cublasH);   
    if (cudenseH) cusolverDnDestroy(cudenseH);   

    cudaDeviceReset();

    return;
}