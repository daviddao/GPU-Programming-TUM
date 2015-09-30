//Example 2. Application Using C and CUBLAS: 0-based indexing
//-----------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "mex.h"

#define M 6
#define N 5
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

static __inline__ void modify (cublasHandle_t handle, float *m, int ldm, int n, int p, int q, float alpha, float beta){
    cublasSscal (handle, n-p, &alpha, &m[IDX2C(p,q,ldm)], ldm);
    cublasSscal (handle, ldm-p, &beta, &m[IDX2C(p,q,ldm)], 1);
}

/*
 * 
 * This function scales the vector x by the scalar α and overwrites it with the result. 
 * Hence, the performed operation is x [ j ] = α × x [ j ] for i = 1 , … , n and j = 1 + ( i - 1 ) *  incx .
 
 cublasStatus_t  cublasSscal(cublasHandle_t handle, int n,
                            const float           *alpha,
                            float           *x, int incx)

handle: handle to the cuBLAS library context. (input)
n: number of elements in the vector x			(input)
alpha: <type> scalar used for multiplication.	(input)
x: <type> vector with n elements. 				(in/out)
incx: stride between consecutive elements of x. (input)
 * 
 */


int test (){
    cudaError_t cudaStat;    
    cublasStatus_t stat;
    cublasHandle_t handle;
    int i, j;
    float* devPtrA;
    float* a = 0;
    a = (float *)malloc (M * N * sizeof (*a));
    if (!a) {
    	mexPrintf ("host memory allocation failed");
        return EXIT_FAILURE;
    }
    for (j = 0; j < N; j++) {
        for (i = 0; i < M; i++) {
            a[IDX2C(i,j,M)] = (float)(i * M + j + 1);
        }
    }
    cudaStat = cudaMalloc ((void**)&devPtrA, M*N*sizeof(*a));
    if (cudaStat != cudaSuccess) {
    	mexPrintf ("device memory allocation failed");
        return EXIT_FAILURE;
    }
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
    	mexPrintf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }
    stat = cublasSetMatrix (M, N, sizeof(*a), a, M, devPtrA, M);
    /*
     * cublasStatus_t
cublasSetMatrix(int rows, int cols, int elemSize,
                const void *A, int lda, void *B, int ldb)

This function copies a tile of rows x cols elements from a matrix A in host memory space to a matrix B in GPU memory space. 
It is assumed that each element requires storage of elemSize bytes and that both matrices are stored in column-major format, 
with the leading dimension of the source matrix A and destination matrix B given in lda and ldb, respectively. 
The leading dimension indicates the number of rows of the allocated matrix, even if only a submatrix of it is being used. 
In general, B is a device pointer that points to an object, or part of an object, that was allocated in GPU memory space via cublasAlloc().

     * 
     */
    
    if (stat != CUBLAS_STATUS_SUCCESS) {
    	mexPrintf ("data download failed");
        cudaFree (devPtrA);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    modify (handle, devPtrA, M, N, 1, 2, 16.0f, 12.0f); //inline method
    stat = cublasGetMatrix (M, N, sizeof(*a), devPtrA, M, a, M);
    
    /*
     * This function copies a tile of rows x cols elements from a matrix A in GPU memory space to a matrix B in host memory space. 
     * It is assumed that each element requires storage of elemSize bytes and that both matrices are stored in column-major format,
     *  with the leading dimension of the source matrix A and destination matrix B given in lda and ldb, respectively. 
     *  The leading dimension indicates the number of rows of the allocated matrix, even if only a submatrix of it is being used. 
     *  In general, A is a device pointer that points to an object, or part of an object, that was allocated in GPU memory space via cublasAlloc().

cublasStatus_t
cublasGetMatrix(int rows, int cols, int elemSize,
                const void *A, int lda, void *B, int ldb)



     * 
     */
    
    if (stat != CUBLAS_STATUS_SUCCESS) {
    	mexPrintf ("data upload failed");
        cudaFree (devPtrA);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    cudaFree (devPtrA);
    cublasDestroy(handle);
    for (j = 0; j < N; j++) {
        for (i = 0; i < M; i++) {
        	mexPrintf ("%7.0f", a[IDX2C(i,j,M)]);
        }
        mexPrintf ("\n");
    }
    free(a);
    return EXIT_SUCCESS;
}



/*=================================================================
 * mexfunction.c 
 *
 * This example demonstrates how to use mexFunction.  It returns
 * the number of elements for each input argument, providing the 
 * function is called with the same number of output arguments
 * as input arguments.
 
 * This is a MEX-file for MATLAB.  
 * Copyright 1984-2011 The MathWorks, Inc.
 * All rights reserved.
 *=================================================================*/
#include "mex.h"

void
mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
    int        i;
       
    /* Examine input (right-hand-side) arguments. */
    mexPrintf("\nThere are %d right-hand-side argument(s).", nrhs);
    for (i=0; i<nrhs; i++)  {
        mexPrintf("\n\tInput Arg %i is of type:\t%s ",i,mxGetClassName(prhs[i]));
    }
    
    /* Examine output (left-hand-side) arguments. */
    mexPrintf("\n\nThere are %d left-hand-side argument(s).\n", nlhs);
    if (nlhs > nrhs)
      mexErrMsgIdAndTxt( "MATLAB:mexfunction:inputOutputMismatch",
              "Cannot specify more outputs than inputs.\n");
    
    for (i=0; i<nlhs; i++)  {
        plhs[i]=mxCreateDoubleMatrix(1,1,mxREAL);
        *mxGetPr(plhs[i])=(double)mxGetNumberOfElements(prhs[i]);
    }
    
    test();
}

