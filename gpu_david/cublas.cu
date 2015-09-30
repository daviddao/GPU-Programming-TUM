#include <stdio.h>
#include <stdlib.h>
#include "mex.h"
#include <cuda_runtime.h>
#include "cublas_v2.h"

#define n 6

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray*prhs[]) {
    cudaError_t cudaStat; //Cuda Malloc Status
    cublasStatus_t stat; //Cublas function status
    cublasHandle_t handle; //Cublas context

    int j;
    float* x;
    float* y;

    x = (float*) malloc(n * sizeof(*x));
    for(j=0;j<n;j++)
        x[j] = (float) j;

    //Print x
    printf("x: ");
    for(j=0;j<n;j++)
        printf("%2.0f,",x[j]);
    printf("\n");

    y = (float*) malloc(n * sizeof(*y));

    float* d_y;
    float* d_x;

    cudaStat = cudaMalloc((void**)&d_x ,n* sizeof (*x)); 
    cudaStat = cudaMalloc((void**)&d_y ,n* sizeof (*y));

    stat = cublasCreate(&handle); //initialise CUBLAS context
    stat = cublasSetVector(n, sizeof(*x) ,x ,1 ,d_x ,1); //cp x->d_x
    stat = cublasSetVector(n, sizeof(*x) ,d_x ,1 ,d_y ,1); //cp d_x->d_y
    //stat = cublasScopy(handle,n,d_x,1,d_y,1); //copy the vector d_x into d_y : d_x -> d_y
    
    float result;
    stat = cublasSdot(handle,n,d_x,1,d_y,1,&result);
    
    
    //stat = cublasGetVector(n, sizeof(float) ,d_y ,1 ,y ,1); // copy d_y -> y
    
    printf("result after dot product! :\n");
//     for(j=0;j<n;j++)
//         printf (" %2.0f,",x[j]); 
//     printf("\n");
    printf("%2.0f \n",result);

    cudaFree(d_x);
    cudaFree(d_y);

    cublasDestroy(handle);
    
    free(x);
    free(y);

}








