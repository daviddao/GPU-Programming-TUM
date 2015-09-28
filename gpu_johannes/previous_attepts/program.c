/**
 * David Dao, Johannes Rausch, Michal Szymczak
 * TU Munich
 * Sep 2015
 */

#include <stdio.h>

#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cula_device.h>
#include <cula_lapack_device.h>
#include <cula_lapack.h>

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




/* Gateway routine */
int main(void)
{

  
  double *d_matrix_A, *d_matrix_B;
  int     N= 6890;
  int D = 30;
  
  double *matrix_A = malloc
  *matrix_B;
  
  srand();
  
  for(int i=0;i<N;i++)
	  for(int j=0;j<N;j++)	
		  A
  
  /* Create the new arrays and set the output pointers to them */
//  OUT_X     = mxCreateDoubleMatrix(N, D, mxREAL);
  
  /* Assign pointers to the output arguments */
//  matrix_X      = mxGetPr(OUT_X);

  int matrix_pivot[N];
  int *d_matrix_pivot;
  culaStatus status;
  status = culaInitialize();
  checkStatus(status);
  
  cudaMalloc(&d_matrix_A, N * N * sizeof(double)); CUDA_CHECK;
  cudaMalloc(&d_matrix_B, N * D * sizeof(double)); CUDA_CHECK;
  cudaMalloc(&d_matrix_pivot, N * sizeof(int)); CUDA_CHECK;
  cudaMemcpy(d_matrix_A, matrix_A, N * N * sizeof(double), cudaMemcpyHostToDevice); CUDA_CHECK;
  cudaMemcpy(d_matrix_B, matrix_B, N * D * sizeof(double), cudaMemcpyHostToDevice); CUDA_CHECK;
  
  //status = culaDgetrs(char trans, int n, int nrhs, culaDouble* a, int lda, culaInt* ipiv, culaDouble* b, int ldb);
  printf("N: %d, D: %d\n", N, D);
  
  //status = culaDeviceDgetrf(N, N, d_matrix_A, N, d_matrix_pivot);
//  status = culaDeviceDgetrf(N, N, d_matrix_A, N, d_matrix_pivot);
//  checkStatus(status);
  status = culaDeviceDgesv(N, D, d_matrix_A, N, d_matrix_pivot, d_matrix_B, N);
  
//  status = culaDeviceDgetrs('N', N, D, d_matrix_A, N, d_matrix_pivot, d_matrix_B, N);
  checkStatus(status);
  culaShutdown();
  
/*  
  for(int i = 0; i<N; i++)
  {
	  for(int j = 0; j<N; j++)
	  {
		  printf("%.2f ", matrix_A[j+i*N]);
}
  	  printf("\n");
}*/
  
  cudaMemcpy(matrix_B, d_matrix_B, N * D * sizeof(double), cudaMemcpyDeviceToHost); CUDA_CHECK;
  
//  for(int i = 0; i<N; i++)
//    printf("%.2f ", matrix_B[i]);
//  printf("\n");
  
//  matrix_X = matrix_B;
  
  cudaFree(d_matrix_A);
  cudaFree(d_matrix_B);
  cudaFree(d_matrix_pivot);
  
  return 0;
}