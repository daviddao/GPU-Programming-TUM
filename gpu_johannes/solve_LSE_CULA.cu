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
#include <cula_lapack.h>

using namespace std;

void checkStatus(culaStatus status)
{
char buf[80];

if(!status)
return;

culaGetErrorInfoString(status, culaGetErrorInfo(), buf, sizeof(buf));
printf("%s\n", buf);

culaShutdown();
//exit(EXIT_FAILURE);
}


/* Input arguments */
#define IN_A		prhs[0]

/* Gateway routine */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
  culaFloatComplex *matrix_A;
  int     N;
  
  /* Get the sizes of each input argument */
  N = mxGetM(IN_A);
  int matrix_pivot[N];
    /* Assign pointers to the input arguments */
  matrix_A      = (culaFloatComplex*)mxGetPr(IN_A);
  
  culaStatus status;
  status = culaInitialize();
  checkStatus(status);
  
  status = culaCgetrf(N, N, matrix_A, N, matrix_pivot);
  //checkStatus(status);
  //printf("Status(0 - ok): %d\n");
/*  
  for(int i = 0; i<N; i++)
  {
	  for(int j = 0; j<N; j++)
	  {
		  printf("%.2f ", matrix_A[j+i*N]);
}
  	  printf("\n");
}*/
  
  culaShutdown();
  
  return;
}



