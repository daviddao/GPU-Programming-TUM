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
#include <cula.h>
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

/* Input arguments */
#define IN_A		prhs[0]
#define IN_B		prhs[1]
#define IN_X		prhs[2]

/* Output arguments */
#define OUT_X		plhs[0]

/* Gateway routine */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
  double *matrix_A, *matrix_B, *matrix_X;
  int     N, D;
  int iter = 0;
  
  /* Get the sizes of each input argument */
  N = mxGetM(IN_A);
  D = mxGetN(IN_B);
  
    /* Assign pointers to the input arguments */
  matrix_A      = mxGetPr(IN_A);
  matrix_B      = mxGetPr(IN_B);
  matrix_X      = mxGetPr(IN_X);

  int matrix_pivot[N];
  culaStatus status;
  status = culaInitialize();
  checkStatus(status);
  
  printf("N: %d, D: %d\n", N, D); 
  status = culaDsgesv(N, D, matrix_A, N, matrix_pivot, matrix_B, N, matrix_X, N, &iter);
  printf("Iterations = %d\n", iter);
  checkStatus(status);
  culaShutdown();
  
  return;
}



