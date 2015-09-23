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

using namespace std;

void checkStatus(culaStatus status)
{
char buf[80];

if(!status)
return;

culaGetErrorInfoString(status, culaGetErrorInfo(), buf, sizeof(buf));
printf("%s\n", buf);

culaShutdown();
exit(EXIT_FAILURE);
}











/* Input arguments */
#define IN_A		prhs[0]
#define IN_B		prhs[1]

/* Output arguments */
#define OUT_X		plhs[0]

/* Gateway routine */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
  culaFloatComplex *matrix_A, *matrix_B;
  double *matrix_X;
  int     N, D;
  int matrix_pivot[N*N];
  /* Get the sizes of each input argument */
  N = mxGetM(IN_A);
  D = mxGetN(IN_B);
  
  /* Create the new arrays and set the output pointers to them */
  //OUT_X     = mxCreateDoubleMatrix(N, D, mxREAL);
  OUT_X     = mxCreateDoubleMatrix(N, N, mxREAL);


    /* Assign pointers to the input arguments */
  matrix_A      = (culaFloatComplex*)mxGetPr(IN_A);
  matrix_B      = (culaFloatComplex*)mxGetPr(IN_B);
  
  /* Assign pointers to the output arguments */
  matrix_X      = mxGetPr(OUT_X);
  
  culaStatus status;
  status = culaInitialize();
  checkStatus(status);
  
  status = culaCgetrf(N, N, matrix_A, N, matrix_pivot);
  checkStatus(status);
  
  culaShutdown();
  
  return;
}



