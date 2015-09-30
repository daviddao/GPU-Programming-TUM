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
#include <Eigen/Dense>
using namespace std;
using namespace Eigen;

#define	max(A, B)	((A) > (B) ? (A) : (B))
#define	min(A, B)	((A) < (B) ? (A) : (B))

/* Input arguments */
#define IN_A		prhs[0]
#define IN_B		prhs[1]

/* Output arguments */
#define OUT_X		plhs[0]

/* Gateway routine */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
//  double *matrix_A, *matrix_B, *matrix_X;
  int     N, D;
  
  /* Get the sizes of each input argument */
  N = mxGetM(IN_A);
  D = mxGetN(IN_B);
//  D = mxGetN(IN_B);
  
  /* Create the new arrays and set the output pointers to them */
  //OUT_X     = mxCreateDoubleMatrix(N, D, mxREAL);
  OUT_X     = mxCreateDoubleMatrix(N, D, mxREAL);


//    /* Assign pointers to the input arguments */
//  matrix_A      = mxGetPr(IN_A);
//  matrix_B      = mxGetPr(IN_B);
//  /* Assign pointers to the output arguments */
//  matrix_X      = mxGetPr(OUT_X);
//  
  
//  Matrix3f A;
//  Vector3f b;
//  A << 1,2,3,  4,5,6,  7,8,10;
//  b << 3, 3, 4;
//  cout << "Here is the matrix A:\n" << A << endl;
//  cout << "Here is the vector b:\n" << b << endl;
//  Vector3f x = A.colPivHouseholderQr().solve(b);
//  cout << "The solution is:\n" << x << endl;
  
  /* Do the actual computations */
  mexPrintf("\n Creating matrixXd..");

  MatrixXd matrix_A=Map<MatrixXd>(mxGetPr(IN_A),N,N);
  MatrixXd matrix_B=Map<MatrixXd>(mxGetPr(IN_B),N,D);

  mexPrintf("\n Starting Eigen solver");


  MatrixXd matrix_X = matrix_A.partialPivLu().solve(matrix_B);
  mexPrintf("\n Finished Eigen solver \n");

  //
//  
//  
//  
  Map<MatrixXd>(mxGetPr(OUT_X),N,D)=matrix_X;//.array();
  
  
 
  
  /* Assign pointers to the output arguments */
//  matrix_X      = mxGetPr(OUT_X);
    
  
  return;
}



