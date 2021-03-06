/**
 * Emanuele Rodola
 * TU Munich
 * Jun 2015
 */
#include "mex.h"
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <cmath>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray*prhs[])
{
	if (nrhs != 3 || nlhs > 1)
      mexErrMsgTxt("Usage: [G] = cpd_G_mex(x, y, beta).");

	const double* const x = mxGetPr(prhs[0]);
	const int dim = int( mxGetM(prhs[0]) );
	const int n = int( mxGetN(prhs[0]) );
	
	const double* const y = mxGetPr(prhs[1]);
	const int m = int( mxGetN(prhs[1]) );
    
    /* Create the new arrays and set the output pointers to them */
    OUT_P1 = mxCreateDoubleMatrix(n, m, mxREAL);
	
	if (dim != int( mxGetM(prhs[1]) ))
		mexErrMsgTxt("The embedding dimensions of x and y must be the same.");
	
	if (n < dim || m < dim)
		mexErrMsgTxt("The number of embedding dimensions cannot be >= number of points.");
	
	
}
