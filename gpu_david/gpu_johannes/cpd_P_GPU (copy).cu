/**
 * Emanuele Rodola
 * TU Munich
 * Jun 2015
 */
#include <stdio.h>
#include "mex.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

#define	max(A, B)	((A) > (B) ? (A) : (B))
#define	min(A, B)	((A) < (B) ? (A) : (B))

__global__
void magickernel(
		double* d_X,
		double* d_T, 
        double d_sigma2,
		double d_outlier,
        double* d_P1,
        double* d_Pt1,
        double* d_Px,
    	double* d_E,
        int d_N,
		int d_M,
        int d_D,
        double* d_P,
		double* d_temp_x
        ) {
	
	
		int		n, m, d;
		double	ksig, diff, razn, outlier_tmp, sp;
	  	  
	  ksig = -2.0 * d_sigma2;
	  outlier_tmp=(d_outlier*d_M*pow (-ksig*3.14159265358979,0.5*d_D))/((1-d_outlier)*d_N); 
	 /* printf ("ksig = %lf\n", *sigma2);*/
	  /* outlier_tmp=*outlier*N/(1- *outlier)/M*(-ksig*3.14159265358979); */
	  
	  
	  for (n=0; n < d_N; n++) {
	      
	      sp=0;
	      for (m=0; m < d_M; m++) {
	          razn=0;
	          for (d=0; d < d_D; d++) {
	             diff= d_X[n+d*d_N]-d_T[m+d*d_M];
	             diff=diff*diff;
	             razn+=diff;
	          }
	          
	          d_P[m]=exp(razn/ksig);
	          sp+=d_P[m];
	      }
	      
	      sp+=outlier_tmp;
	      d_Pt1[n]=1-outlier_tmp / sp;
	      
	      for (d=0; d < d_D; d++) {
	    	  d_temp_x[d] = d_X[n+d*d_N] / sp;
//	    	  *(temp_x+d)=*(x+n+d*N)/ sp;
	      }
	         
	      for (m=0; d_M < d_M; d_M++) {
	         
	          d_P1[m] += d_P[m] / sp;
//	    	  *(P1+m)+=*(P+m)/ sp;
	          
	          for (d=0; d < d_D; d++) {
	        	  d_Px[m+d*d_M] += d_temp_x[d] * d_P[m];	          }
	          
	      }
	      
	   *d_E +=  -log(sp);     
	  }
	  *d_E +=d_D*d_N*log(d_sigma2)/2;
	    
	 

	  return;
		
}


void cpd_comp(
		double* x,
		double* y, 
        double* sigma2,
		double* outlier,
        double* P1,
        double* Pt1,
        double* Px,
    	double* E,
        int N,
		int M,
        int D
        )

{
  int		n, m, d;
  double	ksig, diff, razn, outlier_tmp, sp;
  double	*P, *temp_x;
  
  P = (double*) calloc(M, sizeof(double));
  temp_x = (double*) calloc(D, sizeof(double));
  
  ksig = -2.0 * *sigma2;
  outlier_tmp=(*outlier*M*pow (-ksig*3.14159265358979,0.5*D))/((1-*outlier)*N); 
 /* printf ("ksig = %lf\n", *sigma2);*/
  /* outlier_tmp=*outlier*N/(1- *outlier)/M*(-ksig*3.14159265358979); */
  
  
  for (n=0; n < N; n++) {
      
      sp=0;
      for (m=0; m < M; m++) {
          razn=0;
          for (d=0; d < D; d++) {
             diff=*(x+n+d*N)-*(y+m+d*M);  diff=diff*diff;
             razn+=diff;
          }
          
          *(P+m)=exp(razn/ksig);
          sp+=*(P+m);
      }
      
      sp+=outlier_tmp;
      *(Pt1+n)=1-outlier_tmp/ sp;
      
      for (d=0; d < D; d++) {
       *(temp_x+d)=*(x+n+d*N)/ sp;
      }
         
      for (m=0; m < M; m++) {
         
          *(P1+m)+=*(P+m)/ sp;
          
          for (d=0; d < D; d++) {
          *(Px+m+d*M)+= *(temp_x+d)**(P+m);
          }
          
      }
      
   *E +=  -log(sp);     
  }
  *E +=D*N*log(*sigma2)/2;
    
  
  free((void*)P);
  free((void*)temp_x);

  return;
}

/* Input arguments */
#define IN_X		prhs[0]
#define IN_T		prhs[1]
#define IN_sigma2	prhs[2]
#define IN_outlier	prhs[3]


/* Output arguments */
#define OUT_P1		plhs[0]
#define OUT_Pt1		plhs[1]
#define OUT_Px		plhs[2]
#define OUT_E		plhs[3]


/* Gateway routine */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
  double *x, *t, *sigma2, *outlier, *P1, *Pt1, *Px, *E;
  int     N, M, D;
  
  /* Get the sizes of each input argument */
  N = mxGetM(IN_X);
  M = mxGetM(IN_T);
  D = mxGetN(IN_X);
  
  /* Create the new arrays and set the output pointers to them */
  OUT_P1     = mxCreateDoubleMatrix(M, 1, mxREAL);
  OUT_Pt1    = mxCreateDoubleMatrix(N, 1, mxREAL);
  OUT_Px    = mxCreateDoubleMatrix(M, D, mxREAL);
  OUT_E       = mxCreateDoubleMatrix(1, 1, mxREAL);

    /* Assign pointers to the input arguments */
  x      = mxGetPr(IN_X);
  t       = mxGetPr(IN_T);
  sigma2       = mxGetPr(IN_sigma2);
  outlier    = mxGetPr(IN_outlier);

 
  
  /* Assign pointers to the output arguments */
  P1      = mxGetPr(OUT_P1);
  Pt1      = mxGetPr(OUT_Pt1);
  Px      = mxGetPr(OUT_Px);
  E     = mxGetPr(OUT_E);
   
  mexPrintf("size of N: %d, M: %d, D: %d", N, M, D);

  
  /* Do the actual computations in a subroutine */

  //cpd_comp(x, t, sigma2, outlier, P1, Pt1, Px, E, N, M, D);
  
  //allocate memory on GPU
   double* d_X;
   double* d_T;
   double* d_P1;
   double* d_Pt1;
   double* d_Px;
   double* d_E;
   double* d_P;
   double* temp_x;
   
   int sizeX = M*D;
   int sizeT = N*D;
   
   cudaMalloc(&d_X, sizeX * sizeof(double));
   cudaMalloc(&d_T, sizeT * sizeof(double));
   cudaMalloc(&d_P1, M  * sizeof(double));
   cudaMalloc(&d_Pt1,N  * sizeof(double));
   cudaMalloc(&d_Px, sizeX * sizeof(double));
   cudaMalloc(&d_E,1 * sizeof(double));
   cudaMalloc(&d_P, M * sizeof(double));
   cudaMalloc(&temp_x, D * sizeof(double));
   
   
   cudaMemcpy(d_X, x, sizeX * sizeof(double), cudaMemcpyHostToDevice);
   cudaMemcpy(d_T, t, sizeT * sizeof(double), cudaMemcpyHostToDevice);
   
   cudaMemset (&d_P1,0, M  * sizeof(double));
   cudaMemset(&d_Pt1,0,N  * sizeof(double));
   cudaMemset(&d_Px,0, sizeX * sizeof(double));
   cudaMemset(&d_E,0,1 * sizeof(double));
   cudaMemset(&d_P,0, M * sizeof(double));
   cudaMemset(&temp_x,0, D * sizeof(double));   
   
   //cudaMemcpy(d_P1, P1, M * sizeof(double), cudaMemcpyHostToDevice);
   //cudaMemcpy(d_Pt1, Pt1, N * sizeof(double), cudaMemcpyHostToDevice);
   //cudaMemcpy(d_Px, Px, sizeX * sizeof(double), cudaMemcpyHostToDevice);
   //cudaMemcpy(d_E, E, 1 * sizeof(double), cudaMemcpyHostToDevice);
  
   
   dim3 block = dim3(1, 1, 1);
   dim3 grid = dim3(1,1,1);

   magickernel <<<grid, block>>> (d_X, d_T, *sigma2, *outlier, d_P1, d_Pt1, d_Px, d_E, N, M, D, d_P, temp_x);
   
   cudaMemcpy(P1, d_P1, M * sizeof(double), cudaMemcpyDeviceToHost);
   cudaMemcpy(Pt1, d_Pt1, N * sizeof(double), cudaMemcpyDeviceToHost);
   cudaMemcpy(Px, d_Px, sizeX * sizeof(double), cudaMemcpyDeviceToHost);
   cudaMemcpy(E, d_E, 1 * sizeof(double), cudaMemcpyDeviceToHost);
   
  
   cudaFree(d_X);
   cudaFree(d_T);
   cudaFree(d_P1);
   cudaFree(d_Pt1);
   cudaFree(d_Px);
   cudaFree(d_E);
   cudaFree(d_P);
   cudaFree(temp_x);
  
  
  
  
  
  return;
}



