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
#include "cublas_v2.h"

#define IDX2C(i,j,ld) (((j)*(ld))+(i)) //modify index for 0-based indexing

#define	max(A, B)	((A) > (B) ? (A) : (B))
#define	min(A, B)	((A) < (B) ? (A) : (B))

static __inline__ void modify (cublasHandle_t handle, float *m, int ldm, int n, int p, int q, float alpha, float beta){
    cublasSscal (handle, n-p, &alpha, &m[IDX2C(p,q,ldm)], ldm);
    cublasSscal (handle, ldm-p, &beta, &m[IDX2C(p,q,ldm)], 1);
}




__global__ void calc_nominator(double* devPtrX, double* devPtrY, double* devPtrPSlice, double ksig, int N, int M, int D, int slice_size){
	
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	//d_imgIn[ind_x + ind_y * w + ind_z * w * h] = 0;
	if (x<M && y < slice_size){
		double diff = 0;
		double razn = 0;
		for (int d=0; d < D; d++) { //iterate through D dimensions
        	
            diff=devPtrX[x+d*N] - devPtrY[y+d*M];
            diff=diff*diff; //take the square of the euclidean norm of one scalar -> square the scalar
            razn+=diff; //proposed name: eucl_dist_sqr; add up the differences for each dimension to get the scalar length of the high-dimensional vector
        }
		devPtrPSlice[x+M*y]=exp(razn/ksig); //nominator

	}
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
  double *PSlice;
  double diffXY;
  int slice_size = 50;

  ksig = -2.0 * *sigma2;
  outlier_tmp=(*outlier*M*pow (-ksig*3.14159265358979,0.5*D))/((1-*outlier)*N); 
 /* printf ("ksig = %lf\n", *sigma2);*/
  /* outlier_tmp=*outlier*N/(1- *outlier)/M*(-ksig*3.14159265358979); */
  
  P = (double*) calloc(M, sizeof(double));
  temp_x = (double*) calloc(D, sizeof(double));
  PSlice = (double*) calloc(slice_size*M, sizeof(double));

  
  
  
  
  
  cudaError_t cudaStat;    
  cublasStatus_t stat;
  cublasHandle_t handle;
  double* devPtrX;
  double* devPtrY;
  double* devPtrPSlice; 
  double* devPtrP1;
  double* devPtrPt1;
  double* devPtrPx;
  double* devPtrE;
  
  
  cudaStat = cudaMalloc ((void**)&devPtrX, N*D*sizeof(double));
  cudaStat = cudaMalloc ((void**)&devPtrY, M*D*sizeof(double));
  cudaStat = cudaMalloc ((void**)&devPtrPSlice, M*slice_size*sizeof(double));
  cudaStat = cudaMalloc ((void**)&devPtrP1, N*sizeof(double));
  cudaStat = cudaMalloc ((void**)&devPtrPt1, M*sizeof(double));
  cudaStat = cudaMalloc ((void**)&devPtrPx, M*D*sizeof(double));
  cudaStat = cudaMalloc ((void**)&devPtrE, M*D*sizeof(double));

  
  stat = cublasCreate(&handle);
  stat = cublasSetMatrix (N, D, sizeof(*x), x, N, devPtrX, N);
  stat = cublasSetMatrix (M, D, sizeof(*y), y, M, devPtrY, M);
  stat = cublasSetMatrix (N, D, sizeof(*x), P, N, devPtrPSlice, N);
//  stat = cublasSetMatrix (M, D, sizeof(*y), y, M, devPtrP1, M);
//  stat = cublasSetMatrix (M, D, sizeof(*y), y, M, devPtrPt1, M);
//  stat = cublasSetMatrix (M, D, sizeof(*y), y, M, devPtrPx, M);



  
  
  

	dim3 block = dim3(64, 4, 1);
	dim3 grid = dim3((M + block.x - 1) / block.x,
			(slice_size + block.y - 1) / block.y);

	calc_nominator <<<grid, block>>> (devPtrX, devPtrY, devPtrPSlice, ksig, N, M, D, slice_size);
  
  cudaMemcpy(PSlice, devPtrPSlice,  M*slice_size* sizeof(double), cudaMemcpyDeviceToHost);  
  
  
 

  
  
  cudaFree(devPtrX);
  cudaFree(devPtrY);
  cudaFree(devPtrPSlice);
  cudaFree(devPtrP1);
  cudaFree(devPtrPt1);
  cudaFree(devPtrPx);
  cudaFree(devPtrE);

  
  
  
  
  
  
  
  
  

  
  
  
  
  for (n=0; n < N; n++) {
      
      sp=0;
      for (m=0; m < M; m++) { //iterate through all points (M: width of y)
          razn=0;
          for (d=0; d < D; d++) { //iterate through D dimensions
             diff=*(x+n+d*N)-*(y+m+d*M);  
             diff=diff*diff; //take the square of the euclidean norm of one scalar -> square the scalar
             razn+=diff; //proposed name: eucl_dist_sqr; add up the differences for each dimension to get the scalar length of the high-dimensional vector
          }
          
          P[m]=exp(razn/ksig); //nominator
          if(n< 2){
//        	  mexPrintf ("P(%d, %d): %f , fraction: %f  \n", n, m, P[m], razn/ksig);

        	  if (P[m] != PSlice[m + n * M]){
        		  mexPrintf ("Assertion failed at (%d, %d): %f and %f  \n", n, m, P[m], PSlice[m + n * M]);
        	  }
          }
          sp+=P[m]; //sum in the denominator
      }
      
      sp+=outlier_tmp; //for this particular x point, we calculate the complete denominator
      Pt1[n]=1-outlier_tmp/ sp; //see documentation: (1 - ca)
      
      for (d=0; d < D; d++) {
       temp_x[d]=x[n+d*N]/ sp;
      }
         
      for (m=0; m < M; m++) {
         
          P1[m]+=P[m]/ sp; //P1 is the P * sum_vector from the equation (see documentation) 
          
          for (d=0; d < D; d++) {
          Px[m+d*M]+= temp_x[d]*P[m]; 
          }
          
      }
      
   *E +=  -log(sp);	//entropy: measure of overall change
  }
  *E +=D*N*log(*sigma2)/2;
    
  
  free((void*)P);
  free((void*)PSlice);

  free((void*)temp_x);

  return;
}

/* Input arguments */
#define IN_x		prhs[0]
#define IN_y		prhs[1]
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
    cudaError_t cudaStat;    
    cublasStatus_t stat;
    cublasHandle_t handle;
	
  double *x, *y, *sigma2, *outlier, *P1, *Pt1, *Px, *E;
  int     N, M, D;
  
  /* Get the sizes of each input argument */
  N = mxGetM(IN_x);
  M = mxGetM(IN_y);
  D = mxGetN(IN_x);
  
  /* Create the new arrays and set the output pointers to them */
  OUT_P1     = mxCreateDoubleMatrix(M, 1, mxREAL);
  OUT_Pt1    = mxCreateDoubleMatrix(N, 1, mxREAL);
  OUT_Px    = mxCreateDoubleMatrix(M, D, mxREAL);
  OUT_E       = mxCreateDoubleMatrix(1, 1, mxREAL);

    /* Assign pointers to the input arguments */
  x      = mxGetPr(IN_x);
  y       = mxGetPr(IN_y);
  sigma2       = mxGetPr(IN_sigma2);
  outlier    = mxGetPr(IN_outlier);

 
  
  /* Assign pointers to the output arguments */
  P1      = mxGetPr(OUT_P1);
  Pt1      = mxGetPr(OUT_Pt1);
  Px      = mxGetPr(OUT_Px);
  E     = mxGetPr(OUT_E);
   
  /* Do the actual computations in a subroutine */
  cpd_comp(x, y, sigma2, outlier, P1, Pt1, Px, E, N, M, D);
  
  return;
}



