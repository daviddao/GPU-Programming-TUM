/**
 * Johannes and David
 * TU Munich
 * Sep 2015
 */
#include <stdio.h>
#include "mex.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "time.h"

#define IDX2C(i,j,ld) (((j)*(ld))+(i)) //modify index for 0-based indexing

#define	max(A, B)	((A) > (B) ? (A) : (B))
#define	min(A, B)	((A) < (B) ? (A) : (B))

// measuring time 
class Timer
{
    public:
	Timer() : tStart(0), running(false), sec(0.f)
	{
	}
	void start()
	{
		tStart = clock();
		running = true;
	}
	void end()
	{
		if (!running) { sec = 0; return; }
        cudaDeviceSynchronize();
		clock_t tEnd = clock();
		sec = (float)(tEnd - tStart) / CLOCKS_PER_SEC;
		running = false;
	}
	float get()
	{
		if (running) end();
		return sec;
	}
    private:
	clock_t tStart;
	bool running;
	float sec;
};


// Kernel calculating the nominators of each entry of P (for 6980 x 6980 it takes 160ms)
__global__ 
void calc_nominator(double* devPtrX, double* devPtrY, double* devPtrPSlice, double ksig, int N, int M, int D, int slice_size){
	
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
    
    // TODO: include starting and ending index of slice
	if (i < slice_size && j < M){
		double diff = 0;
		double razn = 0;
		for (int d=0; d < D; d++) { //iterate through D dimensions
        	
            diff=devPtrX[i+d*N] - devPtrY[j+d*M];
            diff=diff*diff; //take the square of the euclidean norm of one scalar -> square the scalar
            razn+=diff; //proposed name: eucl_dist_sqr; add up the differences for each dimension to get the scalar length of the high-dimensional vector
        }
        // Set it using a row-major -> column major translator (for CUBLAS and MATLAB)
		devPtrPSlice[IDX2C(i,j,slice_size)]=exp(razn/ksig); //nominator
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
  int slice_size = 6890;

  ksig = -2.0 * *sigma2;
  outlier_tmp=(*outlier*M*pow (-ksig*3.14159265358979,0.5*D))/((1-*outlier)*N); 
 /* printf ("ksig = %lf\n", *sigma2);*/
  /* outlier_tmp=*outlier*N/(1- *outlier)/M*(-ksig*3.14159265358979); */
  
  
  P = (double*) calloc(M, sizeof(double));
  temp_x = (double*) calloc(D, sizeof(double));
  PSlice = (double*) calloc(slice_size*M, sizeof(double));

  // CUBLAS Stuff
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
  
  // We need that to calculate P1
  double* devPtrOnes;
  cudaMalloc(&devPtrOnes, M * sizeof(double));
  //TODO: Finish Matrix Vector Multiplication
  
  
  // Just for marking entries (stay -1 if they are unedited by the kernel)
  for (int i=0; i<M*slice_size; i++){
	  PSlice[i] = -1;
  }
  
  // Allocate memory on the device
  cudaStat = cudaMalloc (&devPtrX, N*D*sizeof(double));
  cudaStat = cudaMalloc (&devPtrY, M*D*sizeof(double));
  cudaStat = cudaMalloc (&devPtrPSlice, M*slice_size*sizeof(double));
  cudaStat = cudaMalloc (&devPtrP1, N*sizeof(double));
  cudaStat = cudaMalloc (&devPtrPt1, M*sizeof(double));
  cudaStat = cudaMalloc (&devPtrPx, M*D*sizeof(double));
  cudaStat = cudaMalloc (&devPtrE, M*D*sizeof(double));

  // Create CUBLAS Context
  stat = cublasCreate(&handle);
  // TODO: Load data in the beginning instead of every time!
  cudaMemcpy(devPtrX,  x, N*D* sizeof(double), cudaMemcpyHostToDevice);  
  cudaMemcpy(devPtrY,  y, M*D* sizeof(double), cudaMemcpyHostToDevice);  
  cudaMemcpy(devPtrPSlice,PSlice,  M*slice_size* sizeof(double), cudaMemcpyHostToDevice);  

//  stat = cublasSetMatrix (N, D, sizeof(*x), x, N, devPtrX, N);
//  stat = cublasSetMatrix (M, D, sizeof(*y), y, M, devPtrY, M);
//  stat = cublasSetMatrix (N, D, sizeof(*x), P, N, devPtrPSlice, N);

//  stat = cublasSetMatrix (M, D, sizeof(*y), y, M, devPtrP1, M);
//  stat = cublasSetMatrix (M, D, sizeof(*y), y, M, devPtrPt1, M);
//  stat = cublasSetMatrix (M, D, sizeof(*y), y, M, devPtrPx, M);

  dim3 block = dim3(4, 32, 1);
  dim3 grid = dim3((slice_size + block.x - 1) / block.x,
			(M + block.y - 1) / block.y);
	
  Timer timer; timer.start();
  calc_nominator <<<grid, block>>> (devPtrX, devPtrY, devPtrPSlice, ksig, N, M, D, slice_size);
  timer.end();  float t = timer.get();  // elapsed time in seconds	
  mexPrintf("\n GPU Time: %f ms \n",t * 1000);
    
  cudaMemcpy(PSlice, devPtrPSlice,  M*slice_size* sizeof(double), cudaMemcpyDeviceToHost);  
  
  // Free Device Space, so MATLAB doesnt crash
  cudaFree(devPtrX);
  cudaFree(devPtrY);
  cudaFree(devPtrPSlice);
  cudaFree(devPtrP1);
  cudaFree(devPtrPt1);
  cudaFree(devPtrPx);
  cudaFree(devPtrE);
  
  //mexPrintf ("Print P[m] in original code: \n");

  for (n=0; n < N; n++) {
	  //mexPrintf ("\n"); // use for printing P[m]

      sp=0;
      for (m=0; m < M; m++) { //iterate through all points (M: width of y)
          razn=0;
          for (d=0; d < D; d++) { //iterate through D dimensions
             diff=x[n+d*N] - y[m+d*M];// *(x+n+d*N)-*(y+m+d*M);  
             diff=diff*diff; //take the square of the euclidean norm of one scalar -> square the scalar
             razn+=diff; //proposed name: eucl_dist_sqr; add up the differences for each dimension to get the scalar length of the high-dimensional vector
          }

          P[m]=exp(razn/ksig); //nominator
          
          //mexPrintf ("%f ", P[m]); // use for printing P[m]
          
          // Check if P[m] CPU is the same as P_slice[m] from GPU
          if(n<slice_size){
        	  //mexPrintf ("P(%d, %d): %f , %f  \n", n, m, P[m],  PSlice[IDX2C(n,m,slice_size)]);

        	  if ((float) P[m] != (float) PSlice[IDX2C(n,m,slice_size)]){
        		  mexPrintf ("Assertion failed at (%d, %d): %f and %f  \n", n, m, P[m], PSlice[IDX2C(n,m,slice_size)]);
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
  OUT_Px     = mxCreateDoubleMatrix(M, D, mxREAL);
  OUT_E      = mxCreateDoubleMatrix(1, 1, mxREAL);

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



