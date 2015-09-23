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
#include <cstdlib>
#include <iostream>
//using std::string;
//using std::cout;
//using std::endl;

#define IDX2C(i,j,ld) (((j)*(ld))+(i)) //modify index for 0-based indexing

#define	max(A, B)	((A) > (B) ? (A) : (B))
#define	min(A, B)	((A) < (B) ? (A) : (B))


// error check macros
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

// for CUBLAS V2 API
#define cublasCheckErrors(fn) \
    do { \
        cublasStatus_t __err = fn; \
        if (__err != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "Fatal cublas error: %d (at %s:%d)\n", \
                (int)(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

// data filler
void fillvector(double *data, int N, double value){
    for(int i=0; i<N; i++){
        data[i] = value;
    }
}

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
void calc_nominator(double* d_X, double* d_Y, double* d_PSlice, double ksig, int N, int M, int D, int slice_size){
	
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
    
    // TODO: include starting and ending index of slice
	if (i < slice_size && j < M){
		
		
		double diff = 0;
		double razn = 0;
		for (int d=0; d < D; d++) { //iterate through D dimensions
        	
            diff=d_X[i+d*N] - d_Y[j+d*M];
            diff=diff*diff; //take the square of the euclidean norm of one scalar -> square the scalar
            razn+=diff; //proposed name: eucl_dist_sqr; add up the differences for each dimension to get the scalar length of the high-dimensional vector
        }
        // Set it using a row-major -> column major translator (for CUBLAS and MATLAB)
		d_PSlice[IDX2C(i,j,slice_size)]=exp(razn/ksig); //nominator
		
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
  double *ones;
  double *tmp;
  
  P = (double*) calloc(M, sizeof(double));
  tmp = (double*) calloc(slice_size, sizeof(double));
  temp_x = (double*) calloc(D, sizeof(double));
  PSlice = (double*) calloc(slice_size*M, sizeof(double));
  ones = (double*) calloc(M, sizeof(double));
  
  fillvector(ones, M, 1);
  
  ksig = -2.0 * *sigma2;
  outlier_tmp=(*outlier*M*pow (-ksig*3.14159265358979,0.5*D))/((1-*outlier)*N); 
 /* printf ("ksig = %lf\n", *sigma2);*/
  /* outlier_tmp=*outlier*N/(1- *outlier)/M*(-ksig*3.14159265358979); */
  
  
  
  // CUBLAS Stuff
  cudaError_t cudaStat;    
  cublasStatus_t stat;
  cublasHandle_t handle;
  
  double* d_X;
  double* d_Y;
  double* d_PSlice; 
  double* d_PSlice_mat; 

  double* d_P1;
  double* d_Pt1;
  double* d_Px;
  double* d_E;
  // We need that to calculate P1
  double* d_ones;
  double* d_temp;
  double* d_temp2;
  
  double* slice_tmp;
  slice_tmp = (double *)malloc(slice_size*sizeof(double));
  double* d_sp;
  //TODO: Finish Matrix Vector Multiplication
  
  
  
  
  // Just for marking entries (stay -1 if they are unedited by the kernel)
  for (int i=0; i<M*slice_size; i++){
	  PSlice[i] = -1;
  }
  
  // Allocate memory on the device
  cudaMalloc (&d_X, N*D*sizeof(double));
  cudaMalloc (&d_Y, M*D*sizeof(double));
  cudaMalloc (&d_PSlice, M*slice_size*sizeof(double));
  cudaMalloc (&d_PSlice_mat, M*slice_size*sizeof(double));

  cudaMalloc (&d_P1, N*sizeof(double));
  cudaMalloc (&d_Pt1, M*sizeof(double));
  cudaMalloc (&d_Px, M*D*sizeof(double));
  cudaMalloc (&d_E, M*D*sizeof(double));
  cudaMalloc (&d_ones, M * sizeof(double));
  cudaMalloc (&d_temp, slice_size*sizeof(double));
  cudaMalloc (&d_temp2, slice_size*sizeof(double));
  cudaMalloc (&d_sp, N*sizeof(double));

  cudaCheckErrors("cuda malloc fail");
  
  // Create CUBLAS Context
  stat = cublasCreate(&handle);
  
  // TODO: Load data in the beginning instead of every time!
  cudaMemcpy(d_X,  x, N*D* sizeof(double), cudaMemcpyHostToDevice);  
  cudaMemcpy(d_Y,  y, M*D* sizeof(double), cudaMemcpyHostToDevice);  
//  cudaMemcpy(d_PSlice, PSlice,  M*slice_size* sizeof(double), cudaMemcpyHostToDevice);  

  
  
//  stat = cublasSetMatrix (N, D, sizeof(*x), x, N, d_X, N);
//  stat = cublasSetMatrix (M, D, sizeof(*y), y, M, d_Y, M);
//  stat = cublasSetMatrix (N, D, sizeof(*x), P, N, d_PSlice, N);

//  stat = cublasSetMatrix (M, D, sizeof(*y), y, M, d_P1, M);
//  stat = cublasSetMatrix (M, D, sizeof(*y), y, M, d_Pt1, M);
//  stat = cublasSetMatrix (M, D, sizeof(*y), y, M, d_Px, M);

  dim3 block = dim3(4, 32, 1);
  dim3 grid = dim3((slice_size + block.x - 1) / block.x,
			(M + block.y - 1) / block.y);
	
  Timer timer; timer.start();
  calc_nominator <<<grid, block>>> (d_X, d_Y, d_PSlice, ksig, N, M, D, slice_size);
  timer.end();  float t = timer.get();  // elapsed time in seconds	
  mexPrintf("\n GPU Time: %f ms \n",t * 1000);
    
//  cudaMemcpy(PSlice, d_PSlice,  M*slice_size* sizeof(double), cudaMemcpyDeviceToHost);  
  
  
  const int one = 1;
  
  
//  cudaMemset(d_ones,one,M*sizeof(double));

  // Flatten d_Pslice and d_Ones
  cublasSetVector(M, sizeof(double), &(ones[0]), 1, d_ones, 1);
  cublasSetMatrix(slice_size, M, sizeof(double), d_PSlice, slice_size, d_PSlice_mat, slice_size);

  
  double alpha = 1.0f;
  double beta = 0.0f;
  int rowsA = slice_size;
  int columnsA = M;
  
  
  cublasCheckErrors(cublasDgemv(handle, CUBLAS_OP_N, rowsA, columnsA, &alpha, d_PSlice_mat, slice_size, d_ones, 1, &beta, d_temp, 1));
  //d_temp now contains the sum in the nominator 
  //now, we add outlier_tmp to all rows
  
  cudaMemset(d_temp2, outlier_tmp, slice_size*sizeof(double));

  alpha = 1;
  beta = 1;
  cublasCheckErrors(cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, slice_size, 1,
                            &alpha, d_temp, slice_size, &beta,
                            d_temp2, slice_size,
                            d_sp+0, slice_size));


  
  
//  cublasCheckErrors(cublasGetVector(slice_size, sizeof(double), d_temp, 1, d_P1, 1));
  
  
  cudaMemcpy(slice_tmp, d_sp, slice_size* sizeof(double), cudaMemcpyDeviceToHost);  

//  mexPrintf("sp on GPU: \n");
//  for (int i=0; i<slice_size; i++){
//	  mexPrintf("entry %d: %f \n",i, slice_tmp[i]);
//  }
  
  
  // Free Device Space, so MATLAB doesnt crash
  cudaFree(d_X);
  cudaFree(d_Y);
  cudaFree(d_PSlice);
  cudaFree(d_PSlice_mat);
  cudaFree(d_P1);
  cudaFree(d_Pt1);
  cudaFree(d_Px);
  cudaFree(d_E);
  cudaFree(d_ones);
  cudaFree(d_temp);
  cudaFree(d_temp2);
  cudaFree(d_sp);

//  mexPrintf ("Print P[m] in original code: \n");
//  mexPrintf("sp ons CPU: \n");

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
          
          sp+=P[m]; //sum in the denominator
      }
      
      sp+=outlier_tmp; //for this particular x point, we calculate the complete denominator
      if(abs(slice_tmp[n]-sp) > 1e-6){
          mexPrintf("Assertion failed! %d - sp on GPU/CPU: %f - %f \n",n, slice_tmp[n], sp);
//          mexPrintf("sp on CPU: %f \n",sp);
      }

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
  free((void*)tmp);
  free((void*)ones);
  free((void*)slice_tmp);
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



