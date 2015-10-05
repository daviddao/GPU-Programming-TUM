/**magicCUDA('cpd_P_GPU')magicCUDA('cpd_P_GPU')
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
void fillvector(float *data, int N, float value){
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

// Calculates Pt1 using threads
__global__
void calc_Pt1(float* d_Pt1, float* d_sp, float outlier_tmp, int N) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (idx < N) {
        d_Pt1[idx] = 1.0f - (outlier_tmp/(d_sp[idx] + outlier_tmp));
    }
}
// Use threads to calculate E, later we sum up
__global__
void calc_E(float* d_E, float* d_sp, float outlier_tmp, int N) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (idx < N) {
        d_E[idx] = -log(d_sp[idx] + outlier_tmp);
    }
}

// Calculates Px using threads
__global__
void calc_X_tmp(float* d_Xtemp, float* d_X, float* d_denom, int starting_index, int slice_size, int D, int N) {
    
     int idx = threadIdx.x + blockDim.x * blockIdx.x;
     int d = threadIdx.y + blockDim.y * blockIdx.y;
    
     if (idx < slice_size && d < D) {
        // Create d_Xtemp (slice_size * D)
        d_Xtemp[IDX2C(idx,d,slice_size)] = d_denom[idx] * d_X[IDX2C(idx + starting_index,d,N)];   
     } 
}
    
// Calculates slice_size denominator using threads
__global__
void calc_denominator(float* d_denom, float* d_sp, float outlier_tmp, int slice_size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (idx < slice_size) {
        d_denom[idx] = 1.0f / (d_sp[idx] + outlier_tmp);
    }
}

// Kernel calculating the nominators of each entry of P (for 6980 x 6980 it takes 160ms)
__global__ 
void calc_nominator(float* d_X, float* d_Y, float* d_PSlice, float ksig, int N, int M, int D, int slice_size, int slice_nr){
	
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
    
	int i = idx + (slice_size*slice_nr);
	if (idx < slice_size && i<N && j < M){
		
		
		float diff = 0;
		float razn = 0;
		for (int d=0; d < D; d++) { //iterate through D dimensions
        	
            diff=d_X[i+d*N] - d_Y[j+d*M];
            diff=diff*diff; //take the square of the euclidean norm of one scalar -> square the scalar
            razn+=diff; //proposed name: eucl_dist_sqr; add up the differences for each dimension to get the scalar length of the high-dimensional vector
        }
        // Set it using a row-major -> column major translator (for CUBLAS and MATLAB)
		d_PSlice[IDX2C(i%slice_size,j,slice_size)]=exp(razn/ksig); //nominator
		
	}
}


void cpd_comp(
		float* x,
		float* y, 
        float* sigma2,
		float* outlier,
        float* P1,
        float* Pt1,
        float* Px,
    	float* E,
        int N,
		int M,
        int D
        )
{
  int		n, m, d;
  float	ksig, diff, razn, outlier_tmp, sp;
  float	*P, *temp_x;
  float *PSlice;
  int slice_size = N/10;
  float *ones;
  float *filler;
  
  P = (float*) calloc(M, sizeof(float));
  temp_x = (float*) calloc(D, sizeof(float));
  PSlice = (float*) calloc(slice_size*M, sizeof(float));
  ones = (float*) calloc(M, sizeof(float));
  filler = (float*) calloc(N,sizeof(float));
  ksig = -2.0 * *sigma2;
  //outlier_tmp=(*outlier*M*pow (-ksig*M_PI,0.5*D))/((1-*outlier)*N);
  outlier_tmp=(*outlier*pow (-ksig*M_PI,0.5*D))/((1-*outlier));
  
  // Trick like for Andrey
  if(outlier_tmp == 0) outlier_tmp = 1e-38;
  //mexPrintf("OUTLIER %.64f \n", outlier_tmp);
  
  fillvector(ones, M, 1);
  fillvector(filler,N,0);
  
 /* printf ("ksig = %lf\n", *sigma2);*/
  /* outlier_tmp=*outlier*N/(1- *outlier)/M*(-ksig*3.14159265358979); */
  
  
  
  // CUBLAS Stuff
  cudaError_t cudaStat;    
  cublasStatus_t stat;
  cublasHandle_t handle;
  
  float* d_X;
  float* d_Y;
  float* d_PSlice; 
  float* d_PSlice_mat; 
  float* d_P1;
  float* d_P1_tmp;
  float* d_Pt1;
  float* d_Px;
  float* d_E;
  float* d_ones;
  float* slice_tmp;
  slice_tmp = (float *)malloc(M*D*sizeof(float));
  float* d_sp;
  
  float* d_denom; //stores a denominator vector
  float* d_X_tmp; //stores a sliced X * denom version of X
  
  
  //TODO: Finish Matrix Vector Multiplication
 
  // Allocate memory on the device
  cudaMalloc (&d_X, N*D*sizeof(float));
  cudaMalloc (&d_Y, M*D*sizeof(float));
  cudaMalloc (&d_PSlice, M*slice_size*sizeof(float));
  
  cudaMalloc (&d_P1, N*sizeof(float));
  cudaMalloc (&d_P1_tmp, N*sizeof(float));
  
  cudaMalloc (&d_Pt1, M*sizeof(float));
  cudaMalloc (&d_Px, M*D*sizeof(float));
  cudaMalloc (&d_E, N*sizeof(float));
  cudaMalloc (&d_ones, M * sizeof(float));
  cudaMalloc (&d_sp, N*sizeof(float));
  
  cudaMalloc (&d_denom, slice_size*sizeof(float));
  cudaMalloc (&d_X_tmp, slice_size*D*sizeof(float));

  cudaCheckErrors("cuda malloc fail");
  
  // Create CUBLAS Context
  stat = cublasCreate(&handle);
  
  // TODO: Load data in the beginning instead of every time!
  cudaMemcpy(d_X,  x, N*D* sizeof(float), cudaMemcpyHostToDevice);  
  cudaMemcpy(d_Y,  y, M*D* sizeof(float), cudaMemcpyHostToDevice);  
  cudaMemcpy(d_ones,  ones, M*sizeof(float), cudaMemcpyHostToDevice);  
  cudaMemcpy(d_sp,  filler, N*sizeof(float), cudaMemcpyHostToDevice);
  // Cpy Px to GPU once!
  cudaMemcpy(d_Px, Px, N*D*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_P1, P1, N*sizeof(float),cudaMemcpyHostToDevice);

  int numSlices = N / slice_size;
  
  dim3 block;
  dim3 grid;
  
  block = dim3(4, 32, 1);
  grid = dim3((slice_size + block.x - 1) / block.x,
			(M + block.y - 1) / block.y);

  
  
  Timer timer; timer.start();
  for (int s=0; s<numSlices; s++){
	  
      //mexPrintf("\n Iteration %i \n",s);
      
      block = dim3(4, 32, 1);
      grid = dim3((slice_size + block.x - 1) / block.x,
			(M + block.y - 1) / block.y);
      
	  calc_nominator <<<grid, block>>> (d_X, d_Y, d_PSlice, ksig, N, M, D, slice_size, s); 
	   
	   float alpha = 1.0f;
	   float beta = 0.0f;
	   int rowsA = slice_size;
	   int columnsA = M;
	   
       // Calculates sp without outlier
       stat = cublasSgemv(handle, CUBLAS_OP_N, rowsA, columnsA, &alpha, d_PSlice, slice_size, d_ones, 1, &beta, d_sp+(s*slice_size), 1);
	   cublasCheckErrors(stat);
       
       // Get the denominator as 1/sp + outlier in d_denom
       block = dim3(256, 1, 1);
       grid = dim3((slice_size + block.x - 1) / block.x,1);
       // denominator correctly calculates! (tested for 6890)
       calc_denominator  <<<grid, block>>> (d_denom, d_sp+(s*slice_size), outlier_tmp, slice_size);
       
       // Calculate P1 using PSlice_t * denom
       stat = cublasSgemv(handle, CUBLAS_OP_T, rowsA, columnsA, &alpha, d_PSlice, slice_size,  d_denom, 1, &beta, d_P1_tmp, 1);
       cublasCheckErrors(stat);
       
       // Add P1_tmp to P1
       stat = cublasSaxpy(handle, M, &alpha, d_P1_tmp, 1, d_P1, 1);
       cublasCheckErrors(stat);
       
       // Calculate Px
       block = dim3(64, 4, 1);
       grid = dim3((slice_size + block.x - 1) / block.x,
			(D + block.y - 1) / block.y);
       
       // First calculate X_temp_sliced (takes 50ms)
       calc_X_tmp <<<grid, block>>> (d_X_tmp, d_X, d_denom, (s*slice_size), slice_size, D, N); 
       
       // Do PSlice_t * X_tmp =+ Px 
       beta = 1.0f;
       stat = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, D, slice_size, &alpha, d_PSlice, slice_size, d_X_tmp, slice_size, &beta, d_Px, M);
       cublasCheckErrors(stat);
	   
  }
  
  // Calculates the complete P1
  block = dim3(256, 1, 1);
  grid = dim3((N + block.x - 1) / block.x,1);
  calc_Pt1  <<<grid, block>>> (d_Pt1, d_sp, outlier_tmp, N);
  // Calculate E
  calc_E  <<<grid, block>>> (d_E, d_sp, outlier_tmp, N);
  // Sum up E
  stat = cublasSasum(handle, N, d_E, 1, &*E);
    
  *E +=D*N*-log(*sigma2)/2;
  
  
   //mexPrintf("\n E: %f \n",*E);

  
  cudaMemcpy(Pt1, d_Pt1, M*sizeof(float), cudaMemcpyDeviceToHost);  
  
  cudaMemcpy(Px, d_Px, M*D*sizeof(float), cudaMemcpyDeviceToHost);  
  
  cudaMemcpy(P1, d_P1, N* sizeof(float), cudaMemcpyDeviceToHost);  

  // Free Device Space, so MATLAB doesnt crash
  cudaFree(d_X);
  cudaFree(d_Y);
  cudaFree(d_PSlice);
  cudaFree(d_P1);
  cudaFree(d_P1_tmp);
  cudaFree(d_Pt1);
  cudaFree(d_Px);
  cudaFree(d_E);
  cudaFree(d_ones);
  cudaFree(d_sp);
  
  cudaFree(d_denom);
  cudaFree(d_X_tmp);

  timer.end();  float t = timer.get();  // elapsed time in seconds
  
  mexPrintf("\n GPU Time: %f ms \n",t * 1000);
  
  //for (n=0; n < N; n++) {
	  //mexPrintf ("\n"); // use for printing P[m]

//       sp=0;
//       for (m=0; m < M; m++) { //iterate through all points (M: width of y)
//           razn=0;
//           for (d=0; d < D; d++) { //iterate through D dimensions
//              diff=x[n+d*N] - y[m+d*M];// *(x+n+d*N)-*(y+m+d*M);  
//              diff=diff*diff; //take the square of the euclidean norm of one scalar -> square the scalar
//              razn+=diff; //proposed name: eucl_dist_sqr; add up the differences for each dimension to get the scalar length of the high-dimensional vector
//           }
// 
//           P[m]=exp(razn/ksig); //nominator
//           
//           sp+=P[m]; //sum in the denominator
//       }
      
      
      //sp+=outlier_tmp; //for this particular x point, we calculate the complete denominator
      
      // Test out if everything works with Pt1 from GPU
      //Pt1[n] = 1-outlier_tmp/ sp; //see documentation: (1 - ca)
      
//       if((float)slice_tmp[n] != (float)(1.0f/sp)){
//     	  
//         	  mexPrintf("Assertion failed! %d - denom[n] on GPU/CPU: %f - %f \n",n, slice_tmp[n],  1.0f/sp);
// //          mexPrintf("sp on CPU: %f \n",sp);
//       }
      
//       for (d=0; d < D; d++) {
//        temp_x[d]=x[n+d*N]/ sp;
//       }
//          
//       for (m=0; m < M; m++) {
//          
//           P1[m]+=P[m]/ sp; //P1 is the P * sum_vector from the equation (see documentation) 
//           
//           for (d=0; d < D; d++) {
//             Px[m+d*M]+= temp_x[d]*P[m]; 
//           }
//           
//       }

      
   //*E +=  -log(sp);	//entropy: measure of overall change
  //}
  
  // Test for P1
//   for (int i = 0; i < M; i++) {
//      for (int d =0; d < D; d++) {
//      if((float)slice_tmp[IDX2C(i,d,M)] != (float)Px[IDX2C(i,d,M)]){
//     	  
//        mexPrintf("Assertion failed! %d - Px[n] on GPU/CPU: %f - %f \n",i,slice_tmp[IDX2C(i,d,M)], Px[IDX2C(i,d,M)]);
// //          mexPrintf("sp on CPU: %f \n",sp);
//         }
//      }
//   }
//   
  //*E +=D*N*log(*sigma2)/2;
  
  free((void*)P);
  free((void*)PSlice);
  free((void*)temp_x);
  free((void*)ones);
  free((void*)filler);
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
	
  float *x, *y, *sigma2, *outlier, *P1, *Pt1, *Px, *E;
  int     N, M, D;
  
  /* Get the sizes of each input argument */
  N = mxGetM(IN_x);
  M = mxGetM(IN_y);
  D = mxGetN(IN_x);
  
  /* Create the new arrays and set the output pointers to them */
  OUT_P1     = mxCreateNumericMatrix(M, 1, mxSINGLE_CLASS, mxREAL);
  OUT_Pt1    = mxCreateNumericMatrix(N, 1, mxSINGLE_CLASS, mxREAL);
  OUT_Px     = mxCreateNumericMatrix(M, D, mxSINGLE_CLASS, mxREAL);
  OUT_E      = mxCreateNumericMatrix(1, 1, mxSINGLE_CLASS, mxREAL);

    /* Assign pointers to the input arguments */
  x      = (float*) mxGetData(IN_x);
  y       = (float*) mxGetData(IN_y);
  sigma2       = (float*) mxGetData(IN_sigma2);
  outlier    = (float*) mxGetData(IN_outlier);
  
  /* Assign pointers to the output arguments */
  P1      = (float*) mxGetData(OUT_P1);
  Pt1      = (float*) mxGetData(OUT_Pt1);
  Px      = (float*) mxGetData(OUT_Px);
  E     = (float*) mxGetData(OUT_E);
   
  /* Do the actual computations in a subroutine */
  cpd_comp(x, y, sigma2, outlier, P1, Pt1, Px, E, N, M, D);
  
  return;
}



