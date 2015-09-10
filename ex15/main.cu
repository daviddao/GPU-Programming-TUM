// ###
// ###
// ### Practical Course: GPU Programming in Computer Vision
// ###
// ###
// ### Technical University Munich, Computer Vision Group
// ### Summer Semester 2015, September 7 - October 6
// ###
// ###
// ### Thomas Moellenhoff, Robert Maier, Caner Hazirbas
// ###
// ###
// ###
// ### THIS FILE IS SUPPOSED TO REMAIN UNCHANGED
// ###
// ###


#include "aux.h"
#include <iostream>
#include "math.h"
using namespace std;

// uncomment to use the camera
//#define CAMERA


//CPU Computation


__global__ void mykernel(float *d_imgIn, int *d_hist, int w, int h, int nc)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    
	if (x < w && y < h && z < nc)
	{
		int ind = x + w*y + w*h*z; 
		int index = d_imgIn[ind]*255.f;
		atomicAdd(&d_hist[index], 1);
	}
}

__global__ void mykernel_shared(float *d_imgIn, int *d_hist, int w, int h, int nc)
{
   __shared__ 
   int shared_hist[256];

   // first thread init the shared memory
   if (threadIdx.x < 256) {
        shared_hist[threadIdx.x] = 0;
   }

   __syncthreads();


   int x = threadIdx.x + blockDim.x * blockIdx.x;
   int y = threadIdx.y + blockDim.y * blockIdx.y;
   int z = threadIdx.z + blockDim.z * blockIdx.z;
   
	if (x < w && y < h && z < nc)
	{
		int ind = x + w*y + w*h*z; 
		int index = d_imgIn[ind]*255.f;
		atomicAdd(&shared_hist[index], 1);
	}

    __syncthreads();

    // first block thread updates global histogramm
    if (threadIdx.x < 256) {
        atomicAdd(&d_hist[threadIdx.x],shared_hist[threadIdx.x]);
    }
}

int main(int argc, char **argv)
{
    // Before the GPU can process your kernels, a so called "CUDA context" must be initialized
    // This happens on the very first call to a CUDA function, and takes some time (around half a second)
    // We will do it right here, so that the run time measurements are accurate
    cudaDeviceSynchronize();  CUDA_CHECK;




    // Reading command line parameters:
    // getParam("param", var, argc, argv) looks whether "-param xyz" is specified, and if so stores the value "xyz" in "var"
    // If "-param" is not specified, the value of "var" remains unchanged
    //
    // return value: getParam("param", ...) returns true if "-param" is specified, and false otherwise

#ifdef CAMERA
#else
    // input image
    string image = "";
    bool ret = getParam("i", image, argc, argv);
    if (!ret) cerr << "ERROR: no image specified" << endl;
    if (argc <= 1) { cout << "Usage: " << argv[0] << " -i <image> [-repeats <repeats>] [-gray]" << endl; return 1; }
#endif
    
    // number of computation repetitions to get a better run time measurement
    int repeats = 1;
    getParam("repeats", repeats, argc, argv);
    cout << "repeats: " << repeats << endl;
    
    // load the input image as grayscale if "-gray" is specifed
    bool gray = false;
    getParam("gray", gray, argc, argv);
    cout << "gray: " << gray << endl;

    // ### Define your own parameters here as needed    
    float gamma = 1;
    getParam("gamma", gamma, argc, argv);
    cout << "gamma: " << gamma << endl;

    float blockX = 32;
    getParam("blockX", blockX, argc, argv);
    cout << "blockX: " << blockX << endl;

    float blockY = 8;
    getParam("blockY", blockY, argc, argv);
    cout << "blockY: " << blockY << endl;



    // Init camera / Load input image
#ifdef CAMERA

    // Init camera
  	cv::VideoCapture camera(0);
  	if(!camera.isOpened()) { cerr << "ERROR: Could not open camera" << endl; return 1; }
    int camW = 640;
    int camH = 480;
  	camera.set(CV_CAP_PROP_FRAME_WIDTH,camW);
  	camera.set(CV_CAP_PROP_FRAME_HEIGHT,camH);
    // read in first frame to get the dimensions
    cv::Mat mIn;
    camera >> mIn;
    
#else
    
    // Load the input image using opencv (load as grayscale if "gray==true", otherwise as is (may be color or grayscale))
    cv::Mat mIn = cv::imread(image.c_str(), (gray? CV_LOAD_IMAGE_GRAYSCALE : -1));
    // check
    if (mIn.data == NULL) { cerr << "ERROR: Could not load image " << image << endl; return 1; }
    
#endif

    // convert to float representation (opencv loads image values as single bytes by default)
    mIn.convertTo(mIn,CV_32F);
    // convert range of each channel to [0,1] (opencv default is [0,255])
    mIn /= 255.f;
    // get image dimensions
    int w = mIn.cols;         // width
    int h = mIn.rows;         // height
    int nc = mIn.channels();  // number of channels
    cout << "image: " << w << " x " << h << endl;




    // Set the output image format
    // ###
    // ###
    // ### TODO: Change the output image format as needed
    // ###
    // ###
    cv::Mat mOut(h,w,mIn.type());  // mOut will have the same number of channels as the input image, nc layers
    //cv::Mat mOut(h,w,CV_32FC3);    // mOut will be a color image, 3 layers
    //cv::Mat mOut(h,w,CV_32FC1);    // mOut will be a grayscale image, 1 layer
    // ### Define your own output images here as needed




    // Allocate arrays
    // input/output image width: w
    // input/output image height: h
    // input image number of channels: nc
    // output image number of channels: mOut.channels(), as defined above (nc, 3, or 1)

    // allocate raw input image array
    float *imgIn  = new float[(size_t)w*h*nc];

    // allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
    float *imgOut = new float[(size_t)w*h*mOut.channels()];

    int hist_size = 256;
    int *hist = new int[hist_size];
    
	//allocate memory on device
	float *d_imgIn = NULL;
	int *d_hist = NULL;
	int imgSize = (size_t)w*h*nc;
	
	cudaMalloc(&d_imgIn, imgSize*sizeof(float)); CUDA_CHECK;
	cudaMalloc(&d_hist, hist_size*sizeof(int)); CUDA_CHECK;

    dim3 block = dim3(32, 8, 1);
    dim3 grid = dim3((w + block.x - 1) / block.x,
            (h + block.y - 1) / block.y, (nc));
    
    Timer timer; float t = 0;

    // For camera mode: Make a loop to read in camera frames
#ifdef CAMERA
    // Read a camera image frame every 30 milliseconds:
    // cv::waitKey(30) waits 30 milliseconds for a keyboard input,
    // returns a value <0 if no key is pressed during this time, returns immediately with a value >=0 if a key is pressed
    while (cv::waitKey(30) < 0)
    {
    // Get camera image
    camera >> mIn;
    // convert to float representation (opencv loads image values as single bytes by default)
    mIn.convertTo(mIn,CV_32F);
    // convert range of each channel to [0,1] (opencv default is [0,255])
    mIn /= 255.f;
#endif

    // Init raw input image array
    // opencv images are interleaved: rgb rgb rgb...  (actually bgr bgr bgr...)
    // But for CUDA it's better to work with layered images: rrr... ggg... bbb...
    // So we will convert as necessary, using interleaved "cv::Mat" for loading/saving/displaying, and layered "float*" for CUDA computations
    convert_mat_to_layered (imgIn, mIn);
    
    for(int i=0;i<hist_size;i++)
    	hist[i] = 0;
	
	//copy host memory to device
	cudaMemcpy(d_imgIn, imgIn, imgSize*sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
	cudaMemcpy(d_hist, hist, hist_size*sizeof(int), cudaMemcpyHostToDevice); CUDA_CHECK;

    timer.start();
    for (int i=0; i< repeats ; i++)
    {
        
            mykernel_shared <<<grid,block>>> (d_imgIn, d_hist, w, h, nc);
        
    }
    timer.end();
    t = timer.get();  // elapsed time in seconds
    cout << "Average time (shared) for " << repeats << " repeat(s): " << t * 1000 / repeats << " ms" << endl;    

    cudaMemcpy(d_hist, hist, hist_size*sizeof(int), cudaMemcpyHostToDevice); CUDA_CHECK;
    
    timer.start();
	for (int i=0; i< repeats ; i++)
	{
		
			mykernel <<<grid,block>>> (d_imgIn, d_hist, w, h, nc);
		
	}
    timer.end();
    t = timer.get();  // elapsed time in seconds	
	cout << "Average time (naive) for " << repeats << " repeat(s): " << t * 1000 / repeats << " ms" << endl;  
	
	
//    for(int i=0;i<hist_size;i++)
//    	hist[i] = 0;
//	
//	t = 0;
//	for (int i=0; i< repeats ; i++)
//	{
//		timer.start();
//			mykernel_shared <<<grid,block>>> (d_imgIn, d_hist, w, h, nc);
//		timer.end();
//		t += timer.get();  // elapsed time in seconds
//	}	
//	cout << "SHARED - Average time for " << repeats << " repeat(s): " << t * 1000 / repeats << " ms" << endl;	
	
	//copy result back to host memory
	cudaMemcpy(hist, d_hist, hist_size * sizeof(int), cudaMemcpyDeviceToHost); CUDA_CHECK;
	
	showHistogram256("Histogram", hist, 1000, 100);
	
    // show input image
    showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)

    // show output image: first convert to interleaved opencv format from the layered raw array
    //convert_layered_to_mat(mOut, imgOut);
    //showImage("Histogram", mOut, 100+w+40, 100);

    // ### Display your own output images here as needed

#ifdef CAMERA
    // end of camera loop
    }
#else
    // wait for key inputs
    cv::waitKey(0);
#endif

	//free memory
	cudaFree(d_imgIn);
	cudaFree(d_hist);	


    // save input and result
    cv::imwrite("image_input.png",mIn*255.f);  // "imwrite" assumes channel range [0,255]
    cv::imwrite("image_result.png",mOut*255.f);

    // free allocated arrays
    delete[] imgIn;
    delete[] imgOut;
    delete[] hist;

    // close all opencv windows
    cvDestroyAllWindows();
    return 0;
}




