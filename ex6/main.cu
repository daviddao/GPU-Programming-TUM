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
#include <stdio.h>
#include "math.h"
using namespace std;
//static const float BLOCK_SIZE = 256;

// uncomment to use the camera
//#define CAMERA

__device__ int idxBlockToPadded(int x,int y,int blockdimX,int radius){
	int newX = x + radius;
	int newY = y + radius;
	return (newX + newY * (blockdimX+2*radius));
}

__global__ void convolutionkernel(float *d_imgIn, float *d_imgOut,
		float *d_kernel, int w, int h, int nc, int radius) {
	int x0 = blockDim.x * blockIdx.x;
	int y0 = blockDim.y * blockIdx.y;

	int x = threadIdx.x + x0;
	int y = threadIdx.y + y0;

	//allocate shared array
	extern __shared__ float sh_imgIn[];
	int w_blockpadded = blockDim.x + 2 * radius;
	int h_blockpadded = blockDim.y + 2 * radius;
	int sh_size = w_blockpadded * h_blockpadded;

//	int x0_padded = x0 - radius;
//	int y0_padded = y0 - radius;

	int nr_threads = blockDim.x * blockDim.y;
	int pixelsthread = sh_size / nr_threads;
	int threadoffset = (sh_size % nr_threads);
	int threadNr = threadIdx.x + blockDim.x * threadIdx.y;
	if (threadNr == 0){
		pixelsthread = pixelsthread + threadoffset;
		threadoffset = 0;
	}

	bool isActive = (x < w && y < h);

	int kernelwidth = 2 * radius + 1;
	//size_t nOmega = (size_t) w*h;
	
	for (int z = 0; z < nc; z++) {
		
		for (int i=0; i<pixelsthread; i++){
			int pos = threadNr*pixelsthread + threadoffset + i;
			int xi = (pos%w_blockpadded) + (x0-radius); //convert to pixel coordinates
			int yi = (pos/w_blockpadded) + (y0-radius);
			xi = max(0, min(w-1,xi)); //clamping
			yi = max(0, min(h-1,yi));
			float val = d_imgIn[xi + (size_t)w*yi + w*h*z];
			sh_imgIn[pos] = val;
		}
		
//		for (int ptT=threadIdx.x+blockDim.x*threadIdx.y; ptT<sh_size; ptT += blockDim.x*blockDim.y){
//			int xi = (ptT%w_blockpadded) + (x0-radius); //convert to pixel coordinates
//			int yi = (ptT/w_blockpadded) + (y0-radius);
//			xi = max(0, min(w-1,xi)); //clamping
//			yi = max(0, min(h-1,yi));
//			float val = d_imgIn[xi + (size_t)w*yi + nOmega*z];
//			sh_imgIn[ptT] = val;
//		}
		
		__syncthreads();
		
		if (isActive) {

			float outputPixel = 0;
			for (int diffX = -radius; diffX < radius + 1; diffX++) {
				for (int diffY = -radius; diffY < radius + 1; diffY++) {
//					int tmpi = x + diffX;
//					int tmpj = y + diffY;
//					if (tmpi < 0)
//						tmpi = 0;
//					if (tmpi >= w)
//						tmpi = w - 1;
//					if (tmpj < 0)
//						tmpj = 0;
//					if (tmpj >= h)
//						tmpj = h - 1;
//					int indexOffset = tmpi + tmpj * w + z * h * w;
					
					
//					int ptT =threadIdx.x+blockDim.x*threadIdx.y;

					
					//					int xi = (ptT%w_blockpadded);
//					int yi = (ptT/w_blockpadded);
//
//					xi = xi + radius + diffX;
//					yi = yi + radius + diffY;
//					xi = max(0, min(w_blockpadded-1,xi));
//					yi = max(0, min(h_blockpadded-1,yi));
					int indexOffset = idxBlockToPadded(threadIdx.x+diffX, threadIdx.y+diffY, blockDim.x, radius);
					
					//d_imgOut[index] += d_imgIn[indexOffset] * d_kernel[(diffX+radius + (kernelwidth)*(diffY+radius))];
					//sh_data[iBlock] += d_imgIn[indexOffset] * d_kernel[(diffX+radius + (kernelwidth)*(diffY+radius))];
//					outputPixel += d_imgIn[indexOffset] * d_kernel[(diffX + radius + (kernelwidth) * (diffY + radius))];
					outputPixel += sh_imgIn[indexOffset] *  d_kernel[(diffX + radius + (kernelwidth) * (diffY + radius))];

				}
			}

			//ensure all threads are done before continuing
			__syncthreads();
			
			int index =x + y * w + z * w * h;
			d_imgOut[index] = outputPixel;

		}

	}
}

int main(int argc, char **argv) {
	// Before the GPU can process your kernels, a so called "CUDA context" must be initialized
	// This happens on the very first call to a CUDA function, and takes some time (around half a second)
	// We will do it right here, so that the run time measurements are accurate
	cudaDeviceSynchronize();
	CUDA_CHECK;

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
	if (!ret)
		cerr << "ERROR: no image specified" << endl;
	if (argc <= 1) {
		cout << "Usage: " << argv[0]
				<< " -i <image> [-repeats <repeats>] [-gray]" << endl;
		return 1;
	}
#endif

	// number of computation repetitions to get a better run time measurement
	int repeats = 1;
	getParam("repeats", repeats, argc, argv);
	cout << "repeats: " << repeats << endl;

	// load the input image as grayscale if "-gray" is specifed
	bool gray = false;
	getParam("gray", gray, argc, argv);
	cout << "gray: " << gray << endl;

	float sigma = 1;
	getParam("sigma", sigma, argc, argv);
	cout << "sigma: " << sigma << endl;
	// ### Define your own parameters here as needed    

	// Init camera / Load input image
#ifdef CAMERA

	// Init camera
	cv::VideoCapture camera(0);
	if(!camera.isOpened()) {cerr << "ERROR: Could not open camera" << endl; return 1;}
	int camW = 640;
	int camH = 480;
	camera.set(CV_CAP_PROP_FRAME_WIDTH,camW);
	camera.set(CV_CAP_PROP_FRAME_HEIGHT,camH);
	// read in first frame to get the dimensions
	cv::Mat mIn;
	camera >> mIn;

#else

	// Load the input image using opencv (load as grayscale if "gray==true", otherwise as is (may be color or grayscale))
	cv::Mat mIn = cv::imread(image.c_str(),
			(gray ? CV_LOAD_IMAGE_GRAYSCALE : -1));
	// check
	if (mIn.data == NULL) {
		cerr << "ERROR: Could not load image " << image << endl;
		return 1;
	}

#endif

	// convert to float representation (opencv loads image values as single bytes by default)
	mIn.convertTo(mIn, CV_32F);
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
	cv::Mat mOut(h, w, mIn.type()); // mOut will have the same number of channels as the input image, nc layers
	//cv::Mat mOut(h,w,CV_32FC3);    // mOut will be a color image, 3 layers
	//cv::Mat mOut(h,w,CV_32FC1);    // mOut will be a grayscale image, 1 layer
	// ### Define your own output images here as needed

	// Allocate arrays
	// input/output image width: w
	// input/output image height: h
	// input image number of channels: nc
	// output image number of channels: mOut.channels(), as defined above (nc, 3, or 1)

	// allocate raw input image array
	float *imgIn = new float[(size_t) w * h * nc];

	// allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
	float *imgOut = new float[(size_t) w * h * mOut.channels()];

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
	convert_mat_to_layered(imgIn, mIn);

	Timer timer;
	timer.start();
	// ###
	// ###
	// ### TODO: Main computation

	//compute gaussian kernel
//	float sigma = 2;
	int radius = ceil(3 * sigma);
	float factor = 1 / (2 * M_PI * sigma * sigma);
	//float gaussian[(2*radius+1)*(2*radius+1)];
	int kernelsize = (2 * radius + 1) * (2 * radius + 1);
	float* gaussian = new float[kernelsize];
	int diffX;
	int diffY;
	float entry;
	float sum = 0;
	for (int i = 0; i < 2 * radius + 1; i++) {
		for (int j = 0; j < 2 * radius + 1; j++) {
			diffX = i - radius;
			diffY = j - radius;
			entry = factor
					* exp(
							-((diffX * diffX + diffY * diffY)
									/ (2 * sigma * sigma)));
			if (entry < 0.001f)
				entry = 0;
			gaussian[i + (2 * radius + 1) * j] = entry;
			sum += entry;
		}
	}
	//normalize kernel
	for (int j = 0; j < 2 * radius + 1; j++) {
		for (int i = 0; i < 2 * radius + 1; i++) {
			gaussian[i + (2 * radius + 1) * j] = gaussian[i
					+ (2 * radius + 1) * j] / sum;
			cout << gaussian[i + (2 * radius + 1) * j] << "  ";
		}
		cout << endl;
	}

	//create kernel
	cv::Mat kernel(radius * 2 + 1, radius * 2 + 1, CV_32FC1, gaussian);

	//make max value 1
//	for (int j = 0; j < 2 * radius+1; j++) {
//		for (int i=0; i< 2 *radius+1; i++) {
//			gaussian[i + (2*radius+1)*j] = gaussian[i + (2*radius+1)*j]/gaussian[radius + (2*radius+1)*radius];
//		}
//	}
//	cv::Mat kernel(radius*2+1, radius*2+1, CV_32FC1, gaussian);

	//normalize kernel
	//cv::normalize(kernel, kernel, 1);
	showImage("Kernel", kernel, 0, 0);

//	sum = 0;
//	for (int j = 0; j < 2 * radius+1; j++) {
//		for (int i=0; i< 2 *radius+1; i++) {
//			sum += gaussian[i + (2*radius+1)*j];
//		}
//	}
//	cout << "sum: " << sum << endl;

	for (int i = 0; i < w * h * nc; i++) {
		imgOut[i] = 0;
	}

	/*
	 //implement convolution
	 for (int c=0; c<nc; c++){
	 for (int i=0; i<w; i++){
	 for (int j=0; j<h; j++){
	 int index = i+j*w+c*h*w;
	 
	 for (int diffX = -radius; diffX < radius+1; diffX++) {
	 for (int diffY =-radius; diffY < radius+1; diffY++) {
	 int tmpi = i + diffX;
	 int tmpj = j + diffY;
	 if (tmpi<0) tmpi = 0;
	 if (tmpi>=w) tmpi = w-1;
	 if (tmpj<0) tmpj = 0;
	 if (tmpj>=h) tmpj = h-1;
	 int indexOffset = tmpi+tmpj*w+c*h*w;
	 
	 imgOut[index] += imgIn[indexOffset] * gaussian[(diffX+radius + (2*radius+1)*(diffY+radius))];
	 }
	 }

	 
	 }
	 }
	 }
	 */
	//Convolute on GPU
	//allocate memory on device
	float *d_imgIn;
	float *d_imgOut;
	float *d_kernel;
	int imgSize = w * h * nc;
	cudaMalloc(&d_imgIn, imgSize * sizeof(float));
	cudaMalloc(&d_imgOut, imgSize * sizeof(float));
	cudaMalloc(&d_kernel, kernelsize * sizeof(float));
	CUDA_CHECK;

//	cudaDeviceSynchronize();
	cudaMemcpy(d_imgIn, imgIn, imgSize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_kernel, gaussian, kernelsize * sizeof(float),
			cudaMemcpyHostToDevice);

	CUDA_CHECK;

	int dimx = 32;
	int dimy = 4;
	dim3 block = dim3(dimx, dimy);
	dim3 grid = dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
	size_t smBytes = (dimx+2*radius) * (dimy+2*radius) * sizeof(float);
	
	convolutionkernel <<<grid,block,smBytes>>> (d_imgIn, d_imgOut, d_kernel, w, h, nc, radius);
	CUDA_CHECK;

	cudaMemcpy(imgOut, d_imgOut, imgSize * sizeof(float),
			cudaMemcpyDeviceToHost);

	CUDA_CHECK;

	//free memory
	cudaFree(d_imgIn);
	cudaFree(d_imgOut);
	cudaFree(d_kernel);

// ###
// ###
	timer.end();
	float t = timer.get();  // elapsed time in seconds
	cout << "time: " << t * 1000 << " ms" << endl;

	// show input image
	showImage("Input", mIn, 100, 100); // show at position (x_from_left=100,y_from_above=100)

	// show output image: first convert to interleaved opencv format from the layered raw array
	convert_layered_to_mat(mOut, imgOut);
	showImage("Output", mOut, 100 + w + 40, 100);

	// ### Display your own output images here as needed

#ifdef CAMERA
	// end of camera loop
}
#else
	// wait for key inputs
	cv::waitKey(0);
#endif

	// save input and result
	cv::imwrite("image_input.png", mIn * 255.f); // "imwrite" assumes channel range [0,255]
	cv::imwrite("image_result.png", mOut * 255.f);

	// free allocated arrays
	delete[] imgIn;
	delete[] imgOut;
	delete[] gaussian;

	// close all opencv windows
	cvDestroyAllWindows();
	return 0;
}

