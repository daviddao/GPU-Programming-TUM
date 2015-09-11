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
using namespace std;

// uncomment to use the camera
//#define CAMERA

__host__ __device__ float diffusity(float gradientX, float gradientY) {
	float gHat = 1;
//	float gradientnorm = sqrtf(gradientX*gradientX+gradientY*gradientY);
	return gHat;
	//	return (gHat * gradientnorm);
}

__host__ __device__ float diffusity2(float gradientX, float gradientY,
		float eps) {
	float gHat = 1;
	float gradientnorm = sqrtf(gradientX * gradientX + gradientY * gradientY);
	gHat = gHat / max(eps, gradientnorm);
	return gHat;
//	return (gHat * gradientnorm);
}

__host__ __device__ float diffusity3(float gradientX, float gradientY,
		float eps) {
	float gradientnorm = sqrtf(gradientX * gradientX + gradientY * gradientY);
	float gHat = expf(-gradientnorm * gradientnorm / eps) / eps;
	return gHat;
//	return (gHat * gradientnorm);
}

//not necessary, because we use ghatType2
//__device__ h_eps(float gradientnorm, float eps){
//	if (gradientnorm < eps){
//		return (gradientnorm*gradientnorm/(2*eps));
//	}else{
//		return (gradientnorm - eps/2.0f);
//	}
//}

__global__ void gradientsdiffusitykernel(float *d_imgIn, float *d_imgOutX,
		float *d_imgOutY, float *d_diffusity, int w, int h, int nc, float eps) {
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;
	if (x < w && y < h && z < nc) {
		int index = x + y * w + z * w * h;
		float gradX = 1.f;
		float gradY = 1.f;
		if (x + 1 < w) {
			int index2 = x + 1 + y * w + z * w * h;
			gradX = d_imgIn[index2] - d_imgIn[index];
		} else {
			gradX = 0;
		}
		d_imgOutX[index] = gradX;
		if (y + 1 < h) {
			int index3 = x + (y + 1) * w + z * w * h;
			gradY = d_imgIn[index3] - d_imgIn[index];
		} else {
			gradY = 0;
		}
		d_imgOutY[index] = gradY;

		d_diffusity[index] = diffusity2(gradX, gradY, eps);

	}
}

//__global__ void diffusifygradientsKernel(float *d_imgGradX, float *d_imgGradY,  int w, int h, int nc, int gHatType, float eps)
//{
//	int x = threadIdx.x + blockDim.x * blockIdx.x;
//	int y = threadIdx.y + blockDim.y * blockIdx.y;
//	int z = threadIdx.z + blockDim.z * blockIdx.z;
//	//d_imgIn[ind_x + ind_y * w + ind_z * w * h] = 0;
//	if (x<w && y < h && z < nc)
//	{
//		int index = x + y * w + z * w * h;
//		float gradX = d_imgGradX[index];
//		float gradY = d_imgGradY[index];
//		float g = 1.0f;
//		if (gHatType == 1){
//			g = diffusity(gradX, gradY);
//		} else if (gHatType == 2){
//			g = diffusity2(gradX, gradY, eps);
//		} else if (gHatType == 3){
//			g = diffusity3(gradX, gradY, eps);
//
//		}
//		gradX = g * gradX;
//		gradY = g * gradY;
//		d_imgGradX[index] = gradX;
//		d_imgGradY[index] = gradY;
//	}
//}

__global__ void divkernel(float *d_imgIn, float *d_imgIn2, float *d_imgOut,
		int w, int h, int nc) {
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;
	//d_imgIn[ind_x + ind_y * w + ind_z * w * h] = 0;
	if (x < w && y < h && z < nc) {
		int index = x + y * w + z * w * h;
		if (x > 0) {
			int index2 = x - 1 + y * w + z * w * h;
			d_imgOut[index] = d_imgIn[index] - d_imgIn[index2];
		} else {
			d_imgOut[index] = d_imgIn[index];
		}

		if (y > 0) {
			int index3 = x + (y - 1) * w + z * w * h;
			d_imgOut[index] += d_imgIn2[index] - d_imgIn2[index3];
		} else {
			d_imgOut[index] += d_imgIn2[index];
		}
	}
}

__global__ void updatestepKernel(float *d_original, float * d_imgIn,
		float * d_diffusity, float * d_imgOut, int w, int h, int nc, float tau,
		float lambda) {
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int z = threadIdx.z + blockDim.z * blockIdx.z;
	//d_imgIn[ind_x + ind_y * w + ind_z * w * h] = 0;
	if (x < w && y < h && z < nc) {
		int index = x + y * w + z * w * h;
		int index_xminus1 = (x - 1) + y * w + z * w * h;
		int index_yminus1 = x + (y - 1) * w + z * w * h;
		int index_xplus1 = (x + 1) + y * w + z * w * h;
		int index_yplus1 = x + (y + 1) * w + z * w * h;
		float g_l = 0;
		float g_r = 0;
		float g_u = 0;
		float g_d = 0;
		float g = d_diffusity[index];
		if (x + 1 < w) {
			g_r = g;
		}
		if (y + 1 < h) {
			g_u = g;
		}
		if (x > 0) {
			g_l = d_diffusity[index_xminus1];
		}
		if (y > 0) {
			g_d = d_diffusity[index_yminus1];
		}

		float updated = (2*d_original[index] + lambda * (g_r * d_imgIn[index_xplus1] +
			g_l * d_imgIn[index_xminus1] + g_u * d_imgIn[index_yplus1] + g_d * d_imgIn[index_yminus1])) / 
					(2 + lambda * (g_r + g_l + g_u + g_d));
		d_imgIn[index] = updated;
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

	int N = 1;
	getParam("N", N, argc, argv);
	cout << "N: " << N << endl;

	float tau = 0.01;
	getParam("tau", tau, argc, argv);
	cout << "tau: " << tau << endl;

	float eps = 0.01;
	getParam("eps", eps, argc, argv);
	cout << "eps: " << eps << endl;

	float lambda = 0.5;
	getParam("lambda", lambda, argc, argv);
	cout << "lambda: " << lambda << endl;

	int gHatType = 2;
	getParam("gHatType", gHatType, argc, argv);
	cout << "gHatType: " << gHatType << endl;

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

	//allocate memory on device
	int imgSize = h * w * nc;

	float *d_imgIn;
	float *d_original;
	float *d_imgOut;
	float *d_imgOut2;
	float *d_imgOut3;
	float *d_diffusity;
	cudaMalloc(&d_imgIn, imgSize * sizeof(float));
	cudaMalloc(&d_original, imgSize * sizeof(float));
	cudaMalloc(&d_imgOut, imgSize * sizeof(float));
	cudaMalloc(&d_imgOut2, imgSize * sizeof(float));
	cudaMalloc(&d_imgOut3, imgSize * sizeof(float));
	cudaMalloc(&d_diffusity, imgSize * sizeof(float));

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

	//add noise:
	float noiseSigma = 0.1;
	addNoise(mIn, noiseSigma);

	convert_mat_to_layered(imgIn, mIn);

	Timer timer;
	float t = 0.0f;
	// ###
	// ###
	// ### TODO: Main computation

	float avgTime = 0.0f;
	for (int r = 0; r < repeats; r++) {
		timer.start();

		CUDA_CHECK;

		cudaMemcpy(d_imgIn, imgIn, imgSize * sizeof(float),
				cudaMemcpyHostToDevice);
		CUDA_CHECK;

		dim3 block = dim3(32, 8, 1);
		dim3 grid = dim3((w + block.x - 1) / block.x,
				(h + block.y - 1) / block.y, (nc));

		for (int iteration = 0; iteration < N; iteration++) {
			gradientsdiffusitykernel <<<grid,block>>> (d_imgIn, d_imgOut, d_imgOut2, d_diffusity, w, h, nc, eps);

			cudaDeviceSynchronize();

//			diffusifygradientsKernel <<<grid,block>>> (d_imgOut, d_imgOut2, w, h, nc, gHatType, eps);
			CUDA_CHECK;

//			divkernel <<<grid,block>>> (d_imgOut, d_imgOut2, d_imgOut3, w, h, nc);

			CUDA_CHECK;

			//perform update step
			//tau = 0.2f/gHat;
		updatestepKernel <<<grid,block>>> (d_original, d_imgIn, d_diffusity, d_imgOut, w, h, nc, tau, lambda);
	}
	cudaMemcpy(imgOut, d_imgIn, imgSize * sizeof(float),
			cudaMemcpyDeviceToHost);

	timer.end();
	t = timer.get();  // elapsed time in seconds
	avgTime += t;
//		cout << "time GPU: " << t * 1000 << " ms" << endl;
}

avgTime = avgTime / repeats;
//	cout << "avg time GPU: " << avgTime * 1000 << " ms" << endl;
// ###
// ###

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

//free memory
cudaFree(d_imgIn);
cudaFree(d_original);
cudaFree(d_imgOut2);
cudaFree(d_imgOut);
cudaFree(d_imgOut3);
cudaFree(d_diffusity);

// close all opencv windows
cvDestroyAllWindows();
return 0;
}

