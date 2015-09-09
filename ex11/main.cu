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
#include <math.h>
#include <iostream>
using namespace std;

// uncomment to use the camera
//#define CAMERA


__global__ void mykernel(float *imgIn, int w, int h, float gamma, size_t pitch)
{
	int ind_x = threadIdx.x + blockDim.x * blockIdx.x;
	int ind_y = threadIdx.y + blockDim.y * blockIdx.y;

	if (ind_x<w && ind_y<h) 
		{
			int ind = ind_x + pitch*ind_y;
			imgIn[ind] = powf(imgIn[ind], gamma);
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
    int gamma = 1;
    getParam("gamma", gamma, argc, argv);
    cout << "gamma: " << gamma << endl;

    int dim1_size = 1;
    getParam("dim1_size", dim1_size, argc, argv);
    cout << "dim1_size: " << dim1_size << endl;
    
    int dim2_size = 1;
    getParam("dim2_size", dim2_size, argc, argv);
    cout << "dim2_size: " << dim2_size << endl;
    
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

    float *d_imgIn = NULL;
    dim3 block = dim3(dim1_size,dim2_size,1);
	dim3 grid = dim3((w+block.x-1)/block.x,(h*nc+block.y-1)/block.y,1);
	Timer timer; 
	float t;
	float t_sum = 0;
	size_t pitch = 0;
	//*pitch = 0;
	
	cudaMallocPitch(&d_imgIn, &pitch, w*sizeof(float), h*nc); CUDA_CHECK;
	

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


	//for(int no_repeat = 0; no_repeat < repeats; no_repeat++)
	//{
//		timer.start();
//			for(size_t i=0;i<(size_t)w*h*nc;i++)
//				imgOut[i] = powf(imgIn[i],(float)gamma);
//		timer.end(); t = timer.get();  // elapsed time in seconds
//		
//		cout << no_repeat+1 << ".CPU time: " << t*1000 << " ms" << endl;
		
		cudaMemcpy2D(d_imgIn, pitch, imgIn, w*sizeof(float), w*sizeof(float), h*nc, cudaMemcpyHostToDevice); CUDA_CHECK;
		
		timer.start();		
			mykernel <<<grid, block>>> (d_imgIn, w, h*nc, gamma, pitch/sizeof(float)); CUDA_CHECK;		
		timer.end(); t = timer.get();  // elapsed time in seconds
		
		cudaMemcpy2D(imgOut, w*sizeof(float), d_imgIn, pitch, w*sizeof(float), h*nc, cudaMemcpyDeviceToHost); CUDA_CHECK;
		t_sum += t;
	//}
	//cout << "GPU time: " << 1000*t_sum/repeats << " ms" << endl;



    // show input image
    showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)

    // show output image: first convert to interleaved opencv format from the layered raw array
    convert_layered_to_mat(mOut, imgOut);
    showImage("Output", mOut, 100+w+40, 100);

    // ### Display your own output images here as needed

#ifdef CAMERA
    // end of camera loop
    }
#else
    // wait for key inputs
    cv::waitKey(0);
#endif



	cudaFree(d_imgIn);
    // save input and result
    cv::imwrite("image_input.png",mIn*255.f);  // "imwrite" assumes channel range [0,255]
    cv::imwrite("image_result.png",mOut*255.f);

    // free allocated arrays
    delete[] imgIn;
    delete[] imgOut;

    // close all opencv windows
    cvDestroyAllWindows();
    return 0;
}



