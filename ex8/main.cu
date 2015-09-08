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

// uncomment to use the camera
//#define CAMERA

// calculate eigenvalues
__device__
void calculate2DEigen(float x11, float x12, float x21, float x22, float& eigen1, float& eigen2) {
    
    float tmp1;
    float tmp2;
    // Calculate the trace
    float trace = x11 + x22;
    // Calculate the determinant
    float determinant = x11 * x22 - x12 * x21;

    // Calculate Eigenvalue
    tmp1 = trace / 2.f + sqrt((trace * trace)/(4 - determinant));
    tmp2 = trace / 2.f - sqrt((trace * trace)/(4 - determinant));

    // Sort values
    if (tmp1 > tmp2) {
        eigen1 = tmp2;
        eigen2 = tmp1;
    } else {
        eigen1 = tmp1;
        eigen2 = tmp2;
    }
}

__global__ 
void convolutionkernel(float *d_imgIn, float *d_imgOut, float *d_kernel, int w, int h, int nc, int radius)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x<w && y < h && z < nc)
    {

        int index =x + y * w + z * w * h;
        float res = 0;
        
        d_imgOut[index] = 0;

        for (int diffX = -radius; diffX < radius+1; diffX++) {
            for (int diffY =-radius; diffY < radius+1; diffY++) {
                int tmpi = x + diffX;
                int tmpj = y + diffY;
                if (tmpi<0) tmpi = 0;
                if (tmpi>=w) tmpi = w-1;
                if (tmpj<0) tmpj = 0;
                if (tmpj>=h) tmpj = h-1;
                int indexOffset = tmpi+tmpj*w+z*h*w;
                res += d_imgIn[indexOffset] * d_kernel[(diffX+radius + (2*radius+1)*(diffY+radius))];
            }
        }

        d_imgOut[index] = res;
//      if (x < 10 && y < 10 && z < 2) printf("sum in thread: %d \n", sum);



    }
}

// rotational symmetric derivative discretization for x and y partial derivative
__global__ 
void rotationaldevkernel_x(float *d_imgIn, float *d_imgOut, int w, int h, int nc) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;

    if (x<w && y < h && z < nc)
    {
        int index = x + y * w + z * w * h;
        
        // get the x with clamping
        int x_tmpi = x + 1;
        int x_tmpj = x - 1;

        if (x_tmpi<0) x_tmpi = 0;
        if (x_tmpi>=w) x_tmpi = w-1;

        // get the y with clamping
        int y_tmpi = y + 1;
        int y_tmpj = y - 1;

        if (y_tmpj<0) y_tmpj = 0;
        if (y_tmpj>=h) y_tmpj = h-1;

        // assign the discretisation result to the pixel
        d_imgOut[index] =   3*d_imgIn[x_tmpi + y_tmpi * w + z * w * h]
                            + 10*d_imgIn[x_tmpi + y * w + z * w * h]
                            + 3*d_imgIn[x_tmpi + y_tmpj * w + z * w * h]
                            - 3*d_imgIn[x_tmpj + y_tmpi * w + z * w * h]
                            - 10*d_imgIn[x_tmpj + y * w + z * w * h]
                            - 3*d_imgIn[x_tmpj + y_tmpj * w + z * w * h];

        d_imgOut[index] /= 32.f;
    }
}

__global__ 
void rotationaldevkernel_y(float *d_imgIn, float *d_imgOut, int w, int h, int nc) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;

    if (x<w && y < h && z < nc)
    {
        int index = x + y * w + z * w * h;
        
        // get the x with clamping
        int x_tmpi = x + 1;
        int x_tmpj = x - 1;

        if (x_tmpi<0) x_tmpi = 0;
        if (x_tmpi>=w) x_tmpi = w-1;

        // get the y with clamping
        int y_tmpi = y + 1;
        int y_tmpj = y - 1;

        if (y_tmpj<0) y_tmpj = 0;
        if (y_tmpj>=h) y_tmpj = h-1;

        // assign the discretisation result to the pixel
        d_imgOut[index] =   3*d_imgIn[x_tmpi + y_tmpi * w + z * w * h]
                            + 10*d_imgIn[x + y_tmpi * w + z * w * h]
                            + 3*d_imgIn[x_tmpj + y_tmpi * w + z * w * h]
                            - 3*d_imgIn[x_tmpi + y_tmpj * w + z * w * h]
                            - 10*d_imgIn[x + y_tmpj * w + z * w * h]
                            - 3*d_imgIn[x_tmpj + y_tmpj * w + z * w * h];

        d_imgOut[index] /= 32.f;
    }
}

__global__
void calculate_M(float *d_xOut, float *d_yOut, float* d_m1, float* d_m2, float* d_m3, int w, int h, int nc) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;

    if (x<w && y < h && z < nc)
    {   
        float res_m1 = 0;
        float res_m2 = 0;
        float res_m3 = 0;
        int index;
        for (int i = 0; i < nc; i++) {
            int index = x + y * w + i * w * h;
            res_m1 += (d_xOut[index] * d_xOut[index]);
            res_m2 += (d_yOut[index] * d_yOut[index]);
            res_m3 += (d_yOut[index] * d_xOut[index]);

        }

        index = x + y * w + z * w * h;
        d_m1[index] = res_m1;
        d_m2[index] = res_m2;
        d_m3[index] = res_m3;
    }
}

__global__
void edgedetector(float* m1, float* m2, float* m3, float* edgeOut, int w, int h, int nc, float alpha, float beta) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;

    float eigen1;
    float eigen2;

    if (x<w && y < h && z < nc)
    {   
        int index = x + y * w + z * w * h;
        float x11 = m1[index];
        float x22 = m2[index];
        float x12 = m3[index];
        float x21 = x12;


        int redIndex = x + y * w;
        int greenIndex = x + y * w + 1 * w * h;
        int blueIndex = x + y * w + 2 * w * h;

        // calculate the eigenvalues
        calculate2DEigen(x11, x12, x21, x22, eigen1, eigen2);

        if (eigen1 >= alpha) {
            //Corner, make it red!

            edgeOut[redIndex] = 255;
            edgeOut[greenIndex] = 0;
            edgeOut[blueIndex] = 0;

        } else if (eigen1 <= beta && alpha <= eigen2) {
            //Edge, draw green!

            edgeOut[redIndex] = 0;
            edgeOut[greenIndex] = 255;
            edgeOut[blueIndex] = 0;

        } else {
            edgeOut[index] *= 0.5;
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

    float sigma = 2;
    getParam("sigma", sigma, argc, argv);
    cout << "sigma: " << sigma << endl;

    float scale = 300;
    getParam("scale", scale, argc, argv);
    cout << "scale: " << scale << endl;

    float alpha = 0.01;
    getParam("alpha", alpha, argc, argv);
    cout << "alpha: " << alpha << endl;

    float beta = 0.001;
    getParam("beta", beta, argc, argv);
    cout << "beta: " << beta << endl;
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
    cv::Mat m11(h, w, mIn.type());
    cv::Mat m12(h, w, mIn.type());
    cv::Mat m13(h, w, mIn.type());

    // Allocate arrays
    // input/output image width: w
    // input/output image height: h
    // input image number of channels: nc
    // output image number of channels: mOut.channels(), as defined above (nc, 3, or 1)

    // allocate raw input image array
    float *imgIn = new float[(size_t) w * h * nc];

    // allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
    float *imgOut = new float[(size_t) w * h * mOut.channels()];

    // allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
    float *m1Out = new float[(size_t) w * h * mOut.channels()];

    // allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
    float *m2Out = new float[(size_t) w * h * mOut.channels()];

    // allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
    float *m3Out = new float[(size_t) w * h * mOut.channels()];

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
//  float sigma = 2;
    int radius = ceil(3 * sigma);
    float factor = 1 / (2 * M_PI * sigma * sigma);
    //float gaussian[(2*radius+1)*(2*radius+1)];
    int kernelsize = (2*radius+1)*(2*radius+1);
    float* gaussian = new float[kernelsize];
    int diffX;
    int diffY;
    float entry;
    float sum = 0;
    for (int i = 0; i < 2 * radius+1; i++) {
        for (int j=0; j<2*radius+1; j++) {
            diffX=i-radius;
            diffY=j-radius;
            entry = factor * exp(-((diffX*diffX+diffY*diffY)/(2*sigma*sigma)));
            if (entry < 0.001f) entry = 0;
            gaussian[i+(2*radius+1)*j]=entry;
            sum += entry;
        }
    }
    //normalize kernel
    for (int j = 0; j < 2 * radius+1; j++) {
        for (int i=0; i< 2 *radius+1; i++) {
            gaussian[i + (2*radius+1)*j] = gaussian[i + (2*radius+1)*j]/sum;
            cout <<  gaussian[i + (2*radius+1)*j] << "  ";
        }
        cout << endl;
    }


    //create kernel
    cv::Mat kernel(radius*2+1, radius*2+1, CV_32FC1, gaussian);
    
    //make max value 1
//  for (int j = 0; j < 2 * radius+1; j++) {
//      for (int i=0; i< 2 *radius+1; i++) {
//          gaussian[i + (2*radius+1)*j] = gaussian[i + (2*radius+1)*j]/gaussian[radius + (2*radius+1)*radius];
//      }
//  }
//  cv::Mat kernel(radius*2+1, radius*2+1, CV_32FC1, gaussian);

    
    //normalize kernel
    //cv::normalize(kernel, kernel, 1);
    showImage("Kernel", kernel, 0,0);
    
    
//  sum = 0;
//  for (int j = 0; j < 2 * radius+1; j++) {
//      for (int i=0; i< 2 *radius+1; i++) {
//          sum += gaussian[i + (2*radius+1)*j];
//      }
//  }
//  cout << "sum: " << sum << endl;
    
    
    for (int i = 0; i < w*h*nc; i++){
        imgOut[i]=0;
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
    float *d_xOut;
    float *d_yOut;
    float *d_m1;
    float *d_m2;
    float *d_m3;
    float *d_m1Out;
    float *d_m2Out;
    float *d_m3Out;
    int imgSize = w*h*nc;
    cudaMalloc(&d_imgIn, imgSize * sizeof(float));
    cudaMalloc(&d_imgOut, imgSize * sizeof(float));
    cudaMalloc(&d_kernel, kernelsize * sizeof(float));
    // Storing the partial derivatives
    cudaMalloc(&d_xOut, imgSize * sizeof(float));
    cudaMalloc(&d_yOut, imgSize * sizeof(float));
    // Storing m11, m12, m13
    cudaMalloc(&d_m1, imgSize * sizeof(float));
    cudaMalloc(&d_m2, imgSize * sizeof(float));
    cudaMalloc(&d_m3, imgSize * sizeof(float));
    // Storing the convolved m11, m12, m13
    cudaMalloc(&d_m1Out, imgSize * sizeof(float));
    cudaMalloc(&d_m2Out, imgSize * sizeof(float));
    cudaMalloc(&d_m3Out, imgSize * sizeof(float));
    CUDA_CHECK;

//  cudaDeviceSynchronize();
    cudaMemcpy(d_imgIn, imgIn, imgSize * sizeof(float),
            cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, gaussian, kernelsize * sizeof(float),
            cudaMemcpyHostToDevice);
    
    CUDA_CHECK;

    dim3 block = dim3(32, 8, 1);
    dim3 grid = dim3((w + block.x - 1) / block.x,
            (h + block.y - 1) / block.y, (nc));

    convolutionkernel <<<grid,block>>> (d_imgIn, d_imgOut, d_kernel, w, h, nc, radius);

    // calculate the partial derivatives of the blurred image
    rotationaldevkernel_x <<<grid,block>>> (d_imgOut, d_xOut , w, h, nc);
    rotationaldevkernel_y <<<grid,block>>> (d_imgOut, d_yOut, w, h, nc);

    // now calculate M
    calculate_M <<<grid,block>>> (d_xOut, d_yOut, d_m1, d_m2, d_m3, w, h, nc);

    convolutionkernel <<<grid,block>>> (d_m1, d_m1Out, d_kernel, w, h, nc, radius);
    convolutionkernel <<<grid,block>>> (d_m2, d_m2Out, d_kernel, w, h, nc, radius);
    convolutionkernel <<<grid,block>>> (d_m3, d_m3Out, d_kernel, w, h, nc, radius);

    // copy the input image
    cudaMemcpy(d_imgOut, imgIn, imgSize * sizeof(float),
            cudaMemcpyHostToDevice);

    // edge detector
    edgedetector <<<grid,block>>> (d_m1Out, d_m2Out, d_m3Out, d_imgOut, w, h, nc, alpha, beta);

    cudaMemcpy(imgOut, d_imgOut, imgSize * sizeof(float),
            cudaMemcpyDeviceToHost);
    cudaMemcpy(m1Out, d_m1Out, imgSize * sizeof(float),
            cudaMemcpyDeviceToHost);
    cudaMemcpy(m2Out, d_m2Out, imgSize * sizeof(float),
            cudaMemcpyDeviceToHost);
    cudaMemcpy(m3Out, d_m3Out, imgSize * sizeof(float),
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
convert_layered_to_mat(m11, m1Out);
m11 *= scale;
showImage("m1", m11, 100 + w + 60, 100);

convert_layered_to_mat(m12, m2Out);
m12 *= scale;
showImage("m2", m12, 100 + w + 80, 100);

convert_layered_to_mat(m13, m3Out);
m13 *= scale;
showImage("m3", m13, 100 + w + 100, 100);



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
