#include <time.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <cuda_runtime.h>

#include <gameOfLife.h>
#include <BoardVisualization.hpp>

#include "aux.h"

#define BOARD_SIZE_X 200
#define BOARD_SIZE_Y 200
#define CIRCLE_RADIUS 2


void initBoardAtRandom(cv::Mat& board) {
    srand(time(NULL));
    for (int i = 0; i < board.rows; i++) {
        for (int j = 0; j < board.cols; j++) {
            board.at<unsigned char>(i, j) = static_cast<unsigned char>(rand() % 2);
        }
    }
}


int main(int argc, const char *argv[])
{
    unsigned char* d_src;
    unsigned char* d_dst;

    cv::Mat board = cv::Mat::zeros(BOARD_SIZE_X, BOARD_SIZE_Y, CV_8UC1);
    BoardVisualization viewer(BOARD_SIZE_X, BOARD_SIZE_Y, CIRCLE_RADIUS);

    // Initialize the board randomly
    initBoardAtRandom(board);

    // pointer to the board array
    unsigned char* h_src = board.data;

    /**
     *  YOUR CODE HERE
     *
     *  Here you should perform the proper device memory operations.
     *  Allocate memory for d_src and d_dst.
     *  Copy the initial board from h_dst to d_src.
     *
     */

     // for (int i = 0; i < 10; i++) {
     //        std::cout << "check: " << (h_src[i] == 0) << std::endl;
     // }

     const int ARRAY_BYTES = BOARD_SIZE_X * BOARD_SIZE_Y * sizeof(unsigned char); 

     // Allocate memory on the GPU
     cudaMalloc((void**) &d_src, ARRAY_BYTES);
     CUDA_CHECK;
     cudaMalloc((void**) &d_dst, ARRAY_BYTES);
     CUDA_CHECK;

     // Transfer host memory to device
     cudaMemcpy(d_src, h_src, ARRAY_BYTES, cudaMemcpyHostToDevice);
     CUDA_CHECK;

    int key;
    while (key = cv::waitKey(10)) {

        /**
         *  YOUR CODE HERE
         *
         *  Here you must perform one iteration of the Game of Life and do
         *  the proper memory operations to display the board.
         */

        runGameOfLifeIteration(d_src, d_dst, BOARD_SIZE_X, BOARD_SIZE_Y);

        cudaMemcpy(h_src, d_dst, ARRAY_BYTES, cudaMemcpyDeviceToHost);
        cudaMemcpy(d_src, d_dst, ARRAY_BYTES, cudaMemcpyDeviceToDevice);


        /** This is just for display. You should not touch this.  **/
        viewer.displayBoard(board);
        if (key != -1) break;
    }
 
    /**
     *  YOUR CODE HERE
     *
     *  Remember to free the device memory.
     */

     cudaFree(d_src);
     CUDA_CHECK;
     cudaFree(d_dst);
     CUDA_CHECK;


    return 0;
}

