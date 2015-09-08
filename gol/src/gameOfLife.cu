  #include <cuda_runtime.h>
  #include <gameOfLife.h>
  #include "aux.h"

  #define BLOCK_SIZE_X 16
  #define BLOCK_SIZE_Y 16


  // Calculate number of blocks
  dim3 get_numBlocks(size_t w, size_t h, dim3 threadsPerBlock) {
    dim3 numBlocks( (w + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (h + threadsPerBlock.y - 1) / threadsPerBlock.y);

    return numBlocks;
  }

  __global__ void gameOfLifeKernel(unsigned char* d_src, unsigned char* d_dst, const size_t width, const size_t height) {

    /**
     *  YOUR CODE HERE
     *
     *  You must write here your kernel for one iteration of the game of life.
     *
     *  Input: d_src should contain the board at time 't'
     *  Output: d_dst should contain the board at time 't + 1' after one
     *  iteration of the game of life.
     *
     */


     int x = threadIdx.x + blockIdx.x * blockDim.x;
     int y = threadIdx.y + blockIdx.y * blockDim.y;
     // Calculate local index
     int ind = x + y * width;
     // Store number of neighbours
     int neighbours = 0;

     int neighbour_ind;
     // We are counting all neighbours and the cell itself
     for (int cellX = -1; cellX <= 1; cellX++) {
        for (int cellY = -1; cellY <= 1; cellY++) {
          neighbour_ind = (x + cellX) + (y + cellY) * width;
          // Check if we are still in board
          if (x + cellX < width && (x + cellX) >= 0) {
            if (y + cellY < height && (y + cellY) >= 0) {
              neighbours += d_src[neighbour_ind];
            }
          }
        }
     }

     // Rules
     if (x < width && y < height) {
        // if cell lives
        if (d_src[ind] == 1) {
          // Overcrowded
          if (neighbours > 4) {
            d_dst[ind] = 0;
          }
          // Perfect
          else if (neighbours > 2) {
            d_dst[ind] = 1;
          // Lonely
          } else {
            d_dst[ind] = 0;
          }
        // cell was dead
        } else {
          if (neighbours == 3) d_dst[ind] = 1;
        }
     }


  }

  void runGameOfLifeIteration(unsigned char* d_src, unsigned char* d_dst, const size_t width, const size_t height) {
      
    /**
     *  YOUR CODE HERE 
     *
     *  Here you must calculate the block size and grid size to latter call the
     *  gameOfLifeKernel.
     *
     */

     dim3 block = dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
     dim3 grid = get_numBlocks(width, height, block);

     gameOfLifeKernel<<<grid, block>>>(d_src, d_dst, width, height);

  }

