// ================================================================================================
// Tim Backus
// CIS 450 - High Performance Computing
// 3D Game of Life - CUDA Version
// ================================================================================================

#define GOL_IO_FILENAME "gol3DOutput.dat"
#define GOL_CUDA_THREADS_PER_BLOCK 32

#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <stdlib.h>

// ------------------------------------------------------------------------------------------------
// CUDA kernel (Gather) - Adds up the number of neighbors for a cell in a 3x3x3 cube.
// ------------------------------------------------------------------------------------------------
__global__
void sumNeighborsKernel(const char* const d_in, char* d_out, const unsigned int xs,
                        const unsigned int ys, const unsigned int zs) {
  
  // Calculate block and thread IDs
  const int threadPosX = blockIdx.x * blockDim.x + threadIdx.x;
  const int threadPosY = blockIdx.y * blockDim.y + threadIdx.y;
  const int threadPosZ = blockIdx.z * blockDim.z + threadIdx.z;
  const int stepX = xs * ys;
  const int arrayPos = threadPosX * stepX + threadPosY * ys + threadPosZ;
  
  // Ensure thread bounds
  if(threadPosX > xs - 1) return;
  if(threadPosY > ys - 1) return;
  if(threadPosZ > zs - 1) return;
  
  // X-Axis neighbors
  unsigned char sum = 0;
  int i, realx;
  for(i = arrayPos - stepX; i <= arrayPos + stepX; i += stepX) {
    
    // Wrap X-Axis
    realx = i;
    if(i > xs) {
      realx = threadPosY * ys + threadPosZ;
    } else if(i < 0) {
      realx = (xs - 1) * stepX + threadPosY * ys + threadPosZ;
    }
    
    // Y-Axis neighbors
    int j, realy;
    for(j = arrayPos - ys; j <= arrayPos + ys; j += ys) {
      
      // Wrap Y-Axis
      realy = j;
      if(j > ys) {
        realy = threadPosZ;
      } else if(j < 0) {
        realy = (ys - 1) * ys + threadPosZ;
      }
      
      // Z-Axis neighbors
      int k, realz;
      for(k = arrayPos - 1; k <= arrayPos + 1; k++) {
        
        // Wrap Z-Axis
        realz = k;
        if(k > zs) {
          realz = 0;
        } else if(k < 0) {
          realz = zs - 1;
        }
        
        sum += d_in[realx * stepX + realy * ys + realz];
      }
    }
  }
  
  d_out[arrayPos] = sum;
}

// ------------------------------------------------------------------------------------------------
// CUDA kernel (Map) - Sets each cell to alive or dead depending on its number of neighbors and
//   the rules for this current game.
// ------------------------------------------------------------------------------------------------
__global__
void setAliveDeadKernel(const char* const d_nei, char* d_out, const unsigned int xs, 
                        const unsigned int ys, const unsigned int zs, const unsigned int alow, 
                        const unsigned int ahigh) {
  
  // Calculate block and thread IDs
  const int threadPosX = blockIdx.x * blockDim.x + threadIdx.x;
  const int threadPosY = blockIdx.y * blockDim.y + threadIdx.y;
  const int threadPosZ = blockIdx.z * blockDim.z + threadIdx.z;
  const int stepX = xs * ys;
  const int arrayPos = threadPosX * stepX + threadPosY * ys + threadPosZ;
  
  // Ensure thread bounds
  if(threadPosX > xs - 1) return;
  if(threadPosY > ys - 1) return;
  if(threadPosZ > zs - 1) return;
  
  // Set the cell alive or dead as according to the rules
  if (d_nei[arrayPos] < alow || d_nei[arrayPos] > ahigh) {
    d_out[arrayPos] = 0;
  } else if (d_nei[arrayPos] >= alow && d_nei[arrayPos] <= ahigh) {
    d_out[arrayPos] = 1;
  }
}

// ------------------------------------------------------------------------------------------------
// Returns the 1D position of a simulated 3D array
// ------------------------------------------------------------------------------------------------
int getArrIndex(const unsigned int xp, const unsigned int yp, const unsigned int zp,
                const unsigned int xs, const unsigned int ys) {
  return xp * xs * ys + ys * yp + zp;
}

// ------------------------------------------------------------------------------------------------
// Writes cells to alive or dead, randomly.
// ------------------------------------------------------------------------------------------------
void randomizeGrid(char* grid, unsigned const int size, unsigned const int chance) {
 
  srand(time(NULL));
  int i;
  for(i = 0; i < size; i++) {
    grid[i] = (char)((rand() % 100 <= chance) ? 1 : 0);
  }
  
}

// ------------------------------------------------------------------------------------------------
// Runs the Game of Life.
// ------------------------------------------------------------------------------------------------
void runLife(const unsigned int iterations, const unsigned int xsize, const unsigned int ysize, 
             const unsigned int zsize, const unsigned int initc, const unsigned int alow,
             const unsigned int ahigh, const unsigned int outputToFile) {
  
  // Memory values
  const int arrSize = xsize * ysize * zsize;
  const int arrMem = arrSize * sizeof(char);
  
  // GPU grid dimensions
  const int gx = (int) ceil(xsize / GOL_CUDA_THREADS_PER_BLOCK);
  const int gy = (int) ceil(ysize / GOL_CUDA_THREADS_PER_BLOCK);
  const int gz = (int) ceil(zsize / GOL_CUDA_THREADS_PER_BLOCK);
  dim3 gridDim(gx, gy, gz);
  
  // GPU thread dimensions
  const int tx = (xsize >= GOL_CUDA_THREADS_PER_BLOCK) ? GOL_CUDA_THREADS_PER_BLOCK : xsize;
  const int ty = (ysize >= GOL_CUDA_THREADS_PER_BLOCK) ? GOL_CUDA_THREADS_PER_BLOCK : ysize;
  const int tz = (zsize >= GOL_CUDA_THREADS_PER_BLOCK) ? GOL_CUDA_THREADS_PER_BLOCK : zsize;
  dim3 blockDim(tx, ty, tz);
  
  // Initialize game space
  char *h_in = (char *) malloc(arrMem);
  randomizeGrid(h_in, arrSize, initc);
  
  // Number of neighbors
  char *h_nei = (char *) malloc(arrMem);

  // Allocate X-Size on GPU
  int d_xs;
  cudaMalloc((void **) &d_xs, sizeof(int));
  cudaMemcpy(&d_xs, &xsize, sizeof(int), cudaMemcpyHostToDevice);
  
  // Allocate Y-Size on GPU
  int d_ys;
  cudaMalloc((void **) &d_ys, sizeof(int));
  cudaMemcpy(&d_ys, &ysize, sizeof(int), cudaMemcpyHostToDevice);
  
  // Allocate Z-Size on GPU
  int d_zs;
  cudaMalloc((void **) &d_zs, sizeof(int));
  cudaMemcpy(&d_zs, &zsize, sizeof(int), cudaMemcpyHostToDevice);
  
  // Allocate neighbor count for alive low threshold on GPU
  int d_lw;
  cudaMalloc((void **) &d_lw, sizeof(int));
  cudaMemcpy(&d_lw, &alow, sizeof(int), cudaMemcpyHostToDevice);
  
  // Allocate neighbor count for alive low threshold on GPU
  int d_hg;
  cudaMalloc((void **) &d_hg, sizeof(int));
  cudaMemcpy(&d_hg, &ahigh, sizeof(int), cudaMemcpyHostToDevice);
  
  // Pointers for GPU game data
  char *d_in;
  char *d_out;
  
  // Allocate input array on GPU
  cudaMalloc(&d_in, arrMem);
  
  // Allocate output array on GPU
  cudaMalloc(&d_out, arrMem);
  
  // Do Game of Life iterations
  int itrNum;
  for(itrNum = 0; itrNum < iterations; itrNum++) {
    
    printf("Iteration %d ", itrNum);
    
    clock_t start = clock();
    
    // Run the kernel to add neighbors of all cells
    cudaMemcpy(d_in, h_in, arrMem, cudaMemcpyHostToDevice);
    sumNeighborsKernel<<<gridDim, blockDim>>>(d_in, d_out, d_xs, d_ys, d_zs);
    cudaMemcpy(h_nei, d_out, arrMem, cudaMemcpyDeviceToHost);
    
    // Run the kernel to set the cells' alive or dead states
    cudaMemcpy(d_in, h_nei, arrMem, cudaMemcpyHostToDevice);
    setAliveDeadKernel<<<gridDim, blockDim>>>(d_in, d_out, d_xs, d_ys, d_zs, d_lw, d_hg);
    cudaMemcpy(h_in, d_out, arrMem, cudaMemcpyDeviceToHost);
    
    clock_t end = clock();
    
    printf("took %d ticks.\n", (end - start));
  }
  
  // Free memory
  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(&d_xs);
  cudaFree(&d_ys);
  cudaFree(&d_zs);
  cudaFree(&d_lw);
  cudaFree(&d_hg);
  free(h_in);
}

// ------------------------------------------------------------------------------------------------
// Prints the usage message if a bad number of runtime arguments are passed.
// ------------------------------------------------------------------------------------------------
void printUsage() {
  printf("Usage: <program> MAX_ITERATIONS, SIZE_X, SIZE_Y, SIZE_Z,\nINITIAL_ALIVE_CHANCE, ");
  printf("  ALIVE_THRESHOLD_LOW (inclusive), ALIVE_THRESHOLD_HIGH (inclusive)");
}

// ------------------------------------------------------------------------------------------------
// Main Method
// ------------------------------------------------------------------------------------------------
int main(int argc, char *argv[]) {
  
  // Ensure proper runtime argument count
  if(argc <= 1 || argc > 9) {
    printUsage();
    return EXIT_SUCCESS;
  }
  
  // Parse iteration count
  unsigned const int iterations = atoi(argv[1]);
  
  // Parse X-Size
  unsigned const int sizeX = atoi(argv[2]);
  
  // Parse Y-Size
  unsigned const int sizeY = atoi(argv[3]);
  
  // Parse Z-Size
  unsigned const int sizeZ = atoi(argv[4]);
  
  // Parse initial alive chance
  unsigned const int initChance = atoi(argv[5]);
  
  // Parse alive low threshold (inclusive)
  unsigned const int aliveLow = atoi(argv[6]);
  
  // Parse alive high threshold (inclusive)
  unsigned const int aliveHigh = atoi(argv[7]);
  
  // Parse whether or not to output to file (0 or 1)
  unsigned const int outputEnabled = atoi(argv[8]);
  
  printf("Starting %d iteration Game of Life (CUDA) with sizes x=%d, y=%d, z=%d\n", iterations,
         sizeX, sizeY, sizeZ);
  runLife(iterations, sizeX, sizeY, sizeZ, initChance, aliveLow, aliveHigh, outputEnabled);
  
  return EXIT_SUCCESS;
}