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

__global__
void sumNeighborsKernel(const char* const d_in, char* d_out, const unsigned int xs,
                        const unsigned int ys, const unsigned int zs) {
  
  // Calculate block and thread IDs
  const int threadPosX = blockIdx.x * blockDim.x + threadIdx.x;
  const int threadPosY = blockIdx.y * blockDim.y + threadIdx.y;
  const int threadPosZ = blockIdx.z * blockDim.z + threadIdx.z;
  const int stepX = xs * ys;
  const int arrayPos = threadPosX * stepX + threadPosY * ys + threadPosZ;
  
  unsigned char sum = 0;
  int i;
  for(i = arrayPos - stepX; i <= arrayPos + stepX; i += stepX) {
    int j;
    for(j = arrayPos - ys; j <= arrayPos + ys; j += ys) {
      int k;
      for(k = arrayPos - 1; k <= arrayPos + 1; k++) {
        sum += d_in[arrayPos];
      }
    }
  }
  
  d_out[arrayPos] = sum;
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
  
  const int arrSize = xsize * ysize * zsize;
  const int arrMem = arrSize * sizeof(char);
  
  // GPU grid dimensions GOL_CUDA_THREADS_PER_BLOCK
  const int gx = xsize / GOL_CUDA_THREADS_PER_BLOCK;
  const int gy = ysize / GOL_CUDA_THREADS_PER_BLOCK;
  const int gz = zsize / GOL_CUDA_THREADS_PER_BLOCK;
  dim3 grid(gx, gy, gz);
  
  // GPU thread dimensions
  const int tx = GOL_CUDA_THREADS_PER_BLOCK;
  const int ty = GOL_CUDA_THREADS_PER_BLOCK;
  const int tz = GOL_CUDA_THREADS_PER_BLOCK;
  dim3 block(tx, ty, tz);
  
  // Initialize game space
  char *h_in = (char *) malloc(arrMem);
  randomizeGrid(h_in, arrSize, initc);

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
  
  // Pointers for GPU game data
  char *d_in;
  char *d_out;
  
  // Allocate input array on GPU
  cudaMalloc(&d_in, arrMem);
  
  // Allocate output array on GPU
  cudaMalloc(&d_out, arrMem);
  
  int itrNum;
  for(itrNum = 0; itrNum < iterations; itrNum++) {
    
    cudaMemcpy(d_in, h_in, arrMem, cudaMemcpyHostToDevice);
    
    // Run the kernel to add neighbors of all cells
    sumNeighborsKernel<<<grid, block>>>(d_in, d_out, d_xs, d_ys, d_zs);
    
    cudaMemcpy(h_in, d_out, arrMem, cudaMemcpyDeviceToHost);
  }
  
  // Free memory
  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(&d_xs);
  cudaFree(&d_ys);
  cudaFree(&d_zs);
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