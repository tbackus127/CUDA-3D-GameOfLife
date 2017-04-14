// ================================================================================================
// Tim Backus
// CIS 450 - High Performance Computing
// 3D Game of Life
// ================================================================================================

#define GOL_IO_FILENAME "gol3DOutput.dat"

#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <stdlib.h>

// ------------------------------------------------------------------------------------------------
// Creates and initializes the life grid.
// ------------------------------------------------------------------------------------------------
char ***createGrid(unsigned const int x, unsigned const int y, unsigned const int z) {

  // Allocate the 3D array
  char ***result;
  result = malloc(x * sizeof(char **));
  if(result == NULL) {
    printf("Out of memory.\n");
    return NULL;
  }
  int i;
  for(i = 0; i < x; i++) {
    result[i] = malloc(y * sizeof(char *));
    if(result[i] == NULL) {
      printf("Out of memory.\n");
      return NULL;
    }
    int j;
    for(j = 0; j < y; j++) {
      result[i][j] = malloc(z * sizeof(char));
      if(result[i][j] == NULL) {
        printf("Out of memory.\n");
        return NULL;
      }
      
      // Initialize all to 0
      int k;
      for(k = 0; k < z; k++) {
        result[i][j][k] = (char) 0;
      }
    }
  }
  
  return result;
}

// ------------------------------------------------------------------------------------------------
// Writes cells to alive or dead, randomly.
// ------------------------------------------------------------------------------------------------
void randomizeGrid(char*** grid, unsigned const int xsize, unsigned const int ysize,
                   unsigned const int zsize, unsigned const int chance) {
 
  srand(time(NULL));
  
  int i;
  for(i = 0; i < xsize; i++) {
    int j;
    for(j = 0; j < ysize; j++) {
      int k;
      for(k = 0; k < zsize; k++) {
        grid[i][j][k] = (char)((rand() % 100 <= chance) ? 1 : 0);
      }
    }
  }
  
}

// ------------------------------------------------------------------------------------------------
// Prints a 3D array.
// ------------------------------------------------------------------------------------------------
void print3DArray(char*** arr, unsigned const int x, unsigned const int y, unsigned const int z) {
  int i;
  for(i = 0; i < x; ++i) {
    printf("Dimension %d:\n", i);
    int j;
    for(j = 0; j < y; ++j) {
      int k;
      for(k = 0; k < z; ++k) {
        printf("%d ", arr[i][j][k]);
      }
      printf("\n");
    }
    printf("\n");
  }
}

// ------------------------------------------------------------------------------------------------
// Initializes the game data file.
// Line 1: <iteration count> <x-size> <y-size> <z-size>
// Line 2: Blank
// ------------------------------------------------------------------------------------------------
void initGameFile(const unsigned int itrs, const unsigned int x, const unsigned int y,
                  const unsigned int z) {
  FILE *fp;
  fp = fopen(GOL_IO_FILENAME, "w+");
  fprintf(fp, "%d %d %d %d\n\n", itrs, x, y, z);
  fclose(fp);
}

// ------------------------------------------------------------------------------------------------
// Writes a game to a file for visualization within Java.
// For every iteration, a block of text is created of the format:
//   "<x-coord>:<z-coords for y=0>, <z-coords for y=1>, ..."
//   Z-coords are represented by a 0 or 1 for each z-coordinate
// Example: Game with 5 iterations, x=3, y=7, z=4
//   5 3 7 4
//   
//   0:0000,0000,0000,0000,0000,0000,0000
//   1:0000,0000,0000,0100,0000,0000,0001
//   2:0000,0000,0010,0100,0001,0011,0000
//   0:0000,0000,0000,0000,0000,0000,0000
//   1:0000,0000,0000,0100,0000,0000,0001
//   2:0000,0000,0010,0100,0001,0011,0000
//   0:0000,0000,0000,0000,0000,0000,0000
//   1:0000,0000,0000,0100,0000,0000,0001
//   2:0000,0000,0010,0100,0001,0011,0000
//   0:0000,0000,0000,0000,0000,0000,0000
//   1:0000,0000,0000,0100,0000,0000,0001
//   2:0000,0000,0010,0100,0001,0011,0000
//   0:0000,0000,0000,0000,0000,0000,0000
//   1:0000,0000,0000,0100,0000,0000,0001
//   2:0000,0000,0010,0100,0001,0011,0000
// 
// ------------------------------------------------------------------------------------------------
void writeGameStep(char*** arr, unsigned const int x, unsigned const int y, unsigned const int z) {
  FILE *fp;
  fp = fopen(GOL_IO_FILENAME, "a");
  
  int i;
  for(i = 0; i < x; i++) {
    fprintf(fp, "%d:", i);
    int j;
    for(j = 0; j < y; j++) {
      
      if(j > 0) {
        fprintf(fp, ",");
      }
      // Print Z-Dim values
      int k;
      for(k = 0; k < z; k++) {
        fprintf(fp, "%d", arr[i][j][k]);
      }
    }
    fprintf(fp, "\n");
  }
  fclose(fp);
}

// ------------------------------------------------------------------------------------------------
// Copies the values of a 3D array to another of the same size.
// ------------------------------------------------------------------------------------------------
void copy3DArray(char*** from, char*** to, const unsigned int xsize, const unsigned int ysize,
                 const unsigned int zsize) {
  int i;
  for(i = 0; i < xsize; i++) {
    int j;
    for(j = 0; j < ysize; j++) {
      int k;
      for(k = 0; k < zsize; k++) {
        to[i][j][k] = (char)from[i][j][k];
      }
    }
  }
}

// ------------------------------------------------------------------------------------------------
// Frees a 3D array.
// ------------------------------------------------------------------------------------------------
void free3DArray(char*** arr, const unsigned int x, const unsigned int y) {
  int i;
  for(i = 0; i < x; i++) {
    int j;
    for(j = 0; j < y; j++) {
      free(arr[i][j]);
    }
    free(arr[i]);
  }
  free(arr);
}

// ------------------------------------------------------------------------------------------------
// Sums the number of neighbors a cell has in a 3x3x3 cube.
// ------------------------------------------------------------------------------------------------
char sumNeighbors(char*** arr, const unsigned int cx, const unsigned int cy, const unsigned int cz,
                 const unsigned int xsize, const unsigned int ysize, const unsigned int zsize) {
  
  char result = 0;

  // X axis
  int i;
  for(i = cx - 1; i <= cx + 1; i++) {
    
    int realx = i;
    
    // Wrap around for X
    if(i < 0) realx = xsize - 1;
    if(i >= xsize) realx = 0;
    
    // Y axis
    int j;
    for(j = cy - 1; j <= cy + 1; j++) {
    
      // Wrap around for Y
      int realy = j;
      if(j < 0) realy = ysize - 1;
      if(j >= ysize) realy = 0;
      
      // Z axis
      int k;
      for(k = cz - 1; k <= cz + 1; k++) {
        
        // Wrap around for Z
        int realz = k;
        if(k < 0) realz = zsize - 1;
        if(k >= zsize) realz = 0;

        
        // If there is a cell alive at this position, add it to the total
        if(arr[realx][realy][realz]) {
          result++;
        }
      }
    }
  }
  
  return result;
}

// ------------------------------------------------------------------------------------------------
// Runs the Game of Life.
// ------------------------------------------------------------------------------------------------
void runLife(const unsigned int iterations, const unsigned int xsize, const unsigned int ysize, 
             const unsigned int zsize, const unsigned int initc, const unsigned int alow,
             const unsigned int ahigh) {
  
  printf("Creating initial game state... ");
  char ***grid = createGrid(xsize, ysize, zsize);
  if(grid == NULL) {
    printf("Grid is null!");
    free3DArray(grid, xsize, ysize);
    return;
  }
  
  randomizeGrid(grid, xsize, ysize, zsize, initc);
  
  printf("DONE\n");
  initGameFile(iterations, xsize, ysize, zsize);
  
  int itrNum;
  for(itrNum = 0; itrNum < iterations; ++itrNum) {
    printf(">> Iteration %d\n", itrNum);
    
    char ***tempGrid = createGrid(xsize, ysize, zsize);
    // printf("  Created temporary grid\n");
    
    int i;
    for(i = 0; i < xsize; i++) {
      
      int j;
      for(j = 0; j < ysize; j++) {
        
        int k;
        for(k = 0; k < zsize; k++) {
          // printf("    Getting neighbors of (%d,%d,%d): ", i, j, k);
          const unsigned int numNeighbors = sumNeighbors(grid, i, j, k, xsize, ysize, zsize);
          // printf("%d\n", numNeighbors);
          if (numNeighbors < alow || numNeighbors > ahigh) {
            tempGrid[i][j][k] = 0;
          } else if (numNeighbors >= alow && numNeighbors <= ahigh) {
            tempGrid[i][j][k] = 1;
          }
        }
      }
    }
    
    // DEBUG
    // printf("\nTemp array:\n");
    // print3DArray(tempGrid, xsize, ysize, zsize);
    
    // Once calculations have been completed, copy the temp grid to the current and destroy it
    copy3DArray(tempGrid, grid, xsize, ysize, zsize);
    free3DArray(tempGrid, xsize, ysize);
    
    // printf("\nCopied array:\n");
    // print3DArray(grid, xsize, ysize, zsize);
    // WARNING: The file can get pretty big for large game areas (260MB for 1000itr 64x64x64).
    // I will try to find a way to compact it later, but it is low priority as of now.
    writeGameStep(grid, xsize, ysize, zsize);
  }
  
  free3DArray(grid, xsize, ysize);
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
  if(argc <= 1 || argc > 8) {
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
  
  
  printf("Starting %d iteration Game of Life with sizes x=%d, y=%d, z=%d\n", iterations,
         sizeX, sizeY, sizeZ);
  runLife(iterations, sizeX, sizeY, sizeZ, initChance, aliveLow, aliveHigh);
  
  return EXIT_SUCCESS;
}