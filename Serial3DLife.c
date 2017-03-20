// ================================================================================================
// Tim Backus
// CIS 450 - High Performance Computing
// 3D Game of Life
// ================================================================================================

#define GOL_OPTION_DEATH_THRESHOLD_LOW 2
#define GOL_OPTION_DEATH_THRESHOLD_HIGH 5

#include <stdio.h>
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
  for(i = 0; i < y; i++) {
    result[i] = malloc(y * sizeof(char *));
    if(result[i] == NULL) {
      printf("Out of memory.\n");
      return NULL;
    }
    int j;
    for(j = 0; j < z; j++) {
      result[i][j] = malloc(z * sizeof(char));
      if(result[i][j] == NULL) {
        printf("Out of memory.\n");
        return NULL;
      }
    }
  }
  
  // Fill the grid with either a 0 or 1
  srand(time(NULL));
  for(i = 0; i < x; i++) {
    int j;
    for(j = 0; j < y; j++) {
      int k;
      for(k = 0; k < z; k++) {
        result[i][j][k] = rand() & 1;
      }
    }
  }
  
  return result;
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

  int i;
  for(i = cx - 1; i <= cx + 1; i++) {
    int j;
    for(j = cy - 1; j <= cy + 1; j++) {
      int k;
      for(k = cz - 1; k <= cz + 1; k++) {
        
        // Ignore edge conditions for now
        if(i < 0 || j < 0 || k < 0 || i >= xsize || j >= ysize || k >= zsize) {
          continue;
        }
        
        // If there is a cell alive at this position
        if(arr[i][j][k]) {
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
void runLife(const int iterations, unsigned const int xsize, unsigned const int ysize, 
             unsigned const int zsize) {
  
  char ***grid = createGrid(xsize, ysize, zsize);
  if(grid == NULL) {
    return;
  }
  
  int itrNum;
  for(itrNum = 0; itrNum < iterations; ++itrNum) {
    printf(">> Iteration %d:\n\n", itrNum);
    
    int i;
    for(i = 0; i < xsize; i++) {
      
      int j;
      for(j = 0; j < ysize; j++) {
        
        int k;
        for(k = 0; k < zsize; k++) {
          const unsigned int numNeighbors = sumNeighbors(grid, i, j, k, xsize, ysize, zsize);
          if (numNeighbors <= GOL_OPTION_DEATH_THRESHOLD_LOW || 
              numNeighbors >= GOL_OPTION_DEATH_THRESHOLD_HIGH) {
            grid[i][j][k] = 0;
          }
        }
      }
    }
    
    print3DArray(grid, xsize, ysize, zsize);
  }
  
  free3DArray(grid, xsize, ysize);
}

// ------------------------------------------------------------------------------------------------
// Prints the usage message if a bad number of runtime arguments are passed.
// ------------------------------------------------------------------------------------------------
void printUsage() {
  printf("Usage: <program> MAX_ITERATIONS, SIZE_X, SIZE_Y, SIZE_Z\n");
}

// ------------------------------------------------------------------------------------------------
// Main Method
// ------------------------------------------------------------------------------------------------
int main(int argc, char *argv[]) {
  
  // Ensure proper runtime argument count
  if(argc <= 1 || argc > 5) {
    printUsage();
  }
  
  // Parse iteration count
  unsigned const int iterations = atoi(argv[1]);
  
  // Parse X-Size
  unsigned const int sizeX = atoi(argv[2]);
  
  // Parse Y-Size
  unsigned const int sizeY = atoi(argv[3]);
  
  // Parse Z-Size
  unsigned const int sizeZ = atoi(argv[4]);
  
  printf("Starting %d iteration Game of Life with sizes x=%d, y=%d, z=%d\n", iterations,
         sizeX, sizeY, sizeZ);
  runLife(iterations, sizeX, sizeY, sizeZ);
  
  return 0;
}