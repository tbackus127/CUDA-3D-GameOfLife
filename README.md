# CUDA 3D Life

This program simulates a three-dimensional version of Conway's Game of Life, generated using CUDA for parallelization, and rendered using LWJGL for Java.

## How to Use

1. Compile Serial3DLife.c with a C compiler and Cuda3DLife.cu with nvcc. If you are using Windows and have everything set up already, simply run the build.bat file.
2. Run with the following arguments (separated by spaces): 
    1. Number of iterations to run
    2. X-Size
    3. Y-Size
    4. Z-Size
    5. Initial chance (1 to 100) that a given cell will be alive at the start of the game
    6. The lowest (inclusive) number of neighbors a cell needs to be alive
    7. The highest (inclusive) number of neighbors a cell needs to be alive
    8. Whether or not to print the array as the simulation happens (0 = do not print, 1 = print)
    9. Whether or not to write the entire game out to a file (0 = do not output, 1 = write to file)
    * Warning!: Be careful not to accidentally output to file with large game worlds! *
3. Compile the visualizer with Java, linking the lwjgl libraries when compiling.
4. Run the visualizer with the path to the data file as its runtime argument.
5. Once the LWJGL window has started to render, use the left and right arrow keys to control the iteration number.

## Performance

All times are measured in ticks from time.h's clock() method and are averaged together.

### 64x64x64 Grid

* Serial version: 45 ticks per iteration.
* CUDA version (unoptimized): 1 tick per iteration.
* CUDA version (optimized): 1 tick per iteration.

### 128x128x128 Grid

* Serial version: 350 ticks per iteration.
* CUDA version (unoptimized): 4 ticks per iteration.
* CUDA version (optimized): 2 ticks per iteration.

### 256x256x256 Grid

* Serial version: 2780 ticks per iteration.
* CUDA version (unoptimized): 27 ticks per iteration.
* CUDA version (optimized): 19 ticks per iteration.

### 512x512x512 Grid

* Serial version: 24000 ticks per iteration.
* CUDA version (unoptimized): 205 ticks per iteration.
* CUDA version (optimized): 133 ticks per iteration.

### 1024x1024x1024 Grid

* Serial version: My computer runs out of memory while allocating the temporary grid.
* CUDA version (unoptimized): 1750 ticks per iteration.
* CUDA version (optimized): 1056 ticks per iteration.
