# CUDA 3D Life

This program will create and render a three-dimensional version of Conway's Game of Life, generated using CUDA for parallelization, and rendered using LWJGL for Java.

### // TODO: auto-generated stub

* Set my system locale back to English so MSVC stops flooding the console with Unicode character warnings.
* Verify CUDA version works correctly (test it against serial w/ set random seed)
* Optimize parallelization (shared memory between threads in a block).
* Learn about warps and if using them is worth it for this project.
* Compress game data file.
* Make the renderer prettier.

### How to Use

**!! Warning !!
The serial version of this program will create a potentially large data file (5MB for 20 iterations of 64x64x64, for an example) in the local directory. Be careful running it with large argument values!**

1. Compile Serial3DLife.c.
2. Run with the following arguments: NUMBER_OF_ITERATIONS X_SIZE Y_SIZE Z_SIZE INITIAL_ALIVE_CHANCE NUM_NEIGHBORS_ALIVE_THRESH_LOW NUM_NEIGHBORS_ALIVE_THRESH_HIGH
3. Compile the visualizer with Java, linking the lwjgl libraries when compiling.
4. Run the visualizer with the path to the data file as its runtime argument.
5. Once the LWJGL window has started to render, use the left and right arrow keys to control the iteration number.


