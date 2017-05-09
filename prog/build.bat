@echo off
echo Compiling serial version...
gcc Serial3DLife.c -o Serial3DLife.exe
echo Compiling CUDA version...
nvcc Cuda3DLife.cu -o Cuda3DLife.exe
del Cuda3DLife.lib
del Cuda3DLife.exp
echo Compiling visualizer...
cd src
javac -cp .;lib\* com\tb\gol3d\Main.java
pause