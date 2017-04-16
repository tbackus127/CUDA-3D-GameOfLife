@echo off
gcc Serial3DLife.c -o Serial3DLife.exe
nvcc Cuda3DLife.cu -o Cuda3DLife.exe
del Cuda3DLife.lib
del Cuda3DLife.exp