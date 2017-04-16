#!/bin/bash

gcc Serial3DLife.c -o Serial3DLife
chmod 744 Serial3DLife

nvcc Cuda3DLife.cu -o Cuda3DLife
chmod 744 Cuda3DLife
rm Cuda3DLife.lib
rm Cuda3DLife.exp