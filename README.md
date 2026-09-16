# Multilayer Perceptron
  A multilayer perceptron framework written from scratch in C++20 with support for CUDA and OpenCL.

## Features

  * Build MLP models with chosen layer sizes, activation functions, and loss function
  * Train and test models on compatible datasets, MNIST
  * Benchmarking and profiling for separate tasks such as layer propagation, softmax, layer optimisation, etc
  * Save and load model data to/from a save file
  * Classify individual sample files, with the loss value displayed if a label is given
  * Compute environment chosen at compile time (CUDA, OpenCL, CPU)

## Dependencies and Build Instructions

For all build types, CMake and a C++20 compatible compiler are required.

### CUDA

  * CUDA toolkit
  * NVIDIA GPU

  Build:
  ```
  cmake -B build/cuda -DMLP_COMPUTE=CUDA -DCMAKE_BUILD_TYPE=Debug
  cmake --build build/cuda
  ```
  Run:
  ```
  build/cuda/mlp
  ```

### OpenCL

  * OpenCL driver
  * OpenCL development headers

  Build:
  ```
  cmake -B build/opencl -DMLP_COMPUTE=OPENCL -DCMAKE_BUILD_TYPE=Debug
  cmake --build build/opencl
  ```
  Run:
  ```
  build/opencl/mlp
  ```

### CPU-only/Host-only

  * No additional dependencies.

  Build:
  ```
  cmake -B build/host -DMLP_COMPUTE=HOST -DCMAKE_BUILD_TYPE=Debug
  cmake --build build/host
  ```
  Run:
  ```
  build/host/mlp
  ```