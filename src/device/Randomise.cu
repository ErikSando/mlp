#include <iostream>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "device/DeviceContext.hpp"

#define CUDA_ERROR(err, message)\
    do {\
        std::cout << "\033[31m" << "[Error]\033[0m "\
                  << message\
                  << cudaGetErrorString(err) << '\n'\
                  << "File: " << __FILE__ << '\n'\
                  << "Line: " << __LINE__ << '\n'\
                  << "Function: " << __func__ << '\n';\
        std::abort();\
    } while (0)

__global__ void init_rng(curandState* states, const unsigned long seed, const size_t size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= size) return;

    curand_init(seed, index, 0, &states[index]);
}

__global__ void randomize_kernel(float* M, curandState* states, size_t size, float min, float max) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= size) return;

    M[index] = curand_uniform(&states[index]) * (max - min) + min;
}

namespace mlp {
    void DeviceContext::randomise(Matrix& matrix, float min, float max) {
        cudaError_t err;

        int grid_size = block_count(matrix.size(), BLOCK_SIZE);

        static curandState* states = nullptr;
        static int allocated = 0;

        if (allocated < matrix.size()) {
            if (states) cudaFree(states);

            err = cudaMalloc(&states, matrix.size() * sizeof(curandState));
            if (err != cudaSuccess) CUDA_ERROR(err, "CUDA malloc error: ");

            init_rng<<<grid_size, BLOCK_SIZE>>>(states, time(nullptr), matrix.size());

            err = cudaGetLastError();
            if (err != cudaSuccess) CUDA_ERROR(err, "CUDA init RNG error: ");

            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) CUDA_ERROR(err, "CUDA device synchronise error: ");

            allocated = matrix.size();
        }

        randomize_kernel<<<grid_size, BLOCK_SIZE>>>(matrix.data(), states, matrix.size(), min, max);

        err = cudaGetLastError();
        if (err != cudaSuccess) CUDA_ERROR(err, "CUDA matrix randomise error: ");

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) CUDA_ERROR(err, "CUDA device synchronise error: ");
    }
}