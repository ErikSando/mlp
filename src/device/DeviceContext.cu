#include <cassert>
#include <filesystem>
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

    if (index < size) {
        curand_init(seed, index, 0, &states[index]);
    }
}

__global__ void mat_mul_kernel(const float* A, const float* B, float* C, const size_t M, const size_t N, const size_t K) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row >= M || col >= N) return; 

    float value = 0.0f;

    for (int k = 0; k < K; k++) {
        value += A[row * K + k] * B[k * N + col];
    }

    C[row * N + col] = value;
}

__global__ void randomize_kernel(float* M, curandState* states, size_t size, float min, float max) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        M[index] = curand_uniform(&states[index]) * (max - min) + min;
    }
}

namespace mlp {
    void DeviceContext::multiply(Matrix& matrix_A, Matrix& matrix_B, Matrix& matrix_C) {
        assert(matrix_A.columns() == matrix_B.rows());
        assert(matrix_C.rows() == matrix_A.rows());
        assert(matrix_C.columns() == matrix_B.columns());

        cudaError_t err;

        dim3 block(16, 16);

        dim3 grid(
            (matrix_B.columns() + block.x - 1) / block.x,
            (matrix_A.rows() + block.y - 1) / block.y
        );

        mat_mul_kernel<<<grid, block>>>(matrix_A.data(), matrix_B.data(), matrix_C.data(), matrix_A.rows(), matrix_B.columns(), matrix_A.columns());

        err = cudaGetLastError();
        if (err != cudaSuccess) CUDA_ERROR(err, "CUDA matrix multiplication error: ");

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) CUDA_ERROR(err, "CUDA device synchronise error: ");
    }

    void DeviceContext::randomise(Matrix& matrix, float min, float max) {
        cudaError_t err;

        constexpr int blockSize = 256; // maybe dont hard code this
        int gridSize = (matrix.size() + blockSize - 1) / blockSize;

        static curandState* states = nullptr;
        static int allocated = 0;

        if (allocated < matrix.size()) {
            if (states) cudaFree(states);

            err = cudaMalloc(&states, matrix.size() * sizeof(curandState));
            if (err != cudaSuccess) CUDA_ERROR(err, "CUDA malloc error: ");

            init_rng<<<gridSize, blockSize>>>(states, time(nullptr), matrix.size());

            err = cudaGetLastError();
            if (err != cudaSuccess) CUDA_ERROR(err, "CUDA init RNG error: ");

            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) CUDA_ERROR(err, "CUDA device synchronise error: ");

            allocated = matrix.size();
        }

        randomize_kernel<<<gridSize, blockSize>>>(matrix.data(), states, matrix.size(), min, max);

        err = cudaGetLastError();
        if (err != cudaSuccess) CUDA_ERROR(err, "CUDA matrix randomise error: ");

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) CUDA_ERROR(err, "CUDA device synchronise error: ");
    }

    void DeviceContext::transfer(Matrix& src, float* dest) {
        cudaError_t err = cudaMemcpy(dest, src.data(), src.size() * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) CUDA_ERROR(err, "CUDA memcpy error: ");
    }

    void DeviceContext::transfer(float* src, Matrix& dest) {
        cudaError_t err = cudaMemcpy(dest.data(), src, dest.size() * sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) CUDA_ERROR(err, "CUDA memcpy error: ");
    }
}