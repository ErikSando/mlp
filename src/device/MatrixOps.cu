#include <cassert>
#include <iostream>

#include <cuda_runtime.h>

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

__global__ void mat_add_kernel(const float* A, const float* B, float* C, const size_t size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= size) return;

    C[index] = A[index] + B[index];
}

__global__ void mat_add_biases_kernel(const float* A, const float* B, float* C, const size_t M, const size_t N) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row >= M || col >= N) return;

    C[row * N + col] = A[row * N + col] + B[col];
}

namespace mlp {
    void DeviceContext::multiply(const Matrix& matrix_A, const Matrix& matrix_B, Matrix& matrix_C) {
        assert(matrix_A.columns() == matrix_B.rows());
        assert(matrix_C.rows() == matrix_A.rows());
        assert(matrix_C.columns() == matrix_B.columns());

        cudaError_t err;

        dim3 block(TILE_SIZE, TILE_SIZE);

        dim3 grid(
            block_count(matrix_B.columns(), block.x),
            block_count(matrix_A.rows(), block.y)
        );

        mat_mul_kernel<<<grid, block>>>(matrix_A.data(), matrix_B.data(), matrix_C.data(), matrix_A.rows(), matrix_B.columns(), matrix_A.columns());

        err = cudaGetLastError();
        if (err != cudaSuccess) CUDA_ERROR(err, "CUDA matrix multiplication error: ");

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) CUDA_ERROR(err, "CUDA device synchronise error: ");
    }

    void DeviceContext::add(const Matrix& matrix_A, const Matrix& matrix_B, Matrix& matrix_C) {
        assert(matrix_A.size() == matrix_B.size());
        assert(matrix_B.size() == matrix_C.size());
        assert(matrix_A.rows() == matrix_B.rows());
        assert(matrix_B.rows() == matrix_C.rows());
        assert(matrix_A.columns() == matrix_B.columns());
        assert(matrix_B.columns() == matrix_C.columns());

        cudaError_t err;

        int grid_size = (matrix_A.size() + BLOCK_SIZE - 1) / BLOCK_SIZE;

        mat_add_kernel<<<grid_size, BLOCK_SIZE>>>(matrix_A.data(), matrix_B.data(), matrix_C.data(), matrix_A.size());

        err = cudaGetLastError();
        if (err != cudaSuccess) CUDA_ERROR(err, "CUDA matrix addition error: ");

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) CUDA_ERROR(err, "CUDA device synchronise error: ");
    }

    void DeviceContext::addBiases(const Matrix& layer, const Matrix& biases, Matrix& output) {
        assert(layer.size() == output.size());
        assert(layer.rows() == output.rows());
        assert(layer.columns() == output.columns());
        assert(biases.rows() == 1);
        assert(biases.columns() == layer.columns());

        cudaError_t err;

        dim3 block(TILE_SIZE, TILE_SIZE);

        dim3 grid(
            block_count(output.columns(), block.x),
            block_count(output.rows(), block.y)
        );

        mat_add_biases_kernel<<<grid, block>>>(layer.data(), biases.data(), output.data(), output.rows(), output.columns());

        err = cudaGetLastError();
        if (err != cudaSuccess) CUDA_ERROR(err, "CUDA matrix addition error: ");

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) CUDA_ERROR(err, "CUDA device synchronise error: ");
    }
}