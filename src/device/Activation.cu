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

__global__ void sigmoid_kernel(const float* input, float* output, const size_t size) {
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index >= size) return;

    output[index] = 1 / (1 + expf(-input[index]));
}

__global__ void tanh_kernel(const float* input, float* output, const size_t size) {
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index >= size) return;

    output[index] = tanhf(input[index]);
}

__global__ void relu_kernel(const float* input, float* output, const size_t size) {
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index >= size) return;

    output[index] = fmaxf(0, input[index]);
}

__global__ void leaky_relu_kernel(const float* input, float* output, const size_t size, const float a) {
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index >= size) return;

    output[index] = input[index] >= 0 ? input[index] : a * input[index];
}

namespace mlp {
    void DeviceContext::sigmoid(const Matrix& input, Matrix& output) const {
        assert(input.size() == output.size());
        assert(input.rows() == output.rows());
        assert(input.columns() == output.columns());

        cudaError_t err;

        unsigned int grid_size = block_count(input.size(), BLOCK_SIZE);

        sigmoid_kernel<<<grid_size, BLOCK_SIZE>>>(input.data(), output.data(), input.size());

        err = cudaGetLastError();
        if (err != cudaSuccess) CUDA_ERROR(err, "CUDA sigmoid error: ");

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) CUDA_ERROR(err, "CUDA device synchronise error: ");
    }

    void DeviceContext::tanh(const Matrix& input, Matrix& output) const {
        assert(input.size() == output.size());
        assert(input.rows() == output.rows());
        assert(input.columns() == output.columns());

        cudaError_t err;

        unsigned int grid_size = block_count(input.size(), BLOCK_SIZE);

        tanh_kernel<<<grid_size, BLOCK_SIZE>>>(input.data(), output.data(), input.size());

        err = cudaGetLastError();
        if (err != cudaSuccess) CUDA_ERROR(err, "CUDA tanh error: ");

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) CUDA_ERROR(err, "CUDA device synchronise error: ");
    }

    void DeviceContext::relu(const Matrix& input, Matrix& output) const {
        assert(input.size() == output.size());
        assert(input.rows() == output.rows());
        assert(input.columns() == output.columns());

        cudaError_t err;

        unsigned int grid_size = block_count(input.size(), BLOCK_SIZE);

        relu_kernel<<<grid_size, BLOCK_SIZE>>>(input.data(), output.data(), input.size());

        err = cudaGetLastError();
        if (err != cudaSuccess) CUDA_ERROR(err, "CUDA ReLU error: ");

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) CUDA_ERROR(err, "CUDA device synchronise error: ");
    }

    void DeviceContext::leakyReLU(const Matrix& input, Matrix& output) const {
        assert(input.size() == output.size());
        assert(input.rows() == output.rows());
        assert(input.columns() == output.columns());

        cudaError_t err;

        unsigned int grid_size = block_count(input.size(), BLOCK_SIZE);

        leaky_relu_kernel<<<grid_size, BLOCK_SIZE>>>(input.data(), output.data(), input.size(), 0.01f);

        err = cudaGetLastError();
        if (err != cudaSuccess) CUDA_ERROR(err, "CUDA leaky ReLU error: ");

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) CUDA_ERROR(err, "CUDA device synchronise error: ");
    }
}