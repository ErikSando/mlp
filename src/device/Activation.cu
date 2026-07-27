#include <cassert>
#include <iostream>

#include <cuda_runtime.h>

#include "device/DeviceContext.hpp"

namespace mlp {
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

    void DeviceContext::sigmoid(const Matrix& input, Matrix& output) const {
        assert(input.size() == output.size());
        assert(input.rows() == output.rows());
        assert(input.columns() == output.columns());

        cudaError_t err;

        unsigned int grid_size = block_count(input.size(), BLOCK_SIZE);

        sigmoid_kernel<<<grid_size, BLOCK_SIZE>>>(input.data(), output.data(), input.size());

        err = cudaGetLastError();
        if (err != cudaSuccess) CUDA_ERROR(err, "CUDA sigmoid error: ");
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
    }

    void DeviceContext::leakyReLU(const Matrix& input, Matrix& output) const {
        assert(input.size() == output.size());
        assert(input.rows() == output.rows());
        assert(input.columns() == output.columns());

        cudaError_t err;

        unsigned int grid_size = block_count(input.size(), BLOCK_SIZE);

        CUDATaskID task;
        if (m_profiler) task = m_profiler->startTask("Leaky ReLU");

        leaky_relu_kernel<<<grid_size, BLOCK_SIZE>>>(input.data(), output.data(), input.size(), 0.01f);

        if (m_profiler) m_profiler->endTask(task);

        err = cudaGetLastError();
        if (err != cudaSuccess) CUDA_ERROR(err, "CUDA leaky ReLU error: ");
    }
}