#include <cassert>
#include <cmath>

#include <cuda_runtime.h>

#include "device/Activation.cuh"
// #include "device/Propagation.cuh"
#include "device/Softmax.cuh"
#include "device/DeviceContext.hpp"

namespace mlp {
    struct NoActivation {
        __device__ static float activate(float x) {
            return x;
        }
    };

    struct Sigmoid {
        __device__ static float activate(float x) {
            return 1.0f / (1.0f + expf(-x));
        }
    };

    struct Tanh {
        __device__ static float activate(float x) {
            return tanhf(x);
        }
    };

    struct ReLU {
        __device__ static float activate(float x) {
            return x >= 0.0f ? x : 0.0f;
        }

        __device__ static float derivative(float x) {
            return x >= 0.0f ? 1.0f : 0.0f;
        }
    };

    struct LeakyReLU {
        __device__ static float activate(float x) {
            return x >= 0.0f ? x : 0.01f * x;
        }

        __device__ static float derivative(float x) {
            return x >= 0.0f ? 1.0f : 0.01f;
        }
    };

    template<typename TActivation = NoActivation>
    __global__ void propagate_kernel(
        const float* input, float* output, const float* weights, const float* biases,
        const size_t batch_size, const size_t input_count, const size_t output_count
    ) {
        unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
        unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;

        if (row >= batch_size || col >= output_count) return; 

        float value = biases[col];

        for (unsigned int k = 0; k < input_count; k++) {
            value += input[row * input_count + k] * weights[k * output_count + col];
        }

        output[row * output_count + col] = TActivation::activate(value);
    }

    void DeviceContext::propagate(const Matrix& input, Matrix& output, const Matrix& weights, const Matrix& biases, const Activation activation) const {
            assert(input.columns() == weights.rows());
            assert(output.rows() == input.rows());
            assert(output.columns() == weights.columns());

            cudaError_t err;

            dim3 block(TILE_SIZE, TILE_SIZE);

            dim3 grid(
                block_count(weights.columns(), block.x),
                block_count(input.rows(), block.y)
            );

            CUDATaskID task;
            if (m_profiler) task = m_profiler->startTask("Layer propagation");

            switch (activation) {
                case Activation::SIGMOID:
                    propagate_kernel<Sigmoid><<<grid, block>>>(input.data(), output.data(), weights.data(), biases.data(), input.rows(), input.columns(), output.columns());
                break;

                case Activation::TANH:
                    propagate_kernel<Tanh><<<grid, block>>>(input.data(), output.data(), weights.data(), biases.data(), input.rows(), input.columns(), output.columns());
                break;

                case Activation::RELU:
                    propagate_kernel<ReLU><<<grid, block>>>(input.data(), output.data(), weights.data(), biases.data(), input.rows(), input.columns(), output.columns());
                break;

                case Activation::LEAKY_RELU:
                    propagate_kernel<LeakyReLU><<<grid, block>>>(input.data(), output.data(), weights.data(), biases.data(), input.rows(), input.columns(), output.columns());
                break;

                case Activation::SOFTMAX:
                    propagate_kernel<<<grid, block>>>(input.data(), output.data(), weights.data(), biases.data(), input.rows(), input.columns(), output.columns());

                    CUDATaskID sm_task;
                    if (m_profiler) sm_task = m_profiler->startTask("Softmax"); // this is like a sub task, it overlaps with layer propagation, but i havent accounted for this in the profiler

                    softmax_kernel<<<input.rows(), BLOCK_SIZE>>>(output.data(), output.data(), output.rows(), output.columns());

                    if (m_profiler) m_profiler->endTask(sm_task);
                break;
            }

            if (m_profiler) m_profiler->endTask(task);

            err = cudaGetLastError();
            if (err != cudaSuccess) CUDA_ERROR(err, "CUDA layer propagation error: ");
        }
}