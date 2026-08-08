#include <cassert>
#include <cmath>

#include <cuda_runtime.h>

#include "cuda/kernels/Activation.cuh"
#include "cuda/kernels/Softmax.cuh"
#include "cuda/Context.hpp"
#include "enums/Enums.hpp"

namespace mlp {
    namespace cuda {
        template<typename TActivation = NoActivation>
        __global__ void propagate_kernel_tiled(
            const float* inputs, float* logits, float* activations,
            const float* weights, const float* biases,
            const size_t batch_size, const size_t input_count, const size_t output_count
        ) {
            unsigned int tid = threadIdx.x;

            unsigned int thread_row = tid / TILE_SIZE;
            unsigned int thread_col = tid % TILE_SIZE;

            unsigned int cache_row = blockIdx.y;
            unsigned int cache_col = blockIdx.x;

            unsigned int row = cache_row * TILE_SIZE + thread_row;
            unsigned int col = cache_col * TILE_SIZE + thread_col;

            __shared__ float input_shared[TILE_SIZE * TILE_SIZE];
            __shared__ float weights_shared[TILE_SIZE * TILE_SIZE];

            inputs += cache_row * TILE_SIZE * input_count;
            weights += cache_col * TILE_SIZE;
            logits += cache_row * TILE_SIZE * output_count + cache_col * TILE_SIZE;
            activations += cache_row * TILE_SIZE * output_count + cache_col * TILE_SIZE;

            float temp = biases[col];

            for (unsigned int i = 0; i < input_count; i += TILE_SIZE) {
                float input_shared_value = 0.0f;
                float weights_shared_value = 0.0f;

                if (row < batch_size && i + thread_col < input_count) input_shared_value = inputs[thread_row * input_count + i + thread_col];
                if (col < output_count && i + thread_row < input_count) weights_shared_value = weights[(thread_row + i) * output_count + thread_col];

                input_shared[thread_row * TILE_SIZE + thread_col] = input_shared_value;
                weights_shared[thread_row * TILE_SIZE + thread_col] = weights_shared_value;

                __syncthreads();

                for (unsigned int di = 0; di < TILE_SIZE; di++) {
                    temp += input_shared[thread_row * TILE_SIZE + di] * weights_shared[di * TILE_SIZE + thread_col];
                }

                __syncthreads();
            }

            if (row < batch_size && col < output_count) {
                logits[thread_row * output_count + thread_col] = temp;
                activations[thread_row * output_count + thread_col] = TActivation::activate(temp);
            }
        }

        template<typename TActivation = NoActivation>
        __global__ void propagate_kernel(
            const float* input,
            float* logits, float* activations,
            const float* weights, const float* biases,
            const size_t batch_size, const size_t input_count, const size_t output_count
        ) {
            unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
            unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;

            if (row >= batch_size || col >= output_count) return; 

            float value = biases[col];

            for (unsigned int k = 0; k < input_count; k++) {
                value += input[row * input_count + k] * weights[k * output_count + col];
            }

            logits[row * output_count + col] = value;
            activations[row * output_count + col] = TActivation::activate(value);
        }

        void Context::propagate(
            const Matrix& inputs, Matrix& logits, Matrix& activations, const Matrix& weights, const Matrix& biases, const Activation activation
        ) const {
            assert(inputs.columns() == weights.rows());
            assert(activations.rows() == inputs.rows());
            assert(activations.columns() == weights.columns());
            assert(activations.size() == logits.size());

            cudaError_t err;

            dim3 block(TILE_SIZE * TILE_SIZE);
            // dim3 block(TILE_SIZE, TILE_SIZE); // if using untiled

            dim3 grid(
                block_count(weights.columns(), TILE_SIZE),
                block_count(inputs.rows(), TILE_SIZE)
            );

            TaskID task;
            if (m_profiler) task = m_profiler->startTask("Layer propagation");

            switch (activation) {
                case Activation::SIGMOID:
                    propagate_kernel_tiled<Sigmoid><<<grid, block>>>(inputs.data(), logits.data(), activations.data(), weights.data(), biases.data(), inputs.rows(), inputs.columns(), activations.columns());
                break;

                case Activation::TANH:
                    propagate_kernel_tiled<Tanh><<<grid, block>>>(inputs.data(), logits.data(), activations.data(), weights.data(), biases.data(), inputs.rows(), inputs.columns(), activations.columns());
                break;

                case Activation::RELU:
                    propagate_kernel_tiled<ReLU><<<grid, block>>>(inputs.data(), logits.data(), activations.data(), weights.data(), biases.data(), inputs.rows(), inputs.columns(), activations.columns());
                break;

                case Activation::LEAKY_RELU:
                    propagate_kernel_tiled<LeakyReLU><<<grid, block>>>(inputs.data(), logits.data(), activations.data(), weights.data(), biases.data(), inputs.rows(), inputs.columns(), activations.columns());
                break;

                case Activation::SOFTMAX:
                    propagate_kernel_tiled<<<grid, block>>>(inputs.data(), logits.data(), activations.data(), weights.data(), biases.data(), inputs.rows(), inputs.columns(), activations.columns());

                    TaskID sm_task;
                    if (m_profiler) sm_task = m_profiler->startTask("Softmax"); // this is like a sub task, it overlaps with layer propagation, but i havent accounted for this in the profiler

                    // logits = activations here, but I am using logits as inputs and activations as outputs because it makes sense
                    softmax_kernel<<<inputs.rows(), BLOCK_SIZE>>>(logits.data(), activations.data(), logits.rows(), logits.columns());

                    if (m_profiler) m_profiler->endTask(sm_task);
                break;
            }

            if (m_profiler) m_profiler->endTask(task);

            err = cudaGetLastError();
            if (err != cudaSuccess) CUDA_ERROR(err, "CUDA layer propagation error: ");
        }
    }
}