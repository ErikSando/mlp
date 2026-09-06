#include <cassert>
#include <cmath>

#include <cuda_runtime.h>

#include "context/Context.hpp"
#include "cuda/ContextImpl.hpp"
#include "cuda/impl/Activation.cuh"
#include "cuda/impl/Softmax.cuh"
#include "enums/Enums.hpp"

namespace mlp {
    template<typename TActivation = NoActivation>
    __global__ void propagate_kernel_tiled(
        const float* left_activations, float* logits, float* activations,
        const float* weights, const float* biases,
        const size_t batch_size, const size_t n_left, const size_t n_right
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

        left_activations += cache_row * TILE_SIZE * n_left;
        weights += cache_col * TILE_SIZE;
        logits += cache_row * TILE_SIZE * n_right + cache_col * TILE_SIZE;
        activations += cache_row * TILE_SIZE * n_right + cache_col * TILE_SIZE;

        float temp = biases[col];

        for (unsigned int i = 0; i < n_left; i += TILE_SIZE) {
            float input_shared_value = 0.0f;
            float weights_shared_value = 0.0f;

            if (row < batch_size && i + thread_col < n_left) input_shared_value = left_activations[thread_row * n_left + i + thread_col];
            if (col < n_right && i + thread_row < n_left) weights_shared_value = weights[(thread_row + i) * n_right + thread_col];

            input_shared[thread_row * TILE_SIZE + thread_col] = input_shared_value;
            weights_shared[thread_row * TILE_SIZE + thread_col] = weights_shared_value;

            __syncthreads();

            for (unsigned int di = 0; di < TILE_SIZE; di++) {
                temp += input_shared[thread_row * TILE_SIZE + di] * weights_shared[di * TILE_SIZE + thread_col];
            }

            __syncthreads();
        }

        if (row < batch_size && col < n_right) {
            logits[thread_row * n_right + thread_col] = temp;
            activations[thread_row * n_right + thread_col] = TActivation::activate(temp);
        }
    }

    template<typename TActivation = NoActivation>
    __global__ void propagate_kernel(
        const float* left_activations,
        float* right_logits, float* right_activations,
        const float* weights, const float* biases,
        const size_t batch_size, const size_t n_left, const size_t n_right
    ) {
        unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
        unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;

        if (row >= batch_size || col >= n_right) return; 

        float value = biases[col];

        for (unsigned int k = 0; k < n_left; k++) {
            value += left_activations[row * n_left + k] * weights[k * n_right + col];
        }

        right_logits[row * n_right + col] = value;
        right_activations[row * n_right + col] = TActivation::activate(value);
    }

    void Context::propagate(
        const Matrix& left_activations, Matrix& right_logits, Matrix& right_activations, const Matrix& weights, const Matrix& biases, const Activation activation
    ) const {
        assert(left_activations.columns() == weights.rows());
        assert(right_activations.rows() == left_activations.rows());
        assert(right_activations.columns() == weights.columns());
        assert(right_activations.size() == right_logits.size());

        cudaError_t err;

        dim3 block(TILE_SIZE * TILE_SIZE);
        // dim3 block(TILE_SIZE, TILE_SIZE); // if using untiled

        dim3 grid(
            block_count(weights.columns(), TILE_SIZE),
            block_count(left_activations.rows(), TILE_SIZE)
        );

        if (m_profiler) m_profiler->startTask("Propagation");

        switch (activation) {
            case Activation::SIGMOID:
                propagate_kernel_tiled<Sigmoid><<<grid, block>>>(
                    left_activations.data(), right_logits.data(), right_activations.data(),
                    weights.data(), biases.data(),
                    left_activations.rows(), left_activations.columns(), right_activations.columns()
                );
            break;

            case Activation::TANH:
                propagate_kernel_tiled<Tanh><<<grid, block>>>(
                    left_activations.data(), right_logits.data(), right_activations.data(),
                    weights.data(), biases.data(),
                    left_activations.rows(), left_activations.columns(), right_activations.columns()
                );
            break;

            case Activation::RELU:
                propagate_kernel_tiled<ReLU><<<grid, block>>>(
                    left_activations.data(), right_logits.data(), right_activations.data(),
                    weights.data(), biases.data(),
                    left_activations.rows(), left_activations.columns(), right_activations.columns()
                );
            break;

            case Activation::LEAKY_RELU:
                propagate_kernel_tiled<LeakyReLU><<<grid, block>>>(
                    left_activations.data(), right_logits.data(), right_activations.data(),
                    weights.data(), biases.data(),
                    left_activations.rows(), left_activations.columns(), right_activations.columns()
                );
            break;

            case Activation::SOFTMAX: 
                propagate_kernel_tiled<<<grid, block>>>(
                    left_activations.data(), right_logits.data(), right_activations.data(),
                    weights.data(), biases.data(),
                    left_activations.rows(), left_activations.columns(), right_activations.columns()
                );

                assert(right_logits.columns() < BLOCK_SIZE);

                if (m_profiler) m_profiler->startTask("Softmax"); // this is like a sub task, it overlaps with layer propagation

                softmax_kernel<<<right_logits.rows(), BLOCK_SIZE>>>(right_logits.data(), right_activations.data(), right_logits.rows(), right_logits.columns());

                if (m_profiler) m_profiler->endTask("Softmax");

                break;
            

            default:
                propagate_kernel_tiled<<<grid, block>>>(
                    left_activations.data(), right_logits.data(), right_activations.data(),
                    weights.data(), biases.data(),
                    left_activations.rows(), left_activations.columns(), right_activations.columns()
                );
            break;
        }

        if (m_profiler) m_profiler->endTask("Propagation");

        err = cudaGetLastError();
        if (err != cudaSuccess) CUDA_ERROR(err, "CUDA layer propagation error");
    }
}