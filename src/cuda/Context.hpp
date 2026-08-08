#pragma once

#include <cassert>
#include <iostream>

#include "cuda/matrix/Matrix.hpp"
#include "cuda/profiling/Profiler.hpp"
#include "enums/Enums.hpp"

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

namespace mlp {
    namespace cuda {
        constexpr unsigned int TILE_SIZE = 32;
        constexpr unsigned int BLOCK_SIZE = 256;

        inline unsigned int block_count(unsigned int thread_count, unsigned int block_size) {
            return (thread_count + block_size - 1) / block_size;
        }

        class Context {
            public:

            using Matrix = cuda::Matrix;

            Context(Profiler* profiler = nullptr) : m_profiler(profiler) {}

            // Transfer data from device memory to host memory
            void transfer(const Matrix& src, float* dest) const;

            // Transfer data from host memory to device memory
            void transfer(const float* src, Matrix& dest) const;

            void transfer(const Matrix& src, Matrix& dest) const;

            // Randomise each value in the matrix to a value between min and max
            void randomise(Matrix& matrix, float min, float max) const;

            void softmax(const Matrix& inputs, Matrix& outputs) const;

            /*
                Propagation function (L_n+1 = activation(L_n W + b) fused into one kernel, L_n is the nth layer)
                last_activations: the nodes in the layer being propagated from
                activations: the nodes in the layer being propagated into
                weights: weights matrix
                biases: biases matrix (really a vector)
                activation: activation function type
            */
            void propagate(
                const Matrix& last_activations,
                Matrix& logits, Matrix& activations,
                const Matrix& weights, const Matrix& biases,
                const Activation activation
            ) const;

            // void computeLoss(const Matrix& output, const Matrix& target, Matrix& result, const Loss loss) const;

            // void backPropagate(const Matrix& outputs, const Loss loss) const;

            /*
                Ideas for back propagation

                Use a kernel to compute the gradients for a layer, using the previous layers' result for the next (starting at the output layer)
            */

            void computeOutputGradients(
                const Matrix& last_activations, const Matrix& activations, const Matrix& weights,
                const size_t n_last_activations,
                const std::vector<int>& labels,
                const Activation activation, const Loss loss,
                Matrix& gradients, Matrix& dC_da_next
            ) const; // output layer

            void computeGradients(
                const Matrix& dC_da,
                const Matrix& left_activations, const Matrix& right_activations,
                const Matrix& weights,
                const Activation activation,
                Matrix& gradients, Matrix& dC_da_next
            ) const; // hidden layers

            void optimiseLayer(Matrix& weights, const Matrix& gradients, const float learning_rate) const;

            void checkOutputs(const Matrix& outputs, const std::vector<int>& labels, const size_t n_samples, Matrix& correct, Matrix& classifications) const;

            void computeLoss(const Matrix& outputs, const Matrix& targets, Matrix& result, const Loss loss) const;

            inline void synchronise() const {
                cudaError_t err = cudaDeviceSynchronize();
                if (err != cudaSuccess) CUDA_ERROR(err, "CUDA device synchronise error: ");
            }

            private:

            Profiler* m_profiler = nullptr;
        };
    }
}