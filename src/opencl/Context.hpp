#pragma once

#if !defined(RANDOM_BS) // using this just so intellisense leaves me alone
#define MLP_OPENCL
#endif

#if defined(MLP_OPENCL)

#include <cassert>
#include <iostream>

#include "opencl/matrix/Matrix.hpp"
#include "opencl/profiling/Profiler.hpp"
#include "enums/Enums.hpp"

namespace mlp {
    namespace opencl {
        constexpr unsigned int TILE_SIZE = 32;
        constexpr unsigned int BLOCK_SIZE = 256;

        inline unsigned int block_count(unsigned int thread_count, unsigned int block_size) {
            return (thread_count + block_size - 1) / block_size;
        }

        class Context {
            public:

            using Matrix_t = Matrix;

            Context(Profiler* profiler = nullptr) : m_profiler(profiler) {}

            // Transfer data from device memory to host memory
            void transfer(const Matrix_t& src, float* dest) const;

            // Transfer data from host memory to device memory
            void transfer(const float* src, Matrix_t& dest) const;

            void transfer(const Matrix_t& src, Matrix_t& dest) const;

            // Randomise each value in the Matrix_t to a value between min and max
            void randomise(Matrix_t& Matrix_t, float min, float max) const;

            void softmax(const Matrix_t& inputs, Matrix_t& outputs) const;

            /*
                Propagation function (L_n+1 = activation(L_n W + b) fused into one kernel, L_n is the nth layer)
                last_activations: the nodes in the layer being propagated from
                activations: the nodes in the layer being propagated into
                weights: weights Matrix_t
                biases: biases Matrix_t (really a vector)
                activation: activation function type
            */
            void propagate(
                const Matrix_t& last_activations,
                Matrix_t& logits, Matrix_t& activations,
                const Matrix_t& weights, const Matrix_t& biases,
                const Activation activation
            ) const;

            void computeGradients(
                const Matrix_t& dC_da,
                const Matrix_t& left_activations, const Matrix_t& right_activations,
                const Matrix_t& weights,
                const Activation activation,
                Matrix_t& gradients, Matrix_t& dC_da_next
            ) const; // hidden layers

            void computeOutputGradients(
                const Matrix_t& last_activations, const Matrix_t& output_activations,
                const Matrix_t& weights,
                const std::vector<int>& labels,
                const Activation activation, const Loss loss,
                Matrix_t& gradients, Matrix_t& dC_da_hidden
            ) const; // output layer

            void optimiseLayer(Matrix_t& weights, const Matrix_t& gradients, const float learning_rate) const;

            void checkOutputs(const Matrix_t& outputs, const std::vector<int>& labels, const size_t n_samples, Matrix_t& correct, Matrix_t& classifications) const;

            void computeLoss(const Matrix_t& outputs, const Matrix_t& targets, Matrix_t& result, const Loss loss) const;

            // probably need something equivalent to cuda device synchronise

            private:

            Profiler* m_profiler = nullptr;
        };
    }
}

#endif