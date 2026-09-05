#pragma once

#if !defined(RANDOM_BS) // using this just so intellisense leaves me alone
#define MLP_OPENCL
#endif

#if defined(MLP_OPENCL)

#include <cassert>
#include <iostream>

#include "enums/Enums.hpp"
#include "opencl/memory/Buffer.hpp"
#include "opencl/memory/Matrix.hpp"
#include "opencl/profiling/Profiler.hpp"

#define CL_ERROR(err, message)\
    do {\
        std::cout << "\033[31m" << "[Error]\033[0m "\
                  << message << ": "\
                  << err << '\n'\
                  << "File: " << __FILE__ << '\n'\
                  << "Line: " << __LINE__ << '\n'\
                  << "Function: " << __func__ << '\n';\
        std::abort();\
    } while (0);

namespace mlp {
    namespace opencl {
        class Context {
            public:

            using Buffer_t = Buffer;
            using Matrix_t = Matrix;

            Context(Profiler* profiler = nullptr);

            /*
                TODO:

                initialisation
                get Buffer and Matrix working (and transfer functions)
                matrix multiplication, addition

            */

            void transfer(float* dest, const Matrix_t& src) const;
            void transfer(Matrix_t& dest, const float* src) const;
            void transfer(Matrix_t& dest, const Matrix_t& src) const;

            void transfer(void* dest, const Buffer_t& src) const;
            void transfer(Buffer_t& dest, const void* src) const;
            void transfer(Buffer_t& dest, const Buffer_t& src) const;

            // Randomise each value in the Matrix_t to a value between min and max
            // void randomise(Matrix_t& Matrix_t, float min, float max) const;

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
                Matrix_t& weight_gradients, Matrix_t& bias_gradients, Matrix_t& dC_da_next
            ) const; // hidden layers

            void computeOutputGradients(
                const Matrix_t& last_activations, const Matrix_t& output_activations,
                const Matrix_t& weights,
                const std::vector<int>& labels,
                const OALP al_pair,
                Matrix_t& weight_gradients, Matrix_t& bias_gradients, Matrix_t& dC_da_hidden
            ) const; // output layer

            void optimiseLayer(Matrix_t& weights, Matrix& biases, const Matrix_t& weight_gradients, const Matrix_t& bias_gradients, const float learning_rate) const;

            void checkOutputs(const Matrix_t& outputs, const std::vector<int>& labels, Buffer_t& correct, Buffer_t& classifications, const size_t samples) const;

            void computeLoss(const Matrix_t& outputs, const Matrix_t& targets, Matrix_t& result, const Loss loss) const;

            // probably need something equivalent to cuda device synchronise

            private:

            Profiler* m_profiler = nullptr;
        };
    }
}

#endif