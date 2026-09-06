#pragma once

#include <cassert>
#include <iostream>

#include "enums/Enums.hpp"
#include "opencl/memory/Buffer.hpp"
#include "opencl/memory/Matrix.hpp"
#include "profiling/Profiler.hpp"

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

            using Buffer = Buffer;
            using Matrix = Matrix;

            Context(Profiler* profiler = nullptr);

            /*
                TODO:

                initialisation
                get Buffer and Matrix working (and transfer functions)
                matrix multiplication, addition

            */

            void transfer(float* dest, const Matrix& src) const;
            void transfer(Matrix& dest, const float* src) const;
            void transfer(Matrix& dest, const Matrix& src) const;

            void transfer(void* dest, const Buffer& src) const;
            void transfer(Buffer& dest, const void* src) const;
            void transfer(Buffer& dest, const Buffer& src) const;

            // Randomise each value in the Matrix to a value between min and max
            // void randomise(Matrix& Matrix, float min, float max) const;

            void softmax(const Matrix& inputs, Matrix& outputs) const;

            /*
                Propagation function (L_n+1 = activation(L_n W + b) fused into one kernel, L_n is the nth layer)
                last_activations: the nodes in the layer being propagated from
                activations: the nodes in the layer being propagated into
                weights: weights Matrix
                biases: biases Matrix (really a vector)
                activation: activation function type
            */
            void propagate(
                const Matrix& last_activations,
                Matrix& logits, Matrix& activations,
                const Matrix& weights, const Matrix& biases,
                const Activation activation
            ) const;

            void computeGradients(
                const Matrix& dC_da,
                const Matrix& left_activations, const Matrix& right_activations,
                const Matrix& weights,
                const Activation activation,
                Matrix& weight_gradients, Matrix& bias_gradients, Matrix& dC_da_next
            ) const; // hidden layers

            void computeOutputGradients(
                const Matrix& last_activations, const Matrix& output_activations,
                const Matrix& weights,
                const std::vector<int>& labels,
                const OALP al_pair,
                Matrix& weight_gradients, Matrix& bias_gradients, Matrix& dC_da_hidden
            ) const; // output layer

            void optimiseLayer(Matrix& weights, Matrix& biases, const Matrix& weight_gradients, const Matrix& bias_gradients, const float learning_rate) const;

            void checkOutputs(const Matrix& outputs, const std::vector<int>& labels, Buffer& correct, Buffer& classifications, const size_t samples) const;

            void computeLoss(const Matrix& outputs, const Matrix& targets, Matrix& result, const Loss loss) const;

            void synchronise() const;

            private:

            Profiler* m_profiler = nullptr;
        };
    }
}