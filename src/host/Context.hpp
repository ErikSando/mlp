#pragma once

#include <cassert>

#include "enums/Enums.hpp"
#include "host/memory/Buffer.hpp" 
#include "host/memory/Matrix.hpp"
#include "profiling/Profiler.hpp"

namespace mlp {
    namespace host {
        class Context {
            public:

            Context(Profiler* profiler = nullptr) : m_profiler(profiler) {}

            void transfer(float* dest, const Matrix& src) const;
            void transfer(Matrix& dest, const float* src) const;
            void transfer(Matrix& dest, const Matrix& src) const;

            void transfer(void* dest, const Buffer& src) const;
            void transfer(Buffer& dest, const void* src) const;
            void transfer(Buffer& dest, const Buffer& src) const;

            void randomise(Matrix& matrix, float min, float max) const;

            void softmax(const Matrix& inputs, Matrix& outputs) const;

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
                const Matrix& last_hidden_activations, const Matrix& output_activations,
                const Matrix& weights,
                const std::vector<int>& labels,
                const OALP al_pair,
                Matrix& weight_gradients, Matrix& bias_gradients, Matrix& dC_da_hidden
            ) const; // output layer

            void optimiseLayer(Matrix& weights, Matrix& biases, const Matrix& weight_gradients, const Matrix& bias_gradients, const float learning_rate) const;

            void checkOutputs(const Matrix& outputs, const std::vector<int>& labels, Buffer& correct, Buffer& classifications, const size_t samples) const;

            void computeLoss(const Matrix& outputs, const Matrix& targets, Matrix& result, const Loss loss) const;

            inline void synchronise() const {}

            private:

            Profiler* m_profiler = nullptr;
        };
    }
}