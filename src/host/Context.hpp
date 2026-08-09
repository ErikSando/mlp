#pragma once

#include <cassert>

#include "enums/Enums.hpp"
#include "host/matrix/Matrix.hpp"
#include "host/profiling/Profiler.hpp"

namespace mlp {
    namespace host {
        class Context {
            public:

            using Matrix_t = Matrix;

            Context(Profiler* profiler = nullptr) : m_profiler(profiler) {}

            void transfer(const Matrix_t& src, float* dest) const;
            void transfer(const float* src, Matrix_t& dest) const;
            void transfer(const Matrix_t& src, Matrix_t& dest) const;

            void randomise(Matrix_t& matrix, float min, float max) const;

            void softmax(const Matrix_t& inputs, Matrix_t& outputs) const;

            void propagate(
                const Matrix_t& last_activations,
                Matrix_t& logits, Matrix_t& activations,
                const Matrix_t& weights, const Matrix_t& biases,
                const Activation activation
            ) const;

            void computeOutputGradients(
                const Matrix_t& last_activations, const Matrix_t& activations, const Matrix_t& weights,
                const size_t n_last_activations,
                const std::vector<int>& labels,
                const Activation activation, const Loss loss,
                Matrix_t& gradients, Matrix_t& dC_da_next
            ) const; // output layer

            void computeGradients(
                const Matrix_t& dC_da,
                const Matrix_t& left_activations, const Matrix_t& right_activations,
                const Matrix_t& weights,
                const Activation activation,
                Matrix_t& gradients, Matrix_t& dC_da_next
            ) const; // hidden layers

            void optimiseLayer(Matrix_t& weights, const Matrix_t& gradients, const float learning_rate) const;

            void checkOutputs(const Matrix_t& outputs, const std::vector<int>& labels, const size_t n_samples, Matrix_t& correct, Matrix_t& classifications) const;

            void computeLoss(const Matrix_t& outputs, const Matrix_t& targets, Matrix_t& result, const Loss loss) const;

            inline void synchronise() const {}

            private:

            Profiler* m_profiler = nullptr;
        };
    }
}