#pragma once

#if !defined(RANDOM_BS) // using this just so intellisense leaves me alone
#define MLP_HOST
#endif

#if defined(MLP_HOST)

#include <cassert>

#include "enums/Enums.hpp"
#include "host/memory/Buffer.hpp" 
#include "host/memory/Matrix.hpp"
#include "host/profiling/Profiler.hpp"

namespace mlp {
    namespace host {
        class Context {
            public:

            using Buffer_t = Buffer;
            using Matrix_t = Matrix;

            Context(Profiler* profiler = nullptr) : m_profiler(profiler) {}

            void transfer(float* dest, const Matrix_t& src) const;
            void transfer(Matrix_t& dest, const float* src) const;
            void transfer(Matrix_t& dest, const Matrix_t& src) const;

            void transfer(void* dest, const Buffer_t& src) const;
            void transfer(Buffer_t& dest, const void* src) const;
            void transfer(Buffer_t& dest, const Buffer_t& src) const;

            void randomise(Matrix_t& matrix, float min, float max) const;

            void softmax(const Matrix_t& inputs, Matrix_t& outputs) const;

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
                const Matrix_t& last_hidden_activations, const Matrix_t& output_activations,
                const Matrix_t& weights,
                const std::vector<int>& labels,
                const Activation activation, const Loss loss,
                Matrix_t& gradients, Matrix_t& dC_da_hidden
            ) const; // output layer

            void optimiseLayer(Matrix_t& weights, const Matrix_t& gradients, const float learning_rate) const;

            void checkOutputs(const Matrix_t& outputs, const std::vector<int>& labels, Buffer_t& correct, Buffer_t& classifications) const;

            void computeLoss(const Matrix_t& outputs, const Matrix_t& targets, Matrix_t& result, const Loss loss) const;

            inline void synchronise() const {}

            private:

            Profiler* m_profiler = nullptr;
        };
    }
}

#endif