#pragma once

#include <cassert>
#include <iostream>

#include "context/MemoryConfig.hpp"
#include "enums/Enums.hpp"
#include "profiling/Profiler.hpp"

namespace mlp {
    class Context {
        public:

        Context(Profiler* profiler = nullptr);
        ~Context();

        Matrix createMatrix(const size_t rows, const size_t columns) const;
        Buffer createBuffer(const size_t size) const;

        void zeroMatrix(Matrix& matrix) const;
        void zeroBuffer(Buffer& buffer) const;

        void synchronise() const;

        void transfer(float* dest, const Matrix& src) const;
        void transfer(Matrix& dest, const float* src) const;
        void transfer(Matrix& dest, const Matrix& src) const;

        void transfer(void* dest, const Buffer& src) const;
        void transfer(Buffer& dest, const void* src) const;
        void transfer(Buffer& dest, const Buffer& src) const;

        void randomise(Matrix& Matrix, float min, float max) const;

        void softmax(const Matrix& logits, Matrix& activations) const;

        /*
            Propagation function (L_n+1 = activation(L_n W + b) fused into one kernel, L_n is the nth layer)
            left_activations: the activations buffer for the layer being propagated from
            right_logits: the logits buffer for the layer being propagated into
            right_activations: the activations buffer for the layer being propagated into
            weights: weights Matrix
            biases: biases Matrix (really a vector)
            activation: activation function type
        */
        void propagate(
            const Matrix& left_activations,
            Matrix& right_logits, Matrix& right_activations,
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

        private:

        Profiler* m_profiler = nullptr;

        struct Impl;
        std::unique_ptr<Impl> m_impl;
    };
}