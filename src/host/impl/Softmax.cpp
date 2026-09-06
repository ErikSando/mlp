#include <iostream>

#include <cfloat>
#include <cmath>

#include "context/Context.hpp"

namespace mlp {
    void softmax_op(const float* logits, float* activations, const size_t rows, const size_t cols) {
        for (unsigned int row = 0; row < rows; row++) {
            float max = -FLT_MAX;

            for (unsigned int col = 0; col < cols; col++) {
                float value = logits[row * cols + col];
                if (value > max) max = value;
            }

            float sum = 0.0f;

            float* exponents = new float[cols];

            for (unsigned int col = 0; col < cols; col++) {
                float exponent = expf(logits[row * cols + col] - max);
                sum += exponent;
                exponents[col] = exponent;
            }

            for (unsigned int col = 0; col < cols; col++) {
                activations[row * cols + col] = exponents[col] / sum;
            }

            delete[] exponents;
        }
    }

    void Context::softmax(const Matrix& logits, Matrix& activations) const {
        assert(logits.size() == activations.size());
        assert(logits.rows() == activations.rows());
        assert(logits.columns() == activations.columns());

        if (m_profiler) m_profiler->startTask("Softmax");

        softmax_op(logits.data(), activations.data(), logits.rows(), logits.columns());

        if (m_profiler) m_profiler->endTask("Softmax");
    }
}