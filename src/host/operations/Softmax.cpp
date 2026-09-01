#include <iostream>

#include <cfloat>
#include <cmath>

#include "host/Context.hpp"

namespace mlp {
    namespace host {
        void softmax_op(const float* inputs, float* outputs, const size_t rows, const size_t cols) {
            for (unsigned int row = 0; row < rows; row++) {
                float max = -FLT_MAX;

                for (unsigned int col = 0; col < cols; col++) {
                    float value = inputs[row * cols + col];
                    if (value > max) max = value;
                }

                float sum = 0.0f;

                float* exponents = new float[cols];

                for (unsigned int col = 0; col < cols; col++) {
                    float exponent = expf(inputs[row * cols + col] - max);
                    sum += exponent;
                    exponents[col] = exponent;
                }

                for (unsigned int col = 0; col < cols; col++) {
                    outputs[row * cols + col] = exponents[col] / sum;
                }

                delete[] exponents;
            }
        }

        void Context::softmax(const Matrix_t& inputs, Matrix_t& outputs) const {
            assert(inputs.size() == outputs.size());
            assert(inputs.rows() == outputs.rows());
            assert(inputs.columns() == outputs.columns());

            if (m_profiler) m_profiler->startTask("Softmax");

            softmax_op(inputs.data(), outputs.data(), inputs.rows(), inputs.columns());

            if (m_profiler) m_profiler->endTask();
        }
    }
}