#pragma once

#include "matrix/Matrix.hpp"

namespace mlp {
    class DeviceContext {
        public:

        void multiply(Matrix& matrix_A, Matrix& matrix_B, Matrix& matrix_C);
        void transfer(Matrix& src, float* dest); // device to host
        void transfer(float* src, Matrix& dest); // host to device

        void randomise(Matrix& matrix, float min, float max);
    };
}