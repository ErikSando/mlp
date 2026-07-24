#pragma once

#include "matrix/Matrix.hpp"

namespace mlp {
    constexpr int BLOCK_SIZE = 256;

    class DeviceContext {
        public:
        // Transfer values from host memory to device memory
        void transfer(Matrix& src, float* dest);

        // Transfer values from device memory to host memory
        void transfer(float* src, Matrix& dest);

        // Randomise each value in the matrix to a value between min and max
        void randomise(Matrix& matrix, float min, float max);

        // Calculate C = AB
        void multiply(Matrix& matrix_A, Matrix& matrix_B, Matrix& matrix_C);

        // Calculate C = A + B
        void add(Matrix& matrix_A, Matrix& matrix_B, Matrix& matrix_C);
    };
}