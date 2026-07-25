#pragma once

#include "matrix/Matrix.hpp"

namespace mlp {
    constexpr int TILE_SIZE = 16;
    constexpr int BLOCK_SIZE = 256;

    inline size_t block_count(size_t thread_count, size_t block_size) {
        return (thread_count + block_size - 1) / block_size;
    }

    class DeviceContext {
        public:
        // Transfer data from device memory to host memory
        void transfer(const Matrix& src, float* dest);

        // Transfer data from host memory to device memory
        void transfer(float* src, Matrix& dest);

        // Randomise each value in the matrix to a value between min and max
        void randomise(Matrix& matrix, float min, float max);

        // Calculate C = AB
        void multiply(const Matrix& matrix_A, const Matrix& matrix_B, Matrix& matrix_C);

        // Calculate C = A + B
        void add(const Matrix& matrix_A, const Matrix& matrix_B, Matrix& matrix_C);

        // Perform C = A + b
        void addBiases(const Matrix& layer, const Matrix& biases, Matrix& output);

        // Apply the softmax function on the given input martix, and write the result into the given output matrix
        void softmax(const Matrix& input, Matrix& output);
    };
}