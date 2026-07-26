#pragma once

#include "matrix/Matrix.hpp"

namespace mlp {
    constexpr unsigned int TILE_SIZE = 16;
    constexpr unsigned int BLOCK_SIZE = 256;

    inline unsigned int block_count(unsigned int thread_count, unsigned int block_size) {
        return (thread_count + block_size - 1) / block_size;
    }

    class DeviceContext {
        public:

        // I marked every member function as const so I can use const DeviceContext& in function arguments but I don't know if that has any benefits

        // Transfer data from device memory to host memory
        void transfer(const Matrix& src, float* dest) const;

        // Transfer data from host memory to device memory
        void transfer(float* src, Matrix& dest) const;

        // Randomise each value in the matrix to a value between min and max
        void randomise(Matrix& matrix, float min, float max) const;

        // Calculate C = AB
        void multiply(const Matrix& matrix_A, const Matrix& matrix_B, Matrix& matrix_C) const;

        // Calculate C = A + B
        void add(const Matrix& matrix_A, const Matrix& matrix_B, Matrix& matrix_C) const;

        // Perform C = A + b, where b is a row matrix, added onto each row in A
        void addBiases(const Matrix& layer, const Matrix& biases, Matrix& output) const;

        // Activation functions

        void sigmoid(const Matrix& input, Matrix& output) const;
        void tanh(const Matrix& input, Matrix& output) const;
        void relu(const Matrix& input, Matrix& output) const;
        void leakyReLU(const Matrix& input, Matrix& output) const;

        // Apply the softmax function on the given input martix, and write the result into the given output matrix
        void softmax(const Matrix& input, Matrix& output) const;
    };
}