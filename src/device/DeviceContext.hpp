#pragma once

#include <cassert>
#include <iostream>

#include "matrix/Matrix.hpp"
#include "profiling/CUDAProfiler.hpp"

#define CUDA_ERROR(err, message)\
    do {\
        std::cout << "\033[31m" << "[Error]\033[0m "\
                  << message\
                  << cudaGetErrorString(err) << '\n'\
                  << "File: " << __FILE__ << '\n'\
                  << "Line: " << __LINE__ << '\n'\
                  << "Function: " << __func__ << '\n';\
        std::abort();\
    } while (0)

namespace mlp {
    constexpr unsigned int TILE_SIZE = 16;
    constexpr unsigned int BLOCK_SIZE = 256;

    inline unsigned int block_count(unsigned int thread_count, unsigned int block_size) {
        return (thread_count + block_size - 1) / block_size;
    }

    enum class Activation { // not sure where to put this, leaving it here for now
        NONE,
        SIGMOID, TANH, RELU, LEAKY_RELU, // hidden layer activation functions
        SOFTMAX // output activation functions
    };

    class DeviceContext {
        public:

        DeviceContext(CUDAProfiler* profiler = nullptr) : m_profiler(profiler) {}

        // I marked every member function as const so I can use const DeviceContext& in function arguments but I don't know if that has any benefits

        // Transfer data from device memory to host memory
        void transfer(const Matrix& src, float* dest) const;

        // Transfer data from host memory to device memory
        void transfer(const float* src, Matrix& dest) const;

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
        void softmax(const Matrix& input, Matrix& output) const;

        // Loss functions

        void mse() const;
        void cce(const Matrix& output, const Matrix& target) const;
        void hinge() const;

        /*
            Propagation function (L_n+1 = activation(L_n W + b) fused into one kernel, L_n is the nth layer)
            last_activations: the nodes in the layer being propagated from
            activations: the nodes in the layer being propagated into
            weights: weights matrix
            biases: biases matrix (really a vector)
            activation: activation function type
        */
        void propagate(const Matrix& last_activations, Matrix& activations, const Matrix& weights, const Matrix& biases, const Activation activation) const;

        inline void synchronise() const {
            cudaError_t err = cudaDeviceSynchronize();
            if (err != cudaSuccess) CUDA_ERROR(err, "CUDA device synchronise error: ");
        }

        private:

        CUDAProfiler* m_profiler = nullptr;
    };
}