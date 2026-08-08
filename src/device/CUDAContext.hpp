#pragma once

#include <cassert>
#include <iostream>

#include "device/DeviceContext.hpp"
#include "matrix/CUDAMatrix.hpp"
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
    constexpr unsigned int TILE_SIZE = 32;
    constexpr unsigned int BLOCK_SIZE = 256;

    inline unsigned int block_count(unsigned int thread_count, unsigned int block_size) {
        return (thread_count + block_size - 1) / block_size;
    }

    // enum class Activation { // not sure where to put this, leaving it here for now
    //     NONE,
    //     SIGMOID, TANH, RELU, LEAKY_RELU, // hidden layer activation functions
    //     SOFTMAX // output activation functions
    // };

    // enum class Loss {
    //     MSE, CCE
    // };

    class CUDAContext : public IDeviceContext<CUDAMatrix> {
        public:

        using Matrix = CUDAMatrix;

        CUDAContext(CUDAProfiler* profiler = nullptr) : m_profiler(profiler) {}

        // I marked every member function as const so I can use const DeviceContext& in function arguments but I don't know if that has any benefits

        // Transfer data from device memory to host memory
        void transfer(const CUDAMatrix& src, float* dest) const override;

        // Transfer data from host memory to device memory
        void transfer(const float* src, CUDAMatrix& dest) const override;

        void transfer(const CUDAMatrix& src, CUDAMatrix& dest) const;

        // Randomise each value in the matrix to a value between min and max
        void randomise(CUDAMatrix& matrix, float min, float max) const;

        // Calculate C = AB
        void multiply(const CUDAMatrix& matrix_A, const CUDAMatrix& matrix_B, CUDAMatrix& matrix_C) const;
        // Calculate C = AB
        void multiplyOld(const CUDAMatrix& matrix_A, const CUDAMatrix& matrix_B, CUDAMatrix& matrix_C) const;

        // Calculate C = A + B
        void add(const CUDAMatrix& matrix_A, const CUDAMatrix& matrix_B, CUDAMatrix& matrix_C) const;

        // Perform C = A + b, where b is a row matrix, added onto each row in A
        void addBiases(const CUDAMatrix& layer, const CUDAMatrix& biases, CUDAMatrix& output) const;

        // Activation functions

        void sigmoid(const CUDAMatrix& input, CUDAMatrix& output) const;
        void tanh(const CUDAMatrix& input, CUDAMatrix& output) const;
        void relu(const CUDAMatrix& input, CUDAMatrix& output) const;
        void leakyReLU(const CUDAMatrix& input, CUDAMatrix& output) const;
        void softmax(const CUDAMatrix& input, CUDAMatrix& output) const;

        // Loss functions
        // Might merge these with a back propagation kernel like with the forward propagation kerne;

        void mse() const;
        void cce(const CUDAMatrix& output, const CUDAMatrix& target, CUDAMatrix& result) const;
        void hinge() const;

        /*
            Propagation function (L_n+1 = activation(L_n W + b) fused into one kernel, L_n is the nth layer)
            last_activations: the nodes in the layer being propagated from
            activations: the nodes in the layer being propagated into
            weights: weights matrix
            biases: biases matrix (really a vector)
            activation: activation function type
        */
        virtual void propagate(
            const CUDAMatrix& last_activations,
            CUDAMatrix& logits, CUDAMatrix& activations,
            const CUDAMatrix& weights, const CUDAMatrix& biases,
            const Activation activation
        ) const override;

        void computeLoss(const CUDAMatrix& output, const CUDAMatrix& target, CUDAMatrix& result, const Loss loss) const;

        // void backPropagate(const CUDAMatrix& outputs, const Loss loss) const;

        /*
            Ideas for back propagation

            Use a kernel to compute the gradients for a layer, using the previous layers' result for the next (starting at the output layer)
        */

        virtual void computeOutputGradients(
            const CUDAMatrix& last_activations, const CUDAMatrix& activations, const CUDAMatrix& weights,
            const size_t n_last_activations,
            const std::vector<int>& labels,
            const Activation activation, const Loss loss,
            CUDAMatrix& gradients, CUDAMatrix& dC_da_next
        ) const override; // output layer

        virtual void computeGradients(
            const CUDAMatrix& dC_da,
            const CUDAMatrix& left_activations, const CUDAMatrix& right_activations,
            const CUDAMatrix& weights,
            const Activation activation,
            CUDAMatrix& gradients, CUDAMatrix& dC_da_next
        ) const override; // hidden layers

        virtual void optimiseLayer(CUDAMatrix& weights, const CUDAMatrix& gradients, const float learning_rate) const override;

        virtual void checkOutputs(const CUDAMatrix& outputs, const std::vector<int>& labels, const size_t n_samples, CUDAMatrix& correct, CUDAMatrix& classifications) const override;

        inline void synchronise() const {
            cudaError_t err = cudaDeviceSynchronize();
            if (err != cudaSuccess) CUDA_ERROR(err, "CUDA device synchronise error: ");
        }

        private:

        CUDAProfiler* m_profiler = nullptr;
    };
}