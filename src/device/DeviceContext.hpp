#pragma once

#include <cassert>
#include <iostream>

#include "profiling/CUDAProfiler.hpp"

// #define CUDA_ERROR(err, message)\
//     do {\
//         std::cout << "\033[31m" << "[Error]\033[0m "\
//                   << message\
//                   << cudaGetErrorString(err) << '\n'\
//                   << "File: " << __FILE__ << '\n'\
//                   << "Line: " << __LINE__ << '\n'\
//                   << "Function: " << __func__ << '\n';\
//         std::abort();\
//     } while (0)

namespace mlp {
    // constexpr unsigned int TILE_SIZE = 32;
    // constexpr unsigned int BLOCK_SIZE = 256;

    // inline unsigned int block_count(unsigned int thread_count, unsigned int block_size) {
    //     return (thread_count + block_size - 1) / block_size;
    // }

    enum class Activation { // not sure where to put this, leaving it here for now
        NONE,
        SIGMOID, TANH, RELU, LEAKY_RELU, // hidden layer activation functions
        SOFTMAX // output activation functions
    };

    enum class Loss {
        MSE, CCE
    };

    // class DeviceContext {
    //     public:

    //     DeviceContext(CUDAProfiler* profiler = nullptr) : m_profiler(profiler) {}

    //     // I marked every member function as const so I can use const DeviceContext& in function arguments but I don't know if that has any benefits

    //     // Transfer data from device memory to host memory
    //     void transfer(const Matrix& src, float* dest) const;

    //     // Transfer data from host memory to device memory
    //     void transfer(const float* src, Matrix& dest) const;

    //     // Randomise each value in the matrix to a value between min and max
    //     void randomise(Matrix& matrix, float min, float max) const;

    //     // Calculate C = AB
    //     void multiply(const Matrix& matrix_A, const Matrix& matrix_B, Matrix& matrix_C) const;
    //     // Calculate C = AB
    //     void multiplyOld(const Matrix& matrix_A, const Matrix& matrix_B, Matrix& matrix_C) const;

    //     // Calculate C = A + B
    //     void add(const Matrix& matrix_A, const Matrix& matrix_B, Matrix& matrix_C) const;

    //     // Perform C = A + b, where b is a row matrix, added onto each row in A
    //     void addBiases(const Matrix& layer, const Matrix& biases, Matrix& output) const;

    //     // Activation functions

    //     void sigmoid(const Matrix& input, Matrix& output) const;
    //     void tanh(const Matrix& input, Matrix& output) const;
    //     void relu(const Matrix& input, Matrix& output) const;
    //     void leakyReLU(const Matrix& input, Matrix& output) const;
    //     void softmax(const Matrix& input, Matrix& output) const;

    //     // Loss functions
    //     // Might merge these with a back propagation kernel like with the forward propagation kerne;

    //     void mse() const;
    //     void cce(const Matrix& output, const Matrix& target, Matrix& result) const;
    //     void hinge() const;

    //     /*
    //         Propagation function (L_n+1 = activation(L_n W + b) fused into one kernel, L_n is the nth layer)
    //         last_activations: the nodes in the layer being propagated from
    //         activations: the nodes in the layer being propagated into
    //         weights: weights matrix
    //         biases: biases matrix (really a vector)
    //         activation: activation function type
    //     */
    //     void propagate(
    //         const Matrix& last_activations,
    //         Matrix& logits, Matrix& activations,
    //         const Matrix& weights, const Matrix& biases,
    //         const Activation activation
    //     ) const;

    //     void computeLoss(const Matrix& output, const Matrix& target, Matrix& result, const Loss loss) const;

    //     // void backPropagate(const Matrix& outputs, const Loss loss) const;

    //     /*
    //         Ideas for back propagation

    //         Use a kernel to compute the gradients for a layer, using the previous layers' result for the next (starting at the output layer)
    //     */

    //     void computeOutputGradients(
    //         const Matrix& last_activations, const Matrix& activations, const Matrix& weights,
    //         const size_t n_last_activations,
    //         const std::vector<int>& labels,
    //         const Activation activation, const Loss loss,
    //         Matrix& gradients, Matrix& dC_da_next
    //     ) const; // output layer

    //     void computeGradients(
    //         const Matrix& dC_da,
    //         const Matrix& left_activations, const Matrix& right_activations,
    //         const Matrix& weights,
    //         const Activation activation,
    //         Matrix& gradients, Matrix& dC_da_next
    //     ) const; // hidden layers

    //     void optimiseLayer(Matrix& weights, const Matrix& gradients, const float learning_rate) const;

    //     void checkOutputs(const Matrix& outputs, const std::vector<int>& labels, const size_t n_samples, Matrix& correct, Matrix& classifications);

    //     inline void synchronise() const {
    //         cudaError_t err = cudaDeviceSynchronize();
    //         if (err != cudaSuccess) CUDA_ERROR(err, "CUDA device synchronise error: ");
    //     }

    //     private:

    //     CUDAProfiler* m_profiler = nullptr;
    // };

    template<typename TMatrix>
    class IDeviceContext {
        public:

        IDeviceContext() = default;

        virtual void transfer(const TMatrix& src, float* dest) const = 0;
        virtual void transfer(const float* src, TMatrix& dest) const = 0;
        virtual void transfer(const TMatrix& src, TMatrix& dest) const = 0;

        // Randomise each value in the matrix to a value between min and max
        virtual void randomise(TMatrix& matrix, float min, float max) const = 0;

        /*
            Propagation function (L_n+1 = activation(L_n W + b) fused into one kernel, L_n is the nth layer)
            last_activations: the nodes in the layer being propagated from
            activations: the nodes in the layer being propagated into
            weights: weights matrix
            biases: biases matrix (really a vector)
            activation: activation function type
        */
        virtual void propagate(
            const TMatrix& last_activations,
            TMatrix& logits, TMatrix& activations,
            const TMatrix& weights, const TMatrix& biases,
            const Activation activation
        ) const = 0;

        virtual void computeOutputGradients(
            const TMatrix& last_activations, const TMatrix& activations, const TMatrix& weights,
            const size_t n_last_activations,
            const std::vector<int>& labels,
            const Activation activation, const Loss loss,
            TMatrix& gradients, TMatrix& dC_da_next
        ) const = 0; // output layer

        virtual void computeGradients(
            const TMatrix& dC_da,
            const TMatrix& left_activations, const TMatrix& right_activations,
            const TMatrix& weights,
            const Activation activation,
            TMatrix& gradients, TMatrix& dC_da_next
        ) const = 0; // hidden layers

        virtual void optimiseLayer(TMatrix& weights, const TMatrix& gradients, const float learning_rate) const = 0;

        virtual void checkOutputs(const TMatrix& outputs, const std::vector<int>& labels, const size_t n_samples, TMatrix& correct, TMatrix& classifications) const = 0;
    };
}