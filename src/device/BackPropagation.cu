#include <cassert>

#include "device/Activation.cuh"
#include "device/Loss.cuh"
#include "device/DeviceContext.hpp"

namespace mlp {
    /*
        Notes and stuff

        

        For the output layer:
        dC/dw = -1 x activation of input node x (1 - activation of output node)
        where w connects "input node" (last hidden layer) and "output node" (output layer)

        For prior layers (not sure if this is the easiest way to compute it, just an idea for now):
        dC/dw = dC/da da/dw
        So basically split the derivate into the influence of w on a, then influence of a on C
        I think da/dw will be computed earlier and will account for all the next layers too

    */

    template<typename TActivation, typename TLoss>
    __global__ void compute_gradients_kernel(
        const float* last_gradients, const float* activations,
        const size_t n_last_gradients, const size_t n_activations, const size_t batch_size,
        float* gradients
    ) {
        unsigned int last_gradient_index = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int activation_index = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned int batch = blockIdx.z * blockDim.z + threadIdx.z;

        if (last_gradient_index >= n_last_gradients || activation_index >= n_activations || batch >= batch_size) return;

        // unsigned int weight_index = activation_index * n_activations + last_gradient_index;

        // gradients[weight_index] = weight_index / 10.0f;

        // I'm preeeeetty sure that the gradients are averaged across batches
    }

    template<typename TActivation, typename TLoss>
    __global__ void compute_output_gradients_kernel(
        const float* last_activations, const float* activations,
        const size_t n_activations, const size_t n_last_activations, const size_t batch_size,
        const int* labels, float* gradients
    ) {
        unsigned int last_activation_index = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int activation_index = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned int batch = blockIdx.z * blockDim.z + threadIdx.z;

        if (last_activation_index >= n_last_activations || activation_index >= n_activations || batch >= batch_size) return;

        if (activation_index == labels[batch]) {
            unsigned int weight_index = last_activation_index * n_activations + activation_index;

            gradients[weight_index] += -last_activations[last_activation_index] * (1 - activations[activation_index]);
        }
    }

    __global__ void optimise_kernel(float* weights, const float* gradients, const size_t n_weights, const float learning_rate) {
        unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

        if (index >= n_weights) return;

        weights[index] += gradients[index] * learning_rate;
    }

    void DeviceContext::computeGradients(const Matrix& last_gradients, const Matrix& activations, const Activation activation, const Loss loss, Matrix& gradients) const {
        assert(last_gradients.rows() == activations.rows());

        cudaError_t err;

        dim3 block(8, 8, 8);

        dim3 grid(
            block_count(last_gradients.columns(), block.x),
            block_count(activations.columns(), block.y),
            block_count(activations.rows(), block.z)
        );

        compute_gradients_kernel<LeakyReLU, CCE><<<grid, block>>>(last_gradients.data(), activations.data(), last_gradients.columns(), activations.columns(), activations.rows(), gradients.data());

        err = cudaGetLastError();
        if (err != cudaSuccess) CUDA_ERROR(err, "CUDA compute hidden layer gradients error: ");
    }

    void DeviceContext::computeGradients(
        const Matrix& last_activations, const Matrix& activations,
        const size_t n_last_activations,
        const std::vector<int>& labels,
        const Activation activation, const Loss loss,
        Matrix& gradients) const
    {
        cudaError_t err;

        dim3 block(8, 8, 8);

        dim3 grid(
            block_count(n_last_activations, block.x),
            block_count(activations.columns(), block.y),
            block_count(activations.rows(), block.z)
        );

        compute_output_gradients_kernel<LeakyReLU, CCE><<<grid, block>>>(last_activations.data(), activations.data(), activations.columns(), n_last_activations, activations.rows(), labels.data(), gradients.data());

        err = cudaGetLastError();
        if (err != cudaSuccess) CUDA_ERROR(err, "CUDA compute output layer gradients error: ");
    }

    void DeviceContext::optimiseLayer(Matrix& weights, const Matrix& gradients, const float learning_rate) const {
        assert(weights.size() == gradients.size());

        cudaError_t err;

        unsigned int grid_size = block_count(weights.size(), TILE_SIZE);

        optimise_kernel<<<grid_size, TILE_SIZE>>>(weights.data(), gradients.data(), weights.size(), learning_rate);

        err = cudaGetLastError();
        if (err != cudaSuccess) CUDA_ERROR(err, "CUDA optimise layer error: ");
    }
}