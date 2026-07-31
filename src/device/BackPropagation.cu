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

    // I've gotta change the kernel dimensions I think, I need one thread for each weight connecting two layers

    template<typename TActivation, typename TLoss>
    __global__ void compute_gradients_kernel(const float* last_gradients, const float* activations, const size_t n_last_gradients, const size_t n_activations, const size_t batch_size, float* gradients) {
        unsigned int lgi = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int ai = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned int batch = blockIdx.z * blockDim.z + threadIdx.z;

        if (lgi >= n_last_gradients || ai >= n_activations || batch >= batch_size) return;

        // I'm preeeeetty sure that the gradients are averaged across batches
    }

    template<typename TActivation, typename TLoss>
    __global__ void compute_output_gradients_kernel(const float* activations, const size_t n_activations, const size_t batch_size, float* gradients) {
        unsigned int ai = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int batch = blockIdx.y * blockDim.y + threadIdx.y;

        if (ai >= n_activations || batch >= batch_size) return;

    }

    // void DeviceContext::backPropagate(const Loss loss) const {

    // }

    void DeviceContext::computeGradients(const Matrix& last_gradients, const Matrix& activations, const Activation activation, const Loss loss, Matrix& gradients) const {
        assert(last_gradients.rows() == activations.rows());

        cudaError_t err;

        dim3 block(TILE_SIZE, TILE_SIZE, TILE_SIZE);

        dim3 grid(
            block_count(last_gradients.columns(), block.x),
            block_count(activations.columns(), block.y),
            block_count(activations.rows(), block.z)
        );

        compute_gradients_kernel<LeakyReLU, CCE><<<grid, block>>>(last_gradients.data(), activations.data(), last_gradients.columns(), activations.columns(), activations.rows(), gradients.data());

        err = cudaGetLastError();
        if (err != cudaSuccess) CUDA_ERROR(err, "CUDA compute hidden layer gradients error: ");
    }

    void DeviceContext::computeGradients(const Matrix& activations, const Activation activation, const Loss loss, Matrix& gradients) const {
        cudaError_t err;

        dim3 block(TILE_SIZE, TILE_SIZE);

        dim3 grid(
            block_count(activations.columns(), block.x),
            block_count(activations.rows(), block.y)
        );

        compute_output_gradients_kernel<LeakyReLU, CCE><<<grid, block>>>(activations.data(), activations.columns(), activations.rows(), gradients.data());

        err = cudaGetLastError();
        if (err != cudaSuccess) CUDA_ERROR(err, "CUDA compute output layer gradients error: ");
    }
}