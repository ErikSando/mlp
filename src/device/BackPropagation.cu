#include <cassert>

#include "device/Activation.cuh"
#include "device/Loss.cuh"
#include "device/DeviceContext.hpp"

namespace mlp {
    /*
        Notes and stuff

        I don't know if I'm correctly using the "average across batches"

        For the output layer:
        dC/dw = -1 x activation of hidden node x (1 - activation of output node)
        where w connects "hidden node" (last hidden layer) and "output node" (output layer)

        For prior layers (not sure if this is the easiest way to compute it, just an idea for now):
        dC/dw = dC/da da/dw
        So basically split the derivate into the influence of w on a, then influence of a on C
        I think da/dw will be computed earlier and will account for all the next layers too

        Using cross categorial entropy loss function, softmax output activation function, and leaky ReLU hidden layer activation function

        for the output layer (l = L):

        y = 1 if this is the correct output class, otherwise 0
        w_l = a weight connecting a node from layer l-1 to layer l

        dC/da_L = -1/a_L
        da_L/dz_L = a_L(y - a_L)
        dz_L/dw_L = a_L-1  (the weight connecting a_L-1 and a_L)

        dC/dw_L = dC/da_L da_L/dz_L dz_L/dw_L
              = -1/a_L a_L(y - a_L) a_L-1
              = -a_L-1(y - a_L)
              = a_L-1(a_L - y)

        dC/dz_L = dC/da_L da_L/dz_L   (we need this for the next layer's gradient computation)
              = -1/a_L a_L(y - a_L)
              = a_L - y

        for the last hidden layer:

        z_L = Σ w_L a_L-1           the sum of the products of the preceding layer's activations and the weights connecting those nodes into the current layer's nodes
        dz_L/da_L-1 = w_L           the derivative of a logit in the current layer based on any activation in the previous layer

        da_L-1/dz_L-1 = { 1      if z_L-1 >= 0
                        { 0.01   if z_L-1 < 0

        dC/dz_L-1 = Σ(dC/dz_L dz_L/da_L-1) da_L-1/dz_L-1

        and then repeat the above procedure into every next layer to find the gradients of each weight in the network

        z_L-1 = Σ w_L-1 a_L-2
        dz_L-1/da_L-2 = w_L-1

        da_L-2/dz_L-2 = { 1      if z_L-2 >= 0
                        { 0.01   if z_L-2 < 0

        dC/dz_L-2 = Σ(dC/dz_L-1 dz_L-1/da_L-2) da_L-2/dz_L-2      we will have these pre computed, but it expands to:
                  = Σ(Σ(dC/dz_L dz_L/da_L-1) da_L-1/dz_L-1 dz_L-1/da_L-2) da_L-2/dz_L-2
                  = Σ(Σ(dC/da_L da_L/dz_L dz_L/da_L-1) da_L-1/dz_L-1 dz_L-1/da_L-2) da_L-2/dz_L-2

    */

    template<typename TActivation>
    __global__ void compute_gradients_kernel(
        const float* last_gradients, const float* activations, // last_gradients is dC/da for the nodes in the next layer (left to right), "last" in the sense that we are moving backwards through the layers
        const size_t n_last_gradients, const size_t n_activations, const size_t batch_size,
        float* gradients, float* dC_da
    ) {
        unsigned int last_gradient_index = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int activation_index = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned int batch = blockIdx.z * blockDim.z + threadIdx.z;

        if (last_gradient_index >= n_last_gradients || activation_index >= n_activations || batch >= batch_size) return;

        if (last_gradients[last_gradient_index] == 0) return; // I don't think this will ever be true, every node in the next layer connects to the correct output node

        unsigned int weight_index = activation_index * n_activations + last_gradient_index;

        // I wrote this code hella sleepy i don't know what's going on please fix it

        float da_dz = TActivation::derivative(activations[activation_index]);
        float dz_dw = activations[activation_index];
        float da_dw = da_dz * dz_dw;
        float dC_dw = da_dw * last_gradients[last_gradient_index];// / (float) batch_size;

        gradients[weight_index] = dC_dw;
    }

    template<typename TActivation, typename TLoss>
    __global__ void compute_output_gradients_kernel(
        const float* last_hidden_activations, const float* activations,
        const float* weights,
        const size_t n_activations, const size_t n_last_activations, const size_t batch_size,
        const int* labels, float* gradients, float* dC_da_hidden
    ) {
        unsigned int last_activation_index = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int activation_index = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned int batch = blockIdx.z * blockDim.z + threadIdx.z;

        if (last_activation_index >= n_last_activations || activation_index >= n_activations || batch >= batch_size) return;

        unsigned int weight_index = last_activation_index * n_activations + activation_index;

        float a_L = activations[batch * n_activations + activation_index];
        float a_Lminus1 = last_hidden_activations[batch * n_last_activations + last_activation_index];

        float y = 0.0f;
        if (activation_index == labels[batch]) y = 1.0f;

        float output_delta = a_L - y;

        // need some reduction method for these, fix it later:

        // i think dC_dz is usually used not dC_da, ill maybe change it later
        // i need z_Lminus1 for the activation derivative, works using a_Lminus1 for all the actiation functions i have, but probably is bad practice
        // dC_da_hidden[batch * n_last_activations + last_activation_index] += TActivation::derivative(a_Lminus1) * output_delta * weights[weight_index];
        atomicAdd(&dC_da_hidden[batch * n_last_activations + last_activation_index], TActivation::derivative(a_Lminus1) * output_delta * weights[weight_index]);

        // dC/dw = a_l-1 (a_l - y)
        // gradients[weight_index] += a_Lminus1 * output_delta;
        atomicAdd(&gradients[weight_index], a_Lminus1 * output_delta);
    }

    __global__ void optimise_layer_kernel(float* weights, const float* gradients, const size_t n_weights, const float learning_rate) {
        // unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int index = threadIdx.x;

        if (index >= n_weights) return;

        weights[index] += gradients[index] * learning_rate;
    }

    void DeviceContext::computeGradients(
        const Matrix& last_gradients, const Matrix& activations, 
        const Activation activation,
        Matrix& gradients, Matrix& dC_da_hidden
    ) const {
        assert(last_gradients.rows() == activations.rows());

        cudaError_t err;

        dim3 block(8, 8, 8);

        dim3 grid(
            block_count(last_gradients.columns(), block.x),
            block_count(activations.columns(), block.y),
            block_count(activations.rows(), block.z)
        );

        compute_gradients_kernel<LeakyReLU><<<grid, block>>>(
            last_gradients.data(), activations.data(),
            last_gradients.columns(), activations.columns(), activations.rows(),
            gradients.data(), dC_da_hidden.data()
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) CUDA_ERROR(err, "CUDA compute hidden layer gradients error: ");
    }

    void DeviceContext::computeOutputGradients(
        const Matrix& last_activations, const Matrix& activations, const Matrix& weights,
        const size_t n_last_activations,
        const std::vector<int>& labels,
        const Activation activation, const Loss loss,
        Matrix& gradients, Matrix& dC_da_hidden
    ) const {
        cudaError_t err;

        dim3 block(8, 8, 8);

        dim3 grid(
            block_count(n_last_activations, block.x),
            block_count(activations.columns(), block.y),
            block_count(activations.rows(), block.z)
        );

        compute_output_gradients_kernel<LeakyReLU, CCE><<<grid, block>>>(
            last_activations.data(), activations.data(), weights.data(),
            activations.columns(), n_last_activations, activations.rows(),
            labels.data(),
            gradients.data(), dC_da_hidden.data()
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) CUDA_ERROR(err, "CUDA compute output layer gradients error: ");
    }

    void DeviceContext::optimiseLayer(Matrix& weights, const Matrix& gradients, const float learning_rate) const {
        assert(weights.size() == gradients.size());

        cudaError_t err;

        unsigned int grid_size = block_count(weights.size(), TILE_SIZE);

        optimise_layer_kernel<<<grid_size, TILE_SIZE>>>(weights.data(), gradients.data(), weights.size(), learning_rate);

        err = cudaGetLastError();
        if (err != cudaSuccess) CUDA_ERROR(err, "CUDA optimise layer error: ");
    }
}