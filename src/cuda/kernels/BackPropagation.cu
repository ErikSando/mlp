#include <cassert>

#include "cuda/kernels/Activation.cuh"
#include "cuda/kernels/Loss.cuh"
#include "cuda/Context.hpp"
#include "enums/Enums.hpp"

namespace mlp {
    namespace cuda {
        /*
            Notes and stuff

            Explanation using cross categorial entropy loss function, softmax output activation function, and leaky ReLU hidden layer activation function:

            * I may have made mistakes here

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

            dC/dz_L = dC/da_L da_L/dz_L
                = -1/a_L a_L(y - a_L)
                = a_L - y

            z_L = Σ(w_L a_L-1)

            dz_L/da_L-1 = w_L     (summed for every weight connecting nodes between the two layers)

            dC/da_L-1 = dC/dz_L dz_L/da_L-1       (we need this for the next layer's gradient computation)
                    = w_L(a_L - y)

            for the last hidden layer:

            da_L-1/dz_L-1 = { 1      if z_L-1 >= 0
                            { 0.01   if z_L-1 < 0

            dz_L-1/dw_L-1 = a_L-2

            dC/dz_L-1 = dC/da_L-1 da_L-1/dz_L-1

            dC/dw_L-1 = dC/dz_L-1 dz_L-1/dw_L-1
                    = a_L-2 dC/da_L-1 da_L-1/dz_L-1

            dz_L-1/da_L-2 = w_L-1

            dC/da_L-2 = dC/dz_L-1 dz_L-1/da_L-2
                    = w_L-1 Σ(dC/dz_L-1)

            and repeat through, in general it looks something like:

            for every layer l, we will have dC/da_l pre computed
            * I'm saying "layer", but each step is looking at the weights between layers so it doesn't really make sense to say for every layer
            I think I should say we have dC/da pre computed for the layer on the right or the succeeding layer

            da_l/dz_l = { 1      if z_L-1 >= 0
                        { 0.01   if z_L-1 < 0

            dz_l/dw_l = a_l-1

            dC/dz_l = dC/da_l da_l/dz_l
            dC/dw_l = dC/da_l da_l/dz_l dz_l/dw_l
                    = a_l-1 dC/da_l da_l/dz_l

            dz_l/da_l-1 = w_l    for every pair of nodes in layers l-1 and l

            dC/da_l-1 = dC/dz_l dz_l/da_l-1

        */

        template<typename TActivation>
        __global__ void compute_hidden_gradients_kernel(
            const float* dC_da_gradients,
            const float* a_left, const float* a_right,
            const float* weights,
            const size_t n_left, const size_t n_right, const size_t batch_size,
            float* weight_gradients, float* bias_gradients, float* dC_da_next
        ) {
            unsigned int left_index = blockIdx.x * blockDim.x + threadIdx.x;
            unsigned int right_index = blockIdx.y * blockDim.y + threadIdx.y;
            unsigned int sample = blockIdx.z * blockDim.z + threadIdx.z;

            if (left_index >= n_left || right_index >= n_right || sample >= batch_size) return;

            unsigned int weight_index = left_index * n_right + right_index;

            float a = a_right[sample * n_right + right_index];
            float aleft = a_left[sample * n_left + left_index]; // preceding layer activation

            float da_dz = TActivation::derivative(a); // works for leaky relu, switch to using z later, need to add logits into the kernel launch
            float dz_dw = aleft;
            float dC_da = dC_da_gradients[sample * n_right + right_index];

            float dC_dz = dC_da * da_dz;
            float dC_dw = dC_dz * dz_dw;
            float dC_db = dC_dz;

            float dz_da_left = weights[weight_index];
            float dC_da_left = dC_dz * dz_da_left;

            // need some reduction method for these, fix it later:

            atomicAdd(&dC_da_next[sample * n_left + left_index], dC_da_left);
            atomicAdd(&weight_gradients[weight_index], dC_dw / batch_size);

            if (left_index == 0) {
                atomicAdd(&bias_gradients[right_index], dC_db / batch_size);
            }
        }

        template<typename TActivation, typename TLoss>
        // template<OALP TOALP>
        __global__ void compute_output_gradients_kernel(
            const float* a_hidden_list, const float* a_output_list,
            const float* weights,
            const size_t n_output, const size_t n_hidden, const size_t batch_size,
            const int* labels, float* weight_gradients, float* bias_gradients, float* dC_da_hidden_list
        ) {
            unsigned int hidden_index = blockIdx.x * blockDim.x + threadIdx.x;
            unsigned int output_index = blockIdx.y * blockDim.y + threadIdx.y;
            unsigned int sample = blockIdx.z * blockDim.z + threadIdx.z;

            if (hidden_index >= n_hidden || output_index >= n_output || sample >= batch_size) return;

            unsigned int weight_index = hidden_index * n_output + output_index;

            float a_output = a_output_list[sample * n_output + output_index];
            float a_hidden = a_hidden_list[sample * n_hidden + hidden_index];

            // this is using softmax + cce right now, need to generalise

            // if constexpr (std::is_same_v<TOALP, SOFTMAX_NONE> && std::is_same_v) {}
            float y = 0.0f;
            if (output_index == labels[sample]) y = 1.0f;

            float dC_dz_output = a_output - y; // dC/dz_L = a_L - y
            float dC_dw_output = a_hidden * dC_dz_output; // dC/dw_L = a_L-1 (a_L - y)    where w_L connects a_L-1 and a_L
            float dC_db_output = dC_dz_output;

            float dz_output_da_hidden = weights[weight_index]; // dz_L/da_L-1 = w_L

            float dC_da_hidden = dC_dz_output * dz_output_da_hidden; // dC/da_L-1 = Σ(dC/dz_L dz_L/da_L-1)   we don't have to worry about summing here, we do that later

            // need some reduction method for these, fix it later:

            atomicAdd(&dC_da_hidden_list[sample * n_hidden + hidden_index], dC_da_hidden);
            atomicAdd(&weight_gradients[weight_index], dC_dw_output / batch_size);

            if (hidden_index == 0) {
                atomicAdd(&bias_gradients[output_index], dC_db_output / batch_size);
            }
        }

        __global__ void optimise_layer_kernel(float* weights, float* biases, const float* weight_gradients, const float* bias_gradients, const size_t n_weights, const size_t n_biases, const float learning_rate) {
            unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

            if (index >= n_weights) return;

            weights[index] -= weight_gradients[index] * learning_rate;

            if (index < n_biases) biases[index] -= bias_gradients[index] * learning_rate;
        }

        void Context::computeGradients(
            const Matrix_t& dC_da,
            const Matrix_t& left_activations, const Matrix_t& right_activations,
            const Matrix_t& weights,
            const Activation activation,
            Matrix_t& weight_gradients, Matrix_t& bias_gradients, Matrix_t& dC_da_next
        ) const {
            assert(left_activations.rows() == right_activations.rows());
            assert(weights.size() == left_activations.columns() * right_activations.columns());
            assert(dC_da.size() == right_activations.size());
            assert(dC_da_next.size() == left_activations.size());
            assert(weight_gradients.size() == weights.size());
            assert(bias_gradients.size() == right_activations.columns());

            cudaError_t err;

            if (m_profiler) m_profiler->startTask("Compute Hidden Layer Gradients");

            dim3 block(8, 8, 8);

            dim3 grid(
                block_count(left_activations.columns(), block.x),
                block_count(right_activations.columns(), block.y),
                block_count(right_activations.rows(), block.z)
            );

            compute_hidden_gradients_kernel<LeakyReLU><<<grid, block>>>(
                dC_da.data(),
                left_activations.data(), right_activations.data(),
                weights.data(),
                left_activations.columns(), right_activations.columns(), right_activations.rows(),
                weight_gradients.data(), bias_gradients.data(), dC_da_next.data()
            );

            if (m_profiler) m_profiler->endTask("Compute Hidden Layer Gradients");

            err = cudaGetLastError();
            if (err != cudaSuccess) CUDA_ERROR(err, "CUDA compute hidden layer gradients error: ");
        }

        void Context::computeOutputGradients(
            const Matrix_t& last_hidden_activations, const Matrix_t& output_activations, const Matrix_t& weights,
            const std::vector<int>& labels,
            const OALP al_pair,
            Matrix_t& weight_gradients, Matrix_t& bias_gradients, Matrix_t& dC_da_hidden
        ) const {
            assert(last_hidden_activations.rows() == output_activations.rows());
            assert(weights.size() == last_hidden_activations.columns() * output_activations.columns());
            assert(dC_da_hidden.size() == last_hidden_activations.size());
            assert(weight_gradients.size() == weights.size());
            assert(bias_gradients.size() == output_activations.columns());

            cudaError_t err;

            dim3 block(8, 8, 8);

            dim3 grid(
                block_count(last_hidden_activations.columns(), block.x),
                block_count(output_activations.columns(), block.y),
                block_count(output_activations.rows(), block.z)
            );

            Buffer_t device_labels(labels.size() * sizeof(int));
            transfer(device_labels, (void*) labels.data());

            // intelli sense was acting up, so ill leave the template parameters how they were and try change it later so i dont need this switch case
            // i want to use compile time branching in the kernel to split behaviour based on the output activation and loss function

            if (m_profiler) m_profiler->startTask("Compute Output Layer Gradients");

            switch (al_pair) {
                case OALP::NONE_CCE:
                    compute_output_gradients_kernel<NoActivation, CCE><<<grid, block>>>(
                        last_hidden_activations.data(), output_activations.data(), weights.data(),
                        output_activations.columns(), last_hidden_activations.columns(), output_activations.rows(),
                        (int*) device_labels.data(),
                        weight_gradients.data(), bias_gradients.data(), dC_da_hidden.data()
                    );
                break;

                case OALP::NONE_MSE:
                    compute_output_gradients_kernel<NoActivation, CCE><<<grid, block>>>(
                        last_hidden_activations.data(), output_activations.data(), weights.data(),
                        output_activations.columns(), last_hidden_activations.columns(), output_activations.rows(),
                        (int*) device_labels.data(),
                        weight_gradients.data(), bias_gradients.data(), dC_da_hidden.data()
                    );
                break;

                case OALP::SOFTMAX_CCE:
                    compute_output_gradients_kernel<Softmax, CCE><<<grid, block>>>(
                        last_hidden_activations.data(), output_activations.data(), weights.data(),
                        output_activations.columns(), last_hidden_activations.columns(), output_activations.rows(),
                        (int*) device_labels.data(),
                        weight_gradients.data(), bias_gradients.data(), dC_da_hidden.data()
                    );
                break;

                case OALP::SOFTMAX_MSE:
                    compute_output_gradients_kernel<Softmax, MSE><<<grid, block>>>(
                        last_hidden_activations.data(), output_activations.data(), weights.data(),
                        output_activations.columns(), last_hidden_activations.columns(), output_activations.rows(),
                        (int*) device_labels.data(),
                        weight_gradients.data(), bias_gradients.data(), dC_da_hidden.data()
                    );
                break;

                default:
                    throw std::runtime_error("How did we get here?");
                break;
            }

            if (m_profiler) m_profiler->endTask("Compute Output Layer Gradients");

            err = cudaGetLastError();
            if (err != cudaSuccess) CUDA_ERROR(err, "CUDA compute output layer gradients error: ");
        }

        void Context::optimiseLayer(Matrix_t& weights, Matrix& biases, const Matrix_t& weight_gradients, const Matrix_t& bias_gradients, const float learning_rate) const {
            assert(weights.size() == weight_gradients.size());
            assert(biases.size() == bias_gradients.size());

            cudaError_t err;

            if (m_profiler) m_profiler->startTask("Optimise Layer");

            unsigned int grid_size = block_count(weights.size(), TILE_SIZE);

            optimise_layer_kernel<<<grid_size, TILE_SIZE>>>(weights.data(), biases.data(), weight_gradients.data(), bias_gradients.data(), weights.size(), biases.size(), learning_rate);

            if (m_profiler) m_profiler->endTask("Optimise Layer");

            err = cudaGetLastError();
            if (err != cudaSuccess) CUDA_ERROR(err, "CUDA optimise layer error: ");
        }
    }
}