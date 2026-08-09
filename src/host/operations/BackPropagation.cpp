#include "host/Context.hpp"
#include "host/operations/Activation.hpp"
#include "host/operations/Loss.hpp"

namespace mlp {
    namespace host {
        template<typename TActivation>
        void compute_hidden_gradients(
            const float* dC_da_gradients,
            const float* a_left, const float* a_right,
            const float* weights,
            const size_t n_left, const size_t n_right, const size_t batch_size,
            float* gradients, float* dC_da_next
        ) {
             for (unsigned int batch = 0; batch < batch_size; batch++) {
                for (unsigned int left_index = 0; left_index < n_left; left_index++) {
                    for (unsigned int right_index = 0; right_index < n_right; right_index++) {
                        unsigned int weight_index = left_index * n_right + right_index;

                        float a = a_right[batch * n_right + right_index];
                        float aleft = a_left[batch * n_left + left_index]; // preceding layer activation

                        float da_dz = TActivation::derivative(a); // works for leaky relu, switch to using z later, need to add logits into the argument list
                        float dz_dw = aleft;
                        float dC_da = dC_da_gradients[batch * n_right + right_index];

                        float dC_dz = dC_da * da_dz;
                        float dC_dw = dC_dz * dz_dw;

                        float dz_da_left = weights[weight_index];
                        float dC_da_left = dC_dz * dz_da_left;

                        dC_da_next[batch * n_left + left_index] += dC_da_left;
                        gradients[weight_index] += dC_dw / batch_size;
                    }
                }
            }
        }

        template<typename TActivation, typename TLoss>
        void compute_output_gradients(
            const float* a_hidden_list, const float* a_output_list,
            const float* weights,
            const size_t n_output, const size_t n_hidden, const size_t batch_size,
            const int* labels, float* gradients, float* dC_da_hidden_list
        ) {
            for (unsigned int batch = 0; batch < batch_size; batch++) {
                for (unsigned int hidden_index = 0; hidden_index < n_hidden; hidden_index++) {
                    for (unsigned int output_index = 0; output_index < n_output; output_index++) {
                        unsigned int weight_index = hidden_index * n_output + output_index;

                        float a_output = a_output_list[batch * n_output + output_index];
                        float a_hidden = a_hidden_list[batch * n_hidden + hidden_index];

                        float y = 0.0f;
                        if (output_index == labels[batch]) y = 1.0f;

                        float dC_dz_output = a_output - y; // dC/dz_L = a_L - y
                        float dC_dw_output = a_hidden * dC_dz_output; // dC/dw_L = a_L-1 (a_L - y)    where w_L connects a_L-1 and a_L

                        float dz_output_da_hidden = weights[weight_index]; // dz_L/da_L-1 = w_L

                        float dC_da_hidden = dC_dz_output * dz_output_da_hidden;

                        dC_da_hidden_list[batch * n_hidden + hidden_index] += dC_da_hidden;
                        gradients[weight_index] += dC_dw_output / batch_size;
                    }
                }
            }
        }

        void optimise_layer(float* weights, const float* gradients, const size_t size, const float learning_rate) {
            for (size_t i = 0; i < size; i++) {
                weights[i] -= gradients[i] * learning_rate;
            }
        }

        void Context::computeGradients(
            const Matrix_t& dC_da,
            const Matrix_t& left_activations, const Matrix_t& right_activations,
            const Matrix_t& weights,
            const Activation activation,
            Matrix_t& gradients, Matrix_t& dC_da_next
        ) const {
            assert(left_activations.rows() == right_activations.rows());
            assert(weights.size() == left_activations.columns() * right_activations.columns());
            assert(dC_da.size() == right_activations.size());
            assert(dC_da_next.size() == left_activations.size());
            assert(gradients.size() == weights.size());

            switch (activation) {
                case Activation::SIGMOID:

                break;

                case Activation::TANH:

                break;

                case Activation::RELU:

                break;

                case Activation::LEAKY_RELU:
                    compute_hidden_gradients<LeakyReLU>(
                        dC_da.data(),
                        left_activations.data(), right_activations.data(),
                        weights.data(),
                        left_activations.columns(), right_activations.columns(), left_activations.rows(),
                        gradients.data(), dC_da_next.data()
                    );
                break;

                default:

                break;
            }
        }

        void Context::computeOutputGradients(
            const Matrix_t& last_hidden_activations, const Matrix_t& output_activations,
            const Matrix_t& weights,
            const std::vector<int>& labels,
            const Activation activation, const Loss loss,
            Matrix_t& gradients, Matrix_t& dC_da_hidden
        ) const {
            assert(last_hidden_activations.rows() == output_activations.rows());
            assert(weights.size() == last_hidden_activations.columns() * output_activations.columns());
            assert(dC_da_hidden.size() == last_hidden_activations.size());
            assert(gradients.size() == weights.size());

            compute_output_gradients<Softmax, CCE>(
                last_hidden_activations.data(), output_activations.data(),
                weights.data(),
                output_activations.columns(), last_hidden_activations.columns(), output_activations.rows(),
                labels.data(), gradients.data(), dC_da_hidden.data()
            );
        }

        void Context::optimiseLayer(Matrix_t& weights, const Matrix_t& gradients, const float learning_rate) const {
            assert(weights.size() == gradients.size());

            optimise_layer(weights.data(), gradients.data(), weights.size(), learning_rate);
        }
    }
}