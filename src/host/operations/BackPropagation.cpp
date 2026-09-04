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
            float* weight_gradients, float* bias_gradients, float* dC_da_next
        ) {
             for (unsigned int sample = 0; sample < batch_size; sample++) {
                for (unsigned int left_index = 0; left_index < n_left; left_index++) {
                    for (unsigned int right_index = 0; right_index < n_right; right_index++) {
                        unsigned int weight_index = left_index * n_right + right_index;

                        float a = a_right[sample * n_right + right_index];
                        float aleft = a_left[sample * n_left + left_index]; // preceding layer activation

                        // i think using a to find the derivate instead of z is better, less computations
                        float da_dz = TActivation::derivative_from_a(a);
                        float dz_dw = aleft;
                        float dC_da = dC_da_gradients[sample * n_right + right_index];

                        float dC_dz = dC_da * da_dz;
                        float dC_dw = dC_dz * dz_dw;
                        float dC_db = dC_dz; // dC/db = dC/dz * dz/db, and dz/db = 1 therefore dC/db = dC/dz

                        float dz_da_left = weights[weight_index];
                        float dC_da_left = dC_dz * dz_da_left;

                        dC_da_next[sample * n_left + left_index] += dC_da_left;
                        weight_gradients[weight_index] += dC_dw / batch_size;
                        if (left_index == 0) bias_gradients[right_index] += dC_db / batch_size;
                    }
                }
            }
        }

        template<typename TActivation, typename TLoss>
        void compute_output_gradients(
            const float* a_hidden_list, const float* a_output_list,
            const float* weights,
            const size_t n_output, const size_t n_hidden, const size_t batch_size,
            const int* labels, float* weight_gradients, float* bias_gradients, float* dC_da_hidden_list
        ) {
            for (unsigned int sample = 0; sample < batch_size; sample++) {
                for (unsigned int hidden_index = 0; hidden_index < n_hidden; hidden_index++) {
                    for (unsigned int output_index = 0; output_index < n_output; output_index++) {
                        unsigned int weight_index = hidden_index * n_output + output_index;

                        float a_output = a_output_list[sample * n_output + output_index];
                        float a_hidden = a_hidden_list[sample * n_hidden + hidden_index];

                        // this is using softmax + cce right now, need to generalise

                        float y = 0.0f;
                        if (output_index == labels[sample]) y = 1.0f;

                        float dC_dz_output = a_output - y; // dC/dz_L = a_L - y
                        float dC_dw_output = a_hidden * dC_dz_output; // dC/dw_L = a_L-1 (a_L - y)    where w_L connects a_L-1 and a_L
                        float dC_db_output = dC_dz_output; // dC/db = dC/dz * dz/db, and dz/db = 1 therefore dC/db = dC/dz

                        float dz_output_da_hidden = weights[weight_index]; // dz_L/da_L-1 = w_L

                        float dC_da_hidden = dC_dz_output * dz_output_da_hidden;

                        dC_da_hidden_list[sample * n_hidden + hidden_index] += dC_da_hidden;
                        weight_gradients[weight_index] += dC_dw_output / batch_size;
                        if (hidden_index == 0) bias_gradients[output_index] += dC_db_output / batch_size;
                    }
                }
            }
        }

        void optimise_layer(float* weights, float* biases, const float* weight_gradients, const float* bias_gradients, const size_t rows, const size_t columns, const float learning_rate) {
            for (size_t c = 0; c < columns; c++) {
                biases[c] -= bias_gradients[c] * learning_rate;

                for (size_t r = 0; r < rows; r++) {
                    size_t i = r * columns + c;
                    weights[i] -= weight_gradients[i] * learning_rate;
                }
            }
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

            if (m_profiler) m_profiler->startTask("Compute Hidden Layer Gradients");

            switch (activation) {
                case Activation::SIGMOID:

                break;

                case Activation::TANH:

                break;

                case Activation::RELU:
                    compute_hidden_gradients<ReLU>(
                        dC_da.data(),
                        left_activations.data(), right_activations.data(),
                        weights.data(),
                        left_activations.columns(), right_activations.columns(), left_activations.rows(),
                        weight_gradients.data(), bias_gradients.data(), dC_da_next.data()
                    );
                break;

                case Activation::LEAKY_RELU:
                    compute_hidden_gradients<LeakyReLU>(
                        dC_da.data(),
                        left_activations.data(), right_activations.data(),
                        weights.data(),
                        left_activations.columns(), right_activations.columns(), left_activations.rows(),
                        weight_gradients.data(), bias_gradients.data(), dC_da_next.data()
                    );
                break;

                default:
                    compute_hidden_gradients<NoActivation>(
                        dC_da.data(),
                        left_activations.data(), right_activations.data(),
                        weights.data(),
                        left_activations.columns(), right_activations.columns(), left_activations.rows(),
                        weight_gradients.data(), bias_gradients.data(), dC_da_next.data()
                    );
                break;
            }

            if (m_profiler) m_profiler->endTask("Compute Hidden Layer Gradients");
        }

        void Context::computeOutputGradients(
            const Matrix_t& last_hidden_activations, const Matrix_t& output_activations,
            const Matrix_t& weights,
            const std::vector<int>& labels,
            const OALP al_pair,
            Matrix_t& weight_gradients, Matrix_t& bias_gradients, Matrix_t& dC_da_hidden
        ) const {
            assert(last_hidden_activations.rows() == output_activations.rows());
            assert(weights.size() == last_hidden_activations.columns() * output_activations.columns());
            assert(dC_da_hidden.size() == last_hidden_activations.size());
            assert(weight_gradients.size() == weights.size());
            assert(bias_gradients.size() == output_activations.columns());

            if (m_profiler) m_profiler->startTask("Compute Output Layer Gradients");

            switch (al_pair) {
                case OALP::NONE_CCE:
                    compute_output_gradients<NoActivation, CCE>(
                        last_hidden_activations.data(), output_activations.data(),
                        weights.data(),
                        output_activations.columns(), last_hidden_activations.columns(), output_activations.rows(),
                        labels.data(), weight_gradients.data(), bias_gradients.data(), dC_da_hidden.data()
                    );
                break;

                case OALP::NONE_MSE:
                    compute_output_gradients<NoActivation, MSE>(
                        last_hidden_activations.data(), output_activations.data(),
                        weights.data(),
                        output_activations.columns(), last_hidden_activations.columns(), output_activations.rows(),
                        labels.data(), weight_gradients.data(), bias_gradients.data(), dC_da_hidden.data()
                    );

                break;

                case OALP::SOFTMAX_CCE:
                    compute_output_gradients<Softmax, CCE>(
                        last_hidden_activations.data(), output_activations.data(),
                        weights.data(),
                        output_activations.columns(), last_hidden_activations.columns(), output_activations.rows(),
                        labels.data(), weight_gradients.data(), bias_gradients.data(), dC_da_hidden.data()
                    );
                break;

                case OALP::SOFTMAX_MSE:
                    compute_output_gradients<Softmax, MSE>(
                        last_hidden_activations.data(), output_activations.data(),
                        weights.data(),
                        output_activations.columns(), last_hidden_activations.columns(), output_activations.rows(),
                        labels.data(), weight_gradients.data(), bias_gradients.data(), dC_da_hidden.data()
                    );
                break;

                default:
                    throw std::runtime_error("How did we get here?");
                break;
            }

            if (m_profiler) m_profiler->endTask("Compute Output Layer Gradients");
        }

        void Context::optimiseLayer(Matrix_t& weights, Matrix_t& biases, const Matrix_t& weight_gradients, const Matrix_t& bias_gradients, const float learning_rate) const {
            assert(weights.size() == weight_gradients.size());
            assert(biases.size() == bias_gradients.size());

            if (m_profiler) m_profiler->startTask("Optimise Layer");

            optimise_layer(weights.data(), biases.data(), weight_gradients.data(), bias_gradients.data(), weights.rows(), weights.columns(), learning_rate);

            if (m_profiler) m_profiler->endTask("Optimise Layer");
        }
    }
}