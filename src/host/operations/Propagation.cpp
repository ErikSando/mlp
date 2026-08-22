#include <cassert>

#include "host/Context.hpp"
#include "host/operations/Activation.hpp"
#include "host/operations/Softmax.hpp"

namespace mlp {
    namespace host {
        template<typename TActivation = NoActivation>
        void propagate_op(
            const float* inputs, float* logits, float* activations,
            const float* weights, const float* biases,
            const size_t batch_size, const size_t input_count, const size_t output_count
        ) {
            for (unsigned int sample = 0; sample < batch_size; sample++) {
                for (unsigned int output_column = 0; output_column < output_count; output_column++) {
                    unsigned int output_index = sample * output_count + output_column;

                    logits[output_index] = biases[output_column];

                    for (unsigned int input_column = 0; input_column < input_count; input_column++) {
                        unsigned int input_index = sample * input_count + input_column;
                        unsigned int weight_index = input_column * output_count + output_column;

                        assert(input_index < batch_size * input_count);
                        assert(output_index < batch_size * output_count);
                        assert(weight_index < input_count * output_count);

                        logits[output_index] += inputs[input_index] * weights[weight_index];
                    }

                    activations[output_index] = TActivation::activate(logits[output_index]);
                }
            }
        }

        void Context::propagate(
            const Matrix_t& inputs,
            Matrix_t& logits, Matrix_t& activations,
            const Matrix_t& weights, const Matrix_t& biases,
            const Activation activation
        ) const {
            assert(inputs.columns() == weights.rows());
            assert(activations.rows() == inputs.rows());
            assert(activations.columns() == weights.columns());
            assert(activations.size() == logits.size());

            if (m_profiler) m_profiler->startTask("Propagation");

            switch (activation) {
                case Activation::SIGMOID:
                    propagate_op<Sigmoid>(inputs.data(), logits.data(), activations.data(), weights.data(), biases.data(), inputs.rows(), inputs.columns(), activations.columns());
                break;

                case Activation::TANH:
                    propagate_op<Tanh>(inputs.data(), logits.data(), activations.data(), weights.data(), biases.data(), inputs.rows(), inputs.columns(), activations.columns());
                break;

                case Activation::RELU:
                    propagate_op<ReLU>(inputs.data(), logits.data(), activations.data(), weights.data(), biases.data(), inputs.rows(), inputs.columns(), activations.columns());
                break;

                case Activation::LEAKY_RELU:
                    propagate_op<LeakyReLU>(inputs.data(), logits.data(), activations.data(), weights.data(), biases.data(), inputs.rows(), inputs.columns(), activations.columns());
                break;

                case Activation::SOFTMAX:
                    propagate_op(inputs.data(), logits.data(), activations.data(), weights.data(), biases.data(), inputs.rows(), inputs.columns(), activations.columns());
                    softmax_op(logits.data(), activations.data(), logits.rows(), logits.columns());
                break;

                default:
                    propagate_op<NoActivation>(inputs.data(), logits.data(), activations.data(), weights.data(), biases.data(), inputs.rows(), inputs.columns(), activations.columns());
                break;
            }

            if (m_profiler) m_profiler->endTask();
        }
    }
}