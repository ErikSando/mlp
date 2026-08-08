#include "host/Context.hpp"
#include "host/operations/Activation.hpp"

namespace mlp {
    namespace host {
        template<typename TActivation>
        void propagate(
            const float* inputs, float* logits, float* activations,
            const float* weights, const float* biases,
            const size_t batch_size, const size_t input_count, const size_t output_count
        ) {
            for (unsigned int batch = 0; batch < batch_size; batch++) {
                for (unsigned int output_column = 0; output_column < output_count; output_column++) {
                    unsigned int output_index = batch * output_count + output_column;

                    logits[output_index] = biases[output_column];

                    for (unsigned int input_column = 0; input_column < input_count; input_column++) {
                        unsigned int input_index = batch * input_count + input_column;
                        unsigned int weight_index = input_index * input_count + output_index;

                        logits[output_index] += inputs[input_index] * weights[weight_index];
                    }

                    activations[output_index] = TActivation::activate(logits[output_index]);
                }
            }
        }

        void Context::propagate(
            const Matrix& inputs,
            Matrix& logits, Matrix& activations,
            const Matrix& weights, const Matrix& biases,
            const Activation activation
        ) const {
            assert(inputs.columns() == weights.rows());
            assert(activations.rows() == inputs.rows());
            assert(activations.columns() == weights.columns());
            assert(activations.size() == logits.size());

            switch (activation) {
                case Activation::SIGMOID:

                break;

                case Activation::TANH:
                    
                break;

                case Activation::RELU:
                    
                break;

                case Activation::LEAKY_RELU:
                    
                break;

                case Activation::SOFTMAX:
                    
                break;
            }
        }
    }
}