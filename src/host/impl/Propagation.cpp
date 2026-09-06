#include <cassert>

#include "context/Context.hpp"
#include "host/impl/Activation.hpp"
#include "host/impl/Softmax.hpp"

namespace mlp {
    template<typename TActivation = NoActivation>
    void propagate_op(
        const float* left_activations, float* right_logits, float* right_activations,
        const float* weights, const float* biases,
        const size_t batch_size, const size_t input_count, const size_t output_count
    ) {
        for (unsigned int sample = 0; sample < batch_size; sample++) {
            for (unsigned int output_column = 0; output_column < output_count; output_column++) {
                unsigned int output_index = sample * output_count + output_column;

                right_logits[output_index] = biases[output_column];

                for (unsigned int input_column = 0; input_column < input_count; input_column++) {
                    unsigned int input_index = sample * input_count + input_column;
                    unsigned int weight_index = input_column * output_count + output_column;

                    assert(input_index < batch_size * input_count);
                    assert(output_index < batch_size * output_count);
                    assert(weight_index < input_count * output_count);

                    right_logits[output_index] += left_activations[input_index] * weights[weight_index];
                }

                right_activations[output_index] = TActivation::activate(right_logits[output_index]);
            }
        }
    }

    void Context::propagate(
        const Matrix& left_activations,
        Matrix& right_logits, Matrix& right_activations,
        const Matrix& weights, const Matrix& biases,
        const Activation activation
    ) const {
        assert(right_logits.size() == right_activations.size());
        assert(right_logits.rows() == right_activations.rows());
        assert(right_logits.columns() == right_activations.columns());
        assert(left_activations.columns() == weights.rows());
        assert(right_activations.rows() == left_activations.rows());
        assert(right_activations.columns() == weights.columns());

        if (m_profiler) m_profiler->startTask("Propagation");

        switch (activation) {
            case Activation::SIGMOID:
                propagate_op<Sigmoid>(
                    left_activations.data(), right_logits.data(), right_activations.data(),
                    weights.data(), biases.data(),
                    left_activations.rows(), left_activations.columns(), right_activations.columns());
            break;

            case Activation::TANH:
                propagate_op<Tanh>(
                    left_activations.data(), right_logits.data(), right_activations.data(),
                    weights.data(), biases.data(),
                    left_activations.rows(), left_activations.columns(), right_activations.columns()
                );
            break;

            case Activation::RELU:
                propagate_op<ReLU>(left_activations.data(), right_logits.data(), right_activations.data(),
                    weights.data(), biases.data(),
                    left_activations.rows(), left_activations.columns(), right_activations.columns()
                );
            break;

            case Activation::LEAKY_RELU:
                propagate_op<LeakyReLU>(
                    left_activations.data(), right_logits.data(), right_activations.data(),
                    weights.data(), biases.data(),
                    left_activations.rows(), left_activations.columns(), right_activations.columns()
                );
            break;

            case Activation::SOFTMAX:
                propagate_op(left_activations.data(), right_logits.data(), right_activations.data(),
                    weights.data(), biases.data(),
                    left_activations.rows(), left_activations.columns(), right_activations.columns()
                );

                if (m_profiler) m_profiler->startTask("Softmax");
                softmax_op(right_logits.data(), right_activations.data(), right_logits.rows(), right_logits.columns());
                if (m_profiler) m_profiler->endTask("Softmax");
            break;

            default:
                propagate_op<NoActivation>(
                    left_activations.data(), right_logits.data(), right_activations.data(),
                    weights.data(), biases.data(),
                    left_activations.rows(), left_activations.columns(), right_activations.columns()
                );
            break;
        }

        if (m_profiler) m_profiler->endTask("Propagation");
    }
}