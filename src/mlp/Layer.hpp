#pragma once

namespace mlp {
    template<typename TContext>
    struct Layer {
        using Matrix_t = typename TContext::Matrix_t;

        Layer(const size_t batch_size, const size_t node_count, const size_t previous_count, const Activation activation = Activation::LEAKY_RELU)
        : logits(batch_size, node_count), activations(batch_size, node_count), weights(previous_count, node_count), biases(1, node_count), activation(activation) {}

        void propagate(const TContext& context, const Matrix_t& previous_layer) {
            context.propagate(previous_layer, logits, activations, weights, biases, activation);
        }

        Matrix_t logits;
        Matrix_t activations;
        Matrix_t weights;
        Matrix_t biases;

        Activation activation;
    };
}