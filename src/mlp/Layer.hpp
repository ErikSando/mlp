#pragma once

#include "matrix/Matrix.hpp"

namespace mlp {
    struct Layer {
        Layer(const size_t batch_size, const size_t node_count, const size_t previous_count, const Activation activation = Activation::LEAKY_RELU)
        : nodes(batch_size, node_count), weights(previous_count, node_count), biases(1, node_count), activation(activation) {}

        void uploadNodes(const DeviceContext& context, const float* new_nodes);

        void propagate(const DeviceContext& context, const Matrix& previous_layer);

        Matrix nodes; // in the future, seperate activations and logits
        Matrix weights;
        Matrix biases;

        Activation activation;
    };
}