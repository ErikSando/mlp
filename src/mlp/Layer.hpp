#pragma once

#include "matrix/Matrix.hpp"

namespace mlp {
    enum class Activation {
        NONE,
        SIGMOID, TANH, RELU, LEAKY_RELU, // hidden layer activation functions
        SOFTMAX // output activation functions
    };

    struct Layer {
        Layer(const size_t batch_size, const size_t node_count, const size_t previous_count, const Activation activation = Activation::LEAKY_RELU)
        : nodes(batch_size, node_count), weights(previous_count, node_count), biases(1, node_count), activation(activation) {}

        Matrix nodes;
        Matrix weights;
        Matrix biases;

        Activation activation;

        void pass(const DeviceContext& context, const Matrix& previous_layer);
    };
}