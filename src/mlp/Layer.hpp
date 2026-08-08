#pragma once

// #include "device/DeviceContext.hpp"

namespace mlp {
    template<typename TDeviceContext>
    struct Layer {
        using Matrix = typename TDeviceContext::Matrix;

        Layer(const size_t batch_size, const size_t node_count, const size_t previous_count, const Activation activation = Activation::LEAKY_RELU)
        : logits(batch_size, node_count), activations(batch_size, node_count), weights(previous_count, node_count), biases(1, node_count), activation(activation) {}

        // void propagate(const DeviceContext& context, const Matrix& previous_layer);
        void propagate(const TDeviceContext& context, const Matrix& previous_layer) {
            context.propagate(previous_layer, logits, activations, weights, biases, activation);
        }

        // Matrix nodes; // in the future, seperate activations and logits
        Matrix logits;
        Matrix activations;
        Matrix weights;
        Matrix biases;

        Activation activation;
    };
}