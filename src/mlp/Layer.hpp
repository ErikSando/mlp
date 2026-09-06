#pragma once

#include "context/Context.hpp"

namespace mlp {
    struct Layer {
        Layer(const Context& context, const size_t batch_size, const size_t node_count, const size_t previous_count, const Activation activation = Activation::NONE)
        : logits(context.createMatrix(batch_size, node_count)),
          activations(context.createMatrix(batch_size, node_count)),
          weights(context.createMatrix(previous_count, node_count)),
          biases(context.createMatrix(1, node_count)),
          activation(activation) {}

        // Layer(const size_t batch_size, const size_t node_count, const size_t previous_count, const Activation activation = Activation::NONE)
        // : logits(batch_size, node_count), activations(batch_size, node_count), weights(previous_count, node_count), biases(1, node_count), activation(activation) {}

        // void propagate(const Context& context, const Matrix& previous_layer) {
        //     context.propagate(previous_layer, logits, activations, weights, biases, activation);
        // }

        Matrix logits;
        Matrix activations;
        Matrix weights;
        Matrix biases;

        Activation activation;
    };
}